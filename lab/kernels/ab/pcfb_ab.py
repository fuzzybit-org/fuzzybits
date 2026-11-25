"""A/B of two fuzzybits runtime images on compiled .pcfb models kept on the docker host (for private models:
no model data leaves that host; only timings, n and hits come back).

Per round, base and head run in fresh containers in alternating order (A-B, B-A, ...). Each container
runs the bench harness's in-process runner (benchmarks/harness/execute.py INNER_INPROCESS): AdaptiveCpp
init, a warm-up run, then one timed fuzzybits.run. GPU runs hold the host's /tmp/gpu-lock-<host>-<gpu>.
Results (timings, n, hits) go to a local JSON file; the summary prints timing ratios only.

Usage: pcfb_ab.py REPO HOST WORKDIR TARGET OUT --base IMG --head IMG --models a b --trials T --warmup W --rounds R
       [--pin-cudart]. REPO is a fuzzybits checkout (for the harness's runner text), HOST an ssh host whose
       docker runs the images, WORKDIR the directory on HOST that holds <model>.pcfb, TARGET gpu:<nvidia-smi
       index>, hip (the host's AMD GPU, through scripts/ci/amd-docker-args.sh, lock /tmp/gpu-lock-HOST-amd-0) or
       cpu. --pin-cudart preloads the image's libcudart
       on both sides, so neither re-creates its CUDA context per run (AdaptiveCpp plugin load order).
       Commands go to HOST's bash on stdin, whatever its login shell.
"""
import argparse
import json
import math
import shlex
import statistics
import subprocess
import sys
import time

sys.path.insert(0, sys.argv.pop(1))  # fuzzybits checkout, for the harness's runner text
from benchmarks.harness.execute import INNER_INPROCESS  # noqa: E402

IMAGES = {}
AMD_ARGS = []
CUDART = "/usr/local/lib/python3.12/dist-packages/fuzzybits/lib/libcudart.so.12"
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def run_one(host, workdir, target, image, model, trials, warmup, timeout, pin_cudart):
    payload = {"backend": {"cpu": "omp", "hip": "hip"}.get(target, "cuda"), "pcfb": f"/work/{model}.pcfb", "trials": trials,
               "warmup_trials": warmup, "timeout": timeout}
    # The harness's container_env (benchmarks/harness/execute.py), as the image's default user.
    env = ["-e", "ACPP_DEBUG_LEVEL=0", "-e", "ACPP_ADAPTIVITY_LEVEL=1", "-e", "ACPP_PERSISTENT_RUNTIME=0",
           "-e", "ACPP_APPDB_DIR=/home/coder/.cache/fuzzybits/acpp"]
    if target == "cpu":
        dev = ["--cpuset-cpus", "0-7", "-e", "OMP_NUM_THREADS=8", "-e", "OMP_PROC_BIND=close", "-e",
               "OMP_PLACES=cores", "-e", "ACPP_VISIBILITY_MASK=omp"]
        lock = ""
    elif target == "hip":
        dev = AMD_ARGS + ["-e", "ACPP_VISIBILITY_MASK=hip"]
        lock = f"flock /tmp/gpu-lock-{host}-amd-0 "
    else:
        gpu = target.split(":")[1]
        dev = ["--gpus", f"device={gpu}", "-e", "ACPP_VISIBILITY_MASK=cuda"]
        if pin_cudart:  # keeps the CUDA context across AdaptiveCpp runtime restarts on both sides
            dev += ["-e", f"LD_PRELOAD={CUDART}"]
        lock = f"flock /tmp/gpu-lock-{host}-{gpu} "
    docker = (["docker", "run", "--rm", "-v", f"{workdir}:/work:ro"] + dev + env +
              ["--entrypoint", "python3", image, "-c", INNER_INPROCESS, json.dumps(payload)])
    cmd = lock + " ".join(shlex.quote(a) for a in docker)
    t0 = time.time()
    proc = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "bash", "-s"], input=cmd, capture_output=True,
                          text=True, timeout=timeout + 600)
    out = proc.stdout.strip().splitlines()
    try:
        res = json.loads(out[-1])
        inner = json.loads(res["stdout"])
        q = inner["queries"][0] if inner["queries"] else {}
        return {"sample_s": res["sample_s"], "rc": res["rc"], "n": q.get("n"), "hits": q.get("hits"),
                "container_s": time.time() - t0, "stderr": res.get("stderr", "")[-300:]}
    except Exception as exc:  # noqa: BLE001
        return {"sample_s": None, "rc": proc.returncode, "error": f"{exc}", "stderr": proc.stderr[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("workdir")
    ap.add_argument("target")
    ap.add_argument("out")
    ap.add_argument("--models", nargs="+", default=["a", "b"])
    ap.add_argument("--trials", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=1 << 20)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--pin-cudart", action="store_true")
    a = ap.parse_args()
    IMAGES.update(base=a.base, head=a.head)
    if a.target == "hip":
        script = open(sys.path[0] + "/scripts/ci/amd-docker-args.sh").read()
        proc = subprocess.run(["ssh", "-o", "BatchMode=yes", a.host, "bash", "-s"], input=script, capture_output=True,
                              text=True, check=True)
        AMD_ARGS.extend(proc.stdout.split())
    results = {"host": a.host, "target": a.target, "trials": a.trials, "warmup": a.warmup, "images": IMAGES,
               "pin_cudart": a.pin_cudart, "models": {}}
    for model in a.models:
        rows = []
        for r in range(a.rounds):
            order = ["base", "head"] if r % 2 == 0 else ["head", "base"]
            row = {"round": r}
            for side in order:
                row[side] = run_one(a.host, a.workdir, a.target, IMAGES[side], model, a.trials, a.warmup, a.timeout,
                                    a.pin_cudart)
            rows.append(row)
            b, h = row["base"].get("sample_s"), row["head"].get("sample_s")
            print(f"{a.target} model={model} round={r} base={b} head={h}", flush=True)
            json.dump(results | {"models": results["models"] | {model: rows}}, open(a.out, "w"), indent=1)
        results["models"][model] = rows
        logs = [math.log(x["head"]["sample_s"] / x["base"]["sample_s"]) for x in rows
                if x["head"].get("sample_s") and x["base"].get("sample_s")]
        if len(logs) >= 2:
            m, sd = statistics.mean(logs), statistics.stdev(logs)
            half = T_975.get(len(logs) - 1, 2.0) * sd / math.sqrt(len(logs))
            print(f"SUMMARY {a.target} model={model} ratio={math.exp(m):.4f} "
                  f"ci=[{math.exp(m - half):.4f}, {math.exp(m + half):.4f}] rounds={len(logs)} "
                  f"base_med={statistics.median(x['base']['sample_s'] for x in rows if x['base'].get('sample_s')):.3f}s "
                  f"head_med={statistics.median(x['head']['sample_s'] for x in rows if x['head'].get('sample_s')):.3f}s",
                  flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
