"""Private dataset entries in the harness (#184).

A private entry never reaches published output. Three rules follow from that:

1. **Discovery is content-addressed.** The entry holds the sorted sha256 of the
   model's ``.ftp`` files; :mod:`benchmarks.datasets.v2` matches it against the
   directories under ``$FUZZYBITS_PRIVATE_FIXTURES_DIR/ftp``. Zero or two
   matches is a skip with a reason code, never a path.
2. **Published output carries one line per entry and mode** —
   ``private <id> <mode>: pass|fail|skipped[ (<reason code>)]`` — and nothing
   else: no trial counts, estimates, timings or node counts.
3. **Details go to the private ``.bench.json``** next to the model, or nowhere.
   That file also holds the perf baseline, so a rerun compares against what the
   same host measured before.

Modes, as the manifest declares them:

``compile``
    ``fzb compile`` wall time at each level the compiler exposes for a CAFTA
    FTP project (``-c 0``: no ``ir::pc`` passes; ``-c 1``: the pass pipeline),
    with node and edge counts before and after each pass from ``--report``.
``perf``
    ``sample_s`` of ``fuzzybits.run`` after AdaptiveCpp init and a 2^20
    warm-up, as for public models, at the entry's trial count. The verdict
    compares the rounds against the baseline in the ``.bench.json`` (the first
    run records one) with the public CPU rule: warn above +5%, never block.
``accuracy-boolean``
    Every Boolean variant of the private card that carries a reference
    (a probability with an uncertainty) is judged at |z| ≤ 4, z from the
    fuzzybits estimate and the combined uncertainty. A variant whose card row
    has no reference is ``skipped (no-reference)``.
``hull``
    Reserved for #183 and rejected: ``skipped (hull)``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .execute import container_env, run_inner
from .stats import iqr, paired_log_ratio_ci
from .verdict import cpu_wall_verdict

# The private directory as ``ws.sh --private`` mounts it in the runtime container.
CONTAINER_ROOT = Path("/private")
CPU_WARN = 0.05
DEFAULT_TRIALS = 1 << 20
COMPILE_LEVELS = (0, 1)

# Runs inside the container. The model directory arrives as an argv value, so
# nothing under /private is ever interpolated into code or a shell string.
INNER = r"""
import json, os, subprocess, sys, time

payload = json.loads(sys.argv[1])
mode = payload["mode"]
work = "/work-private"
os.makedirs(work, exist_ok=True)
pcfb = os.path.join(work, "model.pcfb")


def run(cmd, timeout=None):
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"rc": 124, "stdout": "", "stderr": "timeout",
                "wall_s": time.perf_counter() - started}
    return {"rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
            "wall_s": time.perf_counter() - started}


def compile_at(level, overrides, duplicate_import):
    report = os.path.join(work, "report-c%d.json" % level)
    cmd = ["fzb", "compile", payload["model"], "-c", str(level),
           "--negation", payload["compile"]["negation"],
           "--duplicate-import", duplicate_import,
           "--complement-events", payload["compile"]["complement_events"],
           "--report", report, "-o", pcfb, "--quiet"]
    for override in overrides:
        cmd += ["--override", override]
    one = run(cmd, timeout=payload["compile"]["time_budget_s"])
    one["report"] = {}
    if os.path.isfile(report):
        try:
            with open(report) as handle:
                one["report"] = json.load(handle)
        except (OSError, ValueError):
            one["report"] = {}
        os.remove(report)
    inspected = run(["fzb", "inspect", pcfb, "--json"])
    one["inspect"] = {}
    if inspected["rc"] == 0:
        try:
            one["inspect"] = json.loads(inspected["stdout"])
        except ValueError:
            one["inspect"] = {}
    return one


SAMPLE = (
    "import json,sys,time\n"
    "import fuzzybits\n"
    "fuzzybits.init(backend='omp')\n"
    "pcfb=sys.argv[1]; trials=int(sys.argv[2]); warm=int(sys.argv[3])\n"
    "fuzzybits.run(pcfb, trials=warm, backend='omp')\n"
    "t0=time.perf_counter()\n"
    "rows=fuzzybits.run(pcfb, trials=trials, backend='omp')\n"
    "sample=time.perf_counter()-t0\n"
    "json.dump({'sample_s':sample,'queries':[{'name':r.name,'estimate':r.estimate,"
    "'ci_low':r.ci_low,'ci_high':r.ci_high,'n':r.n,'hits':r.hits} for r in rows]}, sys.stdout)\n"
)


def sample(trials, warmup, budget):
    one = run([sys.executable, "-c", SAMPLE, pcfb, str(int(trials)), str(int(warmup))],
              timeout=budget)
    if one["rc"] != 0:
        return {"error": one["stderr"][-500:]}
    try:
        return json.loads(one["stdout"])
    except ValueError:
        return {"error": "no JSON from the sample run"}


def summary(one, level=None):
    report = one["report"] or {}
    row = {
        "rc": one["rc"],
        "wall_s": one["wall_s"],
        "circuit": report.get("circuit"),
        "lowering": report.get("lowering"),
        "passes": report.get("passes"),
        "tensors": one["inspect"].get("tensors"),
        "operators": one["inspect"].get("operators"),
        "samples": one["inspect"].get("samples"),
        "opcodes": one["inspect"].get("opcodes"),
    }
    if level is not None:
        row["level"] = level
    return row


compile_block = payload["compile"]
base = compile_at(compile_block["level"], compile_block.get("overrides") or [],
                  compile_block["duplicate_import"])
result = {"mode": mode, "compile": summary(base), "ok": False}
if base["rc"] != 0:
    result["error"] = base["stderr"][-500:]
    json.dump(result, sys.stdout)
    sys.exit(0)

if mode == "compile":
    result["levels"] = [summary(compile_at(level, compile_block.get("overrides") or [],
                                           compile_block["duplicate_import"]), level)
                        for level in compile_block["levels"]]
    result["ok"] = all(row["rc"] == 0 for row in result["levels"])
elif mode == "perf":
    rounds = []
    for _ in range(int(payload["perf"]["rounds"])):
        one = sample(payload["perf"]["trials"], payload["perf"]["warmup"],
                     payload["perf"]["time_budget_s"])
        if "error" in one:
            result["error"] = one["error"]
            break
        rounds.append(one)
    result["rounds"] = rounds
    result["ok"] = len(rounds) == int(payload["perf"]["rounds"])
elif mode == "accuracy-boolean":
    runs = []
    for variant in payload["variants"]:
        built = compile_at(payload["accuracy"]["level"], variant.get("overrides") or [],
                           variant["duplicate_import"])
        if built["rc"] != 0:
            runs.append({"variant": variant["name"], "error": built["stderr"][-400:]})
            continue
        one = sample(payload["accuracy"]["trials"], payload["accuracy"]["warmup"],
                     payload["accuracy"]["time_budget_s"])
        if "error" in one:
            runs.append({"variant": variant["name"], "error": one["error"]})
            continue
        runs.append({"variant": variant["name"], "queries": one["queries"]})
    result["variants"] = runs
    result["ok"] = all("error" not in row for row in runs)
else:
    result["error"] = "unknown mode"

json.dump(result, sys.stdout)
"""


@dataclass
class PrivateOutcome:
    """One entry and mode. ``detail`` stays local; ``row`` may be published."""

    id: str
    mode: str
    verdict: str
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"id": self.id, "mode": self.mode, "verdict": self.verdict}
        if self.reason is not None:
            row["reason"] = self.reason
        return row

    @property
    def line(self) -> str:
        tail = f" ({self.reason})" if self.reason else ""
        return f"private {self.id} {self.mode}: {self.verdict}{tail}"


def overrides_from(variant: Mapping[str, Any]) -> list[str]:
    """``NAME=0|1`` words for the variant's overrides, sorted for determinism."""
    pairs = sorted((variant.get("overrides") or {}).items())
    return [f"{name}={int(bool(value))}" for name, value in pairs]


def duplicate_policy(variant: Mapping[str, Any], card: Mapping[str, Any]) -> str:
    """The variant's duplicate-import policy, else the card's. First word only:
    cards may annotate it (``last (measured with the vendor tools)``)."""
    block = variant.get("policy") or card.get("policy") or {}
    words = str(block.get("duplicate_import") or "last").split()
    word = words[0] if words else "last"
    return word if word in {"error", "first", "last"} else "last"


def boolean_variants(card: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The card's Boolean variants the harness can compile.

    A variant that needs a file edit, an extra delete term or an initiator rule
    is not expressible as compile flags, so it is left out; the FTP reference
    tests report those. ``reference`` is ``None`` for a variant whose card row
    has only an observed value.
    """
    out: list[dict[str, Any]] = []
    for name, variant in (card.get("variants") or {}).items():
        if not isinstance(variant, Mapping) or variant.get("semantics") != "boolean":
            continue
        if "edit" in variant or "extra_delete_term" in variant:
            continue
        if "max_initiators_per_cutset" in variant:
            continue
        reference = reference_of(variant)
        out.append(
            {
                "name": str(name),
                "overrides": overrides_from(variant),
                "duplicate_import": duplicate_policy(variant, card),
                "reference": reference,
            }
        )
    return out


def reference_of(variant: Mapping[str, Any]) -> dict[str, Any] | None:
    """``{"probability": p, "uncertainty": sd}`` for a variant, or None.

    The card's ``expected`` block: ``kind: exact`` has uncertainty 0, ``se`` is
    taken directly, and ``rel_tolerance_95`` becomes ``p·tol/1.96``. A variant
    with only an ``observed`` row has no independent reference (#159), so it
    cannot be judged.
    """
    expected = variant.get("expected")
    if not isinstance(expected, Mapping) or expected.get("probability") is None:
        return None
    probability = float(expected["probability"])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None
    if expected.get("kind") == "exact":
        return {"probability": probability, "uncertainty": 0.0}
    if expected.get("se") is not None:
        return {"probability": probability, "uncertainty": float(expected["se"])}
    tolerance = float(expected.get("rel_tolerance_95") or 0.0)
    return {"probability": probability, "uncertainty": probability * tolerance / 1.96}


def z_for(estimate: float, n: int, probability: float, uncertainty: float) -> float:
    """Wald z of a Monte Carlo estimate against a reference with its own
    uncertainty: ``(p̂ − p) / sqrt(p̂(1−p̂)/n + σ_ref²)``, using the null
    variance when ``p̂`` is 0 or 1."""
    if n <= 0:
        return float("nan")
    var = max(estimate * (1.0 - estimate), 0.0) / float(n)
    if var == 0.0:
        var = max(probability * (1.0 - probability), 0.0) / float(n)
    sigma = math.sqrt(var + max(uncertainty, 0.0) ** 2)
    if sigma == 0.0:
        if estimate == probability:
            return 0.0
        return math.copysign(float("inf"), estimate - probability)
    return (estimate - probability) / sigma


def accuracy_verdict(
    estimates: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    *,
    z_max: float = 4.0,
) -> tuple[str, str | None, dict[str, Any]]:
    """Judge every Boolean variant. ``(verdict, reason, detail)``."""
    judged: list[dict[str, Any]] = []
    worst = "pass"
    for variant, estimate in zip(variants, estimates, strict=False):
        reference = variant.get("reference")
        row: dict[str, Any] = {"variant": variant.get("name")}
        if not reference:
            row["verdict"] = "skipped"
            row["reason"] = "no-reference"
            judged.append(row)
            continue
        if "error" in estimate:
            row["verdict"] = "fail"
            row["error"] = estimate["error"]
            worst = "fail"
            judged.append(row)
            continue
        queries = estimate.get("queries") or []
        if not queries:
            row["verdict"] = "fail"
            worst = "fail"
            judged.append(row)
            continue
        query = queries[0]
        z = z_for(
            float(query["estimate"]),
            int(query["n"]),
            float(reference["probability"]),
            float(reference["uncertainty"]),
        )
        row |= {
            "estimate": float(query["estimate"]),
            "n": int(query["n"]),
            "hits": int(query["hits"]),
            "reference": float(reference["probability"]),
            "uncertainty": float(reference["uncertainty"]),
            "z": z,
            "verdict": "pass" if math.isfinite(z) and abs(z) <= z_max else "fail",
        }
        if row["verdict"] == "fail":
            worst = "fail"
        judged.append(row)
    detail = {"variants": judged}
    if not judged:
        return "skipped", "no-reference", detail
    if all(row.get("verdict") == "skipped" for row in judged):
        return "skipped", "no-reference", detail
    return worst, None, detail


def perf_verdict(
    samples: Sequence[float], baseline: Mapping[str, Any] | None
) -> tuple[str, str | None, dict[str, Any]]:
    """``(verdict, reason, detail)`` for one perf run against its baseline.

    The first run has none: it records one and passes. Later runs compare the
    median against the baseline median with the public CPU rule
    (``cpu_wall_verdict``: warn above +5%, never block). Fewer than two rounds
    on either side still gives a ratio, only without an interval.
    """
    if not samples:
        return "fail", "error", {"reason": "no samples"}
    detail: dict[str, Any] = {"samples": [float(value) for value in samples], "median_s": _median(samples)}
    old = [float(value) for value in (baseline or {}).get("samples") or ()]
    if not old:
        detail["baseline"] = "recorded"
        return "pass", None, detail
    detail |= {"baseline_median_s": _median(old), "baseline_rounds": len(old)}
    if len(old) < 2 or len(samples) < 2:
        ratio = _median(samples) / _median(old) if _median(old) > 0.0 else float("inf")
        detail["ratio"] = ratio
        return ("pass" if ratio <= 1.0 + CPU_WARN else "warn"), None, detail
    paired = min(len(old), len(samples))
    try:
        ratio, lo, hi, t_stat = paired_log_ratio_ci(old[:paired], list(samples)[:paired])
    except ValueError as exc:
        detail["reason"] = str(exc)
        return "fail", "error", detail
    detail |= {
        "ratio": ratio,
        "ratio_ci": [lo, hi],
        "t": t_stat,
        "iqr": iqr([h / b for b, h in zip(old[:paired], list(samples)[:paired], strict=True)]),
    }
    return cpu_wall_verdict(ratio, lo, hi), None, detail


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


class PrivateRunner:
    """Run private entries against one runtime image, keeping details local.

    The container sees the private directory either at :data:`CONTAINER_ROOT`
    (``ws.sh --private``) or wherever the caller points ``root``; the model
    directory is passed to the inner script as a positional argument.

    ``executor="local"`` runs the same inner script with the current
    interpreter, which is how the harness tests exercise the real code path
    without docker: ``fzb`` and ``fuzzybits`` must be on ``PATH``.
    """

    def __init__(
        self,
        *,
        context: str | None,
        image: str,
        volumes: Sequence[str],
        root: str | Path = CONTAINER_ROOT,
        env: Mapping[str, str] | None = None,
        backend: str = "omp",
        executor: str = "container",
    ) -> None:
        self.context = context
        self.image = image
        self.volumes = list(volumes)
        self.root = Path(root)
        self.env = dict(env or container_env(backend=backend))
        self.backend = backend
        self.executor = executor

    def model_path(self, directory: Path, local_root: str | Path) -> str:
        """The model directory as the container sees it."""
        try:
            return str(self.root / Path(directory).relative_to(local_root))
        except ValueError:
            return str(directory)

    def run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":"))
        if self.executor == "local":
            return self._run_local(body)
        result = run_inner(
            context=self.context,
            image=self.image,
            inner_cmd=["true"],
            volumes=self.volumes,
            env=self.env,
            gpus=None,
            timeout_s=int(payload["time_budget_s"]) + 60,
            inprocess={"mode": "script", "script": INNER, "payload": body},
        )
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error") or "inner runner failed"}
        document = result.get("document")
        if isinstance(document, Mapping):
            return dict(document)
        return _parse_document(result.get("stdout"))

    def _run_local(self, body: str) -> dict[str, Any]:
        import subprocess as sp

        try:
            proc = sp.run(
                [sys.executable, "-c", INNER, body],
                capture_output=True,
                text=True,
                check=False,
                timeout=int(json.loads(body)["time_budget_s"]) + 60,
                env={**os.environ, **self.env},
            )
        except (OSError, sp.TimeoutExpired) as exc:
            return {"ok": False, "error": f"local executor: {type(exc).__name__}"}
        if proc.returncode != 0:
            # Local executor only (tests): the tail explains a harness bug and
            # never reaches a published output, since only `detail` carries it.
            tail = (proc.stderr or "").strip().splitlines()[-6:]
            return {"ok": False, "error": "local executor: " + " / ".join(tail)}
        return _parse_document(proc.stdout)


def _parse_document(text: str | None) -> dict[str, Any]:
    try:
        document = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {"ok": False, "error": "unreadable inner result"}
    return document if isinstance(document, dict) else {"ok": False, "error": "inner result is not an object"}


def _compile_block(settings: Mapping[str, Any], *, level: int) -> dict[str, Any]:
    block = dict(settings.get("compile") or {})
    return {
        "level": int(level),
        "negation": str(block.get("negation") or "boolean"),
        "duplicate_import": str(block.get("duplicate_import") or "last"),
        "complement_events": str(block.get("complement_events") or "negation"),
        "overrides": list(block.get("overrides") or ()),
        "time_budget_s": int(block.get("time_budget_s") or 600),
    }


def compile_payload(model: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    levels = [int(level) for level in (settings.get("compile") or {}).get("levels") or COMPILE_LEVELS]
    block = _compile_block(settings, level=max(levels) if levels else 0)
    block["levels"] = levels
    return {
        "mode": "compile",
        "model": model,
        "compile": block,
        "time_budget_s": block["time_budget_s"] * (1 + len(levels)),
    }


def perf_payload(model: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    perf = dict(settings.get("perf") or {})
    block = _compile_block(settings, level=int(perf.get("level", 0)))
    budget = int(perf.get("time_budget_s") or 600)
    rounds = int(perf.get("rounds") or 5)
    return {
        "mode": "perf",
        "model": model,
        "compile": block,
        "perf": {
            "rounds": rounds,
            "trials": int(perf.get("trials") or DEFAULT_TRIALS),
            "warmup": int(perf.get("warmup") or DEFAULT_TRIALS),
            "time_budget_s": budget,
        },
        "time_budget_s": block["time_budget_s"] + budget * rounds,
    }


def accuracy_payload(
    model: str, settings: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    accuracy = dict(settings.get("accuracy") or {})
    block = _compile_block(settings, level=int(accuracy.get("level", 0)))
    block["negation"] = "boolean"
    budget = int(accuracy.get("time_budget_s") or 900)
    return {
        "mode": "accuracy-boolean",
        "model": model,
        "compile": block,
        "accuracy": {
            "level": block["level"],
            "trials": int(accuracy.get("trials") or DEFAULT_TRIALS),
            "warmup": int(accuracy.get("warmup") or DEFAULT_TRIALS),
            "time_budget_s": budget,
        },
        "variants": [dict(variant) for variant in variants],
        "time_budget_s": block["time_budget_s"] + budget * max(1, len(variants)),
    }


def bench_payload(settings: Mapping[str, Any], results: Mapping[str, Any]) -> dict[str, Any]:
    """The private ``.bench.json``: bench settings plus the recorded results."""
    payload = {key: value for key, value in settings.items() if key != "results"}
    payload["results"] = dict(results)
    return payload


def write_bench(path: Path, payload: Mapping[str, Any]) -> None:
    """Write the private results file. It never lives in the worktree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
