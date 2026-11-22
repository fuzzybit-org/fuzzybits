#!/usr/bin/env bash
# In-image smoke: mcscram-smoke + fzb compile/run (chinese) + Python import.
# Every Monte Carlo invocation is wrapped in `timeout` (default 60s).
set -euo pipefail

TIMEOUT_SEC="${SMOKE_TIMEOUT:-60}"
FZB_TRIALS="${FZB_TRIALS:-1e8}"
FZB_OMP_TRIALS="${FZB_OMP_TRIALS:-1e6}"
INPUT_DIR="${SMOKE_INPUT_DIR:-/app/input}"
SMOKE_MC="${SMOKE_MC:-1}"
SMOKE_FZB_CUDA="${SMOKE_FZB_CUDA:-1}"
SMOKE_FZB_OMP="${SMOKE_FZB_OMP:-1}"
# AMD GPU via the HIP backend; needs `--device /dev/kfd --device /dev/dri` and
# the host's kfd/render group ids (`--group-add`).
SMOKE_FZB_HIP="${SMOKE_FZB_HIP:-0}"
FZB_HIP_TRIALS="${FZB_HIP_TRIALS:-${FZB_TRIALS}}"
SMOKE_Z_MAX="${SMOKE_Z_MAX:-4}"
CHINESE_REF="${CHINESE_REF:-1.17058e-3}"

echo "fuzzybits-image-smoke  timeout=${TIMEOUT_SEC}s  fzb_trials=${FZB_TRIALS}  fzb_omp_trials=${FZB_OMP_TRIALS}"
echo "uid=$(id -u) user=$(id -un) ACPP_APPDB_DIR=${ACPP_APPDB_DIR:-unset}"
test "$(id -u)" -ne 0
test -n "${ACPP_APPDB_DIR:-}"
test -w "${ACPP_APPDB_DIR}"
test -z "${PYTHONPATH:-}"
test -z "${LD_PRELOAD:-}"

echo "== binaries =="
command -v scram-cli
command -v fzb
command -v acpp-info
command -v mcscram-smoke
scram-cli --help >/dev/null
fzb --help >/dev/null
acpp-info -l || true
if [[ "${SMOKE_FZB_OMP}" == "1" ]]; then
  test -x /usr/lib/llvm-20/bin/ld.lld || test -x /usr/local/lib/hipSYCL/ext/llvm/bin/ld.lld
fi

echo "== python (import fuzzybits, no PYTHONPATH/LD_PRELOAD; native /usr/local/bin/fzb) =="
python3 - <<'PY'
import os, sys
assert not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset"
assert not os.environ.get("LD_PRELOAD"), "LD_PRELOAD must be unset"
magic = open("/usr/local/bin/fzb", "rb").read(4)
assert magic == b"\x7fELF", f"/usr/local/bin/fzb is not a native ELF (got {magic!r})"
print("native fzb ELF ok")
import fuzzybits
fuzzybits.init()
print("fuzzybits", fuzzybits.__version__,
      "variant", fuzzybits.active_variant(),
      "backend", fuzzybits.active_backend())
print("devices", fuzzybits.list_devices())
assert hasattr(fuzzybits, "compile") and hasattr(fuzzybits, "inspect")
print("fuzzybits ok")
PY
python3 -m fuzzybits._cli --help >/dev/null

if [[ "${SMOKE_MC}" == "1" ]]; then
  echo "== mcscram-smoke =="
  SMOKE_TIMEOUT="${TIMEOUT_SEC}" mcscram-smoke
fi

check_fzb_z() {
  local backend="$1" log="$2" n="$3"
  python3 - "$backend" "$log" "$n" "${CHINESE_REF}" "${SMOKE_Z_MAX}" <<'PY'
import json, math, sys
backend, log_path, n_s, pref_s, zmax_s = sys.argv[1:6]
pref = float(pref_s)
n = float(n_s)
zmax = float(zmax_s)
text = open(log_path, encoding="utf-8", errors="replace").read()
start = text.find("{")
if start < 0:
    sys.stderr.write(f"error: fzb {backend}: no JSON object in output\n")
    sys.exit(3)
data = json.loads(text[start:])
queries = data.get("queries") or []
if not queries:
    sys.stderr.write(f"error: fzb {backend}: no queries in JSON\n")
    sys.exit(3)
q = queries[0]
phat = float(q["estimate"])
n_out = float(q.get("n") or n)
se = math.sqrt(max(phat * (1.0 - phat), 0.0) / n_out) if n_out > 0 else float("nan")
z = (phat - pref) / se if se > 0 else (0.0 if phat == pref else float("inf"))
absz = abs(z)
status = "PASS" if absz <= zmax else "FAIL"
print(f"fzb {backend:4s} chinese  p̂={phat:.6e}  p_bdd={pref:.6e}  n={n_out:.0f}  |z|={absz:.3f}  {status}")
sys.exit(0 if absz <= zmax else 4)
PY
}

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

if [[ "${SMOKE_FZB_CUDA}" == "1" || "${SMOKE_FZB_OMP}" == "1" || "${SMOKE_FZB_HIP}" == "1" ]]; then
  echo "== fzb compile chinese =="
  fzb compile "${INPUT_DIR}/chinese.xml" -c 2 --no-kn -o "${tmp}/chinese.pcfb"
fi

if [[ "${SMOKE_FZB_CUDA}" == "1" ]]; then
  echo "== fzb run chinese cuda =="
  log="${tmp}/cuda.json"
  timeout "${TIMEOUT_SEC}" fzb run "${tmp}/chinese.pcfb" --trials "${FZB_TRIALS}" --backend cuda --json | tee "${log}"
  check_fzb_z cuda "${log}" "${FZB_TRIALS}"
fi

if [[ "${SMOKE_FZB_HIP}" == "1" ]]; then
  # fzb's queue selector scores every non-HIP device -1 under a `hip` mask, so
  # a successful run cannot have silently fallen back to the OpenMP host.
  echo "== HIP devices =="
  case ";${ACPP_VISIBILITY_MASK:-};" in
    *";hip;"* | ";;") ;;
    *) echo "error: ACPP_VISIBILITY_MASK=${ACPP_VISIBILITY_MASK} does not load the HIP backend" >&2; exit 5 ;;
  esac
  ls -l /dev/kfd /dev/dri
  info="$(ACPP_DEBUG_LEVEL=2 acpp-info -l 2>&1)"
  printf '%s\n' "${info}"
  if ! printf '%s\n' "${info}" | awk '/^Loaded backend [0-9]+: HIP/{hip=1; next} /^Loaded backend/{hip=0} hip && /Found device/{ok=1} END{exit !ok}'; then
    echo "error: AdaptiveCpp enumerated no HIP device" >&2
    exit 5
  fi

  echo "== fzb run chinese hip =="
  log="${tmp}/hip.json"
  timeout "${TIMEOUT_SEC}" fzb run "${tmp}/chinese.pcfb" --trials "${FZB_HIP_TRIALS}" --backend hip --json | tee "${log}"
  check_fzb_z hip "${log}" "${FZB_HIP_TRIALS}"

  echo "== python fuzzybits on hip =="
  timeout "${TIMEOUT_SEC}" python3 - "${tmp}/chinese.pcfb" "${FZB_HIP_TRIALS}" "${CHINESE_REF}" "${SMOKE_Z_MAX}" <<'PY'
import math, sys
import fuzzybits
pcfb, trials, pref, zmax = sys.argv[1], int(float(sys.argv[2])), float(sys.argv[3]), float(sys.argv[4])
fuzzybits.init(backend="hip")
devices = fuzzybits.list_devices()
gpus = [d for d in devices if d["backend"] == "hip"]
print("hip devices", [(d["name"], d["vendor"], d["version"]) for d in gpus])
assert gpus, f"no hip device in {devices}"
assert fuzzybits.active_backend() == "hip", fuzzybits.active_backend()
(q,) = fuzzybits.run(pcfb, trials=trials, backend="hip")
z = abs(q.estimate - pref) / math.sqrt(q.estimate * (1.0 - q.estimate) / q.n)
print(f"python hip chinese  p̂={q.estimate:.6e}  n={q.n}  |z|={z:.3f}")
assert z <= zmax, f"|z|={z:.3f} > {zmax}"
PY
fi

if [[ "${SMOKE_FZB_OMP}" == "1" ]]; then
  echo "== fzb run chinese omp =="
  log="${tmp}/omp.json"
  timeout "${TIMEOUT_SEC}" fzb run "${tmp}/chinese.pcfb" --trials "${FZB_OMP_TRIALS}" --backend omp --json | tee "${log}"
  check_fzb_z omp "${log}" "${FZB_OMP_TRIALS}"
fi

echo "IMAGE SMOKE OK"
