"""Throughput of the real pipeline (fuzzybits.Session) versus passes queued per host wait.

Session.step(n) enqueues n passes and waits once, which is what fuzzybits-server's
FUZZYBITS_PASSES_PER_STEP does; fuzzybits.run waits after every pass (n = 1).
Usage: python3 pps_sweep.py PCFB TOTAL_TRIALS [TRIALS_PER_PASS|auto] [N ...]
"""
import ctypes
import sys
import time

import fuzzybits

pcfb, total = sys.argv[1], int(sys.argv[2])
tpp = None if len(sys.argv) < 4 or sys.argv[3] == "auto" else int(sys.argv[3])
ns = [int(a) for a in sys.argv[4:]] or [1, 4, 16, 64, 0, 1]
ctypes.CDLL("/usr/local/lib/python3.12/dist-packages/fuzzybits/lib/libcudart.so.12")
fuzzybits.init(backend="cuda")
with fuzzybits.Session(pcfb, backend="cuda", trials_per_pass=tpp) as s:
    first = s.step(1)  # JIT and first pass
    per_pass = first.trials_per_pass
    passes = max(1, total // per_pass)
    shape = s.sample_shape
    keys = {k: shape[k] for k in shape if isinstance(shape[k], (int, float))}
    print(f"trials_per_pass={per_pass} passes={passes} shape={keys}", flush=True)
    for n in ns:
        step = passes if n == 0 else n
        t0 = time.perf_counter()
        done = 0
        while done < passes:
            k = min(step, passes - done)
            s.step(k)
            done += k
        dt = time.perf_counter() - t0
        print(f"passes_per_wait={step:5d} wall={dt:7.3f}s  {passes * per_pass / dt / 1e6:8.1f} Mtrials/s", flush=True)
