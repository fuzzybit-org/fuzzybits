"""In-process timing of repeated fuzzybits.run calls (the A/B harness's sample_s path), per call.

Usage (inside a runtime container): python3 inproc_diag.py PCFB BACKEND TRIALS [REPEATS]
"""
import sys
import time

import fuzzybits

pcfb, backend, trials = sys.argv[1], sys.argv[2], int(sys.argv[3])
repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3
t0 = time.perf_counter()
fuzzybits.init(backend=backend)
print(f"init {time.perf_counter() - t0:.3f}s", flush=True)
for label, n in [("warmup", 1 << 20)] + [(f"timed{i}", trials) for i in range(repeats)] + [("small", 1 << 20)]:
    t0 = time.perf_counter()
    rows = fuzzybits.run(pcfb, trials=n, backend=backend)
    dt = time.perf_counter() - t0
    print(f"{label:8s} trials={n:>11d} {dt:8.3f}s n={rows[0].n}", flush=True)
