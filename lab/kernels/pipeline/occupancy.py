"""Time one fuzzybits.run inside a cudaProfilerStart/Stop range (nsys --capture-range=cudaProfilerApi).

Usage (runtime container, libcudart preloaded): python3 occupancy.py PCFB WARMUP_TRIALS TRIALS
"""
import ctypes
import sys
import time

import fuzzybits

pcfb, warm, trials = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cudart = ctypes.CDLL("/usr/local/lib/python3.12/dist-packages/fuzzybits/lib/libcudart.so.12")
fuzzybits.init(backend="cuda")
fuzzybits.run(pcfb, trials=warm, backend="cuda")
cudart.cudaProfilerStart()
t0 = time.perf_counter()
fuzzybits.run(pcfb, trials=trials, backend="cuda")
wall = time.perf_counter() - t0
cudart.cudaProfilerStop()
print(f"WALL {wall:.4f}", flush=True)
