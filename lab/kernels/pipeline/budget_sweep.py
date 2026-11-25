"""Throughput versus the sample shaper's device budget. Needs a build with patches/shaper-budget.patch, which
reads FZB_LAB_MAX_DEVICE_BYTES in place of the 1 GiB MAX_ALLOCATABLE_DEVICE_BYTES.

For each budget (GiB): the pass shape a Session picks, then fuzzybits.run of TRIALS twice (the first run
after a change of budget may compile a new launch shape), printing wall time and Mtrials/s.
Usage: python3 cap_sweep.py PCFB TRIALS GIB [GIB ...]; a budget written fNN is NN% of the device memory free
just before that budget's runs (cudaMemGetInfo). The shaper rounds a pass down to a power of two bitpacks
per node, so the memory a pass actually uses is between half and all of the budget.
"""
import ctypes
import os
import sys
import time

import fuzzybits

pcfb, trials = sys.argv[1], int(sys.argv[2])
budgets = sys.argv[3:]
cudart = ctypes.CDLL("/usr/local/lib/python3.12/dist-packages/fuzzybits/lib/libcudart.so.12")


def free_total():
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    return free.value, total.value


fuzzybits.init(backend="cuda")
fuzzybits.run(pcfb, trials=1 << 20, backend="cuda")  # JIT
for spec in budgets:
    free, total = free_total()
    budget = int(float(spec[1:]) / 100 * free) if spec.startswith("f") else int(float(spec) * (1 << 30))
    gib = budget / (1 << 30)
    os.environ["FZB_LAB_MAX_DEVICE_BYTES"] = str(budget)
    with fuzzybits.Session(pcfb, backend="cuda", num_trials=trials) as s:
        shape = s.sample_shape
    tpp = shape["trials_per_pass"]
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        fuzzybits.run(pcfb, trials=trials, backend="cuda")
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"budget={spec:>4s}={gib:5.1f}GiB free={free / (1 << 30):5.1f}/{total / (1 << 30):4.1f}GiB trials_per_pass={tpp:>9d} passes={trials // tpp:>5d} "
          f"run_s={' '.join(f'{t:.2f}' for t in times)} best={best:.2f}s {trials / best / 1e6:6.1f} Mtrials/s",
          flush=True)
