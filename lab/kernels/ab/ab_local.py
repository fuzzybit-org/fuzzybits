"""Run `python -m benchmarks ab` with locally built images (no registry pin).

The harness pins A/B images from a registry. The head image here is built locally from a tree object
and loaded on every bench host from one saved tarball, so this wrapper lets pin_image fall back to the
local image ID. With FZB_AB_PRELOAD_CUDART=1, CUDA containers of both sides preload the image's
libcudart, so neither side re-creates its CUDA context on every run (AdaptiveCpp unloads libcudart
with its plugins when the omp plugin happens to load first).
"""
import os
import sys

import benchmarks.harness.cli as cli
import benchmarks.harness.dockerutil as dockerutil

_pin_image = dockerutil.pin_image


def _pin_local(image, *, allow_local=False, context=None):
    return _pin_image(image, allow_local=True, context=context)


dockerutil.pin_image = _pin_local

if os.environ.get("FZB_AB_PRELOAD_CUDART") == "1":
    _container_env = cli.container_env
    _cudart = "/usr/local/lib/python3.12/dist-packages/fuzzybits/lib/libcudart.so.12"

    def _env(*, backend, extra=None):
        env = _container_env(backend=backend, extra=extra)
        if backend == "cuda":
            env["LD_PRELOAD"] = _cudart
        return env

    cli.container_env = _env

raise SystemExit(cli.main(sys.argv[1:]))
