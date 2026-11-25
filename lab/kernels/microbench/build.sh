#!/usr/bin/env bash
# Build the sampler microbenchmark inside a running toolchain container, with the production sampler headers
# from this checkout's src/mc/prng.
#
# Usage: lab/kernels/microbench/build.sh CONTAINER   (docker uses the current context; set DOCKER_CONTEXT for
# another host). Start the container from the toolchain image CI uses, with GPUs if the benchmark runs on one:
#   image="ghcr.io/fuzzybit-org/fuzzybits-toolchain:$(sed -n 's/^ACPP_REF=//p' toolchain.env)-generic-$(scripts/ci/toolchain-sha.sh)"
#   docker run -d --name kernels-lab --gpus all --cap-add SYS_ADMIN --entrypoint sleep "${image}" infinity
# Then: docker exec kernels-lab bash -lc 'cd /lab && ACPP_VISIBILITY_MASK=cuda ./bench --p 0.01'
set -euo pipefail
ctr="$1"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "${here}/../../.." && pwd)"
docker exec "${ctr}" mkdir -p /lab/src
COPYFILE_DISABLE=1 tar -C "${root}/src" -cf - mc/prng | docker exec -i "${ctr}" tar -xf - -C /lab/src
COPYFILE_DISABLE=1 tar -C "${here}" -cf - bench.cpp samplers.h sweep.sh | docker exec -i "${ctr}" tar -xf - -C /lab
COPYFILE_DISABLE=1 tar -C "${here}/../inspect" -cf - . | docker exec -i "${ctr}" tar -xf - -C /lab
docker exec "${ctr}" bash -lc 'cd /lab && acpp -O3 -std=c++20 --acpp-targets=generic -I src -o bench bench.cpp'
