#!/usr/bin/env bash
# Build the cardinality and gate microbenchmarks inside a running toolchain container.
#
# Usage:
#   lab/kernels/cardinality/build.sh CONTAINER
#       bench (bench.cpp, k-of-n gates) and bench_gate (bench_gate.cpp, AND, OR, XOR, NAND, NOR, NOT, NULL).
#       Self-contained: card.h holds their kernels.
#   lab/kernels/cardinality/build.sh --network CONTAINER [SRC [BUILD]]
#       bench_network (bench_network.cpp), on the production kernels of src/mc/kernel (#274, #275). It needs a
#       checkout with them at SRC (default /src) configured in BUILD (default SRC/build/generic) inside
#       CONTAINER, as in a scripts/dev/ws.sh workspace container.
#
# docker uses the current context; set DOCKER_CONTEXT for another host. Then, for example:
#   docker exec CONTAINER bash -lc 'cd /lab/cardinality && ACPP_VISIBILITY_MASK=cuda ./bench --reps 15'
#   docker exec CONTAINER bash -lc 'cd /lab/cardinality && ACPP_VISIBILITY_MASK=omp ./bench_gate --B 16384 --reps 9'
#   docker exec CONTAINER bash -lc 'cd /lab/cardinality && ACPP_VISIBILITY_MASK=cuda ./bench_network 3:2 8:4 64:64'
# and summarize with summarize.py (bench, bench_gate), net_summary.py and netexp_summary.py (bench_network).
# AMD: build in a ROCm toolchain container and run the binary in a HIP runtime image with
# ACPP_VISIBILITY_MASK=hip.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

if [[ "${1:-}" == "--network" ]]; then
  ctr="$2"
  src="${3:-/src}"
  build="${4:-${src}/build/generic}"
  inc="-I${src} -I${src}/src -I${src}/src/mc -I${build} -I${build}/include"
  inc+=" -I${build}/_deps/pcfb-build/src/generated/include -I${src}/extern/pcfb/include"
  inc+=" -I${build}/_deps/flatbuffers-src/include -I${build}/_deps/libxml2-build -I${build}/_deps/libxml2-src/include"
  inc+=" -I${build}/_deps/indicators-src/include"
  docker exec "${ctr}" mkdir -p /lab/cardinality
  COPYFILE_DISABLE=1 tar -C "${here}" -cf - bench_network.cpp | docker exec -i "${ctr}" tar -xf - -C /lab/cardinality
  docker exec "${ctr}" bash -lc "cd /lab/cardinality && acpp -O3 -std=c++23 -Wno-psabi --acpp-targets=generic \
    -DNDEBUG -DPROJECT_SOURCE_DIR=\\\"${src}\\\" ${inc} -o bench_network bench_network.cpp"
  exit 0
fi

ctr="$1"
docker exec "${ctr}" mkdir -p /lab/cardinality
COPYFILE_DISABLE=1 tar -C "${here}" -cf - bench.cpp bench_gate.cpp card.h | docker exec -i "${ctr}" tar -xf - -C /lab/cardinality
docker exec "${ctr}" bash -lc 'cd /lab/cardinality \
  && acpp -O3 -std=c++20 -Wno-psabi --acpp-targets=generic -o bench bench.cpp \
  && acpp -O3 -std=c++20 -Wno-psabi --acpp-targets=generic -o bench_gate bench_gate.cpp'
