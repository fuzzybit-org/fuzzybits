#!/usr/bin/env bash
# Build a fuzzybits runtime image exactly as bench.yml's "Build and push head" step does
# (docker/runtime.Dockerfile --target runtime, generic SSCP, the immutable toolchain image that
# toolchain.env pins), from a commit or tree object plus extern/pcfb at the superproject's gitlink.
# The context is unpacked on the docker host (Docker 29 refuses `docker build -f FILE -` with a tar).
# Usage: build_image_ci.sh REPO HOST TAG TREEISH
set -euo pipefail
repo="$1"; host="$2"; tag="$3"; treeish="$4"
work="$(mktemp -d "${TMPDIR:-/tmp}/fzb-img.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
export COPYFILE_DISABLE=1
git -C "${repo}" archive --format=tar "${treeish}" | tar xf - -C "${work}"
pcfb_rev="$(git -C "${repo}" rev-parse "${treeish}:extern/pcfb")"
git -C "${repo}/extern/pcfb" archive --format=tar --prefix=extern/pcfb/ "${pcfb_rev}" | tar xf - -C "${work}"
set -a; . "${work}/toolchain.env"; set +a
toolchain_sha="$(cd "${work}" && bash scripts/ci/toolchain-sha.sh)"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "${work}/pyproject.toml" | head -1)"
toolchain="ghcr.io/fuzzybit-org/fuzzybits-toolchain:${ACPP_REF}-generic-${toolchain_sha}"
echo "toolchain=${toolchain} version=${version}"
remote="/tmp/fzb-kernels-ctx/${tag//[:\/]/_}"
ssh -o BatchMode=yes "${host}" "rm -rf '${remote}' && mkdir -p '${remote}'"
(cd "${work}" && tar cf - --no-xattrs .) | ssh -o BatchMode=yes "${host}" "tar xf - -C '${remote}' 2>/dev/null; find '${remote}' -name '._*' -delete"
ssh -o BatchMode=yes "${host}" "docker image inspect '${toolchain}' >/dev/null 2>&1 || docker pull -q '${toolchain}'; \
  docker build --target runtime -f '${remote}/docker/runtime.Dockerfile' \
  --build-arg TOOLCHAIN_IMAGE='${toolchain}' \
  --build-arg CMAKE_PRESET=generic --build-arg FUZZYBITS_VARIANT=generic --build-arg INCLUDE_SSCP_JIT=1 \
  --build-arg ACPP_TARGETS=generic --build-arg CUDA_EMBED_PTX=n/a --build-arg CLANG_VERSION=${CLANG_VERSION} \
  --build-arg ACPP_REF=${ACPP_REF} --build-arg ACPP_COMMIT=${ACPP_COMMIT} \
  --build-arg CUDA_VERSION=${CUDA_VERSION} --build-arg ROCM_VERSION=${ROCM_VERSION} \
  --build-arg FUZZYBITS_REVISION=${treeish} --build-arg FUZZYBITS_VERSION=${version} \
  --label org.opencontainers.image.revision=${treeish} --label dev.fuzzybits.variant=generic \
  -t '${tag}' '${remote}' 2>&1 | tail -5; rm -rf '${remote}'; docker image inspect '${tag}' --format '{{.Id}} {{.Size}}'"
