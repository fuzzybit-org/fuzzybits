#!/usr/bin/env bash
# Smoke every manylinux wheel in dist/ in its own clean Ubuntu 24.04 container
# (scripts/ci/linux-wheel-smoke.sh), one CPython version at a time.
# Versions: PYTHON_VERSIONS, else scripts/ci/python-versions.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ci/python-versions.sh
. "${here}/python-versions.sh"
for py in "${pythons[@]}"; do
  echo "=== linux smoke CPython ${py} ==="
  docker run --rm \
    -v "$PWD:/src" \
    -w /src \
    -e "PYTHON_VERSION=${py}" \
    ubuntu:24.04 \
    bash scripts/ci/linux-wheel-smoke.sh
done
