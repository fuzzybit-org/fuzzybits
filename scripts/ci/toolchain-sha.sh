#!/usr/bin/env bash
# Content address of the CI toolchain image (`<ACPP_REF>-generic-<sha12>`):
# the pins and the Dockerfile that consumes them. Hashing toolchain.env alone
# let a Dockerfile-only change overwrite an "immutable" tag in place.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cat "${root}/toolchain.env" "${root}/docker/toolchain.Dockerfile" | sha256sum | cut -c1-12
