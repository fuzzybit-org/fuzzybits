#!/usr/bin/env bash
# Check (default) or apply (--fix) .clang-format on tracked C++ sources.
#
# The formatter is pinned to LLVM 20, the same major as the toolchain (toolchain.env) and Homebrew
# llvm@20 (scripts/dev/macos-dev.sh). It uses $CLANG_FORMAT when that is version 20.1.8, else
# runs the PyPI wheel through uvx.
#
# Usage: scripts/ci/clang-format.sh [--fix] [FILE...]
set -euo pipefail

version=20.1.8
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

fix=0
if [[ "${1:-}" == "--fix" ]]; then
  fix=1
  shift
fi

cf=()
if [[ -n "${CLANG_FORMAT:-}" && -x "${CLANG_FORMAT}" ]] \
    && [[ "$("${CLANG_FORMAT}" --version)" == *"version ${version}"* ]]; then
  cf=("${CLANG_FORMAT}")
elif command -v uvx >/dev/null 2>&1; then
  cf=(uvx --quiet "clang-format==${version}")
else
  echo "clang-format.sh: need CLANG_FORMAT (clang-format ${version}) or uvx" >&2
  exit 2
fi

# Vendored and generated code keeps its own style.
files=()
if (( $# )); then
  files=("$@")
else
  while IFS= read -r -d '' f; do
    files+=("${f}")
  done < <(git ls-files -z -- '*.h' '*.hpp' '*.cc' '*.cpp' '*.tpp' '*.inl' \
             ':!:src/ext/**' ':!:input/**' ':!:generated/**')
fi
(( ${#files[@]} )) || exit 0

if (( fix )); then
  printf '%s\0' "${files[@]}" | xargs -0 "${cf[@]}" --style=file -i
else
  printf '%s\0' "${files[@]}" | xargs -0 "${cf[@]}" --style=file --dry-run -Werror \
    || { echo "clang-format.sh: run scripts/ci/clang-format.sh --fix" >&2; exit 1; }
fi
