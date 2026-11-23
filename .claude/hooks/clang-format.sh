#!/usr/bin/env bash
# PostToolUse hook: format C++ files Claude edits with the repository's .clang-format.
# Prefers the pinned LLVM 20 formatter (scripts/dev/macos-dev.sh installs llvm@20 and exports
# CLANG_FORMAT), then any clang-format on PATH. A no-op when none is found, for non-C++ files,
# and for vendored or generated code.
set -euo pipefail
cf=""
for c in "${CLANG_FORMAT:-}" /opt/homebrew/opt/llvm@20/bin/clang-format /usr/local/opt/llvm@20/bin/clang-format \
    "$(command -v clang-format 2>/dev/null || true)"; do
  if [[ -n "${c}" && -x "${c}" ]]; then
    cf="${c}"
    break
  fi
done
[[ -n "${cf}" ]] || exit 0
file="$(python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))' 2>/dev/null || true)"
[[ -n "${file}" && -f "${file}" ]] || exit 0
case "${file}" in
  */src/ext/*|*/_deps/*|*/build/*|*/input/*|*/generated/*) exit 0 ;;
  *.h|*.hpp|*.cc|*.cpp|*.tpp|*.inl) "${cf}" -i --style=file "${file}" ;;
esac
exit 0
