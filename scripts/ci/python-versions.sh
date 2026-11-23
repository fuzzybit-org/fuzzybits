# Sourced: sets the array `pythons` from PYTHON_VERSIONS, else from
# scripts/ci/python-versions (the one list every wheel job uses).
# shellcheck shell=bash
_fzb_versions_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/python-versions"
if [[ -n "${PYTHON_VERSIONS:-}" ]]; then
  read -r -a pythons <<< "${PYTHON_VERSIONS}"
else
  read -r -a pythons <<< "$(grep -v '^[[:space:]]*#' "${_fzb_versions_file}" | tr '\n' ' ')"
fi
unset _fzb_versions_file
if ((${#pythons[@]} == 0)); then
  echo "error: no CPython versions (PYTHON_VERSIONS and scripts/ci/python-versions are empty)" >&2
  exit 2
fi
