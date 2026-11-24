#!/usr/bin/env bash
# Private submodules (extern/) must stay out of CI.
#
# 1. A submodule is under extern/ exactly when .gitmodules gives it `update = none`. Then
#    `git submodule update --init --recursive` (actions/checkout `submodules: recursive`) skips it.
# 2. No such submodule is populated in this checkout.
# 3. This runner cannot read any of them:
#    - over ssh, with the runner user's keys and ssh config;
#    - over https, with the git credentials configured on the runner;
#    - through a gh CLI login on the runner;
#    - over https, with the job's token when GITHUB_TOKEN is set.
#    Any probe that can read one fails the job. A public control repository must be readable, so a
#    network failure cannot pass as "cannot read".
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

control="https://github.com/fuzzybit-org/fuzzybits.git"
fail=0
private=()

bad() {
  echo "FAIL: $*"
  fail=1
}

with_timeout() {
  if command -v timeout >/dev/null; then timeout 60 "$@"; else "$@"; fi
}

# Runs `git ls-remote` with the given environment and git options. Prints the last error line and
# succeeds only when the repository could be read.
can_read() {
  local out status
  set +e
  out="$(with_timeout env GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false SSH_ASKPASS=/bin/false "$@" 2>&1 >/dev/null)"
  status=$?
  set -e
  if ((status == 0)); then
    echo "      readable"
  else
    printf '      %s\n' "$(printf '%s\n' "${out:-exit ${status}}" | tail -n 1)"
  fi
  return "${status}"
}

if [[ -f .gitmodules ]]; then
  while read -r key path; do
    name="${key#submodule.}"
    name="${name%.path}"
    update="$(git config -f .gitmodules --get "submodule.${name}.update" || true)"
    url="$(git config -f .gitmodules --get "submodule.${name}.url" || true)"
    if [[ "${path}" == extern/* ]]; then
      [[ "${update}" == none ]] || bad "${path}: submodules under extern/ need 'update = none' (has '${update:-unset}')"
      private+=("${path}|${url}")
    elif [[ "${update}" == none ]]; then
      bad "${path}: 'update = none' marks a private submodule, which belongs under extern/"
    fi
  done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$')
fi

if ((${#private[@]} == 0)); then
  echo "ok: no private submodules"
  exit "${fail}"
fi

if ! with_timeout git ls-remote "${control}" HEAD >/dev/null 2>&1; then
  echo "FAIL: cannot read the public control ${control}; the probes below would prove nothing" >&2
  exit 1
fi

ssh_cmd="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
for entry in "${private[@]}"; do
  path="${entry%%|*}"
  url="${entry#*|}"
  if [[ -n "$(ls -A "${path}" 2>/dev/null)" ]]; then
    bad "${path} is populated in this checkout"
  fi
  slug="${url#git@github.com:}"
  slug="${slug#https://github.com/}"
  slug="${slug%.git}"
  echo "${path} (${slug}):"

  echo "  ssh, runner keys:"
  if can_read GIT_SSH_COMMAND="${ssh_cmd}" git ls-remote "git@github.com:${slug}.git"; then
    bad "${slug} is readable over ssh from this runner"
  fi

  echo "  https, runner credentials:"
  if can_read git ls-remote "https://github.com/${slug}.git"; then
    bad "${slug} is readable over https with this runner's git credentials"
  fi

  if command -v gh >/dev/null; then
    echo "  gh CLI, runner login:"
    if can_read env -u GH_TOKEN -u GITHUB_TOKEN -u GH_ENTERPRISE_TOKEN gh api "repos/${slug}" --silent; then
      bad "${slug} is readable through the gh CLI login on this runner"
    fi
  fi

  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    echo "  https, job token:"
    auth="$(printf 'x-access-token:%s' "${GITHUB_TOKEN}" | base64 | tr -d '\n')"
    if can_read git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth}" \
      ls-remote "https://github.com/${slug}.git"; then
      bad "${slug} is readable with the job's GITHUB_TOKEN"
    fi
  fi
done

if ((fail)); then
  exit 1
fi
echo "ok: ${#private[@]} private submodules, none populated, none readable from this runner"
