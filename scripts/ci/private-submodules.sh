#!/usr/bin/env bash
# Private submodules must stay out of CI, and no one else's job may reach them.
#
# A submodule is private when .gitmodules gives it `update = none` (extern/private-fixtures,
# extern/pra-solver-images). `git submodule update --init --recursive`, and so actions/checkout with
# `submodules: recursive`, skips it. This script checks:
#
# 1. Every other submodule is readable anonymously. A submodule that is not must be marked
#    `update = none`, or checkouts would try to fetch it.
# 2. In CI, no private submodule is populated.
# 3. The job's GITHUB_TOKEN cannot read a private submodule (any runner).
# 4. Who else could read one, by runner kind (RUNNER_ENVIRONMENT):
#    - github-hosted: fork pull requests run here, so ssh, runner git credentials and a gh login must
#      all fail;
#    - self-hosted: the runner account is trusted and may read them. What keeps other people's jobs
#      off the host is the job-started hook (scripts/ci/runner-job-started.sh), so it must be
#      configured. The account probes are reported, not enforced.
# Outside CI (no GITHUB_ACTIONS) it only reports.
# A public control repository must be readable, so a network failure cannot pass as "cannot read".
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

control="https://github.com/fuzzybit-org/fuzzybits.git"
in_ci="${GITHUB_ACTIONS:-false}"
kind="${RUNNER_ENVIRONMENT:-}"
if [[ "${in_ci}" != true ]]; then
  kind=local
elif [[ "${kind}" != self-hosted ]]; then
  kind=github-hosted # strict when unknown
fi
fail=0
private=()

bad() {
  if [[ "${kind}" == local ]]; then
    echo "WOULD FAIL in CI: $*"
  else
    echo "FAIL: $*"
    fail=1
  fi
}

with_timeout() {
  if command -v timeout >/dev/null; then timeout 60 "$@"; else "$@"; fi
}

# Runs a command with prompts disabled. Prints whether it read the repository (or its last error
# line) and succeeds only when it did.
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

# Reads a repository over https with no credentials at all: no global or system git config, no home.
anonymous() {
  local home
  home="$(mktemp -d)"
  set +e
  with_timeout env -u GH_TOKEN -u GITHUB_TOKEN HOME="${home}" GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
    GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false git -c credential.helper= ls-remote "$1" HEAD >/dev/null 2>&1
  local status=$?
  set -e
  rm -rf "${home}"
  return "${status}"
}

https_url() {
  local url="$1"
  case "${url}" in
    git@github.com:*) url="https://github.com/${url#git@github.com:}" ;;
    ssh://git@github.com/*) url="https://github.com/${url#ssh://git@github.com/}" ;;
  esac
  printf '%s' "${url}"
}

if ! with_timeout git ls-remote "${control}" HEAD >/dev/null 2>&1; then
  echo "FAIL: cannot read the public control ${control}; the probes below would prove nothing" >&2
  exit 1
fi
echo "runner kind: ${kind}"

if [[ -f .gitmodules ]]; then
  while read -r key path; do
    name="${key#submodule.}"
    name="${name%.path}"
    update="$(git config -f .gitmodules --get "submodule.${name}.update" || true)"
    url="$(git config -f .gitmodules --get "submodule.${name}.url" || true)"
    if [[ "${update}" == none ]]; then
      private+=("${path}|${url}")
    elif ! anonymous "$(https_url "${url}")"; then
      bad "${path}: ${url} is not readable anonymously; mark a private submodule 'update = none'"
    fi
  done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$')
fi

if [[ "${kind}" == self-hosted ]]; then
  hook="${ACTIONS_RUNNER_HOOK_JOB_STARTED:-}"
  if [[ -z "${hook}" || ! -f "${hook}" || -z "${FZB_RUNNER_ALLOWED_ACTORS:-}" ]]; then
    bad "this self-hosted runner has no job-started hook with an allowlist (ACTIONS_RUNNER_HOOK_JOB_STARTED, FZB_RUNNER_ALLOWED_ACTORS; see scripts/ci/runner-job-started.sh)"
  else
    echo "job-started hook: ${hook}"
  fi
fi

if ((${#private[@]} == 0)); then
  echo "no private submodules"
  exit "${fail}"
fi

ssh_cmd="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
account() {
  if [[ "${kind}" == github-hosted ]]; then
    bad "$*"
  else
    echo "      note: $* (trusted ${kind} account)"
  fi
}

for entry in "${private[@]}"; do
  path="${entry%%|*}"
  url="${entry#*|}"
  slug="$(https_url "${url}")"
  slug="${slug#https://github.com/}"
  slug="${slug%.git}"
  echo "${path} (${slug}):"
  if [[ -n "$(ls -A "${path}" 2>/dev/null)" ]]; then
    bad "${path} is populated in this checkout"
  fi

  echo "  ssh, runner keys:"
  if can_read GIT_SSH_COMMAND="${ssh_cmd}" git ls-remote "git@github.com:${slug}.git"; then
    account "${slug} is readable over ssh"
  fi

  echo "  https, runner credentials:"
  if can_read git ls-remote "https://github.com/${slug}.git"; then
    account "${slug} is readable over https with the runner's git credentials"
  fi

  if command -v gh >/dev/null; then
    echo "  gh CLI, runner login:"
    if can_read env -u GH_TOKEN -u GITHUB_TOKEN -u GH_ENTERPRISE_TOKEN gh api "repos/${slug}" --silent; then
      account "${slug} is readable through the gh CLI login"
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
echo "ok: ${#private[@]} private submodules (${kind} runner)"
