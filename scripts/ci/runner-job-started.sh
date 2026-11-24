#!/usr/bin/env bash
# Job-started hook for self-hosted runners: only allowed people may run jobs on the host.
#
# Install a copy outside the runner's work directory and add to the runner's .env:
#   ACTIONS_RUNNER_HOOK_JOB_STARTED=/path/to/runner-job-started.sh
#   FZB_RUNNER_ALLOWED_ACTORS=<login> [<login> ...]
#   FZB_RUNNER_ALLOWED_REPOS=<owner/repo glob> [...]   (default: any repository)
# then restart the runner service. The runner runs this before the job's first step; a non-zero exit
# fails the job, so no workflow code from a rejected job runs. Workflow files cannot change it:
# a fork or a branch can edit `runs-on`, but not this file or the runner's .env.
#
# A job runs only if all of these hold:
# - FZB_RUNNER_ALLOWED_ACTORS is set (fail closed);
# - GITHUB_ACTOR and GITHUB_TRIGGERING_ACTOR (who started or re-ran it) are both allowed;
# - GITHUB_REPOSITORY matches FZB_RUNNER_ALLOWED_REPOS;
# - the event is not pull_request_target or workflow_run;
# - for pull_request, the head repository is the base repository (no fork code, even when approved).
# scripts/ci/private-submodules.sh checks in CI that self-hosted runners have this hook.
set -euo pipefail

reject() {
  echo "::error title=runner job guard::job rejected on this self-hosted runner: $*"
  exit 1
}

in_list() {
  local item="$1" pattern
  shift
  for pattern in "$@"; do
    # shellcheck disable=SC2053 # glob patterns are intended
    [[ "${item}" == ${pattern} ]] && return 0
  done
  return 1
}

read -r -a actors <<<"${FZB_RUNNER_ALLOWED_ACTORS:-}"
read -r -a repos <<<"${FZB_RUNNER_ALLOWED_REPOS:-*}"
((${#actors[@]} > 0)) || reject "FZB_RUNNER_ALLOWED_ACTORS is not set"

repo="${GITHUB_REPOSITORY:-}"
event="${GITHUB_EVENT_NAME:-}"
actor="${GITHUB_ACTOR:-}"
triggering="${GITHUB_TRIGGERING_ACTOR:-${actor}}"

[[ -n "${repo}" && -n "${event}" && -n "${actor}" ]] || reject "the job context is incomplete"
in_list "${repo}" "${repos[@]}" || reject "repository ${repo} is not allowed"
in_list "${actor}" "${actors[@]}" || reject "actor ${actor} is not allowed"
in_list "${triggering}" "${actors[@]}" || reject "triggering actor ${triggering} is not allowed"

case "${event}" in
  pull_request_target | workflow_run) reject "event ${event} is not allowed" ;;
  pull_request)
    [[ -r "${GITHUB_EVENT_PATH:-}" ]] || reject "no event payload for ${event}"
    if command -v jq >/dev/null; then
      head_repo="$(jq -r '.pull_request.head.repo.full_name // ""' "${GITHUB_EVENT_PATH}")"
    elif command -v python3 >/dev/null; then
      head_repo="$(python3 -c 'import json,sys; e=json.load(open(sys.argv[1])); print((((e.get("pull_request") or {}).get("head") or {}).get("repo") or {}).get("full_name") or "")' "${GITHUB_EVENT_PATH}")"
    else
      reject "neither jq nor python3 is available to read the event payload"
    fi
    [[ "${head_repo}" == "${repo}" ]] || reject "pull request head ${head_repo:-unknown} is not ${repo}"
    ;;
esac

echo "runner job guard: allowed ${event} by ${triggering} on ${repo}"
