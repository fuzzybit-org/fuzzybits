#!/usr/bin/env bash
# Remote build workspaces: build and test a checkout inside the pinned toolchain image on a
# Docker host, one container per workspace. The Docker daemon may be remote, so nothing is
# bind-mounted: the checkout's tracked and untracked (not ignored) files are copied into /src.
# See scripts/dev/README.md.
#
# Usage: scripts/dev/ws.sh [--context CTX] COMMAND NAME [options] [-- command...]
#
# Commands:
#   up NAME                      create and start the workspace container
#   sync NAME                    copy the checkout into /src
#   run NAME -- CMD...           sync, then run CMD (bash -c) in /src
#   configure NAME [PRESET]      sync, then scripts/ci/inner-build.sh PRESET configure
#   build NAME [TARGET...]       sync, then cmake --build --preset PRESET [--target ...]
#   test NAME [REGEX] [-- ARGS]  sync, then ctest --test-dir build/PRESET [-R REGEX] ARGS
#   shell NAME                   interactive shell in /src
#   status [NAME]                list workspaces
#   down NAME                    remove the workspace container (the ccache volume stays)
#   prune                        remove every workspace container owned by $USER
#   doctor                       check the Docker context, image and volume
#
# Options (sync, run, configure, build, test):
#   --worktree DIR   checkout to copy (default: the checkout this script is in)
#   --no-sync        do not copy files first
#   --clean          before copying, delete /src except build/ and .cache/ (after deletes, renames, rebases)
#   --fresh          before copying, delete all of /src including build/ (after toolchain or CMake cache changes)
#   --private[=DIR]  copy DIR (default $FUZZYBITS_PRIVATE_FIXTURES_DIR) to /private and set
#                    FUZZYBITS_PRIVATE_FIXTURES_DIR=/private for the command
#   --env KEY=VALUE  extra environment for the command (repeatable)
#   --tail N         print only the last N lines (the full log is kept, see FZB_WS_LOG_DIR)
#
# Environment:
#   FZB_WS_CONTEXT        Docker context of the build host (else DOCKER_HOST/DOCKER_CONTEXT, else the current context)
#   FZB_WS_PRESET         CMake preset (default generic)
#   FZB_WS_CPUS           CPUs per workspace and build parallelism (default 12)
#   FZB_WS_CCACHE_VOLUME  shared ccache volume (default fzb-agent-ccache)
#   FZB_WS_LOG_DIR        where command logs are kept (default ${XDG_STATE_HOME:-~/.local/state}/fzb-ws)
#   TOOLCHAIN_IMAGE       image override (default: the toolchain.env-pinned tag, else the moving tag)
set -euo pipefail
export COPYFILE_DISABLE=1

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
preset="${FZB_WS_PRESET:-generic}"
cpus="${FZB_WS_CPUS:-12}"
ccache_volume="${FZB_WS_CCACHE_VOLUME:-fzb-agent-ccache}"
log_root="${FZB_WS_LOG_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/fzb-ws}"
context="${FZB_WS_CONTEXT:-}"
label="org.fuzzybit.ws"

die() { echo "ws: $*" >&2; exit 2; }
note() { echo "ws: $*" >&2; }

dk() {
  if [[ -n "${context}" ]]; then
    docker --context "${context}" "$@"
  else
    docker "$@"
  fi
}

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

container_of() { echo "fzb-ws-$1"; }

check_name() {
  [[ "$1" =~ ^[a-z0-9][a-z0-9_.-]{0,62}$ ]] || die "workspace name must match [a-z0-9][a-z0-9_.-]*: '$1'"
}

resolve_image() {
  local wt="$1" acpp sha pinned moving
  if [[ -n "${TOOLCHAIN_IMAGE:-}" ]]; then
    echo "${TOOLCHAIN_IMAGE}"
    return
  fi
  [[ -f "${wt}/toolchain.env" ]] || die "no toolchain.env in ${wt}"
  acpp="$(sed -n 's/^ACPP_REF=//p' "${wt}/toolchain.env" | tail -1)"
  [[ -n "${acpp}" ]] || die "ACPP_REF missing from ${wt}/toolchain.env"
  if command -v sha256sum >/dev/null 2>&1; then
    sha="$(sha256sum "${wt}/toolchain.env" | cut -c1-12)"
  else
    sha="$(shasum -a 256 "${wt}/toolchain.env" | cut -c1-12)"
  fi
  pinned="ghcr.io/fuzzybit-org/fuzzybits-toolchain:${acpp}-generic-${sha}"
  moving="ghcr.io/fuzzybit-org/fuzzybits-toolchain:${acpp}-generic"
  if dk image inspect "${pinned}" >/dev/null 2>&1 || dk pull -q "${pinned}" >/dev/null 2>&1; then
    echo "${pinned}"
    return
  fi
  # A pin change in toolchain.env moves the hash before CI publishes that tag. The image does not
  # contain fetched dependencies (pcfb is fetched at configure time), so the moving tag works.
  note "${pinned} is not available; using ${moving}"
  if ! dk image inspect "${moving}" >/dev/null 2>&1; then
    dk pull -q "${moving}" >/dev/null || die "cannot pull ${moving} (docker login ghcr.io?)"
  fi
  echo "${moving}"
}

ensure_up() {
  local name="$1" wt="$2" cname image
  cname="$(container_of "${name}")"
  if ! dk inspect "${cname}" >/dev/null 2>&1; then
    image="$(resolve_image "${wt}")"
    dk create --name "${cname}" \
      --label "${label}=1" --label "${label}.name=${name}" --label "${label}.owner=${USER:-unknown}" \
      --label "${label}.checkout=$(basename "${wt}")" --label "${label}.image=${image}" \
      -v "${ccache_volume}:/ccache" \
      -e CCACHE_DIR=/ccache -e CCACHE_MAXSIZE=30G -e CCACHE_COMPILERCHECK=content \
      -e CCACHE_COMPILERTYPE=clang -e CCACHE_BASEDIR=/src -e CCACHE_NOHASHDIR=1 \
      -e ACPP_APPDB_DIR=/src/.cache/acpp -e ACPP_VISIBILITY_MASK=omp \
      -e OMP_NUM_THREADS="$(( cpus > 8 ? 8 : cpus ))" -e CMAKE_BUILD_PARALLEL_LEVEL="${cpus}" \
      -e PRESET="${preset}" -e BUILD_TESTING=ON \
      --cpus "${cpus}" --entrypoint sleep "${image}" infinity >/dev/null
    note "created ${cname} from ${image}"
  fi
  if [[ "$(dk inspect -f '{{.State.Running}}' "${cname}")" != "true" ]]; then
    dk start "${cname}" >/dev/null
  fi
}

# Runs bash -c "$1" in the container, serialised per workspace when flock is available.
in_ws() {
  local cname="$1" script="$2"
  shift 2
  dk exec "$@" "${cname}" bash -c "if command -v flock >/dev/null; then exec flock -w 7200 /tmp/fzb-ws.lock bash -c \"\$0\"; else exec bash -c \"\$0\"; fi" "${script}"
}

# Paths of the submodules with `update = none` in .gitmodules: the private repositories
# (extern/private-fixtures, extern/pra-solver-images). A workspace gets them only through --private.
private_submodules() {
  local wt="$1" key value
  [[ -f "${wt}/.gitmodules" ]] || return 0
  git -C "${wt}" config -f .gitmodules --get-regexp '^submodule\..*\.update$' 2>/dev/null |
    while read -r key value; do
      [[ "${value}" == none ]] || continue
      git -C "${wt}" config -f .gitmodules --get "${key%.update}.path" || true
    done
}

# Copies NUL-separated paths from stdin to stdout, dropping those that are, or are inside, an argument.
drop_paths() {
  local f p
  while IFS= read -r -d '' f; do
    for p in "$@"; do
      [[ "${f}" == "${p}" || "${f}" == "${p}/"* ]] && continue 2
    done
    printf '%s\0' "${f}"
  done
}

do_sync() {
  local name="$1" wt="$2" wipe="$3" cname p
  cname="$(container_of "${name}")"
  git -C "${wt}" rev-parse --show-toplevel >/dev/null 2>&1 || die "${wt} is not a git checkout"
  case "${wipe}" in
    clean) in_ws "${cname}" 'mkdir -p /src && cd /src && find . -mindepth 1 -maxdepth 1 ! -name build ! -name .cache -exec rm -rf {} +' ;;
    fresh) in_ws "${cname}" 'rm -rf /src && mkdir -p /src' ;;
  esac
  # git ls-files lists a submodule as its directory, and tar would copy a populated one whole.
  local skip=()
  while IFS= read -r p; do
    [[ -n "${p}" ]] && skip+=("${p}")
  done < <(private_submodules "${wt}")
  local tar_flags=(--null -T - -cf -)
  if tar --version 2>/dev/null | grep -q bsdtar; then
    tar_flags=(--no-mac-metadata --no-xattrs "${tar_flags[@]}")
  fi
  (cd "${wt}" && git ls-files -co --exclude-standard -z | drop_paths ${skip[@]+"${skip[@]}"} |
    tar "${tar_flags[@]}" 2>/dev/null) |
    in_ws "${cname}" 'mkdir -p /src && tar -C /src -xf -' -i
}

copy_private() {
  local cname="$1" dir="$2"
  [[ -d "${dir}" ]] || die "private fixtures directory not found: ${dir}"
  in_ws "${cname}" 'rm -rf /private && mkdir -m 700 /private'
  # The fixtures may be a git checkout (the extern/private-fixtures submodule): never copy its .git.
  local tar_flags=(--exclude=.git -cf - .)
  if tar --version 2>/dev/null | grep -q bsdtar; then
    tar_flags=(--no-mac-metadata --no-xattrs "${tar_flags[@]}")
  fi
  (cd "${dir}" && tar "${tar_flags[@]}") | in_ws "${cname}" 'tar -C /private -xf -' -i
}

run_logged() {
  local name="$1" tail_n="$2" script="$3"
  shift 3
  local cname log rc
  cname="$(container_of "${name}")"
  mkdir -p "${log_root}/${name}"
  log="${log_root}/${name}/$(date +%Y%m%d-%H%M%S)-$$.log"
  set +e
  if [[ -n "${tail_n}" ]]; then
    in_ws "${cname}" "${script}" -w /src "$@" >"${log}" 2>&1
    rc=$?
    tail -n "${tail_n}" "${log}"
  else
    in_ws "${cname}" "${script}" -w /src "$@" 2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
  fi
  set -e
  note "exit ${rc}; log ${log}"
  return "${rc}"
}

[[ $# -gt 0 ]] || usage 2
if [[ "$1" == "--context" ]]; then
  context="${2:?--context needs a value}"
  shift 2
fi
cmd="${1:-}"
shift || true

case "${cmd}" in
  -h|--help|help) usage 0 ;;
  status)
    filter=(--filter "label=${label}=1")
    [[ $# -gt 0 ]] && filter+=(--filter "label=${label}.name=$1")
    dk ps -a "${filter[@]}" --format "table {{.Label \"${label}.name\"}}\t{{.Status}}\t{{.Label \"${label}.checkout\"}}\t{{.Label \"${label}.owner\"}}\t{{.Label \"${label}.image\"}}"
    exit 0 ;;
  prune)
    ids="$(dk ps -aq --filter "label=${label}=1" --filter "label=${label}.owner=${USER:-unknown}")"
    # shellcheck disable=SC2086  # one id per word
    [[ -n "${ids}" ]] && dk rm -f ${ids} >/dev/null
    echo "removed $(echo "${ids}" | grep -c . || true) workspace(s)"
    exit 0 ;;
  doctor)
    dk version --format 'docker client {{.Client.Version}}, server {{.Server.Version}}' || die "cannot reach the Docker daemon (context: ${context:-current})"
    echo "context: ${context:-${DOCKER_CONTEXT:-$(docker context show 2>/dev/null || echo default)}}"
    echo "image: $(resolve_image "${script_root}")"
    if dk volume inspect "${ccache_volume}" >/dev/null 2>&1; then
      echo "ccache volume: ${ccache_volume}"
    else
      echo "ccache volume: ${ccache_volume} (created on first use)"
    fi
    echo "logs: ${log_root}"
    exit 0 ;;
esac

name="${1:-}"
[[ -n "${name}" ]] || die "missing workspace NAME (see --help)"
check_name "${name}"
shift
cname="$(container_of "${name}")"

case "${cmd}" in
  up) ensure_up "${name}" "${script_root}"; echo "${cname} up"; exit 0 ;;
  down) dk rm -f "${cname}" >/dev/null 2>&1 || true; echo "removed ${cname}"; exit 0 ;;
  shell) ensure_up "${name}" "${script_root}"; exec_tty=(-i); [[ -t 0 ]] && exec_tty=(-it); dk exec "${exec_tty[@]}" -w /src "${cname}" bash; exit $? ;;
  sync|run|configure|build|test) ;;
  *) die "unknown command '${cmd}' (see --help)" ;;
esac

worktree="${script_root}"
sync=1
wipe=""
private=""
tail_n=""
extra_env=()
positional=()
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree) worktree="$(cd "${2:?--worktree needs a value}" && pwd)"; shift 2 ;;
    --no-sync) sync=0; shift ;;
    --clean) wipe="clean"; shift ;;
    --fresh) wipe="fresh"; shift ;;
    --private) private="${FUZZYBITS_PRIVATE_FIXTURES_DIR:-}"; [[ -n "${private}" ]] || die "--private needs a directory or FUZZYBITS_PRIVATE_FIXTURES_DIR"; shift ;;
    --private=*) private="${1#--private=}"; shift ;;
    --env) extra_env+=(-e "${2:?--env needs KEY=VALUE}"); shift 2 ;;
    --tail) tail_n="${2:?--tail needs a number}"; shift 2 ;;
    --) shift; passthrough=("$@"); break ;;
    -*) die "unknown option $1" ;;
    *) positional+=("$1"); shift ;;
  esac
done

ensure_up "${name}" "${worktree}"
if [[ ${sync} -eq 1 ]]; then
  do_sync "${name}" "${worktree}" "${wipe}"
elif [[ -n "${wipe}" ]]; then
  die "--clean/--fresh need a sync"
fi
[[ "${cmd}" == "sync" ]] && { echo "synced $(basename "${worktree}") into ${cname}"; exit 0; }

env_args=(${extra_env[@]+"${extra_env[@]}"})
if [[ -n "${private}" ]]; then
  copy_private "${cname}" "${private}"
  env_args+=(-e FUZZYBITS_PRIVATE_FIXTURES_DIR=/private)
fi

quote() { local out="" a; for a in "$@"; do out+=" $(printf '%q' "${a}")"; done; echo "${out# }"; }

case "${cmd}" in
  run)
    [[ ${#passthrough[@]} -gt 0 ]] || die "run needs -- CMD"
    script="${passthrough[*]}" ;;
  configure)
    script="scripts/ci/inner-build.sh $(quote "${positional[0]:-${preset}}") configure" ;;
  build)
    targets=""
    for t in ${positional[@]+"${positional[@]}"}; do targets+=" --target $(quote "${t}")"; done
    script="cmake --build --preset $(quote "${preset}")${targets} -j ${cpus}" ;;
  test)
    regex=""
    [[ ${#positional[@]} -gt 0 ]] && regex=" -R $(quote "${positional[0]}")"
    script="ctest --test-dir build/$(quote "${preset}")${regex} --output-on-failure -j $(( cpus / 2 > 0 ? cpus / 2 : 1 )) $(quote ${passthrough[@]+"${passthrough[@]}"})" ;;
esac

run_logged "${name}" "${tail_n}" "${script}" ${env_args[@]+"${env_args[@]}"}
