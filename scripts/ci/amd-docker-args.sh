#!/usr/bin/env bash
# Print the `docker run` arguments that give a container this host's AMD GPUs,
# one word per line (`mapfile -t args < <(scripts/ci/amd-docker-args.sh)`).
#
# Only /dev/kfd and the render nodes KFD maps to a GPU agent are passed (not
# all of /dev/dri, which may hold an iGPU), with their owning groups so a
# non-root container user can open them. Exits 1 when no AMD GPU is usable.
set -euo pipefail

topology=/sys/class/kfd/kfd/topology/nodes
if [[ ! -c /dev/kfd ]]; then
  echo "error: /dev/kfd missing (amdgpu KFD not loaded)" >&2
  exit 1
fi

args=(--device /dev/kfd)
gids=("$(stat -c %g /dev/kfd)")
gpus=0
for props in "${topology}"/*/properties; do
  gfx="$(awk '$1 == "gfx_target_version" {print $2}' "${props}")"
  minor="$(awk '$1 == "drm_render_minor" {print $2}' "${props}")"
  if [[ "${gfx:-0}" -le 0 || -z "${minor}" ]]; then
    continue
  fi
  node="/dev/dri/renderD${minor}"
  if [[ ! -c "${node}" ]]; then
    echo "warning: KFD agent ${props%/properties} has no ${node}" >&2
    continue
  fi
  echo "amd gpu: gfx_target_version=${gfx} ${node}" >&2
  args+=(--device "${node}")
  gids+=("$(stat -c %g "${node}")")
  gpus=$((gpus + 1))
done

if ((gpus == 0)); then
  echo "error: no KFD GPU agent with a render node" >&2
  exit 1
fi
for gid in $(printf '%s\n' "${gids[@]}" | sort -un); do
  args+=(--group-add "${gid}")
done
printf '%s\n' "${args[@]}"
