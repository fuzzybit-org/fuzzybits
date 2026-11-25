#!/usr/bin/env bash
# Register, spill and LDS figures from the AMDGPU notes of each JIT-compiled code object (the AdaptiveCpp
# app directory copied to /lab/amdjit). Usage (inside a ROCm toolchain container): amd_notes.sh
cd /lab/amdjit/bench-*/jit-cache
for f in *.jit; do
  echo "== $f"
  /usr/lib/llvm-20/bin/llvm-readelf --notes "$f" 2>/dev/null | grep -E "\.(vgpr_count|sgpr_count|wavefront_size|max_flat_workgroup_size|vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size|group_segment_fixed_size|agpr_count)" | tr -s ' ' | tr '\n' ' '
  echo
done
