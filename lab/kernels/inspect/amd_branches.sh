#!/usr/bin/env bash
# Branches of each AMD kernel disassembly (llvm-objdump -d output saved as /lab/amdjit/*.jit.s), to find a
# kernel's main loop. Usage (inside the lab container): amd_branches.sh
cd /lab/amdjit
for f in *.jit.s; do
  echo "== $f"
  # Show branch instructions with line numbers and their targets.
  grep -nE "s_cbranch|s_branch|^[0-9a-f]+ <" "$f" | head -40
done
