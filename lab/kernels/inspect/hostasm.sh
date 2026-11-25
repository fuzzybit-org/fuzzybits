#!/usr/bin/env bash
# Summarise host JIT kernels in the AdaptiveCpp cache: per ELF .jit file and per exported kernel symbol.
dir="${1:-$(ls -d /root/.acpp/apps/*/jit-cache | head -1)}"
mkdir -p /lab/hostasm
for f in "${dir}"/*.jit; do
  head -c 4 "$f" | grep -q ELF || continue
  b=$(basename "$f" .jit)
  objdump -d --no-show-raw-insn "$f" > /lab/hostasm/${b}.s
  syms=$(nm -D --defined-only "$f" | awk '{print $3}' | grep -E 'kernel' | c++filt | grep -oE 'integral_constant<[a-z]+, [0-9a-z]+>|\$_[0-9]+' | sort -u | tr '\n' ' ')
  s=/lab/hostasm/${b}.s
  ins=$(grep -cE '^\s+[0-9a-f]+:' "$s")
  y=$(grep -c ymm "$s"); x=$(grep -c xmm "$s")
  mq=$(grep -c vpmuludq "$s"); ml=$(grep -c vpmulld "$s")
  sc=$(grep -cE '\s(imul|mul|mulx)[lq]?\s' "$s")
  echo "${b} [${syms}] ins=${ins} ymm=${y} xmm=${x} vpmuludq=${mq} vpmulld=${ml} scalar_mul=${sc}"
done
