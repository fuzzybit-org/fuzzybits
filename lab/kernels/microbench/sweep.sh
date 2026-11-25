#!/usr/bin/env bash
# Sweep the persistent variant's run length K and the work-group size. Run inside the lab container after
# build.sh. Usage: sweep.sh [BITPACKS]
B="${1:-131072}"
cd /lab
for wg in 128 256 512 1024; do
  line="wg=${wg}"
  for K in 8 16 32 64; do
    v=$(./bench --bitpacks "$B" --reps 7 --no-verify --wg "$wg" --persist "$K" pers16 2>&1 | awk '/^pers16/{print $3}')
    line="${line}  K${K}=${v}"
  done
  v=$(./bench --bitpacks "$B" --reps 7 --no-verify --wg "$wg" bs16 2>&1 | awk '/^bs16/{print $3}')
  echo "${line}  bs16=${v}"
done
