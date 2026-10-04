#!/usr/bin/env bash
# PU2: resident-matrix and CPU references (no network).  Produces the .chk files
# that the streamed runs must reproduce.
set -u
OUT=${OUT:-results/stream}; BIN=${BIN:-./bin/$(uname -m)}
mkdir -p "$OUT"
for inst in random mattis; do
  "$BIN/st_ref" -N 10000  -B 8 -K 20 -i $inst -s 42 --cpu -c "$OUT/ref_cpu_${inst}_1e4.chk" | tee "$OUT/ref_cpu_${inst}_1e4.log"
  "$BIN/st_ref" -N 10000  -B 8 -K 20 -i $inst -s 42       -c "$OUT/ref_gpu_${inst}_1e4.chk" | tee "$OUT/ref_gpu_${inst}_1e4.log"
  "$BIN/st_ref" -N 100000 -B 8 -K 20 -i $inst -s 42       -c "$OUT/ref_gpu_${inst}_1e5.chk" | tee "$OUT/ref_gpu_${inst}_1e5.log"
done
for inst in random mattis; do
  cmp "$OUT/ref_cpu_${inst}_1e4.chk" "$OUT/ref_gpu_${inst}_1e4.chk" && echo "OK: CPU == GPU resident at n=1e4 ($inst)" || echo "MISMATCH: CPU vs GPU resident at n=1e4 ($inst)"
done
