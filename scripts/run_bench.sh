#!/usr/bin/env bash
# GPU-only sweep (no NIC): single-launch, GPU-resident persistent pipeline and
# CPU-proxied persistent pipeline, on the server GPU.
#   GPU=0 scripts/run_bench.sh            -> results/bench_<host>.csv
# Reruns never overwrite: give them a new OUT (results/bench_<host>_r<k>.csv);
# analysis/make_tables.py reads it after the original and reports the newest row.
#   MODES="proxy" PROXY_ARGS="--window 4096 --drain" OUT=results/bench_pu2_r2.csv GPU=0 scripts/run_bench.sh
set -u
BIN=${BIN:-./bin/$(uname -m)}
GPU=${GPU:-0}
OUT=${OUT:-results/bench_$(hostname -s).csv}
FORMATS=${FORMATS:-"pm1 ternary int8"}
N_LIST=${N_LIST:-"64 128 192 256 512"}
B_LIST=${B_LIST:-"1 8 32"}
K_LIST=${K_LIST:-"50 100 200 400"}
COUNT=${COUNT:-200000}
RATES=${RATES:-"0"}          # persistent-mode offered rates (req/s), 0 = saturate
MODES=${MODES:-"single persistent proxy"}
PROXY_ARGS=${PROXY_ARGS:-""}   # e.g. "--window 4096 --drain"; empty = original proxy loop
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT (use a new OUT=..._r<k>.csv)"; exit 1; }
has() { case " $MODES " in *" $1 "*) return 0;; esac; return 1; }
mkdir -p "$(dirname "$OUT")"
"$BIN/dsb_bench" single --header --n 64 --B 1 --K 2 --reps 1 --device "$GPU" | head -1 > "$OUT"
for f in $FORMATS; do for n in $N_LIST; do for B in $B_LIST; do for K in $K_LIST; do
  args="--format $f --n $n --B $B --K $K --device $GPU"
  if has single; then
    "$BIN/dsb_bench" single $args --reps 500 >> "$OUT" 2>/dev/null || { echo "skip $args"; continue; }
  fi
  if has persistent; then
    for r in $RATES; do
      "$BIN/dsb_bench" persistent $args --count "$COUNT" --rate "$r" >> "$OUT" || { echo "skip $args (does not fit or failed, see message above)"; continue 2; }
    done
  fi
  if has proxy; then
    "$BIN/dsb_bench" proxy $args --count $((COUNT / 2)) $PROXY_ARGS >> "$OUT" || { echo "skip $args (does not fit or failed, see message above)"; continue; }
  fi
  echo "done $args"
done; done; done; done
echo "wrote $OUT"
