#!/usr/bin/env bash
# PU2: kernel-stack receiver (baseline), AF_PACKET on ens7np0.  st_recv must NOT be running.
#   sudo stream/scripts/pu2_cpu_recv.sh <tag> -N 1000000 -B 8 -K 2 -t 8 [-G]
set -u
TAG=${1:?tag}; shift
IFACE=${IFACE:-ens7np0}
OUT=${OUT:-results/stream}; BIN=${BIN:-./bin/$(uname -m)}
mkdir -p "$OUT"
echo "# $(date -Is) st_cpu_recv $TAG $*" | tee -a "$OUT/$TAG.log"
"$BIN/st_cpu_recv" -I "$IFACE" -o "$OUT/$TAG.csv" "$@" 2>&1 | tee -a "$OUT/$TAG.log"
