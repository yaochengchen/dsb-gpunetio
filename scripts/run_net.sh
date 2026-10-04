#!/usr/bin/env bash
# Network sweep from the client host against ONE running server.
# Start the server first (on the server host), then e.g.
#   sudo TAG=doca   SERVER=58:a2:e1:db:db:80 IFACE=enP2s2np0 scripts/run_net.sh
#   sudo TAG=launch SERVER=58:a2:e1:db:db:80 IFACE=enP2s2np0 scripts/run_net.sh
#   sudo TAG=proxy  ...
# Appends SUMMARY lines to results/net.csv and keeps per-request CSVs.
# PARTS selects the groups (default all three): lat = unloaded grid, win =
# closed-loop windows 4/16/64, load = Poisson rates.  Repeats use a new tag
# <series>_r<k>; a tag already present in OUT is refused (never overwrite).
#   sudo TAG=doca_w2t_r3 PARTS=load RATES="400000 600000 700000 800000" scripts/run_net.sh
set -u
BIN=${BIN:-./bin/$(uname -m)}
TAG=${TAG:?set TAG=doca|launch|proxy}
SERVER=${SERVER:-58:a2:e1:db:db:80}
IFACE=${IFACE:-enP2s2np0}
OUT=${OUT:-results/net.csv}
RAW=${RAW:-results/raw}
TX_CPU=${TX_CPU:-2}; RX_CPU=${RX_CPU:-3}
COUNT=${COUNT:-100000}
# 1. unloaded latency (closed loop, one outstanding request).  The default set
#    is the scaling grid of the paper: int8 portfolio n x K at B=8, and pm1 n at
#    K=100, B=8 (limits at 48 KB/CTA: int8 n<=192, ternary n<=384, pm1 n<=512).
if [ -z "${LAT_CFGS:-}" ]; then
  LAT_CFGS=""
  for n in 64 128 192; do for K in 50 100 200 400; do LAT_CFGS+=" int8:$n:8:$K"; done; done
  for n in 64 128 256 512; do LAT_CFGS+=" pm1:$n:8:100"; done
  LAT_CFGS+=" ternary:256:8:100 int8:128:1:100 int8:128:32:100"
fi
# 2. load: open loop Poisson at increasing rates for one configuration
LOAD_CFG=${LOAD_CFG:-"int8:128:8:100"}
RATES=${RATES:-"10000 50000 100000 200000 400000 800000"}
PARTS=${PARTS:-"lat win load"}
has() { case " $PARTS " in *" $1 "*) return 0;; esac; return 1; }
mkdir -p "$RAW" "$(dirname "$OUT")"
if [ -f "$OUT" ] && grep -q "^SUMMARY,$TAG," "$OUT"; then
  echo "tag $TAG already in $OUT; use a new tag (<series>_r<k>)"; exit 1
fi
[ -f "$OUT" ] || echo "SUMMARY,tag,format,n,B,K,models,rate,window,ok,err,lost,kreq_s,p50_us,p99_us,p999_us,mean_us,srv_p50_us,srv_p99_us,net_p50_us,verify_bad,ref_bad" > "$OUT"
run() {  # fmt n B K extra... ; tag suffix
  local f=$1 n=$2 B=$3 K=$4; shift 4
  local name="${TAG}_${f}_n${n}_B${B}_K${K}_$*"; name=${name// /_}
  "$BIN/dsb_client" --iface "$IFACE" --server "$SERVER" --format "$f" --n "$n" --B "$B" --K "$K" \
     --count "$COUNT" --tx-cpu "$TX_CPU" --rx-cpu "$RX_CPU" --tag "$TAG" --csv "$RAW/$name.csv" "$@" \
     | tee "$RAW/$name.log" | grep '^SUMMARY' >> "$OUT"
}
if has lat; then for c in $LAT_CFGS; do IFS=: read -r f n B K <<< "$c"; run "$f" "$n" "$B" "$K" --window 1; done; fi
if has win; then for w in 4 16 64; do IFS=: read -r f n B K <<< "$LOAD_CFG"; run "$f" "$n" "$B" "$K" --window "$w"; done; fi
if has load; then for r in $RATES; do IFS=: read -r f n B K <<< "$LOAD_CFG"; run "$f" "$n" "$B" "$K" --rate "$r" --poisson; done; fi
echo "appended to $OUT"
