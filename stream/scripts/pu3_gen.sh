#!/usr/bin/env bash
# PU3: streaming generator on GH200 GPU 1.  Usage (from the repository root):
#   sudo stream/scripts/pu3_gen.sh <tag> -N 10000000 -K 10 -i mattis -s 42 [-q 4] [-L 380] [...]
# Start the receiver on PU2 FIRST and wait for its "waiting for pass 0" line.  Output: results/stream/<tag>_gen.{log,csv}
# Defaults inside st_gen: NIC 0002:01:00.0, receiver MAC 58:a2:e1:db:db:80 (PU2 ens7np0).
# The port is set to MTU 9000 (frames are 4096 B).  Its PHY counters are logged before, after, and
# every 5 s while st_gen runs ("# phy t=..." lines: tx_packets_phy = frames that left the port,
# rx_packets_phy = frames that came back, i.e. the PASS_ACKs).
# Ctrl+C: st_gen handles it itself; this shell and tee ignore it, so "# phy after" is always written.
set -u
TAG=${1:?tag}; shift
# GH200 GPU 1 (the TVG's transmitter GPU) unless GEN_GPU is given
GEN_GPU=${GEN_GPU:-$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' '$1==1{print tolower($2)}' | sed 's/^0000//')}
[ -n "$GEN_GPU" ] || { echo "cannot find GPU 1 with nvidia-smi; set GEN_GPU=<PCIe address>"; exit 1; }
OUT=${OUT:-results/stream}
BIN=${BIN:-./bin/$(uname -m)}
NIC_PCI=${NIC_PCI:-0002:01:00.0}
IF=$(ls /sys/bus/pci/devices/$NIC_PCI/net 2>/dev/null | head -1)
mkdir -p "$OUT"
LOG=$OUT/${TAG}_gen.log
phy() { [ -n "$IF" ] && ethtool -S "$IF" 2>/dev/null | egrep 'tx_packets_phy|tx_bytes_phy|tx_errors_phy|tx_discards_phy|rx_packets_phy' | tr -s ' ' | paste -sd' '; }
txrx() { ethtool -S "$IF" 2>/dev/null | awk '/tx_packets_phy:/{t=$2} /rx_packets_phy:/{r=$2} END{print t, r}'; }
echo "# $(date -Is) st_gen $TAG (GPU $GEN_GPU) $*" | tee -a "$LOG"
[ -n "$IF" ] && ip link set dev "$IF" mtu 9000 2>/dev/null
{ echo "# link: $(ip -br link show "$IF" 2>/dev/null)"; echo "# phy before: $(phy)"; } | tee -a "$LOG"
trap '' INT TERM
WATCH=
if [ -n "$IF" ]; then
	( read -r t0 r0 < <(txrx); s0=$SECONDS
	  while sleep 5; do
		read -r t r < <(txrx)
		l="# phy t=$((SECONDS - s0)) s  tx_packets_phy $t (+$((t - t0)))  rx_packets_phy $r (+$((r - r0)) since start)"
		echo "$l"; echo "$l" >> "$LOG"
	  done ) 2>/dev/null &
	WATCH=$!
fi
"$BIN/st_gen" -g "$GEN_GPU" -o "$OUT/${TAG}_gen.csv" "$@" 2>&1 | tee -a "$LOG"
if [ -n "$WATCH" ]; then pkill -KILL -P "$WATCH" 2>/dev/null; kill -KILL "$WATCH" 2>/dev/null; wait "$WATCH" 2>/dev/null; fi
echo "# phy after:  $(phy)" | tee -a "$LOG"
