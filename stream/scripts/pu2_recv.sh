#!/usr/bin/env bash
# PU2: GPUNetIO streaming receiver.  Usage (from the repository root):
#   sudo stream/scripts/pu2_recv.sh <tag> -N 10000000 -B 8 -K 10 -i mattis -s 42 [...]
# Everything after <tag> is passed to st_recv.  Output: results/stream/<tag>.{log,csv,chk}
# Defaults inside st_recv: GPU 0000:ab:00.0, NIC 0000:bd:00.0, generator MAC 08:20:e7:47:10:8e (PU3).
# The port is set to MTU 9000 (frames are 4096 B).  Its PHY counters are logged before, after, and
# every 5 s while st_recv runs ("# phy t=..." lines): rx_packets_phy counts EVERY frame that reached
# the port, whether DOCA steered it to the GPU queue or the kernel got it, so these lines tell
# "frames reach PU2" independently of st_recv's own rx_frames.
# Ctrl+C: st_recv handles it itself (prints its RUN line, shuts the queues down); this shell and tee
# ignore it, so the log stays complete and "# phy after" is always written.
set -u
TAG=${1:?tag}; shift
OUT=${OUT:-results/stream}
BIN=${BIN:-./bin/$(uname -m)}
NIC_PCI=${NIC_PCI:-0000:bd:00.0}
IF=$(ls /sys/bus/pci/devices/$NIC_PCI/net 2>/dev/null | head -1)
mkdir -p "$OUT"
LOG=$OUT/$TAG.log
phy() { [ -n "$IF" ] && ethtool -S "$IF" 2>/dev/null | egrep 'rx_packets_phy|rx_bytes_phy|rx_oversize_pkts_phy|rx_discards_phy|rx_crc_errors_phy|rx_out_of_buffer|rx_steer_missed|rx_wqe_err|rx_cqe_err|buf_discard|rx_buffer_passed|tx_packets_phy' | tr -s ' ' | paste -sd' '; }
rxphy() { ethtool -S "$IF" 2>/dev/null | awk '/rx_packets_phy:/{print $2}'; }
echo "# $(date -Is) st_recv $TAG $*" | tee -a "$LOG"
[ -n "$IF" ] && ip link set dev "$IF" mtu 9000 2>/dev/null
{ echo "# link: $(ip -br link show "$IF" 2>/dev/null)"; echo "# phy before: $(phy)"; } | tee -a "$LOG"
trap '' INT TERM
WATCH=
if [ -n "$IF" ]; then
	( p0=$(rxphy); t0=$SECONDS
	  while sleep 5; do
		p=$(rxphy); l="# phy t=$((SECONDS - t0)) s  rx_packets_phy $p  (+$((p - p0)) since start)"
		echo "$l"; echo "$l" >> "$LOG"
	  done ) 2>/dev/null &
	WATCH=$!
fi
"$BIN/st_recv" -o "$OUT/$TAG.csv" -c "$OUT/$TAG.chk" "$@" 2>&1 | tee -a "$LOG"
if [ -n "$WATCH" ]; then pkill -KILL -P "$WATCH" 2>/dev/null; kill -KILL "$WATCH" 2>/dev/null; wait "$WATCH" 2>/dev/null; fi
echo "# phy after:  $(phy)" | tee -a "$LOG"
