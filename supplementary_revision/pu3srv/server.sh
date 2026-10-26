#!/usr/bin/env bash
# pu3srv/server.sh -- PU3: the request-service server on the GH200, on PU3's ConnectX-7 (0002:01:00.0,
# enP2s2np0, the NIC st_gen uses).  The client runs on PU2 (ens7np0) over the existing cable: the roles of
# RUN.md step 4 swapped.  Same binaries and options as on PU2; only the addresses differ.
# Start it BEFORE the client on PU2; stop it with Ctrl+C.
#
#   sudo bash pu3srv/server.sh <tag>                 MODE=doca (default): dsb_doca_server -W 2 (= doca_w2t of the paper)
#   sudo MODE=proxy  bash pu3srv/server.sh <tag>     dsb_base_server --mode proxy
#   sudo MODE=launch bash pu3srv/server.sh <tag>     dsb_base_server --mode launch --streams 4
#   anything after <tag> is passed to the server
#
# GPU: nvidia-smi index SRV_GPU_INDEX (default 1 = the GH200 GPU stream/scripts/pu3_gen.sh uses with this NIC);
#      SRV_GPU=<PCIe address> overrides it for the DOCA server.
# CPU: server host thread on core SRV_CPU (default 2, as taskset -c 2 for the baselines on PU2).
# Log: results/pu3srv/server_<tag>.log (appended), first lines = GPU / driver / NIC firmware / NUMA for Table 2.
set -u
TAG=${1:?usage: sudo bash pu3srv/server.sh <tag> [server options]}; shift
MODE=${MODE:-doca}
BIN=${BIN:-./bin/$(uname -m)}
OUT=${OUT:-results/pu3srv}
NIC_PCI=${NIC_PCI:-0002:01:00.0}
SRV_GPU_INDEX=${SRV_GPU_INDEX:-1}
SRV_CPU=${SRV_CPU:-2}
TXMODE=${TXMODE:-2}
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # CUDA device index == nvidia-smi index (for --device of dsb_base_server)

GPU_BUS=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v g="$SRV_GPU_INDEX" '$1==g{print tolower($2)}')
# same conversion as stream/scripts/pu3_gen.sh: 00000009:01:00.0 -> 0009:01:00.0
SRV_GPU=${SRV_GPU:-$(echo "$GPU_BUS" | sed 's/^0000//')}
[ -n "$SRV_GPU" ] || { echo "cannot find GPU $SRV_GPU_INDEX with nvidia-smi; set SRV_GPU=<PCIe address>"; exit 1; }
IF=$(ls /sys/bus/pci/devices/$NIC_PCI/net 2>/dev/null | head -1)
[ -n "$IF" ] || { echo "no netdev for NIC $NIC_PCI"; exit 1; }
mkdir -p "$OUT"
LOG=$OUT/server_${TAG}.log

if pgrep -a -f 'dsb_doca_server|dsb_base_server|st_gen|st_recv' >/dev/null; then
	echo "another server / DOCA program is running on this host (one at a time, RUN.md step 4):"
	pgrep -a -f 'dsb_doca_server|dsb_base_server|st_gen|st_recv'; exit 1
fi

{
	echo "# $(date -Is) $(hostname -s) server MODE=$MODE tag=$TAG args: $*"
	echo "# GPU index $SRV_GPU_INDEX = $SRV_GPU ($(nvidia-smi -i "$SRV_GPU_INDEX" --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader 2>/dev/null)), numa_node $(cat /sys/bus/pci/devices/$SRV_GPU/numa_node 2>/dev/null)"
	echo "# NIC $NIC_PCI = $IF  MAC $(cat /sys/class/net/$IF/address)  MTU $(cat /sys/class/net/$IF/mtu)  numa_node $(cat /sys/bus/pci/devices/$NIC_PCI/numa_node 2>/dev/null)  fw $(ethtool -i "$IF" 2>/dev/null | awk '/firmware-version/{print $2}')"
	echo "# server host thread on core $SRV_CPU"
} | tee -a "$LOG"

trap '' INT TERM          # the servers handle Ctrl+C themselves; keep tee alive so the log is complete
case "$MODE" in
	doca)
		taskset -c "$SRV_CPU" "$BIN/dsb_doca_server" -g "$SRV_GPU" -n "$NIC_PCI" -W "$TXMODE" "$@" 2>&1 | tee -a "$LOG" ;;
	proxy)
		taskset -c "$SRV_CPU" "$BIN/dsb_base_server" --iface "$IF" --mode proxy --device "$SRV_GPU_INDEX" "$@" 2>&1 | tee -a "$LOG" ;;
	launch)
		taskset -c "$SRV_CPU" "$BIN/dsb_base_server" --iface "$IF" --mode launch --streams 4 --device "$SRV_GPU_INDEX" "$@" 2>&1 | tee -a "$LOG" ;;
	*)
		echo "MODE must be doca, proxy or launch"; exit 1 ;;
esac
