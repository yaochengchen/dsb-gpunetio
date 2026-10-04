#!/usr/bin/env bash
# host_fast_state.sh -- keep the kernel receive path of a baseline server (or
# the client) in its "fast state": no sleeping core on the interrupt path, no
# interrupt moderation, NIC interrupts on the busy-polling core's neighbour.
#
#   sudo scripts/host_fast_state.sh apply   IFACE=ens7np0  CORE=2  [IRQ_CORE=3]
#   sudo scripts/host_fast_state.sh restore IFACE=ens7np0
#   sudo scripts/host_fast_state.sh show    IFACE=ens7np0
#
# What it does (all reversible at run time; saved to /tmp/host_fast_state.<iface>):
#   1. cpuidle: disable every C-state deeper than C0 on ALL cores
#      (/sys/devices/system/cpu/cpu*/cpuidle/state*/disable = 1).  The core that
#      takes the NIC interrupt is then never asleep when a frame arrives.  This is
#      the run-time equivalent of idle=poll; it does not need a reboot.
#   2. cpufreq: performance governor (if the driver exposes one).
#   3. NIC: adaptive-rx/tx off, rx-usecs 0, rx-frames 1, tx-usecs 0, tx-frames 1
#      (ethtool -C), GRO/LRO off (ethtool -K): a frame is delivered on arrival.
#   4. IRQ affinity: every mlx5 interrupt of the interface pinned to IRQ_CORE
#      (default CORE+1), so the interrupt never lands on a random sleeping core.
#   5. net.core.busy_poll / busy_read = 50 us (AF_PACKET ignores it, UDP uses it).
#
# Then rerun the baselines of the request service exactly as before, with a new
# tag, e.g. on PU3:
#   sudo TAG=launch_fast SERVER=58:a2:e1:db:db:80 IFACE=enP2s2np0 scripts/run_net.sh
#   sudo TAG=proxy_fast  ...
# (the DOCA server does not need any of this; its receive path is on the GPU).
# Apply it on PU3 as well (IFACE=enP2s2np0, CORE=3 = the client's RX thread) to
# remove the client's share of the remainder, and report both.
set -u
MODE=${1:-show}
IFACE=${IFACE:-ens7np0}
CORE=${CORE:-2}
IRQ_CORE=${IRQ_CORE:-$((CORE + 1))}
SAVE=/tmp/host_fast_state.$IFACE

show() {
  echo "== cpuidle (first core):"; for f in /sys/devices/system/cpu/cpu0/cpuidle/state*/; do echo "  $(cat $f/name) disable=$(cat $f/disable)"; done 2>/dev/null
  echo "== governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
  echo "== coalescing:"; ethtool -c "$IFACE" 2>/dev/null | grep -E 'Adaptive|^rx-usecs:|^rx-frames:|^tx-usecs:|^tx-frames:'
  echo "== offloads:"; ethtool -k "$IFACE" 2>/dev/null | grep -E 'generic-receive-offload|large-receive-offload'
  echo "== busy_poll: $(sysctl -n net.core.busy_poll) busy_read: $(sysctl -n net.core.busy_read)"
  echo "== IRQ affinity ($IFACE):"
  for irq in $(grep -E "$IFACE|mlx5_comp" /proc/interrupts | awk -F: '{print $1}' | tr -d ' '); do
    echo "  irq $irq -> cpus $(cat /proc/irq/$irq/smp_affinity_list 2>/dev/null)"
  done | head -20
}

apply() {
  mkdir -p "$SAVE"
  # 1. cpuidle
  for f in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    [ -f "$f" ] || continue
    echo "$f $(cat $f)" >> "$SAVE/cpuidle"
    echo 1 > "$f" 2>/dev/null
  done
  # 2. governor
  if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$SAVE/governor"
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g" 2>/dev/null; done
  fi
  # 3. NIC coalescing / offloads
  ethtool -c "$IFACE" > "$SAVE/coalesce" 2>/dev/null
  ethtool -k "$IFACE" > "$SAVE/features" 2>/dev/null
  ethtool -C "$IFACE" adaptive-rx off adaptive-tx off 2>/dev/null
  ethtool -C "$IFACE" rx-usecs 0 rx-frames 1 tx-usecs 0 tx-frames 1 2>/dev/null || ethtool -C "$IFACE" rx-usecs 0 rx-frames 1 2>/dev/null
  ethtool -K "$IFACE" gro off lro off 2>/dev/null
  # 4. IRQ affinity
  : > "$SAVE/irqs"
  for irq in $(grep -E "$IFACE|mlx5_comp" /proc/interrupts | awk -F: '{print $1}' | tr -d ' '); do
    echo "$irq $(cat /proc/irq/$irq/smp_affinity_list)" >> "$SAVE/irqs"
    echo "$IRQ_CORE" > /proc/irq/$irq/smp_affinity_list 2>/dev/null
  done
  systemctl stop irqbalance 2>/dev/null && echo "irqbalance stopped (restart with: systemctl start irqbalance)"
  # 5. busy poll
  sysctl -n net.core.busy_poll > "$SAVE/busy_poll"; sysctl -n net.core.busy_read > "$SAVE/busy_read"
  sysctl -w net.core.busy_poll=50 net.core.busy_read=50 >/dev/null
  echo "applied on $IFACE (server core $CORE, IRQ core $IRQ_CORE); state saved in $SAVE"
  show
}

restore() {
  [ -d "$SAVE" ] || { echo "nothing saved in $SAVE"; exit 1; }
  [ -f "$SAVE/cpuidle" ] && while read -r f v; do echo "$v" > "$f" 2>/dev/null; done < "$SAVE/cpuidle"
  [ -f "$SAVE/governor" ] && for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do cat "$SAVE/governor" > "$g" 2>/dev/null; done
  if [ -f "$SAVE/coalesce" ]; then
    ar=$(grep -i 'Adaptive RX' "$SAVE/coalesce" | awk '{print $3}'); at=$(grep -i 'Adaptive RX' "$SAVE/coalesce" | awk '{print $6}')
    ru=$(grep '^rx-usecs:' "$SAVE/coalesce" | awk '{print $2}'); rf=$(grep '^rx-frames:' "$SAVE/coalesce" | awk '{print $2}')
    tu=$(grep '^tx-usecs:' "$SAVE/coalesce" | awk '{print $2}'); tf=$(grep '^tx-frames:' "$SAVE/coalesce" | awk '{print $2}')
    ethtool -C "$IFACE" rx-usecs "${ru:-8}" rx-frames "${rf:-128}" tx-usecs "${tu:-8}" tx-frames "${tf:-128}" 2>/dev/null
    ethtool -C "$IFACE" adaptive-rx "${ar:-on}" adaptive-tx "${at:-on}" 2>/dev/null
  fi
  ethtool -K "$IFACE" gro on 2>/dev/null
  [ -f "$SAVE/irqs" ] && while read -r irq cpus; do echo "$cpus" > /proc/irq/$irq/smp_affinity_list 2>/dev/null; done < "$SAVE/irqs"
  [ -f "$SAVE/busy_poll" ] && sysctl -w net.core.busy_poll="$(cat $SAVE/busy_poll)" net.core.busy_read="$(cat $SAVE/busy_read)" >/dev/null
  systemctl start irqbalance 2>/dev/null
  echo "restored from $SAVE"; show
}

case "$MODE" in
  apply) apply ;;
  restore) restore ;;
  *) show ;;
esac
