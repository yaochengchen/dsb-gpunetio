# Streaming mode — dense couplings streamed into the GPU

The coupling matrix is never stored on the receiver. A generator GPU synthesizes the 1-bit coupling
tiles in column-block-major order and sends them as raw 4096-byte Ethernet frames (Ethertype
0xD5B2, 4032 payload bytes); the receiver GPU accumulates the coupling sums for B replicas as the
frames arrive, runs the dSB update at the end of each pass and acknowledges the pass. State is
O(Bn); the O(n²) matrix is consumed once per update.

| program | host | role |
|---|---|---|
| `st_gen` | PU3 (GH200 GPU 1 + ConnectX-7) | generator: N_q GPU send queues; default `-W 2` = one 512-thread block per queue, BLOCK-scope `wait_send` of 512 WQEs with one NOTIFY per burst; `PASS_END` at the end of each pass, next pass after `PASS_ACK`; `-L <Gb/s>` token-bucket limit |
| `st_recv` | PU2 (RTX PRO 6000 + ConnectX-7) | GPUNetIO receiver: the request service's receive path (DOCA Flow on the Ethertype → one GPU RX queue, one receive warp); each received burst is a work item for resident worker blocks that accumulate in place (XOR/popcount); update kernel and `PASS_ACK` at the end of a pass |
| `st_cpu_recv` | PU2 | kernel-stack baseline: AF_PACKET sockets → pinned memory → H2D → the same GPU accumulation (`-G`: receive only) |
| `st_ref` | PU2 | reference with the matrix resident in GPU memory (`--cpu`: CPU reference); writes per-step checksums for bit-for-bit comparison with streamed runs |

Instances (same generator as the dsb-gpu `bit_dense_multi` application): `random` –
u_ij = fmix32(fmix32(hi ^ seed) + lo·0x9e3779b9) & 1, symmetric; `mattis` – J_ij = η_iη_j with a
hashed sign vector η, whose ground state 2E = −n(n−1) is known. The diagonal is transmitted as +1
and removed in the sum, a_i = n − 2·popc − s_i. Initial x = (2u−1)·0.01, ξ = 0.5/√n, Δt = 1.1; replica
b > 0 uses seed + b. Each pass prints `best 2E` and `chk` (an order-independent checksum of the sign
bitmap).

## Build

```bash
# PU2
make -C stream -f stream.mk ARCH=sm_120               # bin/x86_64/st_ref, st_cpu_recv
export PKG_CONFIG_PATH=/opt/mellanox/doca/lib64/pkgconfig:$PKG_CONFIG_PATH
make -C stream -f stream.mk doca ARCH=sm_120          # bin/x86_64/st_recv
# PU3
make -C stream -f stream.mk doca ARCH=sm_90           # bin/aarch64/st_gen
```

Defaults built into the programs and scripts: PU2 `ens7np0` 58:a2:e1:db:db:80, GPU 0000:ab:00.0,
NIC 0000:bd:00.0; PU3 `enP2s2np0` 08:20:e7:47:10:8e, NIC 0002:01:00.0; the generator
uses GPU 1 (found with `nvidia-smi`, or `GEN_GPU=<PCIe>`). The scripts set MTU 9000.

## Running

1. References (PU2, no network): `stream/scripts/pu2_ref.sh` writes
   `results/stream/ref_{cpu,gpu}_{random,mattis}_{1e4,1e5}.chk` and checks CPU == GPU at n = 10^4.
2. Start the receiver on PU2 first and wait for `waiting for pass 0`, then the generator on PU3
   with the same tag (the generator sends the first pass immediately):

```bash
sudo stream/scripts/pu2_recv.sh <tag> -N <n> -B <B> -K <K> -i <random|mattis> -s 42
sudo stream/scripts/pu3_gen.sh  <tag> -N <n> -K <K> -i <random|mattis> -s 42 [-q <queues>] [-L <Gb/s>]
```

Output: `results/stream/<tag>.{csv,log,chk}` (receiver) and `<tag>_gen.{csv,log}` (generator).
Both scripts log the NIC PHY counters every 5 s (`# phy` lines), independently of DOCA.

Runs in `results/stream/` (the ones reported in the paper):

The generator runs with the receiver's arguments minus `-B` (default 4 send queues, no rate limit).

| tag | receiver arguments | purpose |
|---|---|---|
| `x_random_1e4_w2` | `-N 10000 -B 8 -K 20 -i random -s 42` | bit-identical to `ref_cpu_random_1e4.chk` |
| `x_random_1e5_w2`, `x_mattis_1e5_w2` | `-N 100000 -B 8 -K 20 -i {random,mattis} -s 42` | bit-identical to `ref_gpu_{random,mattis}_1e5.chk` |
| `rate_1e6_q1` … `q8` | `-N 1000000 -B 8 -K 2 -i mattis -s 42`, generator `-q {1,2,4,8}` | rate vs. number of send queues |
| `rate_1e6_B1` | `-N 1000000 -B 1 -K 2 -i mattis -s 42` | B = 1 vs. B = 8 |
| `rate_1e6_random` | `-N 1000000 -B 8 -K 2 -i random -s 42` | random synthesis rate |
| `full_mattis_1e7_B1` | `-N 10000000 -B 1 -K 2 -i mattis -s 42` | n = 10^7, one replica |
| `full_mattis_1e7_B8` | `-N 10000000 -B 8 -K 10 -i mattis -s 42` | n = 10^7, ground state reached by all replicas |
| `full_random_1e7_B8` | `-N 10000000 -B 8 -K 10 -i random -s 42` | n = 10^7, random instance |

Summaries: `python3 stream/analysis/summarize.py results/stream` and
`python3 analysis/make_stream_table.py`.

Kernel-stack baseline (stop `st_recv` first; with DOCA Flow active the kernel sees none of these
frames): `sudo stream/scripts/pu2_cpu_recv.sh <tag> -N 1000000 -B 8 -K 2 -t <threads> -C 2 [-G]`.

## Troubleshooting

* Receiver `rx_frames` stays 0: check that the receiver was running before the generator started;
  compare the generator's `tx_packets_phy` with the receiver's `rx_packets_phy`. If frames reach the
  port but not the GPU queue, another DOCA program may hold the port
  (`pgrep -a 'st_recv|dsb_doca_server'`). With no receiver running,
  `sudo tcpdump -i ens7np0 -e -c 3 ether proto 0xd5b2` on PU2 checks link, MAC, Ethertype and MTU.
* DOCA start-up messages `Failed to query NIC_CAP register`, `GDRCopy ... not found` and
  `adapting queue depth to 128` are harmless (they also appear in `results/server_doca.log`).
* Receiver `bad` > 0: frames with a wrong magic/type arrived (version mismatch or unrelated
  traffic; `RUN INVALID` is printed for any nonzero count).
* `lost` ≠ 0 or `drop` > 0: rerun with a lower generator rate (`-L 350`).
* Generator waits on `PASS_ACK`: if PU2 reports `PASS_ACK send failed`, start the receiver with `-W 1`.
