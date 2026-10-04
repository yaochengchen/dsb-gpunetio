# dsb-gpunetio

GPU-initiated discrete simulated bifurcation (dSB) over NVIDIA DOCA GPUNetIO.

This repository contains the code, measured results and analysis scripts for the paper
*GPU-Initiated Discrete Simulated Bifurcation: Low-Latency Requests and Streaming Dense Couplings*
(Y. Chen, IAG-USP). It has two parts:

* **Request service** – small and medium Ising/QUBO problems (n up to ~1000) are sent as raw
  Ethernet frames to a GPU that receives, solves and answers them itself. One cooperative persistent
  kernel runs the receive warp, the solver blocks (one request per CTA) and the transmit thread; no
  CPU core sits on the data path. Two conventional CPU+GPU servers (launch per request, CPU-proxied
  persistent kernel) use the same wire format, client and solver code as baselines.
* **Streaming mode** (`stream/`) – a dense coupling matrix that does not fit in GPU memory
  (n = 10^7, 12.5 TB of 1-bit couplings) is generated on one GPU and streamed over 400 GbE into a
  second GPU, which evaluates the coupling sums tile by tile as the frames arrive and keeps only the
  O(Bn) solver state.

```
                    +-------------------- one cooperative persistent kernel ----------------------+
ConnectX-7 --DMA--> | RX buffer -> RX warp -> work ring -> worker CTAs -> reply ring -> TX thread   | --> ConnectX-7
                    |              |                         ^                                      |
                    |              +-- LOAD --> model store -+                                      |
                    +-------------------------------------------------------------------------------+
```

## Repository layout

| path | contents |
|---|---|
| `include/` | wire format (`dsb_proto.h`), packed model layouts (`dsb_model.h`), solver (`dsb_solve.cuh`), work/reply rings (`dsb_queue.h`), bit-exact CPU reference and instance generators (`dsb_ref.hpp`) |
| `server_doca/` | GPUNetIO server: DOCA Flow steering, GPU RX/TX queues, persistent kernel |
| `server_base/` | baseline servers: `--mode launch` (socket + H2D + launch + D2H per request) and `--mode proxy` (CPU-proxied persistent kernel) |
| `client/` | AF_PACKET load generator used for all three servers: closed/open loop, per-request CSV, energy and trajectory verification |
| `tools/` | `selftest` (GPU vs CPU reference, bit for bit) and `dsb_bench` (GPU-only measurements without the NIC) |
| `tests/` | CPU-only unit tests and a CPU stand-in server for protocol tests over a veth pair |
| `scripts/` | measurement sweeps (`run_bench.sh`, `run_net.sh`) and host tuning for the baselines (`host_fast_state.sh`) |
| `stream/` | streaming mode: generator (`st_gen`), GPUNetIO receiver (`st_recv`), kernel-stack receiver (`st_cpu_recv`), resident-matrix reference (`st_ref`); see `stream/README.md` |
| `client_doca/` | GPUNetIO client set-up kept for reference only (not built, not used in the measurements); the streaming generator's send path is derived from it |
| `results/` | measured data (see below) |
| `analysis/` | scripts that turn `results/` into the tables, figures and numbers of the paper, and their outputs |

## Hardware and software used

| host | role | GPU | NIC | OS / software |
|---|---|---|---|---|
| PU2 (x86_64) | server / stream receiver | RTX PRO 6000 Blackwell (`sm_120`) | ConnectX-7, 400 GbE | kernel 5.14 (EL9.6), driver 595.58, CUDA 13.2, DOCA 3.4, MLNX OFED 26.04 |
| PU3 (aarch64) | client / stream generator | GH200 (`sm_90`), generator uses GPU 1 | ConnectX-7, 400 GbE | kernel 7.0 (nvidia-64k), MLNX OFED 26.04 |

The two hosts are connected back to back. Exact versions are in `results/env_pu2.txt` and
`results/env_pu3.txt`. MAC and PCIe addresses of these machines are the defaults in the scripts
and programs; change them for other hosts.

## Build

Requirements: CUDA 12.1 or newer, g++ (C++17), and for the GPUNetIO programs DOCA (GPUNetIO, Flow,
Eth, ARGP) with meson and ninja. Binaries go to `bin/$(uname -m)/` so hosts of different
architectures can share one checkout.

```bash
make ARCH=sm_120 -j                         # selftest, dsb_bench, dsb_base_server, dsb_client, cpu_test, cpu_server
export PKG_CONFIG_PATH=/opt/mellanox/doca/lib64/pkgconfig:$PKG_CONFIG_PATH
make doca ARCH=sm_120                       # dsb_doca_server (meson build directory build_$(uname -m)/)
make client                                 # client only, e.g. on a load-generator host without CUDA

make -C stream -f stream.mk ARCH=sm_120     # st_ref, st_cpu_recv
make -C stream -f stream.mk doca ARCH=sm_120   # st_recv, st_gen   (use ARCH=sm_90 on GH200)
```

Shell scripts may lose their executable bit when copied from Windows: `chmod +x scripts/*.sh stream/scripts/*.sh`.

## Solver

The request solver uses the two exact fast paths of dSB, reorganised for many small requests:

* **bit** – `PM1` (dense ±1, one bit plane: `n − 2·popc(J_r ⊕ S) + s_r`) and `TERNARY`
  (two planes: `2(popc(P∧S) − popc(M∧S)) − (|P|−|M|)`);
* **int8** – `INT8` couplings |J| ≤ 127 with DP4A (weighted problems such as portfolios).

One request is one CTA; J, signs and h are in shared memory, x/y in registers. A warp holds 32
consecutive rows of one replica, so the new signs are one `__ballot_sync` word; signs are double
buffered with one `__syncthreads()` per step. The coupling sums are exact integers, so the GPU and
the CPU reference produce bit-identical trajectories. Limits: `B · roundup32(n) ≤ 4096` and the
shared-memory plan (`dsb_make_plan`). Largest n per request:

| shared memory / CTA | PM1 B=1 / 8 | TERNARY B=1 / 8 | INT8 B=1 / 8 | any format, B=32 |
|---|---|---|---|---|
| 48 KB (default) | 544 / 512 | 384 / 384 | 192 / 192 | 128 |
| 96 KB | 832 / 512 | 576 / 512 | 288 / 288 | 128 |
| 200 KB (GH200 only) | 1152 / 512 | 864 / 512 | 416 / 416 | 128 |

`--smem KB` (bench/base/selftest) or `-s KB` (DOCA server) trades CTAs per SM for size; the
RTX PRO 6000 (`sm_120`) allows at most ~99 KB per CTA.

## Running the request service

Server on PU2, client on PU3. Run only one server at a time (all use the same Ethertype).

```bash
# 1. correctness first (expects "SELFTEST PASSED")
./bin/$(uname -m)/cpu_test && ./bin/$(uname -m)/selftest --device 0

# 2. GPU-only measurements on PU2 -> results/bench_<host>.csv
GPU=0 scripts/run_bench.sh 2>&1 | tee results/bench.log

# 3. network measurements: start one server on PU2 ...
sudo ./bin/x86_64/dsb_doca_server -g 0000:ab:00.0 -n 0000:bd:00.0 -W 2           # GPUNetIO (tag doca_w2t)
sudo taskset -c 2 ./bin/x86_64/dsb_base_server --iface ens7np0 --mode launch --streams 4
sudo taskset -c 2 ./bin/x86_64/dsb_base_server --iface ens7np0 --mode proxy
#    ... and run the sweep from PU3 with a matching tag
sudo TAG=doca_w2t SERVER=58:a2:e1:db:db:80 IFACE=enP2s2np0 scripts/run_net.sh
```

`run_net.sh` runs the unloaded latency grid, closed-loop windows 4/16/64 and Poisson open-loop
load for the reference configuration (int8, n=128, B=8, K=100), appends one SUMMARY row per run to
`results/net.csv` and writes per-request CSVs and logs to `results/raw/`. A tag already present in
`net.csv` is refused; repeats use `<series>_r<k>`. The tuned-host baselines (`*_fast`) are run
after `scripts/host_fast_state.sh apply` on both hosts (see the script header).

Tags in `results/net.csv`:

| tag | server |
|---|---|
| `launch`, `proxy` | baselines, default host |
| `launch_fast`, `proxy_fast` | baselines, tuned host (`host_fast_state.sh`) |
| `doca`, `doca_fast` | GPUNetIO, per-reply send (`-W 0`), 5 µs receive timeout; default / tuned host |
| `doca_w2` | GPUNetIO, batched send (`-W 2`), 5 µs receive timeout |
| `doca_fast_W2` | GPUNetIO, batched send (`-W 2`), tuned host, reference configuration only (exploratory; not used in the paper) |
| `doca_w2t` | GPUNetIO, batched send (`-W 2`), 500 ns receive timeout – **the server reported in the paper** (current default build) |
| `diag_w2t` | idle-gap diagnostic runs of `doca_w2t` |
| `<tag>_r<k>` | repeat of a configuration; analysis reports the newest run (and medians over repeats at ≥400 kreq/s) |

Troubleshooting (the GPUNetIO server prints one counter line per second on PU2):

| symptom | likely cause |
|---|---|
| `rx_frames` stays 0 | frames do not reach the GPU queue: wrong `--server` MAC, another server still holds the Ethertype, or the link is down (the server prints `[DSB_HINT]` after 10 s) |
| `rx_frames`/`loads` grow, `solved` does not | model fragments incomplete (`bad` grows) |
| `solved` grows, `tx_frames` does not or `txerr` > 0 | reply send path: the server prints `last tx error`; try another `-W` mode |
| `tx_frames` grows, client reports `lost` | replies leave the NIC but do not reach the client: `tcpdump -i enP2s2np0 ether proto 0xd5b1` on PU3 |
| `drop` grows | work ring full (4096 queued requests): offered load above saturation |
| client `Message too long` | MTU smaller than the frame: `ip link set <iface> mtu 9000` |
| `verify_bad` or `ref_bad` ≠ 0 in `net.csv` | wrong results; rerun `selftest` |
| `queue_ns` always 0 | expected for the CPU servers (their CPU time is reported in `tx_ns`) |

A protocol test without GPU or ConnectX uses a veth pair and the CPU stand-in server:

```bash
sudo ip link add vtest0 type veth peer name vtest1
sudo ip link set vtest0 mtu 9000 up; sudo ip link set vtest1 mtu 9000 up
sudo ./bin/$(uname -m)/cpu_server --iface vtest1 &
sudo ./bin/$(uname -m)/dsb_client --iface vtest0 --server $(cat /sys/class/net/vtest1/address) --n 96 --B 4 --K 60 --count 300
sudo ip link del vtest0
```

The streaming experiments are described in `stream/README.md`.

## Results

| path | contents |
|---|---|
| `results/net.csv` | one SUMMARY row per network run (291 runs): throughput, RTT p50/p99/p99.9, server residence, verification counts |
| `results/raw/*.log` | client log of each network run (configuration, loss, percentiles, energy and trajectory verification) |
| `results/bench_pu2*.csv` | GPU-only measurements (`dsb_bench`); `_r2` proxy rerun with `--window 4096 --drain`, `_r3` repeat of single/persistent |
| `results/bench*.log`, `bench_r2_hoststate.txt` | logs of the bench runs and host state of the r2 run |
| `results/selftest.log`, `results/server_doca.log` | correctness check on PU2, and a GPUNetIO server log |
| `results/stream/` | streaming runs used in the paper: receiver (`<tag>.{csv,log,chk}`), generator (`<tag>_gen.{csv,log}`) and the reference checksums (`ref_*.chk`) |
| `results/env_pu{2,3}.txt` | driver, CUDA, kernel and NIC firmware versions |

The per-request CSVs of the network runs (`results/raw/<tag>_<config>.csv`, one row per request,
about 1.5 GB in total) are not included. All statistics derived from them are in
`analysis/raw_summary.csv` / `raw_summary.md`, `analysis/tables/tab_breakdown.tex`,
`analysis/figures/fig_breakdown.pdf`, `fig_cdf.pdf`, `fig_rest_hist.pdf` and `fig_idle.pdf`.
The raw files are available from the author on request.

## Reproducing the tables and figures

Python 3 with numpy and matplotlib; run from the repository root.

```bash
python3 analysis/make_tables.py                  # analysis/tables/tab_*.tex, analysis/figures/fig_*.pdf, analysis/numbers.txt
python3 analysis/compare.py                      # analysis/compare.md
python3 analysis/make_stream_table.py            # analysis/tables/tab_stream.tex (checks .chk files byte for byte)
python3 stream/analysis/summarize.py results/stream   # results/stream/summary.md
python3 analysis/build_stream_fig.py             # analysis/figures/fig_stream.pdf (schematic)
python3 analysis/analyze_raw.py                  # needs the per-request CSVs in results/raw/
```

With the files in this repository, `make_tables.py` regenerates every table except the per-request
breakdown, which it keeps as shipped (see `analysis/README.md`). `analysis/numbers.txt` lists every
number quoted in the text together with its source.

## Notes and limitations

* Raw Ethernet with one RX and one TX queue on the GPU; requests are not retransmitted. A model id
  must not be reloaded while requests for it are in flight (use a new id).
* Frames are ≤ 1472 bytes for n ≤ 704 (LOAD fragments are 1408 B + header), so MTU 1500 works for
  the request service; the streaming mode uses 4096-byte frames and MTU 9000.
* GPUNetIO receive path: DOCA Flow root pipe on the Ethertype → one cyclic GPU RX queue,
  warp-scope `doca_gpu_dev_eth_rxq_recv` with a 500 ns timeout (`-DDSB_RX_TIMEOUT_NS` to change). Send paths: `-W 0`
  per-reply `txq_send` (as in the `gpunetio_simple_send` sample), `-W 1` `wait_send` with timestamp 0,
  `-W 2` batched WQE prepare/submit.
* `%globaltimer` resolution depends on the GPU and driver; server-side time stamps are used only to
  split the latency, the primary metric is the client-measured RTT.

## License

MIT, see `LICENSE`.
