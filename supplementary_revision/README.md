# supplementary_revision/: data and analysis added after the original submission

This folder contains measurements, analysis code and outputs that were **added after the
original submission** of the paper *GPU-Initiated Discrete Simulated Bifurcation:
Low-Latency Requests and Streaming Dense Couplings*. They respond to the reviews of IEEE
Access manuscript Access-2026-50856, received on 26 October 2026, and were measured on the
same day.

**Nothing outside this folder was changed for the supplement.** The original data
(`results/`), scripts (`analysis/`) and programs (`server_doca/`, `server_base/`, `client/`,
`scripts/`, `stream/`) are exactly as in the original release. The supplementary
measurements used the same binaries and scripts, with new run tags. For S3 (second GPU
platform) the same, unchanged sources were additionally built for the GH200 host (aarch64,
`sm_90`); the only new program is the launcher script `pu3srv/server.sh` in this folder.

## Contents

| path | contents |
|---|---|
| `RUNS.md` | exact commands of the supplementary measurements: host states, server and client commands, stop conditions |
| `results/net_supplementary.csv` | the 78 SUMMARY rows of the supplementary load runs (same columns as `results/net.csv`) |
| `results/raw/*.log` | client log of each of the 78 runs, with configuration, loss, percentiles and verification |
| `results/energy_runs.txt` | markers (PU3 epoch seconds) and client SUMMARY rows of the power runs |
| `results/power_gpu_<label>.csv` | server-GPU power from `nvidia-smi`, every 100 ms |
| `results/power_cpu_<label>.csv`, `.start` | CPU package energy from `perf stat -e power/energy-pkg/`, per 1-s interval, and the epoch at which `perf` was started |
| `analysis/make_revision.py` | analysis and plotting for the supplement. It imports the unchanged helpers of `analysis/make_tables.py` |
| `figures/`, `tables/`, `numbers_revision.txt` | outputs of `make_revision.py`. `numbers_revision.txt` lists every number quoted in the revised text, together with its source runs |
| `pu3srv/server.sh` | S3: starts the unchanged request-service server on the GH200 host (GPU and NIC addresses, host-thread pinning, a header with GPU/driver/NIC/NUMA facts in the log) |
| `results/pu3srv/net.csv` | S3: the 8 SUMMARY rows (same columns as `results/net.csv`) |
| `results/pu3srv/raw/*.log` | S3: client log of each of the 8 runs |
| `results/pu3srv/server_*.log` | S3: server logs, `server_srv_doca_w2t.log` with GPU 1 and `server_srv_doca_w2t_gpu0.log` with GPU 0, including the per-second counters (`rx_frames`, `solved`, `drop`, ...) |
| `analysis/compare_pu3srv.py` | S3: comparison with the RTX PRO 6000 server of the paper. It imports the unchanged `analysis/compare.py` |
| `tables/pu3srv_compare.md` | output of `compare_pu3srv.py` |

Power-log labels: `doca_tuned` and `doca_default` are GPUNetIO on the tuned and the default
PU2 host state. `proxy_tuned` is the CPU proxy on the tuned host. `idle_tuned` and
`idle_default` are runs with no server.

## Supplementary measurements

**S1, load sweeps for two more request configurations.** The paper's open-loop Poisson
experiment originally covered only the reference request (int8, n=128, B=8, K=100). S1
adds two more configurations, each measured with GPUNetIO (batched send, `-W 2`) and with
the tuned CPU proxy, three repeats per server:

* `pm1`, n=256, B=8, K=100 (576-byte request frames). Tags: `doca_w2t_pm256_r{1,2,3}` and
  `proxy_fast_pm256_r{1,2,3}`. Offered rates: 200–800 kreq/s.
* `int8`, n=192, B=8, K=400 (a solve of about 0.95 ms). Tags: `doca_w2t_i192k400_r{1,2,3}` and
  `proxy_fast_i192k400_r{1,2,3}`. Offered rates: 100–500 kreq/s.

The series names differ from `doca_w2t` and `proxy_fast`, so `analysis/make_tables.py`
ignores these rows and the original tables are reproduced unchanged.

**S2, power.** This part records CPU package power (RAPL, the whole PU2 host) and the
power of the server GPU used (`nvidia-smi`). Each case has a 20-s period with the server
running but no traffic, followed by the reference request at 200, 400 and 600 kreq/s.
The cases are GPUNetIO on the tuned and on the default host state, and the proxy on the
tuned host. A further 30 s are recorded with no server on each host state.

**S3, a second GPU platform (small check).** The unchanged GPUNetIO server (`-W 2`) runs
on the other host, an NVIDIA GH200 (sm_90, driver 595.45.04, DOCA 3.5.0098; NIC
`0002:01:00.0`, firmware 28.43.8002), and the unchanged client runs on the RTX PRO 6000 host,
over the same 400 GbE link: the roles of the two hosts are swapped. Measured: the unloaded
latency of the reference request and of `pm1` n=256 (closed loop, one outstanding request),
and the reference request under Poisson load at 200 and 400 kreq/s, 100,000 requests per
run. Two placements of the server GPU:

* tag `pu3srv_doca_w2t_r2`: GPU 0, attached to the same Grace CPU as the NIC (NUMA node 0).
  This is the run to cite.
* tag `pu3srv_doca_w2t`: GPU 1, attached to the other Grace CPU (NUMA node 1).

## Observations

These statements come from the files in this folder. `numbers_revision.txt` has the
exact values.

* **int8, n=192, K=400.** Both servers saturate at about 290 kreq/s, the limit of the
  372 worker blocks. Above that rate the round-trip time levels off at about 14 ms
  instead of growing without bound, and the excess requests are dropped and counted.
  The level matches Little's law for the bounded work ring: 4096 queued requests
  divided by the completion rate.
* **pm1, n=256.** Up to 500 kreq/s offered, GPUNetIO has the lower median and
  99th-percentile RTT. The GPUNetIO server then saturates at about 0.595 Mreq/s, while
  the tuned proxy completes up to about 0.63 Mreq/s at a median of about 265 µs.
  GPUNetIO's server residence stays near 220 µs throughout, so the backlog builds
  before the receive stage. In the current prototype a single warp receives all frames
  and processes the frames of a burst one after another, and this stage sets the
  limit. Lane-parallel frame processing or more receive queues would raise it; this
  remains future work and is not part of the measured code.
* **Power.** GPUNetIO's latency is the same on the default and the tuned host state. The
  CPU proxy is measured on the tuned state, which disables CPU idle states; that costs about
  110 W of package power even at idle. Compared this way (GPUNetIO on the default host,
  proxy on the tuned host), total server power (CPU package plus server GPU) is 13–20 %
  lower for GPUNetIO at 200–600 kreq/s. On the same tuned host, GPUNetIO draws 1–7 % more
  than the proxy, because its GPU power is higher. Its resident persistent kernels add
  about 137 W of GPU power even without traffic. Values are in `tables/tab_energy.tex`.
* **Second platform (S3).** The server runs unchanged on the GH200, and every reply of
  every run is verified (`verify_bad` = `ref_bad` = 0). With 48 KB of shared memory per
  block the GH200 holds 260 worker blocks (372 on the RTX PRO 6000). Each request is
  solved by one block, so server residence follows the speed of one SM rather than the
  size of the GPU: with GPU 0 it is 219.5 µs for the reference request and 320.3 µs for
  `pm1` n=256, against 152.7 and 193.0 µs on the RTX PRO 6000. Under load at 200 and
  400 kreq/s every request is served (200.9 and 401.2 kreq/s completed, none lost), with
  median residence 225.3 and 265.8 µs. GPU placement matters for the receive stage: with
  GPU 1, on the other Grace CPU, the unloaded numbers are the same, but at 400 kreq/s only
  319.9 kreq/s complete and 4,682 requests are lost. The server counted no dropped or bad
  frames, and its `rx_frames` is short by exactly those 4,682, so they were lost before the
  GPU receive queue. This is the receive-stage limit described under pm1 above, reached
  earlier on this platform. The client-side remainder (RTT minus server residence,
  about 97 µs unloaded) is much larger than in the original direction (about 13.5 µs), and
  was not investigated. The client host here is the RTX PRO 6000 host, which was in its
  default host state after S2 (RUNS.md, step 6). Only server residence is therefore compared
  across platforms.

## Reproducing

From the repository root:

```bash
python3 analysis/make_tables.py                          # original outputs (unchanged)
python3 supplementary_revision/analysis/make_revision.py # supplementary outputs (S1, S2)
python3 supplementary_revision/analysis/compare_pu3srv.py # S3 -> supplementary_revision/tables/pu3srv_compare.md
```

`compare_pu3srv.py` reads `results/net.csv` (the RTX PRO 6000 server, newest run of each
configuration, as in `analysis/compare.py`) and `supplementary_revision/results/pu3srv/net.csv`
(the GH200 server; the newest run, `pu3srv_doca_w2t_r2` with GPU 0, is reported).

`make_revision.py` reads `results/net.csv` and `supplementary_revision/results/net_supplementary.csv`
(a row present in both is counted once), the power logs in `supplementary_revision/results/`,
the GPU-only benchmarks `results/bench_*.csv` and, when present, the per-request CSVs in
`results/raw/`. Without the per-request CSVs, `figures/fig_decomp.pdf`,
`figures/fig_breakdown2.pdf` and the solve-under-load numbers are kept as shipped.
Everything else is recomputed.

| output | content |
|---|---|
| `figures/fig_load_cfgs.pdf` | completed rate and client RTT (p50, p99) against offered load for the reference and both S1 configurations. Points are medians over repeats; bars show the range. |
| `figures/fig_decomp.pdf` | decomposition of the unloaded round trip into the terms of Eq. (8), mean per request, for all 19 settings of the unloaded grid (tuned launch, tuned proxy, GPUNetIO) |
| `figures/fig_breakdown2.pdf` | two-panel version of `fig_breakdown`. Panel (b) leaves out the solve so that the 2-µs GPU work-ring wait is visible. |
| `figures/fig_pipeline.pdf` | saturated throughput of the GPU-resident and host-mapped rings without the network, against the solve time, with the bound N_w/t_solve |
| `tables/tab_energy.tex` | CPU, GPU and total power, and energy per request, per host state, server and rate |
| `numbers_revision.txt` | every number quoted in the revised manuscript, with its source |

## Per-request data

As for the original runs, the per-request CSVs of the supplementary load runs (one row
per request) are not part of the repository. No supplementary output depends on them:
all supplementary results derive from `results/net_supplementary.csv`,
`results/pu3srv/net.csv`, the client and server logs and the power logs. The per-request
CSVs are available from the author on request.
