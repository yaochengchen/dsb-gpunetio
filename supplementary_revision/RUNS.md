# Supplementary measurements: commands

These measurements were **added after the original submission** of the paper, in response
to the reviews of IEEE Access manuscript Access-2026-50856 (received 26 October 2026). They
were taken on 26 October 2026 (about 17:00–18:00 UTC), with the same hardware, binaries
and scripts as the original campaign. No program in `server_doca/`, `server_base/`,
`client/` or `scripts/` was changed or rebuilt. The original data in `results/` are
unchanged; the new data are in `supplementary_revision/results/`.

There are three parts (S3, a small check on a second GPU platform, follows S1 and S2 and
is described in section 8):

* **S1, load sweeps for two more request configurations.** These repeat the open-loop
  Poisson experiment of the paper, which used only the reference configuration (int8,
  n=128, B=8, K=100), for two further configurations:
  * `pm1:256:8:100`: a dense ±1 model with larger request frames (576 B) and a
    solve time of about 200 µs.
  * `int8:192:8:400`: a long solve of about 0.95 ms. Here the 372 worker blocks
    saturate at about 290 kreq/s, which is within the range of the client.

  Both servers of the load comparison are measured, GPUNetIO (`-W 2`) and the tuned CPU
  proxy, with three repeats each.
* **S2, power.** CPU package power (RAPL) and server-GPU power are recorded while the
  reference request runs at 200, 400 and 600 kreq/s. The cases are: GPUNetIO on the
  tuned host, GPUNetIO on the default host, the proxy on the tuned host, and no server
  on either host state.

Hosts and addresses are those of the main `README.md`: PU2 is the server (RTX PRO 6000,
GPU PCIe `0000:ab:00.0`, NIC `0000:bd:00.0`, interface `ens7np0`, MAC
`58:a2:e1:db:db:80`), and PU3 is the client (interface `enP2s2np0`). All commands run in
the repository root on each host.

## 0. Checks (both hosts)

```bash
chmod +x scripts/*.sh
ls -l bin/x86_64/dsb_doca_server bin/x86_64/dsb_base_server bin/aarch64/dsb_client
grep -c -E "^SUMMARY,(doca_w2t|proxy_fast)_(pm256|i192k400)_r" results/net.csv   # must be 0 (tags unused)
pgrep -a dsb_ ; nvidia-smi                                                     # PU2: no server left, GPU idle
```

## 1. Tuned host state on both hosts (as for the repeated load runs of the paper)

```bash
sudo IFACE=ens7np0 CORE=2 scripts/host_fast_state.sh apply        # PU2
sudo IFACE=enP2s2np0 CORE=3 scripts/host_fast_state.sh apply      # PU3
```

## 2. S1 with GPUNetIO

PU2, terminal A:
```bash
sudo ./bin/x86_64/dsb_doca_server -g 0000:ab:00.0 -n 0000:bd:00.0 -W 2
```

PU3, smoke tests (nothing is written to `net.csv`). Each must report `acknowledged`,
`lost 0` and `reference trajectory 20/20 match`:
```bash
sudo ./bin/aarch64/dsb_client --iface enP2s2np0 --server 58:a2:e1:db:db:80 --format pm1  --n 256 --B 8 --K 100 --count 2000 --window 1 --tag smoke
sudo ./bin/aarch64/dsb_client --iface enP2s2np0 --server 58:a2:e1:db:db:80 --format int8 --n 192 --B 8 --K 400 --count 2000 --window 1 --tag smoke
```

PU3, sweeps (100,000 requests per run after 1,000 warm-up requests, three repeats):
```bash
for k in 1 2 3; do
  sudo TAG=doca_w2t_pm256_r$k    PARTS=load LOAD_CFG=pm1:256:8:100  RATES="200000 400000 500000 600000 700000 800000" scripts/run_net.sh
  sudo TAG=doca_w2t_i192k400_r$k PARTS=load LOAD_CFG=int8:192:8:400 RATES="100000 200000 250000 300000 350000 400000 500000" scripts/run_net.sh
done
grep -E "^SUMMARY,doca_w2t_(pm256|i192k400)_r" results/net.csv | cut -d, -f2,4,6,8,10,12,13,14,15,21,22
```

Stop conditions:
* Stop if `verify_bad` or `ref_bad` (the last two columns) is not 0, or if the server
  prints `[DSB_FATAL]`.
* Lost requests and a growing `drop` counter above the saturation point are expected
  data. They are not errors.

## 3. S2 for GPUNetIO on the tuned host (server still running)

PU2, terminal B:
```bash
nvidia-smi -i 00000000:AB:00.0 --query-gpu=timestamp,power.draw,utilization.gpu,clocks.sm --format=csv -lms 100 > results/power_gpu_doca_tuned.csv
```
PU2, terminal C:
```bash
date +%s.%N > results/power_cpu_doca_tuned.start; sudo perf stat -a -e power/energy-pkg/ -I 1000 -x, -o results/power_cpu_doca_tuned.csv
```
PU3:
```bash
echo "$(date +%s.%N) idle_start doca_tuned" >> results/energy_runs.txt; sleep 20
for r in 200000 400000 600000; do
  echo "$(date +%s.%N) start doca_tuned $r" >> results/energy_runs.txt
  sudo ./bin/aarch64/dsb_client --iface enP2s2np0 --server 58:a2:e1:db:db:80 --format int8 --n 128 --B 8 --K 100 \
       --count 6000000 --rate $r --poisson --tx-cpu 2 --rx-cpu 3 --tag energy_doca_tuned | grep SUMMARY >> results/energy_runs.txt
  echo "$(date +%s.%N) end doca_tuned $r" >> results/energy_runs.txt
  sleep 15
done
```

The first 20 s are the "kernels resident, no traffic" baseline. When the loop has
finished, press Ctrl+C in terminals B and C, then in terminal A. Then confirm that
`pgrep -a dsb_` prints nothing.

## 4. S1 and S2 with the tuned proxy

PU2, terminal A:
```bash
sudo taskset -c 2 ./bin/x86_64/dsb_base_server --iface ens7np0 --mode proxy
```

PU3: run the two smoke tests of step 2, then:
```bash
for k in 1 2 3; do
  sudo TAG=proxy_fast_pm256_r$k    PARTS=load LOAD_CFG=pm1:256:8:100  RATES="200000 400000 500000 600000 700000 800000" scripts/run_net.sh
  sudo TAG=proxy_fast_i192k400_r$k PARTS=load LOAD_CFG=int8:192:8:400 RATES="100000 200000 250000 300000 350000 400000 500000" scripts/run_net.sh
done
```

Power: repeat step 3 with every `doca_tuned` replaced by `proxy_tuned` (in the file
names, the markers and the client tag).

## 5. S2 with no server on the tuned host

PU2, terminals B and C, recording for 30 s and then stopped with Ctrl+C:
```bash
nvidia-smi -i 00000000:AB:00.0 --query-gpu=timestamp,power.draw,utilization.gpu,clocks.sm --format=csv -lms 100 > results/power_gpu_idle_tuned.csv
date +%s.%N > results/power_cpu_idle_tuned.start; sudo perf stat -a -e power/energy-pkg/ -I 1000 -x, -o results/power_cpu_idle_tuned.csv
```

## 6. S2 for GPUNetIO on the default host (PU3 stays tuned)

```bash
# PU2, terminal A
sudo IFACE=ens7np0 scripts/host_fast_state.sh restore; sudo systemctl start irqbalance
sudo ./bin/x86_64/dsb_doca_server -g 0000:ab:00.0 -n 0000:bd:00.0 -W 2
```

Run the reference smoke test on PU3:
```bash
sudo ./bin/aarch64/dsb_client --iface enP2s2np0 --server 58:a2:e1:db:db:80 --format int8 --n 128 --B 8 --K 100 --count 2000 --window 1 --tag smoke
```
Then repeat step 3 with `doca_default` in place of `doca_tuned`. After that, repeat step 5
with no server running and `idle_default` in place of `idle_tuned`.

## 7. Restore both hosts

```bash
sudo IFACE=enP2s2np0 scripts/host_fast_state.sh restore          # PU3 (PU2 was restored in step 6)
```

## Outputs of these runs

| file | contents |
|---|---|
| `results/net.csv` | 78 new SUMMARY rows with the tags `doca_w2t_pm256_r{1,2,3}`, `doca_w2t_i192k400_r{1,2,3}`, `proxy_fast_pm256_r{1,2,3}` and `proxy_fast_i192k400_r{1,2,3}`. They are copied to `supplementary_revision/results/net_supplementary.csv`. |
| `results/raw/<tag>_<cfg>_--rate_<r>_--poisson.{csv,log}` | per-request CSV and client log of each of the 78 runs |
| `results/energy_runs.txt` | start/end markers (PU3 epoch seconds) and the client SUMMARY row of each power run |
| `results/power_gpu_<label>.csv` | `nvidia-smi` samples every 100 ms (PU2 local time) |
| `results/power_cpu_<label>.csv`, `.start` | `perf` RAPL package energy per 1-s interval, and the epoch at which `perf` was started |

The energy runs use `--count 6000000` and no `--csv`, so they write no per-request file
and add no row to `results/net.csv`.

## 8. S3: second GPU platform, roles of the hosts swapped

Taken on 26 October 2026, 18:54–19:04 UTC, after section 7. The server runs on the GH200 host
(PU3) and the client on the RTX PRO 6000 host (PU2), over the same link. No source file was
changed. The same sources were built on PU3 for aarch64 / `sm_90`, which gives
`bin/aarch64/dsb_doca_server` (meson build directory `build_aarch64/`) and `bin/aarch64/selftest`.
The client is the existing `bin/x86_64/dsb_client`.

| role | host | GPU | NIC / interface | MAC |
|---|---|---|---|---|
| server | PU3 | GH200, nvidia-smi index 0 (`0009:01:00.0`, NUMA 0) or 1 (`0019:01:00.0`, NUMA 1) | `0002:01:00.0` / `enP2s2np0` (NUMA 0) | `08:20:e7:47:10:8e` |
| client | PU2 | — | `ens7np0` | `58:a2:e1:db:db:80` |

Host state: PU2 was left in the default state restored in step 6. The GPUNetIO server does
not use the kernel network path on PU3.

PU3, build and correctness:
```bash
export PKG_CONFIG_PATH=/opt/mellanox/doca/lib/aarch64-linux-gnu/pkgconfig:/opt/mellanox/doca/lib64/pkgconfig:$PKG_CONFIG_PATH
make doca ARCH=sm_90
make ARCH=sm_90 bin/aarch64/selftest
CUDA_DEVICE_ORDER=PCI_BUS_ID ./bin/aarch64/selftest --device 1     # SELFTEST PASSED
```

`supplementary_revision/pu3srv/server.sh` starts `dsb_doca_server -g <GPU> -n 0002:01:00.0 -W 2`
pinned to core 2. It writes `results/pu3srv/server_<tag>.log` (the measurement was made with
the same script at `pu3srv/server.sh` of the development tree; the file is byte-identical).

Run 1, GPU 1 (other Grace CPU than the NIC). PU3, terminal A:
```bash
sudo bash supplementary_revision/pu3srv/server.sh srv_doca_w2t        # prints "Worker CTAs: 260 ..."
```
PU2:
```bash
sudo TAG=pu3srv_doca_w2t SERVER=08:20:e7:47:10:8e IFACE=ens7np0 \
     OUT=results/pu3srv/net.csv RAW=results/pu3srv/raw \
     PARTS="lat load" LAT_CFGS="int8:128:8:100 pm1:256:8:100" RATES="200000 400000" \
     scripts/run_net.sh
```
Then Ctrl+C in terminal A.

Run 2, GPU 0 (same Grace CPU as the NIC), the run to cite. PU3, terminal A:
```bash
sudo SRV_GPU_INDEX=0 bash supplementary_revision/pu3srv/server.sh srv_doca_w2t_gpu0
```
PU2: the same command as run 1 with `TAG=pu3srv_doca_w2t_r2`. Then Ctrl+C in terminal A.

Each run sends 100,000 requests after 1,000 warm-up requests per configuration. The
outputs `results/pu3srv/net.csv`, `results/pu3srv/server_*.log` and the client logs
`results/pu3srv/raw/*.log` were copied to `supplementary_revision/results/pu3srv/`. The
per-request CSVs `results/pu3srv/raw/*.csv` are not distributed, as for the other runs.
