# analysis/ — from results/ to the paper's tables, figures and numbers

Run from the repository root. Python 3 with numpy and matplotlib.

```bash
python3 analysis/make_tables.py                       # analysis/tables/tab_*.tex, analysis/figures/fig_*.pdf, analysis/numbers.txt
python3 analysis/compare.py                           # analysis/compare.md
python3 analysis/make_stream_table.py                 # analysis/tables/tab_stream.tex
python3 stream/analysis/summarize.py results/stream   # results/stream/summary.md (streaming runs, per pass)
python3 analysis/build_stream_fig.py                  # analysis/figures/fig_stream.pdf (schematic, no data)
python3 analysis/analyze_raw.py                       # analysis/raw_summary.{csv,md}, figures/fig_rest_hist.pdf, fig_idle.pdf
```

| script | input | output | used for |
|---|---|---|---|
| `make_tables.py` | `results/net.csv`, `results/bench_*.csv`, `results/raw/*.log`; per-request CSVs `results/raw/<tag>_int8_n128_B8_K100_--window_1.csv` (not in the repository) | `tab_single`, `tab_pipeline`, `tab_net`, `tab_decomp`, `tab_ratio`, `tab_window`, `tab_load`, `tab_load_repeats`, `tab_breakdown`; `fig_overhead`, `fig_breakdown`, `fig_cdf`, `fig_scaling`, `fig_load`; `numbers.txt` (every number quoted in the request-service text, with its source) | request-service section |
| `analyze_raw.py` | all per-request CSVs `results/raw/*.csv` (not in the repository) | per-run queue / solve / transmit wait / remainder percentiles, the client's actual send rate, location of the slowest 1 %; remainder histogram; transmit wait and remainder vs. solve time | receive-timeout analysis, client limit, tails |
| `compare.py` | `results/net.csv` | GPUNetIO vs. best tuned CPU server per configuration; receive timeout 5 µs → 500 ns; send path `-W 0` → `-W 2`; host state; reruns against the original; idle-gap diagnostic | "Against a tuned host", "Latency under load" |
| `make_stream_table.py` | `results/stream/*.csv`, `*.log`, `*.chk`, `stream_event_review.json` | `tab_stream.tex` (also checks the `.chk` cross-checks byte for byte) | streaming measurements |
| `build_stream_fig.py` | — | `fig_stream.pdf` | streaming schematic |

## Per-request data

The per-request CSVs of the network runs (one row per request, ~1.5 GB) are not part of the
repository. Without them:

* `make_tables.py` recomputes everything else and keeps the shipped `tables/tab_breakdown.tex`,
  `figures/fig_breakdown.pdf` and `figures/fig_cdf.pdf`; the `[breakdown]` lines of the existing
  `numbers.txt` are carried over. The `[verify]` line is recomputed from the 291 run logs in
  `results/raw/*.log`.
* `analyze_raw.py` does nothing; its outputs (`raw_summary.csv`, `raw_summary.md`,
  `figures/fig_rest_hist.pdf`, `figures/fig_idle.pdf`) are shipped as computed from the full data.

With the CSVs placed in `results/raw/`, every script regenerates all of its outputs.

## Conventions

* Old results are never overwritten. A configuration measured again runs under a new tag
  `<series>_r<k>` (k = 2, 3, …): its SUMMARY row is appended to `results/net.csv` and its log gets a
  new name. Tables and figures report the **newest** run of every configuration under the series
  name; `numbers.txt` lists which configurations come from a rerun, and section 5 of `compare.md`
  compares every rerun with the original.
* Exception: for `doca_w2t` and `proxy_fast` at ≥ 400 kreq/s, Table `tab_load` and `fig_load` use
  the per-column median over the repeats (generation ≥ 2, no lost requests); `tab_load_repeats.tex`
  shows the range. The run `doca_w2t_r4` at 700 kreq/s lost 1.1 % of the replies at the client and
  is excluded.
* Tags are explained in the top-level `README.md`.
* remainder = RTT − queue − solve − tx = the part of the round trip outside the server's own time stamps.
* `stream_event_review.json` documents two receiver passes flagged `RUN INVALID` because frames of
  unrelated traffic arrived; the receiver classified and excluded them (they did not enter the
  accumulation). Only these two reviewed events are exempted.
