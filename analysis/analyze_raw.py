#!/usr/bin/env python3
"""Per-request analysis of the network runs (results/raw/*.csv).

    python3 analysis/analyze_raw.py [--raw results/raw] [--out analysis]

Every analysis quoted in the request-service section that goes beyond the
SUMMARY lines of results/net.csv is produced here:

  analysis/raw_summary.csv   one row per run: percentiles of queue / solve /
                             transmit wait / remainder / RTT, the client's
                             actual send rate (from its own t_send stamps; the
                             open-loop client cannot exceed ~620 kreq/s), and
                             where in the run the slowest 1% of requests fall
  analysis/raw_summary.md    the same, readable, plus the checks below
  analysis/figures/fig_rest_hist.pdf   (newest run of each series)
                             distribution of the remainder (RTT - server
                             residence) of the reference request, per server:
                             the uniform 0-5 us component of the 5-us receive
                             timeout (doca_w2) and its absence at 500 ns (doca_w2t)
  analysis/figures/fig_idle.pdf
                             GPU-initiated server: transmit wait and remainder
                             against the solve time (= idle gap between requests
                             in closed loop), every unloaded run of the DOCA tags;
                             locates the excess of the longest requests

Remainder = rtt - queue - solve - tx, i.e. the part of the round trip outside
the server's own time stamps.  For the CPU servers queue = 0 and tx is the CPU
residence minus the solve (their convention, see dsb_client.cpp).
"""
import argparse
import glob
import os
import re

import numpy as np

REF = "int8_n128_B8_K100"
Q = (1, 25, 50, 75, 99, 99.9)


def parse_name(path):
    """<tag>_<fmt>_n<n>_B<B>_K<K>_--window_<w>.csv or ..._--rate_<r>_--poisson.csv"""
    b = os.path.basename(path)[:-4]
    m = re.match(r"(?P<tag>.+?)_(?P<fmt>int8|pm1|ternary)_n(?P<n>\d+)_B(?P<B>\d+)_K(?P<K>\d+)_--(?P<mode>window|rate)_(?P<v>\d+)", b)
    if not m:
        return None
    d = m.groupdict()
    d["cfg"] = f"{d['fmt']}_n{d['n']}_B{d['B']}_K{d['K']}"
    return d


def load(path):
    a = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if a.size == 0:
        return None
    a = np.atleast_1d(a)
    rtt = a["rtt_ns"].astype(float)
    ok = rtt > 0
    res = {k: a[k].astype(float)[ok] / 1e3 for k in ("queue_ns", "solve_ns", "tx_ns")}
    res["rtt"] = rtt[ok] / 1e3
    res["rest"] = res["rtt"] - res["queue_ns"] - res["solve_ns"] - res["tx_ns"]
    ts = a["t_send_ns"].astype(float)
    span = (ts.max() - ts.min()) / 1e9
    res["send_kreqs"] = (len(ts) / span / 1e3) if span > 0 else float("nan")
    res["req_id"] = a["req_id"][ok]
    res["n_all"] = len(a)
    return res


def summarize(raw):
    rows = []
    for p in sorted(glob.glob(os.path.join(raw, "*.csv"))):
        meta = parse_name(p)
        if not meta:
            continue
        d = load(p)
        if d is None or d["rtt"].size == 0:
            continue
        r = dict(meta)
        r["ok"] = int(d["rtt"].size)
        r["send_kreqs"] = round(d["send_kreqs"], 1)
        for nm, key in (("queue", "queue_ns"), ("solve", "solve_ns"), ("tx", "tx_ns"), ("rest", "rest"), ("rtt", "rtt")):
            for q, v in zip(Q, np.percentile(d[key], Q)):
                r[f"{nm}_p{q:g}"] = round(float(v), 2)
        # where are the slowest 1 %?  fraction of them in the first 5 % of the run
        thr = np.percentile(d["rtt"], 99)
        ids = d["req_id"][d["rtt"] > thr]
        r["tail_first5pct"] = round(float(np.mean(ids < 0.05 * d["n_all"])) if ids.size else 0.0, 2)
        rows.append(r)
    return rows


def write_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print("wrote", path)


def newest(raw, series, suffix):
    """Raw file of the newest run of a series: <series>_r<k>_<suffix> with the
    largest k, else <series>_<suffix>.  Reruns never overwrite older files."""
    best, path = 0, None
    for p in glob.glob(os.path.join(raw, f"{series}*_{suffix}")):
        t = os.path.basename(p)[: -len(suffix) - 1]
        m = re.fullmatch(re.escape(series) + r"(?:_r(\d+))?", t)
        if m:
            k = int(m.group(1) or 1)
            if k > best:
                best, path = k, p
    return path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


COL = {"launch_fast": "#2a78d6", "proxy_fast": "#eb6834", "doca_w2": "#7fd6b4", "doca_w2t": "#1baf7a"}


def fig_rest_hist(raw, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bins = np.arange(8, 24, 0.25)
    for t in ("launch_fast", "proxy_fast", "doca_w2", "doca_w2t"):
        p = newest(raw, t, f"{REF}_--window_1.csv")
        if p is None:
            continue
        d = load(p)
        ax.hist(d["rest"], bins=bins, histtype="step", density=True, color=COL[t], lw=1.3,
                label={"doca_w2": "GPUNetIO, 5-us timeout", "doca_w2t": "GPUNetIO, 500-ns timeout",
                       "launch_fast": "launch, tuned", "proxy_fast": "proxy, tuned"}[t])
    ax.set_xlabel("RTT $-$ server residence ($\\mu$s)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    fig.savefig(os.path.join(out, "figures", "fig_rest_hist.pdf"), bbox_inches="tight")
    print("wrote", os.path.join(out, "figures", "fig_rest_hist.pdf"))


def fig_idle(rows, out):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for t, m in (("doca_w2", "o"), ("doca_w2t", "^"), ("diag_w2t", "s")):
        pts = sorted((r["solve_p50"], r["tx_p50"], r["rest_p50"]) for r in rows
                     if r["tag"] == t and r["mode"] == "window" and r["v"] == "1")
        if not pts:
            continue
        x = [p[0] for p in pts]
        axes[0].plot(x, [p[1] for p in pts], m, label=t)
        axes[1].plot(x, [p[2] for p in pts], m, label=t)
    axes[0].set_ylabel("transmit wait p50 ($\\mu$s)")
    axes[1].set_ylabel("remainder p50 ($\\mu$s)")
    for ax in axes:
        ax.set_xlabel("solve time p50 ($\\mu$s) = idle gap in closed loop")
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig_idle.pdf"), bbox_inches="tight")
    print("wrote", os.path.join(out, "figures", "fig_idle.pdf"))


def write_md(rows, path):
    L = ["# Per-request analysis of the network runs", "",
         "Generated by analysis/analyze_raw.py from results/raw. Times in us; rest = RTT - queue - solve - tx.", ""]
    L += ["## Reference configuration (int8 n=128 B=8 K=100), closed loop, 1 outstanding", "",
          "| tag | queue p50 | tx p50 | rest p1 | rest p50 | rest p99 | RTT p50 | RTT p99 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["cfg"] == REF and r["mode"] == "window" and r["v"] == "1":
            L.append(f"| {r['tag']} | {r['queue_p50']} | {r['tx_p50']} | {r['rest_p1']} | {r['rest_p50']} | {r['rest_p99']} | {r['rtt_p50']} | {r['rtt_p99']} |")
    L += ["", "## Open-loop runs: offered vs actually sent (client limit)", "",
          "| tag | offered kreq/s | sent kreq/s | queue p50 | tx p50 | rest p50 | RTT p99 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["mode"] == "rate" and r["cfg"] == REF and int(r["v"]) >= 400000:
            L.append(f"| {r['tag']} | {int(r['v']) // 1000} | {r['send_kreqs']} | {r['queue_p50']} | {r['tx_p50']} | {r['rest_p50']} | {r['rtt_p99']} |")
    L += ["", "## Tails: unloaded runs whose p99 exceeds the median by more than 20 %", "",
          "| tag | config | RTT p50 | RTT p99 | queue p99 | tx p99 | rest p99 | share of slowest 1 % in first 5 % of run |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["mode"] == "window" and r["v"] == "1" and r["rtt_p99"] > 1.2 * r["rtt_p50"]:
            L.append(f"| {r['tag']} | {r['cfg']} | {r['rtt_p50']} | {r['rtt_p99']} | {r['queue_p99']} | {r['tx_p99']} | {r['rest_p99']} | {r['tail_first5pct']} |")
    L += ["", "## GPU-initiated server: transmit wait and remainder against the solve time", "",
          "| tag | config | solve p50 | tx p50 | tx p99 | rest p50 | rest p99 |", "|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["tag"].startswith(("doca_w2", "diag")) and r["mode"] == "window" and r["v"] == "1"),
                    key=lambda r: (r["tag"], r["solve_p50"])):
        L.append(f"| {r['tag']} | {r['cfg']} | {r['solve_p50']} | {r['tx_p50']} | {r['tx_p99']} | {r['rest_p50']} | {r['rest_p99']} |")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    rows = summarize(a.raw)
    if not rows:
        print("no per-request CSVs in", a.raw, "(not in the repository); shipped raw_summary.* and figures kept")
        return
    write_csv(rows, os.path.join(a.out, "raw_summary.csv"))
    write_md(rows, os.path.join(a.out, "raw_summary.md"))
    fig_rest_hist(a.raw, a.out)
    fig_idle(rows, a.out)


if __name__ == "__main__":
    main()
