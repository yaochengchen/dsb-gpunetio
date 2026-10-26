#!/usr/bin/env python3
"""Supplementary analysis for the resubmission of the paper to IEEE Access
(manuscript Access-2026-50856, reviews received 26 October 2026).

This script and everything under supplementary_revision/ were ADDED AFTER the
original submission.  The original scripts in analysis/ and the original data in
results/ are not modified.  See supplementary_revision/README.md.

Run from the repository root (it imports the shared helpers of analysis/make_tables.py):

    python3 supplementary_revision/analysis/make_revision.py

Defaults: results/net.csv plus supplementary_revision/results/net_supplementary.csv
(rows present in both are counted once), power logs from
supplementary_revision/results/, per-request CSVs from results/raw/, outputs to
supplementary_revision/{figures,tables}/ and supplementary_revision/numbers_revision.txt.

Outputs
  figures/fig_load_cfgs.pdf    open-loop Poisson load for three request configurations,
                               GPUNetIO and tuned proxy, median and range over repeats
  figures/fig_decomp.pdf       Eq. (8) decomposition of the unloaded round trip, all 19
                               settings, mean components             (needs per-request CSVs)
  figures/fig_breakdown2.pdf   two-panel version of fig_breakdown: panel (b) without the
                               solve, so the 2-us GPU queue is visible (needs per-request CSVs)
  figures/fig_pipeline.pdf     saturated pipelines without the network vs. solve time
  tables/tab_energy.tex        server power and energy per request
  numbers_revision.txt         every number quoted in the revised text, with its source

Without the per-request CSVs in results/raw/ (they are not in the repository) the
three marked outputs are kept as shipped and their lines of the existing
numbers_revision.txt are carried over.
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "analysis"))
sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))   # the checkout the script is run from
import make_tables as mt  # noqa: E402  (shared helpers, styles and the reported-run selection, unchanged)

NOTES = []
DOCA = mt.DOCA
SERVERS = [(DOCA, "GPUNetIO"), ("proxy_fast", "proxy, tuned")]
# (configuration key, tag suffix of the revision runs or None for the reference runs, panel title)
LOAD_CFGS = [
    (mt.REF, None, "int8, $n=128$, $K=100$ (reference)"),
    (("pm1", 256, 8, 100), "pm256", "pm1, $n=256$, $K=100$"),
    (("int8", 192, 8, 400), "i192k400", "int8, $n=192$, $K=400$"),
]
STATS = ["kreq_s", "ok", "lost", "p50_us", "p99_us", "p999_us", "srv_p50_us", "srv_p99_us", "net_p50_us"]
WORK_CAP = 4096          # DSB_WORK_CAP, include/dsb_queue.h
N_WORKERS = 372          # GPUNetIO worker blocks (Table 2)


def note(s):
    NOTES.append(s)


def series_gen(tag):
    m = re.match(r"(.+)_r(\d+)$", tag)
    return (m.group(1), int(m.group(2))) if m else (tag, 1)


def summarize(rows):
    out = {f: (statistics.median(float(r[f]) for r in rows), min(float(r[f]) for r in rows),
               max(float(r[f]) for r in rows)) for f in STATS}
    out["runs"] = len(rows)
    out["tags"] = ",".join(r["tag"] for r in rows)
    return out


# ---------------------------------------------------------------- load, three configurations
def load_points(all_net, latest, key, suffix, series):
    """offered rate -> (median, min, max) of every column over the runs.
    Reference configuration: exactly the runs the paper reports (median over the
    valid repeats at >= 400 kreq/s, the latest run below).  Revision configurations:
    all runs <series>_<suffix>_r<k>; runs with lost requests are kept, because loss
    beyond the saturation point is part of the result."""
    pts = {}
    if suffix is None:
        groups, _ = mt.repeat_groups(all_net)
        for r in latest:
            if r["tag"] == series and mt.key_of(r) == key and float(r.get("rate") or 0) > 0 and int(r["ok"]) > 0:
                rate = float(r["rate"])
                g = groups.get((series, rate))
                pts[rate] = summarize([x for _, x in g] if g and len(g) > 1 else [r])
    else:
        by = defaultdict(list)
        for r in all_net:
            if series_gen(r["tag"])[0] == f"{series}_{suffix}" and mt.key_of(r) == key and float(r.get("rate") or 0) > 0:
                by[float(r["rate"])].append(r)
        for rate, rows in by.items():
            pts[rate] = summarize(rows)
    return dict(sorted(pts.items()))


def load_notes(all_net, latest):
    for key, suffix, _ in LOAD_CFGS:
        for series, name in SERVERS:
            for rate, s in load_points(all_net, latest, key, suffix, series).items():
                sent = s["ok"][0] + s["lost"][0]
                note(f"[load {key}] {series} offered {rate:.0f}: runs {s['runs']} ({s['tags']}); achieved {s['kreq_s'][0]:.1f} "
                     f"({s['kreq_s'][1]:.1f}-{s['kreq_s'][2]:.1f}) kreq/s; lost {s['lost'][0]:.0f} ({100 * s['lost'][0] / sent:.1f}%); "
                     f"rtt p50 {s['p50_us'][0]:.1f} ({s['p50_us'][1]:.1f}-{s['p50_us'][2]:.1f}) p99 {s['p99_us'][0]:.1f} "
                     f"({s['p99_us'][1]:.1f}-{s['p99_us'][2]:.1f}); srv p50 {s['srv_p50_us'][0]:.1f} p99 {s['srv_p99_us'][0]:.1f}; "
                     f"rtt-srv p50 {s['net_p50_us'][0]:.1f}")
    # Saturated worker pool (int8 n=192 K=400): the work ring holds at most C requests,
    # so by Little's law the residence of an admitted request is about C / X at
    # completion rate X; the effective solve time under full load is N_w / X.
    key = ("int8", 192, 8, 400)
    for series, _ in SERVERS:
        for rate, s in load_points(all_net, latest, key, "i192k400", series).items():
            x = s["kreq_s"][0] * 1e3
            if rate >= 300000 and x > 0:
                note(f"[little i192k400] {series} offered {rate:.0f}: achieved {x / 1e3:.1f} kreq/s, "
                     f"C/X = {WORK_CAP / x * 1e6:.0f} us, measured srv p50 {s['srv_p50_us'][0]:.0f} us, "
                     f"N_w/X = {N_WORKERS / x * 1e6:.0f} us (effective solve under full load)")


def fig_load_cfgs(all_net, latest, out):
    plt = mt._plt()
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.3), sharex="col")
    for j, (key, suffix, title) in enumerate(LOAD_CFGS):
        ax_t, ax_l = axes[0][j], axes[1][j]
        xmax = 0.0
        for series, name in SERVERS:
            c, m, _, _ = mt.STYLE[series]
            pts = load_points(all_net, latest, key, suffix, series)
            if not pts:
                continue
            x = [r / 1e3 for r in pts]
            xmax = max(xmax, max(x))
            for col, ls, lab, mfc in (("p50_us", "-", "p50", c), ("p99_us", "--", "p99", "white")):
                y = [pts[r][col][0] for r in pts]
                lo = [pts[r][col][0] - pts[r][col][1] for r in pts]
                hi = [pts[r][col][2] - pts[r][col][0] for r in pts]
                ax_l.errorbar(x, y, yerr=[lo, hi], color=c, marker=m, ls=ls, mfc=mfc, capsize=1.5,
                              elinewidth=0.7, lw=1.1, ms=3.2, label=f"{name}, {lab}")
            ax_t.plot(x, [pts[r]["kreq_s"][0] for r in pts], color=c, marker=m, ms=3.2, lw=1.1, label=name)
            lossy = [(r / 1e3, pts[r]["kreq_s"][0]) for r in pts if pts[r]["lost"][0] > 0]
            if lossy:
                ax_t.plot([p[0] for p in lossy], [p[1] for p in lossy], ls="none", marker="x", color="black", ms=4.5)
        if xmax:
            ax_t.plot([0, xmax], [0, xmax], color="gray", lw=0.7, ls=":", label="completed = offered")
        ax_t.set_title(title, fontsize=7)
        ax_l.set_yscale("log")
        from matplotlib.ticker import LogLocator, NullFormatter
        ax_l.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax_l.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax_l.yaxis.set_minor_formatter(NullFormatter())
        ax_l.set_xlabel("offered Poisson load (kreq/s)")
    axes[0][0].set_ylabel("completed (kreq/s)")
    axes[1][0].set_ylabel("client RTT ($\\mu$s)")
    axes[0][0].legend(frameon=False, fontsize=5.5, loc="upper left")
    axes[1][0].legend(frameon=False, fontsize=5.5, loc="upper left")
    fig.tight_layout()
    mt._save(fig, out, "fig_load_cfgs.pdf")


# ---------------------------------------------------------------- Eq. (8) for all settings
DECOMP_TAGS = [("launch_fast", "launch, tuned", "////"), ("proxy_fast", "proxy, tuned", "...."), (DOCA, "GPUNetIO", "")]
PARTS = [("solve", "#c3c2b7", "$t_{\\mathrm{solve}}$"),
         ("queue", "#eda100", "$t_q$ (GPU work ring)"),
         ("io", "#2a78d6", "server I/O: CPU path (baselines) or transmit wait (GPUNetIO)"),
         ("rest", "#1baf7a", "residual: $t_{\\mathrm{wire}}$ and the uninstrumented parts of $t_{\\mathrm{in}}$, $t_{\\mathrm{out}}$")]


def decomp_stats(raw, latest):
    import numpy as np
    by = mt.net_unloaded(latest)
    keys = mt.sort_keys(by)
    res = {}
    for k in keys:
        cfg = f"{k[0]}_n{k[1]}_B{k[2]}_K{k[3]}"
        for t, _, _ in DECOMP_TAGS:
            files = glob.glob(os.path.join(raw, f"{mt.raw_tag(t, cfg)}_{cfg}_--window_1.csv"))
            if not files:
                note(f"[decomp] missing raw file for {t} {cfg}")
                continue
            rows = mt.read_csv(files[:1])
            a = {c: np.array([int(r[c]) for r in rows], dtype=np.float64) / 1e3
                 for c in ("rtt_ns", "queue_ns", "solve_ns", "tx_ns")}
            ok = a["rtt_ns"] > 0
            rest = a["rtt_ns"] - a["queue_ns"] - a["solve_ns"] - a["tx_ns"]
            res[(k, t)] = {"solve": a["solve_ns"][ok].mean(), "queue": a["queue_ns"][ok].mean(),
                           "io": a["tx_ns"][ok].mean(), "rest": rest[ok].mean(), "rtt": a["rtt_ns"][ok].mean()}
            d = res[(k, t)]
            note(f"[decomp] {k} {t} ({os.path.basename(files[0])}): mean rtt {d['rtt']:.1f} = solve {d['solve']:.1f} "
                 f"+ queue {d['queue']:.1f} + io {d['io']:.1f} + rest {d['rest']:.1f} us")
    for t, _, _ in DECOMP_TAGS:
        ov = [res[(k, t)]["rtt"] - res[(k, t)]["solve"] for k in keys if (k, t) in res]
        if ov:
            note(f"[decomp] {t}: mean RTT minus mean solve over {len(ov)} settings: min {min(ov):.1f} "
                 f"median {statistics.median(ov):.1f} max {max(ov):.1f} us")
    return res, keys


def fig_decomp(res, keys, out):
    from matplotlib.patches import Patch
    plt = mt._plt()
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.6), sharex=True)
    w = 0.27
    for ax, comps, title in ((axes[0], PARTS, "(a) all components of Eq. (8)"),
                             (axes[1], PARTS[1:], "(b) all components except $t_{\\mathrm{solve}}$")):
        for si, (t, _, hatch) in enumerate(DECOMP_TAGS):
            xs = [i + (si - 1) * w for i in range(len(keys))]
            bottom = [0.0] * len(keys)
            for part, col, lab in comps:
                vals = [res.get((k, t), {}).get(part, 0.0) for k in keys]
                ax.bar(xs, vals, w * 0.92, bottom=bottom, color=col, edgecolor="white", linewidth=0.3,
                       hatch=hatch, label=lab if si == 0 else None)
                bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(title, fontsize=7, loc="left")
        ax.set_ylabel("mean per request ($\\mu$s)")
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels([f"{k[0]} {k[1]}/{k[2]}/{k[3]}" for k in keys], fontsize=5.5, rotation=35, ha="right")
    axes[1].set_xlabel("format, $n$/$B$/$K$")
    h, l = axes[0].get_legend_handles_labels()
    h += [Patch(facecolor="white", edgecolor="black", hatch=hatch, label=name) for _, name, hatch in DECOMP_TAGS]
    l += [name for _, name, _ in DECOMP_TAGS]
    axes[0].legend(h, l, frameon=False, fontsize=5.5, loc="upper left", ncol=2)
    fig.tight_layout()
    mt._save(fig, out, "fig_decomp.pdf")


# ---------------------------------------------------------------- R1-4: solve time under load
def solve_under_load(raw, all_net):
    """In-kernel solve and work-ring wait of GPUNetIO for the reference request, unloaded
    against the repeated open-loop runs: do the I/O warps take compute from the workers?"""
    import numpy as np
    cfg = "int8_n128_B8_K100"

    def pct(path):
        rows = mt.read_csv([path])
        s = np.array([int(r["solve_ns"]) for r in rows if int(r["rtt_ns"]) > 0], dtype=np.float64) / 1e3
        q = np.array([int(r["queue_ns"]) for r in rows if int(r["rtt_ns"]) > 0], dtype=np.float64) / 1e3
        return np.median(s), np.percentile(s, 99), np.median(q), np.percentile(q, 99)
    unl = glob.glob(os.path.join(raw, f"{mt.raw_tag(DOCA, cfg)}_{cfg}_--window_1.csv"))
    if unl:
        s50, s99, q50, q99 = pct(unl[0])
        note(f"[solve vs load] unloaded ({os.path.basename(unl[0])}): solve p50 {s50:.1f} p99 {s99:.1f}; queue p50 {q50:.2f} p99 {q99:.2f} us")
    groups, _ = mt.repeat_groups(all_net)
    for (series, rate), lst in sorted(groups.items()):
        if series != DOCA:
            continue
        vals = []
        for _, r in lst:
            p = os.path.join(raw, f"{r['tag']}_{cfg}_--rate_{int(rate)}_--poisson.csv")
            if os.path.exists(p):
                vals.append(pct(p))
        if vals:
            note(f"[solve vs load] offered {rate:.0f}, {len(vals)} runs: solve p50 "
                 f"{statistics.median(v[0] for v in vals):.1f} p99 {statistics.median(v[1] for v in vals):.1f}; "
                 f"queue p50 {statistics.median(v[2] for v in vals):.2f} p99 {statistics.median(v[3] for v in vals):.2f} us")


# ---------------------------------------------------------------- pipelines without the network
def fig_pipeline(bench, out):
    plt = mt._plt()
    by = mt.bench_by(bench)
    g_pts, p_pts, W = [], [], None
    for k, d in by.items():
        g, p, s = d.get("persistent-gpu"), d.get("proxy-cpu"), d.get("single-solve")
        if not (g and p and s):
            continue
        ts = float(s["p50_us"])
        W = int(g["workers"])
        g_pts.append((ts, float(g["rate_req_s"]) / 1e3))
        p_pts.append((ts, float(p["rate_req_s"]) / 1e3))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    if not g_pts:
        mt._empty(ax)
    else:
        ax.scatter([p[0] for p in g_pts], [p[1] for p in g_pts], s=12, marker="^", color=mt.STYLE[DOCA][0],
                   label="GPU-resident rings (GPU producer)")
        ax.scatter([p[0] for p in p_pts], [p[1] for p in p_pts], s=12, marker="o", color=mt.STYLE["proxy_fast"][0],
                   label="host-mapped rings (CPU proxy)")
        xs = [min(p[0] for p in g_pts) * 0.9, max(p[0] for p in g_pts) * 1.1]
        ax.plot(xs, [W / x * 1e3 for x in xs], color="gray", lw=0.8, ls=":", label=f"$N_w/t_{{\\mathrm{{solve}}}}$, $N_w={W}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.legend(frameon=False, fontsize=6, loc="lower left")
        note(f"[pipeline] {len(g_pts)} configurations with both pipelines; GPU rings {min(p[1] for p in g_pts):.0f}-"
             f"{max(p[1] for p in g_pts):.0f} kreq/s; CPU rings {min(p[1] for p in p_pts):.0f}-{max(p[1] for p in p_pts):.0f} kreq/s")
    ax.set_xlabel("in-kernel solve time ($\\mu$s)")
    ax.set_ylabel("saturated throughput (kreq/s)")
    fig.tight_layout()
    mt._save(fig, out, "fig_pipeline.pdf")


# ---------------------------------------------------------------- power and energy
def _naive_utc(s, fmt):
    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()


def read_cpu(res, label):
    """perf stat -a -e power/energy-pkg/ -I 1000 -x, : cumulative time, Joules per
    interval.  Returns [(epoch, W)] and the PU2 UTC offset in seconds, derived from
    the '# started on' line (local time) and the .start file (epoch, same host)."""
    p, st = os.path.join(res, f"power_cpu_{label}.csv"), os.path.join(res, f"power_cpu_{label}.start")
    if not (os.path.exists(p) and os.path.exists(st)):
        return None, None
    t0 = float(open(st).read().split()[0])
    pts, prev, tz = [], 0.0, None
    for line in open(p):
        if line.startswith("# started on"):
            local = _naive_utc(line.split("started on", 1)[1].strip(), "%a %b %d %H:%M:%S %Y")
            tz = round((local - t0) / 900.0) * 900
            continue
        f = line.strip().split(",")
        if len(f) < 4 or f[2] != "Joules":
            continue
        try:
            t, j = float(f[0]), float(f[1])
        except ValueError:
            continue
        if t > prev:
            pts.append((t0 + (prev + t) / 2, j / (t - prev)))
        prev = t
    return pts, tz


def read_gpu(res, label, tz):
    p = os.path.join(res, f"power_gpu_{label}.csv")
    if not os.path.exists(p) or tz is None:
        return None
    pts = []
    with open(p) as fh:
        next(fh, None)
        for line in fh:
            f = [x.strip() for x in line.split(",")]
            if len(f) < 2 or not f[1].endswith("W"):
                continue
            pts.append((_naive_utc(f[0], "%Y/%m/%d %H:%M:%S.%f") - tz, float(f[1][:-1])))
    return pts


def wmean(pts, a, b):
    v = [w for t, w in (pts or []) if a <= t <= b]
    return (statistics.mean(v), len(v)) if v else (None, 0)


def read_energy_runs(path, header):
    runs, cur = defaultdict(lambda: {"idle": None, "loads": []}), None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("SUMMARY"):
            if cur is not None:
                cur["summary"] = dict(zip(header, line.split(",")))
            continue
        f = line.split()
        t, what, label = float(f[0]), f[1], f[2]
        if what == "idle_start":
            runs[label]["idle"] = t
        elif what == "start":
            cur = {"rate": float(f[3]), "t0": t}
            runs[label]["loads"].append(cur)
        elif what == "end" and cur is not None:
            cur["t1"] = t
            cur = None
    return runs


ENERGY_ROWS = [  # (host state, server label or None, no-server idle label)
    ("default", None, "idle_default"),
    ("default", "doca_default", None),
    ("tuned", None, "idle_tuned"),
    ("tuned", "doca_tuned", None),
    ("tuned", "proxy_tuned", None),
]


def tab_energy(res, net_path, out):
    if not os.path.exists(os.path.join(res, "energy_runs.txt")):
        note("[energy] results/energy_runs.txt missing")
        return
    header = open(net_path).readline().strip().split(",")
    runs = read_energy_runs(os.path.join(res, "energy_runs.txt"), header)
    lines = [r"\begin{tabular}{llrrrrrrr}\toprule",
             r"host & server & offered & completed & CPU & GPU & total & energy & RTT p50 \\",
             r" & & kreq/s & kreq/s & W & W & W & mJ/req & $\mu$s \\ \midrule"]
    totals = {}
    for state, label, idle_label in ENERGY_ROWS:
        lab = label or idle_label
        cpu, tz = read_cpu(res, lab)
        gpu = read_gpu(res, lab, tz)
        if cpu is None or gpu is None:
            note(f"[energy] {lab}: power logs missing")
            continue
        if idle_label:  # no server: whole log, 3 s trimmed at both ends
            a, b = min(t for t, _ in cpu) + 3, max(t for t, _ in cpu) - 3
            c, nc = wmean(cpu, a, b)
            g, ng = wmean(gpu, a, b)
            totals[(state, "none", 0)] = (c, g)
            lines.append(f"{state} & none & -- & -- & {c:.0f} & {g:.0f} & {c + g:.0f} & -- & -- \\\\")
            note(f"[energy] {lab}: no server, cpu {c:.1f} W ({nc} samples), gpu {g:.1f} W ({ng} samples)")
            continue
        server = "GPUNetIO" if label.startswith("doca") else "proxy"
        r = runs.get(label)
        if not r or not r["loads"]:
            note(f"[energy] {label}: no run markers")
            continue
        if r["idle"] is not None:  # server running, no traffic
            a, b = r["idle"] + 3, r["loads"][0]["t0"] - 2
            c, nc = wmean(cpu, a, b)
            g, ng = wmean(gpu, a, b)
            if c is not None and g is not None:
                totals[(state, server, 0)] = (c, g)
                lines.append(f"{state} & {server} & 0 & -- & {c:.0f} & {g:.0f} & {c + g:.0f} & -- & -- \\\\")
                note(f"[energy] {label}: resident, no traffic: cpu {c:.1f} W ({nc}), gpu {g:.1f} W ({ng})")
        for ld in r["loads"]:
            if "t1" not in ld:
                continue
            a, b = (ld["t0"] + 4, ld["t1"] - 1) if ld["t1"] - ld["t0"] > 8 else (ld["t0"] + 2, ld["t1"] - 0.5)
            c, nc = wmean(cpu, a, b)
            g, ng = wmean(gpu, a, b)
            s = ld.get("summary", {})
            x = float(s.get("kreq_s", "nan"))
            if c is None or g is None:
                note(f"[energy] {label} {ld['rate']:.0f}: no samples in the window")
                continue
            mj = (c + g) / (x * 1e3) * 1e3
            totals[(state, server, ld["rate"])] = (c, g, x)
            lines.append(f" & & {ld['rate'] / 1e3:.0f} & {x:.1f} & {c:.0f} & {g:.0f} & {c + g:.0f} & {mj:.2f} & {float(s.get('p50_us', 'nan')):.1f} \\\\")
            note(f"[energy] {label} offered {ld['rate']:.0f}: completed {x:.1f} kreq/s, cpu {c:.1f} W ({nc}), gpu {g:.1f} W ({ng}), "
                 f"total {c + g:.1f} W, {mj:.3f} mJ/request; rtt p50 {s.get('p50_us')} p99 {s.get('p99_us')} lost {s.get('lost')}")
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule\end{tabular}")
    mt.write(os.path.join(out, "tables", "tab_energy.tex"), "\n".join(lines) + "\n")
    for rate in (200000.0, 400000.0, 600000.0):
        a, b = totals.get(("default", "GPUNetIO", rate)), totals.get(("tuned", "proxy", rate))
        if a and b:
            ta, tb = a[0] + a[1], b[0] + b[1]
            note(f"[energy] offered {rate:.0f}: GPUNetIO/default {ta:.1f} W vs proxy/tuned {tb:.1f} W: "
                 f"{100 * (1 - ta / tb):.1f}% lower; per request {ta / a[2]:.3f} vs {tb / b[2]:.3f} mJ")
    for state in ("default", "tuned"):
        n = totals.get((state, "none", 0))
        d = totals.get((state, "GPUNetIO", 0))
        if n and d:
            note(f"[energy] {state}: resident GPUNetIO kernels without traffic add {d[1] - n[1]:.1f} W GPU, {d[0] - n[0]:.1f} W CPU over no server")
    n_d, n_t = totals.get(("default", "none", 0)), totals.get(("tuned", "none", 0))
    if n_d and n_t:
        note(f"[energy] tuned host state costs {n_t[0] - n_d[0]:.1f} W CPU package at idle (default {n_d[0]:.1f}, tuned {n_t[0]:.1f})")


# ---------------------------------------------------------------- two-panel breakdown (R2-10)
def fig_breakdown2(raw, out, cfg="int8_n128_B8_K100"):
    """Same data as fig_breakdown of analysis/make_tables.py (mean components of the
    unloaded reference request, 100,000 requests per server); panel (b) leaves out
    the solve so that the 2-us GPU work-ring wait of GPUNetIO is visible."""
    import numpy as np
    stats = {}
    for t, name in mt.TAGS:
        files = glob.glob(os.path.join(raw, f"{mt.raw_tag(t, cfg)}_{cfg}_--window_1.csv"))
        if not files:
            continue
        rows = mt.read_csv(files[:1])
        a = {k: np.array([int(r[k]) for r in rows], dtype=np.float64) / 1e3 for k in ("rtt_ns", "queue_ns", "solve_ns", "tx_ns")}
        ok = a["rtt_ns"] > 0
        stats[t] = {"solve": a["solve_ns"][ok].mean(), "queue": a["queue_ns"][ok].mean(), "io": a["tx_ns"][ok].mean(),
                    "rest": (a["rtt_ns"] - a["queue_ns"] - a["solve_ns"] - a["tx_ns"])[ok].mean(), "rtt": a["rtt_ns"][ok].mean()}
        s = stats[t]
        note(f"[breakdown2] {t} ({os.path.basename(files[0])}): mean rtt {s['rtt']:.1f} = solve {s['solve']:.1f} + queue {s['queue']:.2f} "
             f"+ io {s['io']:.2f} + rest {s['rest']:.2f} us")
    if not stats:
        return False
    plt = mt._plt()
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.9))
    servers = [(t, name) for t, name in mt.TAGS if t in stats]
    names = [name for _, name in servers]
    parts = [("solve", "#c3c2b7", "solve (same kernel)"), ("queue", "#eda100", "GPU queue"),
             ("io", "#2a78d6", "server I/O (CPU or GPU)"), ("rest", "#1baf7a", "unmeasured residual")]
    for ax, comps, with_solve in ((axes[0], parts, True), (axes[1], parts[1:], False)):
        left = [0.0] * len(names)
        for key, col, lab in comps:
            vals = [stats[t][key] for t, _ in servers]
            ax.barh(names, vals, left=left, color=col, edgecolor="white", linewidth=1, label=lab, height=0.6)
            left = [l + v for l, v in zip(left, vals)]
        for i, (t, _) in enumerate(servers):
            if with_solve:
                txt = f"{stats[t]['rtt']:.0f} $\\mu$s"
            elif t == DOCA:
                txt = f"{left[i]:.1f} $\\mu$s (GPU queue {stats[t]['queue']:.1f})"
            else:
                txt = f"{left[i]:.1f} $\\mu$s"
            ax.annotate(txt, (left[i], i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
        ax.set_xlim(0, max(left) * 1.2)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    axes[0].set_title("(a) all components", fontsize=7, loc="left")
    axes[1].set_title("(b) all components except the solve", fontsize=7, loc="left")
    axes[0].set_xlabel("mean time per request ($\\mu$s)")
    axes[1].set_xlabel("mean time per request, solve excluded ($\\mu$s)")
    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h, l, frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    fig.tight_layout()
    mt._save(fig, out, "fig_breakdown2.pdf")
    return True


def read_all_net(paths):
    """SUMMARY rows of several net.csv files in order; a row present in more than one
    file (the development tree's results/net.csv already contains the supplementary
    rows) is counted once."""
    rows, seen = [], set()
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r.get("SUMMARY") != "SUMMARY":
                    continue
                k = tuple(sorted(r.items()))
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
    return rows


def main():
    supp = "supplementary_revision"
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="results/net.csv")
    ap.add_argument("--extra-net", default=os.path.join(supp, "results", "net_supplementary.csv"))
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--res", default=os.path.join(supp, "results"),
                    help="directory with energy_runs.txt and power_*")
    ap.add_argument("--bench", nargs="*", default=glob.glob("results/bench_*.csv"))
    ap.add_argument("--out", default=supp, help="figures/ and tables/ are written below this directory")
    ap.add_argument("--numbers", default=os.path.join(supp, "numbers_revision.txt"))
    a = ap.parse_args()
    all_net = read_all_net([p for p in (a.net, a.extra_net) if p and os.path.exists(p)])
    latest = mt.latest_only(all_net)          # also fills mt.RAWTAG for the raw-file lookups
    bench = mt.read_csv(mt.bench_files(a.bench)) if a.bench else []
    load_notes(all_net, latest)
    fig_load_cfgs(all_net, latest, a.out)
    have_raw = bool(glob.glob(os.path.join(a.raw, "*.csv")))
    if have_raw:
        res, keys = decomp_stats(a.raw, latest)
        fig_decomp(res, keys, a.out)
        solve_under_load(a.raw, all_net)
        fig_breakdown2(a.raw, a.out)
    else:
        # per-request CSVs not available: keep the shipped outputs and their numbers
        if os.path.exists(a.numbers):
            NOTES.extend(l.rstrip("\n") for l in open(a.numbers)
                         if l.startswith(("[decomp]", "[solve vs load]", "[breakdown2]")))
        print("no per-request CSVs in", a.raw, "- fig_decomp, fig_breakdown2 and the solve-under-load numbers kept as shipped")
    fig_pipeline(bench, a.out)
    energy_header_src = a.extra_net if a.extra_net and os.path.exists(a.extra_net) else a.net
    tab_energy(a.res, energy_header_src, a.out)
    mt.write(a.numbers, "\n".join(NOTES) + "\n")


if __name__ == "__main__":
    main()
