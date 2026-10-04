#!/usr/bin/env python3
"""Turn the measured CSVs into the paper's tables, figures and quoted numbers.

    python3 analysis/make_tables.py [--bench results/bench_*.csv] [--net results/net.csv]
                                    [--raw results/raw] [--out analysis]

Writes analysis/tables/tab_*.tex, analysis/figures/fig_*.pdf and analysis/numbers.txt
(every number quoted in the manuscript text, with the line of the CSV it comes
from).  A table whose input is missing is written with dashes, so the
manuscript always compiles and never shows a number that was not measured.

The per-request CSVs (results/raw/*.csv, ~1.5 GB) are not part of this
repository.  Without them the per-request breakdown (tab_breakdown.tex,
fig_breakdown.pdf, fig_cdf.pdf) is not recomputed: the files shipped in
analysis/ are kept and the [breakdown] lines of the existing numbers.txt are
carried over.  Everything else is recomputed from results/.
"""
import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

FMT = {"1": "pm1", "2": "ternary", "3": "int8", "pm1": "pm1", "ternary": "ternary", "int8": "int8"}
# The GPU-initiated server is reported with its batched send path (-W 2) and the
# 500-ns receive timeout (tag doca_w2t; doca_w2 is the same with the 5-us timeout,
# kept for the text).  Its latency does not depend on the host state (the -W 0 runs on the
# default and the tuned host, tags doca and doca_fast, agree within 1 us), so it
# has one column.  The CPU-mediated servers are reported on the default host
# (launch, proxy) and after scripts/host_fast_state.sh (launch_fast, proxy_fast).
DOCA = "doca_w2t"
TAGS = [("launch", "launch, default"), ("proxy", "proxy, default"),
        ("launch_fast", "launch, tuned"), ("proxy_fast", "proxy, tuned"), (DOCA, "GPUNetIO")]
# only in the load table/figure: the per-reply send path of the official sample (-W 0)
TAGS_LOAD = TAGS + [("doca_fast", "GPUNetIO, per-reply send")]
# fixed categorical colours (validated palette) and markers; the tuned variant of a
# CPU server shares its colour and uses filled markers / solid lines, the default
# host open markers / dashed lines
STYLE = {"launch": ("#2a78d6", "s", "--", "white"), "launch_fast": ("#2a78d6", "s", "-", None),
         "proxy": ("#eb6834", "o", "--", "white"), "proxy_fast": ("#eb6834", "o", "-", None),
         DOCA: ("#1baf7a", "^", "-", None), "doca_fast": ("#1baf7a", "^", ":", "white")}
DASH = r"\blank"
REF = ("int8", 128, 8, 100)
NOTES = []


def note(s):
    NOTES.append(s)


def f1(x):
    try:
        return str(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    except (TypeError, ValueError, InvalidOperation):
        return DASH


def f2(x):
    try:
        return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (TypeError, ValueError, InvalidOperation):
        return DASH


def read_csv(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                r["_src"] = os.path.basename(p)
                rows.append(r)
    return rows


def bench_files(paths):
    """GPU-only reruns never overwrite: they are written to bench_<host>_r<k>.csv.
    Every given file is read together with its _r<k> siblings, oldest first, so
    that bench_by() (last row wins) reports the newest measurement of every
    (configuration, mode)."""
    import re
    files = set()
    for p in paths:
        files.add(p)
        stem = re.sub(r"_r\d+\.csv$", ".csv", p)
        if os.path.exists(stem):
            files.add(stem)
        files.update(glob.glob(stem[:-4] + "_r*.csv"))

    def order(p):
        m = re.match(r"(.+)_r(\d+)\.csv$", p)
        return (m.group(1) + ".csv", int(m.group(2))) if m else (p, 1)
    return sorted(files, key=order)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    print("wrote", path)


def key_of(r):
    return (FMT.get(r["format"], r["format"]), int(r["n"]), int(r["B"]), int(r["K"]))


RAWTAG = {}   # (series, config, rate, window) -> tag whose raw per-request file belongs to the reported row


def latest_only(net):
    """Old results are never overwritten: a configuration that is measured again is
    run under a new tag <series>_r<k> (k = 2, 3, ...), and its raw per-request CSV
    gets a new file name.  For the tables the NEWEST run of each configuration is
    reported under its series name: rows of <series>_r<k> replace those of
    <series> and of smaller k; among rows of one tag the last one in net.csv wins.
    The tag actually used is remembered in RAWTAG so that per-request analyses
    read the matching raw file."""
    import re
    best = {}
    for i, r in enumerate(net):
        m = re.match(r"(.+)_r(\d+)$", r["tag"])
        series, gen = (m.group(1), int(m.group(2))) if m else (r["tag"], 1)
        k = (series,) + key_of(r) + (r.get("rate"), r.get("window"))
        if k not in best or (gen, i) > best[k][0]:
            best[k] = ((gen, i), r)
    out = []
    superseded = 0
    for k, ((gen, i), r) in best.items():
        rr = dict(r)
        if rr["tag"] != k[0]:
            superseded += 1
        # keyed by rate as well as window: a Poisson row (rate > 0) has window 1 too and
        # must not overwrite the tag of the closed-loop row
        RAWTAG[(k[0],) + key_of(r) + (r.get("rate"), r.get("window"))] = r["tag"]
        rr["tag"] = k[0]
        out.append(rr)
    if superseded:
        note(f"[net] {superseded} configurations reported from a newer rerun (<series>_r<k>): "
             + "; ".join(sorted({f"{r['tag']}" for _, ((g, i), r) in best.items() if g > 1})))
    note(f"[net] rows in net.csv {len(net)}, reported {len(out)}")
    return out


def raw_tag(series, cfg="int8_n128_B8_K100", window="1", rate="0"):
    f, n, b, k = cfg.split("_")
    key = (series, f, int(n[1:]), int(b[1:]), int(k[1:]), rate, window)
    return RAWTAG.get(key, series)


# ---------------------------------------------------------------- GPU only
def bench_by(bench):
    by = defaultdict(dict)
    for r in bench:
        by[key_of(r)][r["mode"]] = r
    return by


def note_bench_reruns(bench):
    by = bench_by(bench)
    used = defaultdict(int)
    for d in by.values():
        for mode, r in d.items():
            used[(r.get("_src", "?"), mode, r.get("extra", ""))] += 1
    for (src, mode, extra), cnt in sorted(used.items()):
        note(f"[bench] {mode}: {cnt} configurations reported from {src}" + (f" ({extra})" if mode == "proxy-cpu" else ""))


def tab_single(bench, out):
    """Solve time per K, and the host-path overheads, for every (format, n, B)."""
    by = bench_by(bench)
    groups = defaultdict(dict)
    for k, d in by.items():
        if "single-solve" in d and "single-roundtrip" in d:
            groups[k[:3]][k[3]] = d
    lines = [r"\begin{tabular}{lrrrrrrrr}\toprule",
             r" & & & \multicolumn{4}{c}{in-kernel solve, p50 ($\mu$s)} & launch & H2D+launch+D2H \\",
             r"\cmidrule(lr){4-7}",
             r"format & $n$ & $B$ & $K{=}50$ & 100 & 200 & 400 & $-$solve & $-$solve ($\mu$s) \\ \midrule"]
    all_ov, all_lo, n_cfg, n_over = [], [], 0, 0
    for g in sorted(groups, key=lambda k: (["pm1", "ternary", "int8"].index(k[0]), k[1], k[2])):
        d = groups[g]
        solves = [f1(d[K]["single-solve"]["p50_us"]) if K in d else DASH for K in (50, 100, 200, 400)]
        ov = [float(x["single-roundtrip"]["p50_us"]) - float(x["single-solve"]["p50_us"]) for x in d.values()]
        lo = [float(x["single-launch"]["p50_us"]) - float(x["single-solve"]["p50_us"]) for x in d.values()]
        for x in d.values():
            n_cfg += 1
            s = float(x["single-solve"]["p50_us"])
            if float(x["single-roundtrip"]["p50_us"]) - s > s:
                n_over += 1
        all_ov += ov
        all_lo += lo
        lines.append(f"{g[0]} & {g[1]} & {g[2]} & " + " & ".join(solves) +
                     f" & {statistics.median(lo):.1f} & {statistics.median(ov):.1f} ({min(ov):.1f}--{max(ov):.1f}) \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    write(os.path.join(out, "tables", "tab_single.tex"), "\n".join(lines) + "\n")
    if all_ov:
        solves = [float(d["single-solve"]["p50_us"]) for d in by.values() if "single-solve" in d]
        note(f"[bench] configurations measured: {n_cfg}; solve p50 range {min(solves):.1f}-{max(solves):.1f} us")
        note(f"[bench] H2D+launch+D2H minus solve: median {statistics.median(all_ov):.1f}, "
             f"min {min(all_ov):.1f}, max {max(all_ov):.1f}, mean {statistics.mean(all_ov):.1f} us")
        note(f"[bench] launch-only overhead (event-timed launch minus in-kernel): median {statistics.median(all_lo):.1f}, "
             f"min {min(all_lo):.1f}, max {max(all_lo):.1f} us")
        note(f"[bench] configurations where overhead > solve: {n_over} of {n_cfg}")
        small = sorted((s, k) for k, d in by.items() if "single-solve" in d for s in [float(d["single-solve"]["p50_us"])])[:3]
        note(f"[bench] smallest solves: {small}")
        for k in [REF, ("pm1", 128, 8, 100), ("pm1", 64, 1, 50)]:
            d = by.get(k)
            if d:
                note(f"[bench] {k}: solve {d['single-solve']['p50_us']} launch {d['single-launch']['p50_us']} "
                     f"roundtrip {d['single-roundtrip']['p50_us']} (p99 {d['single-roundtrip']['p99_us']})")


def tab_pipeline(bench, out):
    """Saturated throughput and latency of the two persistent pipelines, no NIC, B=8."""
    by = bench_by(bench)
    lines = [r"\begin{tabular}{lrrrrrrrr}\toprule",
             r" & & & & \multicolumn{3}{c}{GPU-resident rings (GPU producer/consumer)} & \multicolumn{2}{c}{host-mapped rings (CPU proxy)} \\",
             r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
             r"format & $n$ & $B$ & $K$ & kreq/s & \shortstack{$N_w/t_{\mathrm{solve}}$\\(kreq/s)} & p99 ($\mu$s) & kreq/s & p99 ($\mu$s) \\ \midrule"]
    rates_g, rates_p = [], []
    for k in sorted(by, key=lambda k: (["pm1", "ternary", "int8"].index(k[0]), k[1], k[2], k[3])):
        if k[2] != 8 or k[3] not in (100, 400):
            continue
        d = by[k]
        g, p, s = d.get("persistent-gpu", {}), d.get("proxy-cpu", {}), d.get("single-solve", {})
        if not g or not p:
            continue
        W = int(g["workers"])
        bound = W / float(s["p50_us"]) * 1e3 if s else None  # kreq/s if every worker solved back to back
        kg, kp = float(g["rate_req_s"]) / 1e3, float(p["rate_req_s"]) / 1e3
        rates_g.append((kg, k, bound))
        rates_p.append((kp, k))
        lines.append(f"{k[0]} & {k[1]} & {k[2]} & {k[3]} & {kg:.0f} & {bound:.0f} & {f1(g['p99_us'])} & "
                     f"{kp:.0f} & {f1(p['p99_us'])} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    write(os.path.join(out, "tables", "tab_pipeline.tex"), "\n".join(lines) + "\n")
    if rates_g:
        allg = [float(d["persistent-gpu"]["rate_req_s"]) / 1e3 for d in by.values() if "persistent-gpu" in d]
        allp = [float(d["proxy-cpu"]["rate_req_s"]) / 1e3 for d in by.values() if "proxy-cpu" in d]
        W = int(next(d["persistent-gpu"]["workers"] for d in by.values() if "persistent-gpu" in d))
        note(f"[bench] persistent-gpu workers {W}; rate range over all configs {min(allg):.0f}-{max(allg):.0f} kreq/s; "
             f"max {max(allg):.0f}; configs at >=1900 kreq/s: {sum(1 for r in allg if r >= 1900)} of {len(allg)}")
        note(f"[bench] proxy-cpu rate range {min(allp):.0f}-{max(allp):.0f} kreq/s")
        d = by.get(REF)
        if d:
            note(f"[bench] REF persistent-gpu {d['persistent-gpu']['rate_req_s']} req/s p50 {d['persistent-gpu']['p50_us']} p99 {d['persistent-gpu']['p99_us']}; "
                 f"proxy-cpu {d['proxy-cpu']['rate_req_s']} req/s p50 {d['proxy-cpu']['p50_us']} p99 {d['proxy-cpu']['p99_us']}; "
                 f"W/t_solve bound {W / float(d['single-solve']['p50_us']) * 1e3:.0f} kreq/s")
        # ratio of achieved to W/t_solve for solves > 300 us (worker bound regime)
        fr = [kg / b for kg, k, b in rates_g if b and b < 1500]
        if fr:
            note(f"[bench] achieved / (W/t_solve) when the bound < 1500 kreq/s: {min(fr):.2f}-{max(fr):.2f}")


# ---------------------------------------------------------------- network
def net_unloaded(net):
    by = defaultdict(dict)
    reported_tags = {tag for tag, _ in TAGS}
    for r in net:
        if (r["tag"] in reported_tags and r.get("window") == "1"
                and float(r.get("rate") or 0) == 0):
            by[key_of(r)][r["tag"]] = r
    return by


def sort_keys(keys):
    return sorted(keys, key=lambda k: (["int8", "pm1", "ternary"].index(k[0]), k[2] != 8, k[1], k[2], k[3]))


def tab_net(net, out):
    by = net_unloaded(net)
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for _, name in TAGS)
    lines = [r"\begin{tabular}{lrrr" + "rr" * len(TAGS) + r"}\toprule",
             r" & & & & " + head + r" \\",
             "".join(f"\\cmidrule(lr){{{5 + 2 * i}-{6 + 2 * i}}}" for i in range(len(TAGS))),
             r"format & $n$ & $B$ & $K$ & " + " & ".join(["p50", "p99"] * len(TAGS)) + r" \\ \midrule"]
    for k in sort_keys(by) or [REF]:
        cells = []
        best = None
        for t, _ in TAGS:  # bold the lowest median of the row
            v = by.get(k, {}).get(t, {}).get("p50_us")
            if v is not None and (best is None or float(v) < best):
                best = float(v)
        for t, _ in TAGS:
            d = by.get(k, {}).get(t, {})
            c50 = f1(d.get("p50_us"))
            if d.get("p50_us") is not None and float(d["p50_us"]) == best:
                c50 = r"\textbf{" + c50 + "}"
            cells += [c50, f1(d.get("p99_us"))]
        lines.append(f"{k[0]} & {k[1]} & {k[2]} & {k[3]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule\end{tabular}")
    write(os.path.join(out, "tables", "tab_net.tex"), "\n".join(lines) + "\n")
    for k in by:
        d = by[k]
        for t, _ in TAGS:
            if t in d:
                r = d[t]
                note(f"[net unloaded] {k} {t}: p50 {r['p50_us']} p99 {r['p99_us']} p999 {r['p999_us']} "
                     f"srv {r['srv_p50_us']}/{r['srv_p99_us']} net {r['net_p50_us']} ok {r['ok']} lost {r['lost']}")


def tab_decomp(net, out):
    """Server residence (stamped by the server) and RTT minus residence, per server."""
    by = net_unloaded(net)
    lines = [r"\begin{tabular}{lrrr" + "rr" * len(TAGS) + r"}\toprule",
             r" & & & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for _, name in TAGS) + r" \\",
             "".join(f"\\cmidrule(lr){{{5 + 2 * i}-{6 + 2 * i}}}" for i in range(len(TAGS))),
             r"format & $n$ & $B$ & $K$ & " + " & ".join(["server", "rest"] * len(TAGS)) + r" \\ \midrule"]
    fast, slow = [], []
    for k in sort_keys(by) or [REF]:
        cells = []
        for t, _ in TAGS:
            d = by.get(k, {}).get(t, {})
            cells += [f1(d.get("srv_p50_us")), f1(d.get("net_p50_us"))]
            if d and t != DOCA:
                (fast if float(d["net_p50_us"]) < 40 else slow).append((t, k, float(d["srv_p50_us"]), float(d["net_p50_us"])))
        lines.append(f"{k[0]} & {k[1]} & {k[2]} & {k[3]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule\end{tabular}")
    write(os.path.join(out, "tables", "tab_decomp.tex"), "\n".join(lines) + "\n")
    if by:
        dn = [float(d[DOCA]["net_p50_us"]) for d in by.values() if DOCA in d]
        note(f"[net] GPUNetIO RTT-server p50 over the {len(dn)} unloaded configs: min {min(dn):.1f} median {statistics.median(dn):.1f} max {max(dn):.1f}")
        note(f"[net] baseline runs in the fast state (rest < 40 us): {len(fast)}: " + "; ".join(f"{t} {k} srv {s:.0f} rest {n:.1f}" for t, k, s, n in fast))
        if slow:
            note(f"[net] baseline runs in the slow state: {len(slow)}; rest median {statistics.median([n for *_, n in slow]):.1f} "
                 f"range {min(n for *_, n in slow):.1f}-{max(n for *_, n in slow):.1f}; server residence range {min(s for _, _, s, _ in slow):.0f}-{max(s for _, _, s, _ in slow):.0f}")
        note(f"[net] fast-state server residence max {max(s for _, _, s, _ in fast):.0f} us")
        # server-side data path cost = baseline residence - doca residence, same config
        for t in ("launch", "proxy", "launch_fast", "proxy_fast"):
            diffs = [float(d[t]["srv_p50_us"]) - float(d[DOCA]["srv_p50_us"]) for d in by.values() if t in d and DOCA in d]
            if diffs:
                note(f"[net] residence({t}) - residence({DOCA}), p50, over configs: median {statistics.median(diffs):.1f} min {min(diffs):.1f} max {max(diffs):.1f}")
            rd = [float(d[t]["net_p50_us"]) for d in by.values() if t in d]
            if rd:
                note(f"[net] rest({t}) p50 over configs: min {min(rd):.1f} median {statistics.median(rd):.1f} max {max(rd):.1f}")
        # GPUNetIO against the faster tuned CPU server, per config
        wins = []
        for k, d in by.items():
            if DOCA in d and "launch_fast" in d and "proxy_fast" in d:
                best = min(float(d["launch_fast"]["p50_us"]), float(d["proxy_fast"]["p50_us"]))
                wins.append((float(d[DOCA]["p50_us"]) - best, k))
        if wins:
            wins.sort()
            neg = [w for w in wins if w[0] < 0]
            note(f"[net] {DOCA} p50 - min(launch_fast, proxy_fast) p50: faster in {len(neg)}/{len(wins)}; "
                 + "; ".join(f"{k} {w:+.1f}" for w, k in wins))


def tab_ratio(net, out):
    by = net_unloaded(net)

    def ratio(d, t, q):
        try:
            return float(d[t][q]) / float(d[DOCA][q])
        except (KeyError, ValueError, ZeroDivisionError):
            return None

    lines = [r"\begin{tabular}{lrrrrrrrr}\toprule",
             r" & & & & GPUNetIO & \multicolumn{2}{c}{default host} & \multicolumn{2}{c}{tuned host} \\",
             r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"format & $n$ & $B$ & $K$ & server p50 ($\mu$s) & launch & proxy & launch & proxy \\ \midrule"]
    rows = []
    for k in sort_keys(by) or [REF]:
        d = by.get(k, {})
        solve = d.get(DOCA, {}).get("srv_p50_us")
        rl, rp, rlf, rpf = (ratio(d, "launch", "p50_us"), ratio(d, "proxy", "p50_us"),
                            ratio(d, "launch_fast", "p50_us"), ratio(d, "proxy_fast", "p50_us"))
        if solve is not None and rl is not None:
            rows.append((float(solve), rl, rp, k, rlf, rpf))
        lines.append(f"{k[0]} & {k[1]} & {k[2]} & {k[3]} & {f1(solve)} & {f2(rl)} & {f2(rp)} & {f2(rlf)} & {f2(rpf)} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    write(os.path.join(out, "tables", "tab_ratio.tex"), "\n".join(lines) + "\n")
    if rows:
        rows.sort()
        note("[ratio] by GPUNetIO server residence: " + "; ".join(f"{s:.0f}us L{rl:.2f} P{rp:.2f} Lf{rlf or 0:.2f} Pf{rpf or 0:.2f} {k}" for s, rl, rp, k, rlf, rpf in rows))
        note(f"[ratio] launch/doca p50 max {max(r[1] for r in rows):.2f} at {max(rows, key=lambda r: r[1])[3]}; "
             f"proxy/doca p50 max {max(r[2] for r in rows):.2f} at {max(rows, key=lambda r: r[2])[3]}")
        fl = [r[4] for r in rows if r[4]] + [r[5] for r in rows if r[5]]
        if fl:
            note(f"[ratio] tuned host: launch_fast/doca and proxy_fast/doca p50 range {min(fl):.3f}-{max(fl):.3f}")


def tab_load(net, out):
    """Open-loop Poisson arrivals: achieved rate, loss, server residence and client RTT."""
    rows = [r for r in net if float(r.get("rate") or 0) > 0 and key_of(r) == REF]
    lines = [r"\begin{tabular}{lrrrrrrrr}\toprule",
             r"server & offered & achieved & lost & \multicolumn{2}{c}{server residence ($\mu$s)} & \multicolumn{3}{c}{client RTT ($\mu$s)} \\",
             r"\cmidrule(lr){5-6}\cmidrule(lr){7-9}",
             r" & kreq/s & kreq/s & \% & p50 & p99 & p50 & p99 & p99.9 \\ \midrule"]
    for t, name in TAGS_LOAD:
        first = True
        for r in sorted((r for r in rows if r["tag"] == t), key=lambda r: float(r["rate"])):
            sent = int(r["ok"]) + int(r["lost"]) + int(r["err"])
            loss = 100.0 * int(r["lost"]) / sent if sent else 0.0
            lines.append(f"{name if first else ''} & {float(r['rate']) / 1e3:.0f} & {float(r['kreq_s']):.1f} & {loss:.1f} & "
                         f"{f1(r['srv_p50_us'])} & {f1(r['srv_p99_us'])} & {f1(r['p50_us'])} & {f1(r['p99_us'])} & {f1(r['p999_us'])} \\\\")
            note(f"[load] {t} offered {r['rate']} achieved {r['kreq_s']} ok {r['ok']} lost {r['lost']} ({loss:.1f}%) srv {r['srv_p50_us']}/{r['srv_p99_us']} "
                 f"rtt {r['p50_us']}/{r['p99_us']}/{r['p999_us']} rest {r['net_p50_us']}")
            first = False
        if not first:
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule\end{tabular}")
    write(os.path.join(out, "tables", "tab_load.tex"), "\n".join(lines) + "\n")


# Repeats of the high-load Poisson runs.  Only runs taken with the current code
# count as repeats: the original doca_w2t / proxy_fast rows predate later code
# changes, so the repeats start at _r2 (the reruns of revision 28).  A run with
# lost replies is excluded from the repeats (it occurred once, doca_w2t_r4 at
# 700 kreq/s: the server transmitted every reply, the client NIC dropped 1.1 %);
# it stays in net.csv and is listed in numbers.txt.
REPEAT_SERIES = [(DOCA, "GPUNetIO"), ("proxy_fast", "proxy, tuned")]
REPEAT_MIN_GEN = 2
REPEAT_MIN_RATE = 400000
MEDIAN_FIELDS = ["kreq_s", "ok", "err", "lost", "p50_us", "p99_us", "p999_us", "mean_us",
                 "srv_p50_us", "srv_p99_us", "net_p50_us"]


def repeat_groups(all_net):
    """(series, rate) -> [(gen, row)] of the valid repeats, and the excluded rows."""
    import re
    groups, excluded = defaultdict(list), []
    series_ok = {s for s, _ in REPEAT_SERIES}
    for r in all_net:
        m = re.match(r"(.+)_r(\d+)$", r["tag"])
        series, gen = (m.group(1), int(m.group(2))) if m else (r["tag"], 1)
        if (series not in series_ok or gen < REPEAT_MIN_GEN or key_of(r) != REF
                or float(r.get("rate") or 0) < REPEAT_MIN_RATE):
            continue
        if int(r["lost"]) > 0:
            excluded.append(r)
            continue
        groups[(series, float(r["rate"]))].append((gen, r))
    return groups, excluded


def apply_repeat_medians(net, all_net):
    """For the load table and figure: replace the rows of the repeated (series, rate)
    points by the per-column median over the valid repeats."""
    groups, _ = repeat_groups(all_net)
    out = []
    for r in net:
        k = (r["tag"], float(r.get("rate") or 0))
        if key_of(r) == REF and k in groups and len(groups[k]) > 1:
            rows = [x for _, x in groups[k]]
            rr = dict(r)
            for f in MEDIAN_FIELDS:
                m = statistics.median(float(x[f]) for x in rows)
                rr[f] = f"{m:.0f}" if f in ("ok", "err", "lost") else f"{m:.2f}"
            rr["runs"] = str(len(rows))
            out.append(rr)
        else:
            out.append(r)
    return out


def load_repeats(all_net, out):
    """All valid repeats of the open-loop runs of the reference configuration at
    >= 400 kreq/s offered, per server and rate: number of runs, the range of the
    achieved rate and of the client RTT percentiles, and their medians.  Writes
    tables/tab_load_repeats.tex and the numbers to numbers.txt."""
    groups, excluded = repeat_groups(all_net)
    for r in excluded:
        note(f"[repeats] excluded (lost replies): {r['tag']} offered {r['rate']} ok {r['ok']} lost {r['lost']} "
             f"rtt p50 {r['p50_us']} p99 {r['p99_us']} srv p50 {r['srv_p50_us']}")
    lines = [r"\begin{tabular}{lrrrrrr}\toprule",
             r"server & offered & runs & achieved & \multicolumn{3}{c}{client RTT ($\mu$s), range over runs (median)} \\",
             r"\cmidrule(lr){5-7}",
             r" & kreq/s & & kreq/s & p50 & p99 & p99.9 \\ \midrule"]

    def rng(v, fmt="{:.0f}"):
        return fmt.format(min(v)) if fmt.format(min(v)) == fmt.format(max(v)) else fmt.format(min(v)) + "--" + fmt.format(max(v))

    def cell(v):
        return f"{rng(v)} ({statistics.median(v):.0f})"
    for series, name in REPEAT_SERIES:
        first = True
        for rate in sorted(k[1] for k in groups if k[0] == series):
            rows = [r for _, r in sorted(groups[(series, rate)], key=lambda x: x[0])]
            ach = [float(r["kreq_s"]) for r in rows]
            p50 = [float(r["p50_us"]) for r in rows]
            p99 = [float(r["p99_us"]) for r in rows]
            p999 = [float(r["p999_us"]) for r in rows]
            lines.append(f"{name if first else ''} & {rate / 1e3:.0f} & {len(rows)} & {rng(ach)} & "
                         f"{cell(p50)} & {cell(p99)} & {cell(p999)} \\\\")
            note(f"[repeats] {series} offered {rate:.0f}: {len(rows)} runs ({', '.join(r['tag'] for r in rows)}); "
                 f"achieved {rng(ach, '{:.1f}')} kreq/s; rtt p50 {rng(p50, '{:.1f}')} (median {statistics.median(p50):.1f}) "
                 f"p99 {rng(p99, '{:.1f}')} (median {statistics.median(p99):.1f}) "
                 f"p99.9 {rng(p999, '{:.1f}')} (median {statistics.median(p999):.1f})")
            first = False
        if not first:
            lines.append(r"\addlinespace")
    for rate in sorted({k[1] for k in groups}):
        a = [r for _, r in groups.get((DOCA, rate), [])]
        b = [r for _, r in groups.get(("proxy_fast", rate), [])]
        if a and b:
            for col in ("p50_us", "p99_us"):
                ma = statistics.median(float(r[col]) for r in a)
                mb = statistics.median(float(r[col]) for r in b)
                note(f"[repeats] median {col} at offered {rate:.0f}: {DOCA} {ma:.1f}, proxy_fast {mb:.1f}, ratio {mb / ma:.2f}")
    lines.append(r"\bottomrule\end{tabular}")
    write(os.path.join(out, "tables", "tab_load_repeats.tex"), "\n".join(lines) + "\n")


def tab_window(net, out):
    """Closed loop with 1, 4, 16, 64 outstanding requests."""
    rows = [r for r in net if float(r.get("rate") or 0) == 0 and key_of(r) == REF]
    lines = [r"\begin{tabular}{lrrrrrrr}\toprule",
             r"server & outstanding & kreq/s & \multicolumn{2}{c}{server residence ($\mu$s)} & \multicolumn{3}{c}{client RTT ($\mu$s)} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r" & & & p50 & p99 & p50 & p99 & p99.9 \\ \midrule"]
    for t, name in TAGS_LOAD:
        first = True
        for r in sorted((r for r in rows if r["tag"] == t), key=lambda r: int(r["window"])):
            lines.append(f"{name if first else ''} & {r['window']} & {float(r['kreq_s']):.1f} & {f1(r['srv_p50_us'])} & {f1(r['srv_p99_us'])} & "
                         f"{f1(r['p50_us'])} & {f1(r['p99_us'])} & {f1(r['p999_us'])} \\\\")
            note(f"[window] {t} w={r['window']} kreq/s {r['kreq_s']} srv {r['srv_p50_us']}/{r['srv_p99_us']} rtt {r['p50_us']}/{r['p99_us']}/{r['p999_us']} rest {r['net_p50_us']}")
            first = False
        if not first:
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule\end{tabular}")
    write(os.path.join(out, "tables", "tab_window.tex"), "\n".join(lines) + "\n")


def tab_breakdown(raw, out, cfg="int8_n128_B8_K100"):
    """Per-request decomposition of the unloaded reference configuration (percentiles over 100k requests)."""
    import numpy as np
    lines = [r"\begin{tabular}{lrrrrrrrrrr}\toprule",
             r" & \multicolumn{2}{c}{queue} & \multicolumn{2}{c}{solve} & \multicolumn{2}{c}{server I/O} & \multicolumn{2}{c}{residual} & \multicolumn{2}{c}{RTT} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"server & p50 & p99 & p50 & p99 & p50 & p99 & p50 & p99 & p50 & p99 \\ \midrule"]
    stats = {}
    for t, name in TAGS:
        files = glob.glob(os.path.join(raw, f"{raw_tag(t, cfg)}_{cfg}_--window_1.csv"))
        if not files:
            lines.append(f"{name} & " + " & ".join([DASH] * 10) + r" \\")
            continue
        rows = read_csv(files[:1])
        a = {k: np.array([int(r[k]) for r in rows], dtype=np.float64) / 1e3 for k in ("rtt_ns", "queue_ns", "solve_ns", "tx_ns")}
        ok = a["rtt_ns"] > 0
        rest = a["rtt_ns"] - a["queue_ns"] - a["solve_ns"] - a["tx_ns"]
        cols = [a["queue_ns"], a["solve_ns"], a["tx_ns"], rest, a["rtt_ns"]]
        cells = []
        st = {}
        for nm, c in zip(("queue", "solve", "io", "rest", "rtt"), cols):
            c = c[ok]
            ordered = np.sort(c)
            # Use the client's rounded order-statistic convention consistently.
            p50, p99 = (ordered[int(p * (len(ordered) - 1) + 0.5)] for p in (0.5, 0.99))
            st[nm] = (p50, p99, c.mean())
            cells += [f1(p50), f1(p99)]
        stats[t] = st
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        hist, edges = np.histogram(rest[ok], bins=[0, 40, 1e9])
        note(f"[breakdown] {t}: " + " ".join(f"{nm} p50 {v[0]:.2f} p99 {v[1]:.2f} mean {v[2]:.2f};" for nm, v in st.items()) +
             f" rest<40us {hist[0]} of {ok.sum()} ({100 * hist[0] / ok.sum():.1f}%)")
    lines.append(r"\bottomrule\end{tabular}")
    write(os.path.join(out, "tables", "tab_breakdown.tex"), "\n".join(lines) + "\n")
    return stats


# ---------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
                         "lines.markersize": 4, "pdf.fonttype": 42})
    return plt


def _save(fig, out, name):
    p = os.path.join(out, "figures", name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


def _empty(ax):
    ax.text(0.5, 0.5, "no measurements yet", ha="center", va="center", transform=ax.transAxes)


def fig_scaling(net, out):
    plt = _plt()
    by = net_unloaded(net)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    panels = [(axes[0], lambda k: k[0] == "int8" and k[1] == 128 and k[2] == 8, 3, "steps $K$ (int8 portfolio, $n=128$, $B=8$)"),
              (axes[1], lambda k: k[0] == "pm1" and k[2] == 8 and k[3] == 100, 1, "problem size $n$ (pm1, $B=8$, $K=100$)")]
    for ax, sel, xi, xlabel in panels:
        any_data = False
        for t, name in TAGS:
            c, m, ls, mfc = STYLE[t]
            pts = sorted((k[xi], float(d[t]["p50_us"])) for k, d in by.items() if sel(k) and t in d)
            if not pts:
                continue
            any_data = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=m, color=c, ls=ls, mfc=mfc or c, label=name)
        if not any_data:
            _empty(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("median round-trip time ($\\mu$s)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_yticks([100, 200, 300, 500, 1000])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    axes[0].legend(ncol=1, frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig_scaling.pdf")


def fig_overhead(bench, out):
    plt = _plt()
    by = bench_by(bench)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    any_data = False
    cols = {"pm1": "#2a78d6", "ternary": "#eb6834", "int8": "#1baf7a"}
    for f, m in zip(["pm1", "ternary", "int8"], ["s", "o", "^"]):
        pts = []
        for k, d in by.items():
            if k[0] != f or "single-solve" not in d or "single-roundtrip" not in d:
                continue
            s_, r_ = float(d["single-solve"]["p50_us"]), float(d["single-roundtrip"]["p50_us"])
            pts.append((s_, r_ - s_))
        if pts:
            any_data = True
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=m, s=14, color=cols[f], label=f, alpha=0.85)
    if any_data:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(4, 80)
        ax.set_xlim(25, 2500)
        xs = [25, 2500]
        for frac, lab, xa in [(1.0, "overhead = solve", 30), (0.1, "overhead = 10% of solve", 120)]:
            ax.plot(xs, [frac * x for x in xs], color="gray", lw=0.7, ls=":")
            ax.annotate(lab, (xa, frac * xa), xytext=(3, -9), textcoords="offset points", fontsize=6, color="gray", rotation=33)
        ax.legend(frameon=False, loc="upper right")
        ax.set_yticks([5, 10, 20, 50])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    else:
        _empty(ax)
    ax.set_xlabel("in-kernel solve time ($\\mu$s)")
    ax.set_ylabel("H2D + launch + D2H $-$ solve ($\\mu$s)")
    fig.tight_layout()
    _save(fig, out, "fig_overhead.pdf")


def fig_load(net, out):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    any_data = False
    for ax, (col, ylabel) in zip(axes, [("p50_us", "median round-trip time ($\\mu$s)"),
                                        ("p99_us", "99th-percentile round-trip time ($\\mu$s)")]):
        for t, name in TAGS_LOAD:
            c, m, ls, mfc = STYLE[t]
            pts = sorted((float(r["rate"]) / 1e3, float(r[col]), int(r["lost"]) > 0)
                         for r in net if r["tag"] == t and float(r.get("rate") or 0) > 0 and key_of(r) == REF and int(r["ok"]) > 0)
            if not pts:
                continue
            any_data = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=m, color=c, ls=ls, mfc=mfc or c, label=name)
            lossy = [p for p in pts if p[2]]
            if lossy:  # first run with lost replies
                ax.plot([lossy[0][0]], [lossy[0][1]], marker="x", color="black", ms=6, ls="none")
        ax.set_xlabel("configured Poisson rate (kreq/s)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}" if v >= 1 else f"{v}"))
    if any_data:
        axes[0].legend(frameon=False, ncol=1, fontsize=6)
    else:
        _empty(axes[0])
    fig.tight_layout()
    _save(fig, out, "fig_load.pdf")


def fig_cdf(raw, out, cfg="int8_n128_B8_K100"):
    import numpy as np
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    any_data = False
    for t, name in TAGS:
        files = glob.glob(os.path.join(raw, f"{raw_tag(t, cfg)}_{cfg}_--window_1.csv"))
        if not files:
            continue
        rows = read_csv(files[:1])
        a = {k: np.array([int(r[k]) for r in rows], dtype=np.float64) / 1e3 for k in ("rtt_ns", "queue_ns", "solve_ns", "tx_ns")}
        ok = a["rtt_ns"] > 0
        rtt = np.sort(a["rtt_ns"][ok])
        rest = np.sort((a["rtt_ns"] - a["queue_ns"] - a["solve_ns"] - a["tx_ns"])[ok])
        if rtt.size == 0:
            continue
        any_data = True
        c, _, ls, _ = STYLE[t]
        y = np.arange(1, rtt.size + 1) / rtt.size
        axes[0].plot(rtt, y, color=c, ls=ls, label=name)
        axes[1].plot(rest, y, color=c, ls=ls, label=name)
    axes[0].set_xlabel("round-trip time ($\\mu$s)")
    axes[1].set_xlabel("RTT $-$ server residence ($\\mu$s)")
    for ax in axes:
        ax.set_ylabel("fraction of requests")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    axes[0].set_xscale("linear")
    axes[0].set_xlim(150, 320)
    axes[1].set_xlim(8, 400)
    if any_data:
        axes[0].legend(frameon=False, loc="lower right")
    else:
        _empty(axes[0])
    fig.tight_layout()
    _save(fig, out, "fig_cdf.pdf")


def fig_breakdown(stats, out):
    """Additive mean decomposition; component medians are not additive."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    if not stats:
        _empty(ax)
    else:
        names = [name for t, name in TAGS if t in stats]
        parts = [("solve", "#c3c2b7", "solve (same kernel)"), ("queue", "#eda100", "GPU queue"),
                 ("io", "#2a78d6", "server I/O (CPU or GPU)"), ("rest", "#1baf7a", "unmeasured residual")]
        left = [0.0] * len(names)
        for key, col, lab in parts:
            vals = [stats[t][key][2] for t, _ in TAGS if t in stats]
            ax.barh(names, vals, left=left, color=col, edgecolor="white", linewidth=1, label=lab, height=0.6)
            left = [l + v for l, v in zip(left, vals)]
        for i, (t, _) in enumerate([x for x in TAGS if x[0] in stats]):
            tot = stats[t]["rtt"][2]
            ax.annotate(f"{tot:.0f} $\\mu$s", (left[i], i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
        ax.set_xlim(0, max(left) * 1.18)
        ax.set_xlabel("mean time per request ($\\mu$s)")
        ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig_breakdown.pdf")


# ---------------------------------------------------------------- verification counts
def verify_counts(raw):
    import re
    e_ok = e_tot = r_ok = r_tot = runs = 0
    for p in glob.glob(os.path.join(raw, "*.log")):
        txt = open(p).read()
        m = re.search(r"energy (\d+)/(\d+) match, reference trajectory (\d+)/(\d+) match", txt)
        if m:
            runs += 1
            e_ok += int(m.group(1)); e_tot += int(m.group(2)); r_ok += int(m.group(3)); r_tot += int(m.group(4))
    note(f"[verify] runs with a verify line: {runs}; energies {e_ok}/{e_tot}; trajectories {r_ok}/{r_tot}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=glob.glob("results/bench_*.csv"))
    ap.add_argument("--net", default="results/net.csv")
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    bench = read_csv(bench_files(a.bench)) if a.bench else []
    if bench:
        note_bench_reruns(bench)
    net = all_net = []
    if os.path.exists(a.net):
        with open(a.net) as fh:
            all_net = [r for r in csv.DictReader(fh) if r.get("SUMMARY") == "SUMMARY"]
        net = latest_only(all_net)
        note("[net] reported series: " + ", ".join(sorted({r["tag"] for r in net})))
    tab_single(bench, a.out)
    tab_pipeline(bench, a.out)
    tab_net(net, a.out)
    tab_decomp(net, a.out)
    tab_ratio(net, a.out)
    net_load = apply_repeat_medians(net, all_net)
    tab_load(net_load, a.out)
    tab_window(net, a.out)
    load_repeats(all_net, a.out)
    have_raw = bool(glob.glob(os.path.join(a.raw, "*.csv")))
    if have_raw:
        stats = tab_breakdown(a.raw, a.out)
    else:
        # per-request CSVs not available: keep the shipped breakdown outputs
        old_numbers = os.path.join("analysis", "numbers.txt")
        if os.path.exists(old_numbers):
            NOTES.extend(l.rstrip("\n") for l in open(old_numbers) if l.startswith("[breakdown]"))
        print("no per-request CSVs in", a.raw, "- tab_breakdown / fig_breakdown / fig_cdf kept as shipped")
    verify_counts(a.raw)
    fig_scaling(net, a.out)
    fig_overhead(bench, a.out)
    fig_load(net_load, a.out)
    if have_raw:
        fig_cdf(a.raw, a.out)
        fig_breakdown(stats, a.out)
    write(os.path.join("analysis", "numbers.txt"), "\n".join(NOTES) + "\n")


if __name__ == "__main__":
    main()
