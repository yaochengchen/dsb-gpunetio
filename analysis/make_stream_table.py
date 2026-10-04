#!/usr/bin/env python3
"""Table of the streaming runs (paper Table 'tab:stream') from results/stream.

    python3 analysis/make_stream_table.py [--dir results/stream] [--out analysis/tables/tab_stream.tex]

Per run (receiver CSV <tag>.csv): payload rate and step time as the range over
the passes after start-up (the first pass, and for n <= 1e5 also the second,
include the generator's start-up), all reported validity counters and terminal status, and the check:
  - Mattis: whether 2E = -n(n-1) is reached and at which step;
  - random n >= 1e6: best 2E after the last step;
  - cross-checks: <tag>.chk byte-identical to the reference .chk.
Also prints the CPU kernel-stack receiver numbers from the generator CSVs.
"""
import argparse
import csv
import filecmp
import os
import json
from pathlib import Path

RUNS = [  # tag, instance, n, B, Nq, reference .chk or None
    ("full_mattis_1e7_B1", "Mattis", 10**7, 1, 4, None),
    ("full_mattis_1e7_B8", "Mattis", 10**7, 8, 4, None),
    ("full_random_1e7_B8", "random", 10**7, 8, 4, None),
    (["rate_1e6_q1", "rate_1e6_q2", "rate_1e6_q4", "rate_1e6_q8"], "Mattis", 10**6, 8, "1--8", None),
    ("rate_1e6_B1", "Mattis", 10**6, 1, 4, None),
    ("rate_1e6_random", "random", 10**6, 8, 4, None),
    ("x_mattis_1e5_w2", "Mattis", 10**5, 8, 4, "ref_gpu_mattis_1e5.chk"),
    ("x_random_1e5_w2", "random", 10**5, 8, 4, "ref_gpu_random_1e5.chk"),
    ("x_random_1e4_w2", "random", 10**4, 8, 4, "ref_cpu_random_1e4.chk"),
]


def read(path):
    with open(path) as fh:
        lines = [l for l in fh if l.strip()]
    hdr_idx = max(i for i, l in enumerate(lines) if l.startswith("pass,"))  # the last run in the file
    return list(csv.DictReader(lines[hdr_idx:]))


def rng(v, fmt):
    lo, hi = min(v), max(v)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else fmt.format(lo) + "--" + fmt.format(hi)


def exp10(n):
    return f"$10^{len(str(n)) - 1}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/stream")
    ap.add_argument("--out", default="analysis/tables/tab_stream.tex")
    a = ap.parse_args()
    review_path = Path(__file__).with_name("stream_event_review.json")
    reviews = json.loads(review_path.read_text(encoding="utf-8"))["reviewed_events"] if review_path.exists() else {}
    L = [r"\begin{tabular}{lrrrrrl}\toprule",
         r"Instance & $n$ & $B$ & $N_q$ & Payload (Gb/s) & Receive pass (s) & Evidence \\ \midrule"]
    for tags, inst, n, B, nq, ref in RUNS:
        tags = tags if isinstance(tags, list) else [tags]
        gb, ts, lost, e_ok, best, chk_ok = [], [], 0, None, None, True
        flagged = []
        ignored = []
        skip = 1 if n >= 10**6 else 2
        for t in tags:
            rows = read(os.path.join(a.dir, t + ".csv"))
            lost += sum(int(r["lost"]) for r in rows)
            for r in rows:
                if (any(int(r[k]) != 0 for k in ("lost", "bad", "dropped_bursts"))
                        or int(r["frames"]) != int(r["expected"])):
                    event = reviews.get(t, {})
                    documented = all(int(r[k]) == event.get(k) for k in ("pass", "bad", "lost", "dropped_bursts"))
                    if documented and int(r["frames"]) == int(r["expected"]):
                        ignored.append((t, int(r["pass"])))
                    else:
                        flagged.append((t, int(r["pass"])))
            with open(os.path.join(a.dir, t + ".log"), encoding="utf-8", errors="replace") as log:
                logtext = log.read()
            if "RUN INVALID" in logtext or "RUN OK" not in logtext:
                if not any(x[0] == t for x in flagged + ignored):
                    flagged.append((t, "terminal status"))
            gb += [float(r["payload_gbps"]) for r in rows[skip:]]
            ts += [float(r["t_pass_s"]) for r in rows[skip:]]
            target = -n * (n - 1)
            hit = [int(r["pass"]) for r in rows if int(r["best_2E"]) == target]
            e_ok = min(hit) if hit else None
            best = int(rows[-1]["best_2E"])
            if ref:
                chk_ok &= filecmp.cmp(os.path.join(a.dir, t + ".chk"), os.path.join(a.dir, ref), shallow=False)
        tfmt = "{:.2f}" if n >= 10**7 else ("{:.3f}" if n >= 10**6 else "{:.4f}")
        if ref:
            check = ("GPU records match" if "gpu" in ref else "CPU records match") if chk_ok else r"\textbf{MISMATCH}"
        elif inst == "Mattis":
            check = (f"$2E=-n(n-1)$ at step {e_ok}" if n >= 10**7 else "$2E=-n(n-1)$") if e_ok is not None else "ground state not reached"
        else:
            check = (f"$2E={best / 10**10:.2f}\\times10^{{10}}$ at step {len(rows) - 1}" if n >= 10**7 else "energy reported")
        status = "Flagged" if flagged or not chk_ok else (r"Complete$^{*}$" if ignored else "Complete")
        if flagged:
            check = "flagged pass " + ", ".join(str(x[1]) for x in flagged)
        print(f"[{status}] {tags}: flagged={flagged}, ignored={ignored}, record_match={chk_ok}")
        L.append(f"{inst} & {exp10(n)} & {B} & {nq} & {rng(gb, '{:.1f}')} & {rng(ts, tfmt)} & {check} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", a.out)
    for t in ("cpu_count_t1", "cpu_count_t8", "cpu_full_t8"):
        p = os.path.join(a.dir, t + "_gen.csv")
        if os.path.exists(p):
            with open(p) as fh:
                g = [float(r["payload_gbps"]) for r in csv.DictReader(fh) if float(r["payload_gbps"]) > 300]
            print(f"[cpu] {t}: generator sent {min(g):.1f}-{max(g):.1f} Gb/s while it was sending" if g else f"[cpu] {t}: no full-rate seconds")


if __name__ == "__main__":
    main()
