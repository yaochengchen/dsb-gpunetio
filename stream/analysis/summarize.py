#!/usr/bin/env python3
"""Summarize the streaming-mode runs in results/stream/.

  python3 stream/analysis/summarize.py results/stream

Prints one line per receiver run (payload Gb/s, step time, loss, best 2E,
Mattis check) and the checksum comparisons between streamed runs and the
references (files <tag>.chk vs ref_*.chk: identical => bit-identical
trajectory).  Writes results/stream/summary.md.
"""
import csv, glob, os, re, sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/stream"
lines = []

def read_chk(path):
    d = {}
    for ln in open(path):
        m = re.match(r"pass (\d+) 2E (-?\d+) chk (0x[0-9a-f]+)", ln)
        if m: d[int(m.group(1))] = (int(m.group(2)), m.group(3))
        m = re.match(r"init (0x[0-9a-f]+)", ln)
        if m: d["init"] = m.group(1)
    return d

def header_of(log):
    for ln in open(log, errors="replace"):
        m = re.match(r"# n (\d+) B (\d+) K (\d+) (?:nq|workers) (\d+) instance (\w+) seed (\d+)", ln)
        if m:
            return dict(n=int(m.group(1)), B=int(m.group(2)), K=int(m.group(3)), nq=int(m.group(4)), inst=m.group(5))  # nq = worker blocks for st_recv
    return None

lines.append("| run | n | B | workers | instance | passes | payload Gb/s (median of update passes) | t_step s | frames lost | best 2E (final) | Mattis ground state |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for csvp in sorted(glob.glob(os.path.join(out_dir, "*.csv"))):
    tag = os.path.basename(csvp)[:-4]
    if tag.endswith("_gen"): continue
    log = os.path.join(out_dir, tag + ".log")
    hdr = header_of(log) if os.path.exists(log) else None
    rows = [r for r in csv.DictReader(open(csvp)) if "pass" in r and r.get("pass", "").isdigit()]
    if not rows: continue
    upd = [r for r in rows if r.get("update") == "1"]
    gb = sorted(float(r["payload_gbps"]) for r in upd) if upd else []
    med = gb[len(gb)//2] if gb else float("nan")
    ts = sorted(float(r["t_pass_s"]) for r in upd) if upd else []
    tmed = ts[len(ts)//2] if ts else float("nan")
    lost = sum(abs(int(r["lost"])) for r in rows)
    best = int(rows[-1]["best_2E"])
    n = hdr["n"] if hdr else None
    mattis = ""
    if hdr and hdr["inst"] == "mattis":
        gs = -n * (n - 1)
        mattis = "reached" if best == gs else f"not reached ({best} vs {gs})"
    lines.append(f"| {tag} | {n} | {hdr['B'] if hdr else ''} | {hdr['nq'] if hdr else ''} | {hdr['inst'] if hdr else ''} | {len(rows)} | {med:.1f} | {tmed:.3f} | {lost} | {best} | {mattis} |")

lines.append("")
lines.append("## Checksum comparisons (streamed vs reference; identical => bit-identical trajectory)")
refs = {os.path.basename(p)[:-4]: read_chk(p) for p in glob.glob(os.path.join(out_dir, "ref_*.chk"))}
for p in sorted(glob.glob(os.path.join(out_dir, "*.chk"))):
    tag = os.path.basename(p)[:-4]
    if tag.startswith("ref_"): continue
    d = read_chk(p)
    for rname, rd in sorted(refs.items()):
        common = [k for k in d if k != "init" and k in rd]
        if not common or d.get("init") != rd.get("init"): continue
        same = all(d[k] == rd[k] for k in common)
        lines.append(f"- {tag} vs {rname}: {'IDENTICAL' if same else 'DIFFER'} over {len(common)} passes (init chk {d.get('init')})")
for a, da in sorted(refs.items()):
    for b, db in sorted(refs.items()):
        if a < b and da.get("init") == db.get("init"):
            common = [k for k in da if k != "init" and k in db]
            if common:
                same = all(da[k] == db[k] for k in common)
                lines.append(f"- {a} vs {b}: {'IDENTICAL' if same else 'DIFFER'} over {len(common)} passes")

open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
