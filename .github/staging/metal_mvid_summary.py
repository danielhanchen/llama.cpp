#!/usr/bin/env python3
# Summarise the staging-metal-mvid logs: eval verdicts per arm, engagement of kernel_mul_mv_id_ext, perf A/B table.
# Exit 1 on: any head eval FAIL, no MUL_MAT_ID case run on Metal (VOID), the new kernel not engaged on head,
# the new kernel engaged with the kill switch or on the base.
import glob
import os
import re
import statistics
import sys

LOGS = sys.argv[1] if len(sys.argv) > 1 else "logs"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PERF = re.compile(r"^\s+(MUL_MAT_ID\(.*\)):\s+(\d+) runs -\s+([\d.]+) us/run")
CASE = re.compile(r"^\s+(MUL_MAT_ID\(.*?\)):\s*(.*)$")
ENG = re.compile(r"compiling pipeline: base = '(kernel_mul_mv_id_ext_[A-Za-z0-9_]+)'")


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as f:
        return [ANSI.sub("", l.rstrip("\n")) for l in f]


def eval_stats(path):
    st = {"OK": 0, "FAIL": 0, "not supported": 0}
    fails = []
    for l in read(path):
        m = CASE.match(l)
        if not m:
            continue
        tail = m.group(2)
        if "FAIL" in tail:
            st["FAIL"] += 1
            fails.append(m.group(1))
        elif "not supported" in tail:
            st["not supported"] += 1
        elif tail.strip().endswith("OK"):
            st["OK"] += 1
    return st, fails


def engaged(path):
    return sorted({m.group(1) for l in read(path) for m in [ENG.search(l)] if m})


def perf(paths):
    res = {}
    for p in paths:
        for l in read(p):
            m = PERF.match(l)
            if m:
                res.setdefault(m.group(1), []).append(float(m.group(3)))
    return res


out = []
bad = []

out.append("## Eval (test-backend-ops -o MUL_MAT_ID -b MTL0, vs CPU)\n")
out.append("| arm | OK | FAIL | not supported |")
out.append("|---|---|---|---|")
evals = {}
for arm in ("head", "head_killswitch", "base"):
    st, fails = eval_stats(f"{LOGS}/eval_{arm}.log")
    evals[arm] = (st, fails)
    out.append(f"| {arm} | {st['OK']} | {st['FAIL']} | {st['not supported']} |")
out.append("")
for arm, (st, fails) in evals.items():
    if fails:
        out.append(f"FAIL cases ({arm}):")
        out += [f"- `{c}`" for c in fails[:200]]
        out.append("")

if evals["head"][0]["OK"] == 0:
    bad.append("VOID: no MUL_MAT_ID case passed on MTL0 for head (backend missing or ops unsupported)")
if evals["head"][0]["FAIL"] > 0:
    bad.append(f"head eval has {evals['head'][0]['FAIL']} FAIL cases")

out.append("## Engagement (pipelines compiled)\n")
for arm in ("head", "head_killswitch", "base"):
    files = [f"{LOGS}/eval_{arm}.log", f"{LOGS}/perf_{arm}_full.log"] + glob.glob(f"{LOGS}/perf_{arm}_rep*.log")
    e = sorted({p for f in files for p in engaged(f)})
    out.append(f"- {arm}: {len(e)} kernel_mul_mv_id_ext pipelines" + (f": {', '.join(e)}" if e else ""))
    if arm == "head" and not e:
        bad.append("VOID: kernel_mul_mv_id_ext never compiled on head")
    if arm != "head" and e:
        bad.append(f"kernel_mul_mv_id_ext compiled on {arm}")
out.append("")

arms = {a: perf(sorted(glob.glob(f"{LOGS}/perf_{a}_rep*.log"))) for a in ("base", "head_killswitch", "head")}
keys = [k for k in arms["head"] if k in arms["base"]]
if keys:
    out.append("## Perf, Inkling-like shapes (median us/run over reps; ratio < 1 = head faster)\n")
    out.append("| case | base us | killswitch us | head us | head/base | base spread % |")
    out.append("|---|---|---|---|---|---|")
    for k in keys:
        b = arms["base"][k]
        h = arms["head"][k]
        ks = arms["head_killswitch"].get(k, [])
        bm = statistics.median(b)
        hm = statistics.median(h)
        km = statistics.median(ks) if ks else float("nan")
        spread = 100.0 * (max(b) - min(b)) / bm if len(b) > 1 and bm > 0 else float("nan")
        short = k.replace("MUL_MAT_ID(", "").replace("type_b=f32,", "").replace(",m_v=0,ids_offset=0,skip=0", "").rstrip(")")
        out.append(f"| {short} | {bm:.1f} | {km:.1f} | {hm:.1f} | {hm / bm:.3f} | {spread:.1f} |")
    out.append("")
else:
    bad.append("VOID: no perf cases parsed")

full = {a: perf([f"{LOGS}/perf_{a}_full.log"]) for a in ("base", "head")}
fk = [k for k in full["head"] if k in full["base"]]
if fk:
    out.append("## Full `perf -o MUL_MAT_ID` (one run each)\n")
    out.append("| case | base us | head us | head/base |")
    out.append("|---|---|---|---|")
    for k in fk:
        bm = full["base"][k][0]
        hm = full["head"][k][0]
        short = k.replace("MUL_MAT_ID(", "").replace("type_b=f32,", "").rstrip(")")
        out.append(f"| {short} | {bm:.1f} | {hm:.1f} | {hm / bm:.3f} |")
    out.append("")

out.append("## Verdict\n")
out += [f"- {b}" for b in bad] if bad else ["- PASS"]

text = "\n".join(out) + "\n"
print(text)
with open(f"{LOGS}/summary.md", "w") as f:
    f.write(text)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(text)
sys.exit(1 if bad else 0)
