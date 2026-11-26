"""Metal check of the unslothai/llama.cpp prebuilt tree (pins) and, optionally, one added commit (mvid).

    python3 jobs/mac_metal_check.py [--tag b12722] [--add-commit https://github.com/<o>/llama.cpp@<sha40>]
        [--parts pins,model,bench,mvid] [--quick] [--work DIR] [--out job.json]

Stdlib only. Trees are built the way the nightly does (git merge with diff3, then additive_merge.py for pure add/add
conflicts, nothing else):
  mix  = upstream tag + every pin of scripts/unsloth/pr-set.json (unslothai/llama.cpp master, or --pinset DIR)
  mvid = mix + the --add-commit commit merged on top; if that does not merge, the commit's own tree (reported)
Exit 0 = every gate passed, 1 = a gate failed, 2 = setup failed (clone, merge, build).

Parts (default: pins,model,bench, plus mvid with --add-commit):
  pins   mix build, test-backend-ops test -b MTL0 (Metal against CPU) for every op the pin set changes on Metal:
         FLASH_ATTN_EXT_BANDED, mixed K/V FLASH_ATTN_EXT, all FLASH_ATTN_EXT (shared launcher, skipped by --quick):
         rc 0, 0 FAIL, > 0 OK. Fork-only types (IQ1_XS/XXS/XXXS, MXFP8) for GET_ROWS / MUL_MAT / MUL_MAT_ID:
         rc 0 and 0 FAIL (Metal has no kernels for them, so it must refuse them, not run them).
         On a GPU with the tensor API (M5) the FA ops are run again with GGML_METAL_TENSOR_DISABLE=1.
  model  tiny Inkling 2 (random weights, seed 42; the test fixture is widened to head size 128 so the banded Metal
         kernels engage), llama-perplexity -fa on per K/V type: CPU (-dev none) writes the reference, Metal
         (default ubatch and -ub 16) is scored against it. Gates: mean KLD <= --kld-tol, banded pipelines compiled.
  bench  llama-bench pp512 / tg64 at depth 0 and 4096 on the tiny model, F16 / q8_0 / q8_0-q4_0 (report only).
  mvid   mvid build, test-backend-ops -o MUL_MAT_ID -b MTL0: rc 0, 0 FAIL, > 0 OK, kernel_mul_mv_id_ext pipelines
         compiled; with GGML_METAL_MUL_MV_ID_EXT_DISABLE=1: rc 0, 0 FAIL, no ext pipeline. Perf A/B kill switch
         on (old) / off (new), --mvid-rounds interleaved rounds, filter --mvid-filter, median us per case (report).

GitHub macOS runners expose an "Apple Paravirtual device" that hides the simdgroup reduction / matrix and bfloat
features of the host GPU; Metal then refuses these ops and every case reads "not supported". --ci-hooks (or
MAC_METAL_CI_HOOKS=1) patches env-gated force hooks into the working trees (never committed) and sets them on
every run. A real Mac needs no hooks and runs pristine trees; timing there is real, on a runner it is not.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
UPSTREAM = os.environ.get("XPLAT_UPSTREAM", "https://github.com/ggml-org/llama.cpp")
PINSET_URL = os.environ.get("XPLAT_PINSET_URL", "https://github.com/unslothai/llama.cpp")
KVS = [("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0"), ("q4_1", "q4_1"), ("q5_0", "q5_0"), ("q5_1", "q5_1"),
       ("q8_0", "q4_0"), ("q8_0", "f16"), ("f16", "q8_0"), ("q4_0", "f16")]
BENCH_KVS = [("f16", "f16"), ("q8_0", "q8_0"), ("q8_0", "q4_0")]
FORCE_ENV = {"XPLAT_FORCE_APPLE7": "1", "XPLAT_FORCE_BF16": "1", "GGML_METAL_TENSOR_DISABLE": "1"}
MVID_KILL = {"GGML_METAL_MUL_MV_ID_EXT_DISABLE": "1"}
# Every K != V pair spelled out: libc++ std::regex rejects a back reference inside a lookahead.
_FA_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl")
MIXED_RE = "type_K=(" + "|".join(
    f"{k},type_V=({'|'.join(v for v in _FA_TYPES if v != k)})," for k in _FA_TYPES) + ")"
FORK_TYPES = "(iq1_xs|iq1_xxs|iq1_xxxs|mxfp8)"
PIN_RE = re.compile(r"https://github\.com/([^/]+)/llama\.cpp/pull/(\d+)/commits/([0-9a-f]{40})")
PERF_RE = re.compile(r"^\s+([A-Z_]+\(.*\)):\s+\d+ runs -\s+([0-9.]+) us/run", re.MULTILINE)
EXT_RE = re.compile(r"compiling pipeline: base = '(kernel_mul_mv_id_ext[^']*)'")

DEVICE_ANCHOR = "                // double-check that the tensor API compiles\n"
DEVICE_HOOK = """                // XPLAT TEST HOOK (CI only, never committed): the paravirtual runner GPU hides the simdgroup
                // reduction / matrix and bfloat features of the M-series host GPU
                if (getenv("XPLAT_FORCE_APPLE7") != NULL) {
                    GGML_LOG_WARN("%s: XPLAT_FORCE_APPLE7: forcing simdgroup reduction + matrix mul\\n", __func__);
                    dev->props.has_simdgroup_reduction = true;
                    dev->props.has_simdgroup_mm        = true;
                }
                if (getenv("XPLAT_FORCE_BF16") != NULL) {
                    GGML_LOG_WARN("%s: XPLAT_FORCE_BF16: forcing bfloat\\n", __func__);
                    dev->props.has_bfloat = true;
                }

"""
TINY_OLD = "    } else if (arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP) {\n        n_embd  = 128;\n"
TINY_NEW = "    } else if (arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP) {\n        n_embd  = 512; // xplat: head size 128\n"
G = ["-c", "user.name=xplat", "-c", "user.email=xplat@localhost", "-c", "merge.conflictStyle=diff3"]


class SetupFail(Exception):
    pass


def setup_fail(msg):
    raise SetupFail(msg)


def run(cmd, log, timeout, env_add=None, cwd=None):
    env = dict(os.environ)
    env.update(env_add or {})
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write("$ " + " ".join(f"{k}={v}" for k, v in (env_add or {}).items()) + " " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        try:
            rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env,
                                cwd=cwd, check=False).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
        except FileNotFoundError as e:
            rc = f"MISSING {e}"
    return rc, round(time.time() - t0, 1)


def git(repo, *a, check=True):
    r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=False)
    if check and r.returncode != 0:
        setup_fail(f"git {' '.join(a)}: {r.stderr.strip()[-400:]}")
    return r


def read(p):
    try:
        return ANSI.sub("", Path(p).read_text(encoding="utf-8", errors="replace"))
    except FileNotFoundError:
        return ""


def load_pins(pr_set):
    pins = []
    for e in json.loads(Path(pr_set).read_text())["prs"]:
        url = e if isinstance(e, str) else e["url"]
        m = PIN_RE.match(url)
        if not m:
            setup_fail(f"unparsable pin {url}")
        pins.append((m.group(1), int(m.group(2)), m.group(3)))
    return pins


def parse_add_commit(s):
    m = re.fullmatch(r"(https://github\.com/[^@\s]+?)(?:\.git)?@([0-9a-f]{40})", s or "")
    if not m:
        setup_fail("--add-commit wants https://github.com/<owner>/<repo>@<full 40-hex sha>")
    return m.group(1), m.group(2)


def fresh_worktree(repo, wt, rev):
    if wt.exists():
        git(repo, "worktree", "remove", "--force", str(wt), check=False)
        shutil.rmtree(wt, ignore_errors=True)
    git(repo, "worktree", "prune")
    git(repo, "worktree", "add", "-q", "--detach", str(wt), rev)


def merge_one(wt, rev, msg, am, log):
    """git merge, then additive_merge.py for pure add/add conflicts. Returns 'clean', 'additive' or the files left."""
    r = git(wt, *G, "merge", "-q", "--no-ff", "--no-edit", "-m", msg, rev, check=False)
    if r.returncode == 0:
        return "clean"
    subprocess.run([sys.executable, str(am), "--repo", str(wt)], cwd=wt, capture_output=True, check=False)
    left = git(wt, "diff", "--name-only", "--diff-filter=U").stdout.split()
    if left:
        with open(log, "a") as f:
            f.write(f"REFUSED {msg}: {left}\n{r.stdout}{r.stderr}\n")
        git(wt, "merge", "--abort", check=False)
        return left
    git(wt, *G, "commit", "-q", "--no-edit")
    return "additive"


def patch_tree(src, ci_hooks, tiny):
    if ci_hooks:
        dev = src / "ggml/src/ggml-metal/ggml-metal-device.m"
        t = dev.read_text()
        if t.count(DEVICE_ANCHOR) != 1:
            setup_fail(f"{src.name}: CI hook anchor not found once in {dev.name}")
        dev.write_text(t.replace(DEVICE_ANCHOR, DEVICE_HOOK + DEVICE_ANCHOR))
    if tiny:
        arch = src / "tests/test-llama-archs.cpp"
        a = arch.read_text()
        if a.count(TINY_OLD) != 1:
            setup_fail("inkling2 fixture anchor not found in test-llama-archs.cpp")
        arch.write_text(a.replace(TINY_OLD, TINY_NEW))


def cmake_build(src, bdir, targets, work, jobs):
    metal = sys.platform == "darwin"
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=ON", "-DLLAMA_CURL=OFF", "-DLLAMA_OPENSSL=OFF",
             "-DLLAMA_BUILD_SERVER=OFF"]
    flags += ["-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"] if metal else ["-DGGML_NATIVE=ON"]
    rc, _ = run(["cmake", "-S", src, "-B", bdir, *flags], work / f"cmake_{src.name}.log", 1800)
    t0 = time.time()
    if rc == 0:
        rc, _ = run(["cmake", "--build", bdir, "-j", str(jobs), "--target", *targets], work / f"build_{src.name}.log", 5400)
    if rc != 0:
        txt = read(work / f"build_{src.name}.log") or read(work / f"cmake_{src.name}.log")
        print("\n".join(l for l in txt.splitlines() if "error" in l.lower())[-4000:])
        setup_fail(f"build {src.name} rc={rc}")
    return bdir / "bin", round(time.time() - t0, 1)


def parse_tbo(text):
    lines = text.splitlines()
    ok, fail, unsup = [], [], []
    for line in lines:
        m = re.match(r"^  (\w+\(.*\)): (.*)$", line)
        if not m:
            continue
        name, st = m.group(1), m.group(2).strip()
        if st.endswith("OK"):
            ok.append(name)
        elif "FAIL" in st:
            fail.append(name + " :: " + st[-160:])
        elif st.startswith("not supported"):
            unsup.append(name)
    # every failed case is listed again after "Failing tests:"; a FAIL that landed on a later line is caught here
    block, in_block = [], False
    for line in lines:
        if line.strip() == "Failing tests:":
            in_block = True
        elif in_block and line.startswith("  ") and line.strip():
            block.append(line.strip())
        else:
            in_block = False
    seen = {f.split(" :: ")[0] for f in fail}
    fail += [b + " :: (Failing tests block)" for b in block if b not in seen]
    short = sum(int(t) - int(p) for p, t in re.findall(r"^\s*(\d+)/(\d+) tests passed", text, re.MULTILINE))
    dev = sorted({l.strip()[:160] for l in lines
                  if re.search(r"GPU name|GPU family|simdgroup|has bfloat|has tensor|XPLAT_FORCE", l)})[:16]
    crash = [l.strip()[:200] for l in lines if re.search(r"GGML_ASSERT|GGML_ABORT|failed to compile pipeline|"
                                                         r"Segmentation|Abort trap|libc\+\+abi", l)][:8]
    return {"n_ok": len(ok), "n_fail": max(len(fail), short), "n_unsup": len(unsup), "fail": fail[:60],
            "unsup": unsup[:30], "device": dev, "crash": crash,
            "ext_pipelines": sorted({m.group(1) for m in EXT_RE.finditer(text)})}


def parse_ppl(text):
    r = {}
    for k, pat in (("ppl", r"Final estimate: PPL = (\S+)"), ("ppl_q", r"Mean PPL\(Q\)\s+:\s+(\S+)"),
                   ("kld_mean", r"Mean\s+KLD:\s+(\S+)"), ("kld_max", r"Maximum KLD:\s+(\S+)"),
                   ("top1", r"Same top p:\s+(\S+)")):
        m = re.search(pat, text)
        if m:
            r[k] = m.group(1)
    pipes = sorted(set(re.findall(r"compiling pipeline: base = '[^']*', name = '([^']*flash_attn[^']*)'", text)))
    r["n_rel_pipelines"] = sum(1 for x in pipes if "_rel=1" in x)
    r["fa_pipelines"] = pipes[:16]
    r["errors"] = [l.strip()[:160] for l in text.splitlines()
                   if re.search(r"GGML_ASSERT|error:|failed to|\bnan\b", l, re.IGNORECASE) and not l.startswith("$ ")][:6]
    return r


def corpus_text(path):
    words = ["the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog", "while", "seven", "bright", "stars",
             "rise", "above", "the", "quiet", "harbor", "and", "a", "small", "boat", "drifts", "toward", "the", "old",
             "stone", "lighthouse", "near", "the", "river"]
    x, toks = 12345, []
    for i in range(20000):
        x = (1103515245 * x + 12345) % (1 << 31)
        toks.append(words[x % len(words)])
        if i % 17 == 16:
            toks.append(".\n")
    Path(path).write_text(" ".join(toks), encoding="utf-8")


def kld_ok(p, tol):
    try:
        return p["rc"] == 0 and float(p.get("kld_mean", "nan")) <= tol
    except ValueError:
        return False


def parse_bench(log):
    try:
        txt = read(log)
        return [{x: r.get(x) for x in ("n_prompt", "n_gen", "n_depth", "avg_ts", "stddev_ts")}
                for r in json.loads(txt[txt.index("["):txt.rindex("]") + 1])]
    except ValueError:
        return []


def short_case(c):
    return (c.replace("MUL_MAT_ID(", "").replace("type_b=f32,", "").replace(",m_v=0,ids_offset=0,skip=0", "")
            .rstrip(")"))


class Job:
    def __init__(self, a, work, out):
        self.a, self.work, self.out = a, work, out
        self.env = dict(FORCE_ENV) if a.ci_hooks else {}
        self.res = {"job": "mac_metal_check", "version": 1, "args": vars(a), "steps": {}, "gates": {}, "ok": False,
                    "ci_hooks": a.ci_hooks}

    def save(self):
        self.out.write_text(json.dumps(self.res, indent=1, default=str), encoding="utf-8")

    def gate(self, name, ok, why=""):
        self.res["gates"][name] = bool(ok)
        print(f"GATE {name}: {'PASS' if ok else 'FAIL'}{(' (' + why + ')') if why else ''}", flush=True)
        self.save()

    def tbo(self, key, bin_, op, filt=None, env_add=None, mode="test"):
        cmd = [bin_ / "test-backend-ops", mode, "-o", op, "-b", "MTL0"] + (["-p", filt] if filt else [])
        rc, dt = run(cmd, self.work / f"tbo_{key}.log", 5400, {**self.env, **(env_add or {})})
        r = parse_tbo(read(self.work / f"tbo_{key}.log"))
        r.update(rc=rc, seconds=dt)
        self.res["steps"][f"tbo_{key}"] = r
        print(f"{key}: rc={rc} ok={r['n_ok']} fail={r['n_fail']} unsup={r['n_unsup']} {dt}s", flush=True)
        for f in r["fail"][:12] + r["crash"][:4]:
            print(f"  {f}", flush=True)
        self.save()
        return r

    # ---------------------------------------------------------------- trees
    def trees(self, parts):
        a, work = self.a, self.work
        t0 = time.time()
        if a.pinset:
            pinset = Path(a.pinset).resolve()
        else:
            pinset = work / "pinset"
            if not (pinset / ".git").exists():
                r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", a.pinset_ref, PINSET_URL,
                                    str(pinset)], capture_output=True, text=True, check=False)
                if r.returncode != 0:
                    setup_fail(f"clone {PINSET_URL}: {r.stderr[-400:]}")
            else:
                git(pinset, "fetch", "-q", "--depth", "1", "origin", a.pinset_ref)
                git(pinset, "checkout", "-q", "--detach", "FETCH_HEAD")
        am, pr_set = pinset / "scripts/unsloth/additive_merge.py", pinset / "scripts/unsloth/pr-set.json"
        if not am.is_file() or not pr_set.is_file():
            setup_fail(f"{pinset}: no scripts/unsloth/pr-set.json / additive_merge.py")
        self.res["pinset"] = {"dir": str(pinset), "commit": git(pinset, "rev-parse", "HEAD", check=False).stdout.strip()}
        pins = load_pins(pr_set)
        self.res["pins"] = [f"{o}#{n}@{s[:10]}" for o, n, s in pins]

        repo = work / "repo"
        if not (repo / ".git").exists():
            r = subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", UPSTREAM, str(repo)],
                               capture_output=True, text=True, check=False)
            if r.returncode != 0:
                setup_fail(f"clone {UPSTREAM}: {r.stderr[-400:]}")
        git(repo, "fetch", "-q", "--filter=blob:none", UPSTREAM, f"+refs/tags/{a.tag}:refs/tags/{a.tag}")
        by_org = {}
        for o, _, s in pins:
            by_org.setdefault(o, set()).add(s)
        for o, shas in by_org.items():
            remote = os.environ.get("XPLAT_PIN_REMOTE", f"https://github.com/{o}/llama.cpp")
            git(repo, "fetch", "-q", "--filter=blob:none", remote, *sorted(shas))
        mix = work / "mix"
        fresh_worktree(repo, mix, f"refs/tags/{a.tag}")
        res = []
        for o, n, s in pins:
            m = merge_one(mix, s, f"pin {o}#{n}", am, work / "merge.log")
            if isinstance(m, list):
                setup_fail(f"mix: pin {o}#{n} does not merge ({' '.join(m)})")
            res.append(f"{o}#{n} {m}")
        self.res["steps"]["merge_mix"] = res
        self.res["mix_head"] = git(mix, "rev-parse", "HEAD").stdout.strip()
        md = git(mix, "diff", "--stat=160", f"refs/tags/{a.tag}", "HEAD", "--", "ggml/src/ggml-metal").stdout
        self.res["mix_metal_diffstat"] = md.strip().splitlines()
        self.trees_ = {"mix": mix}
        patch_tree(mix, a.ci_hooks, tiny="model" in parts or "bench" in parts)

        if "mvid" in parts:
            url, sha = self.add
            git(repo, "fetch", "-q", "--filter=blob:none", url, sha)
            mvid = work / "mvid"
            fresh_worktree(repo, mvid, self.res["mix_head"])
            m = merge_one(mvid, sha, f"add {url}@{sha[:10]}", am, work / "merge.log")
            if isinstance(m, list):
                print(f"NOTE: {sha[:10]} does not merge onto the mix tree ({' '.join(m)}); "
                      f"mvid runs on the commit's own tree", flush=True)
                self.res["mvid_tree"] = {"base": "commit-only", "merge_conflicts": m}
                fresh_worktree(repo, mvid, sha)
            else:
                self.res["mvid_tree"] = {"base": "mix+commit", "merge": m}
            self.res["mvid_tree"]["head"] = git(mvid, "rev-parse", "HEAD").stdout.strip()
            self.res["mvid_tree"]["metal_diffstat_vs_mix"] = git(
                mvid, "diff", "--stat=160", self.res["mix_head"], "HEAD", "--", "ggml/src/ggml-metal").stdout.strip().splitlines()
            patch_tree(mvid, a.ci_hooks, tiny=False)
            self.trees_["mvid"] = mvid
        self.res["steps"]["trees_seconds"] = round(time.time() - t0, 1)
        self.save()

    # ---------------------------------------------------------------- parts
    def part_pins(self, b):
        a = self.a
        ops = [("fa_banded", "FLASH_ATTN_EXT_BANDED", None), ("fa_mixed", "FLASH_ATTN_EXT", MIXED_RE)]
        if not a.quick:
            ops.append(("fa_all", "FLASH_ATTN_EXT", None))
        tensor = False
        for key, op, filt in ops:
            r = self.tbo(key, b, op, filt)
            tensor |= any("has tensor" in d and "true" in d for d in r["device"])
            self.gate(f"pins_{key}", r["rc"] == 0 and r["n_fail"] == 0 and r["n_ok"] > 0,
                      f"ok={r['n_ok']} fail={r['n_fail']} rc={r['rc']}")
        if tensor and not a.ci_hooks:      # M5-class GPU: the non-tensor FA kernels are a second code path
            for key, op, filt in ops[:2]:
                r = self.tbo(key + "_notensor", b, op, filt, {"GGML_METAL_TENSOR_DISABLE": "1"})
                self.gate(f"pins_{key}_notensor", r["rc"] == 0 and r["n_fail"] == 0 and r["n_ok"] > 0,
                          f"ok={r['n_ok']} fail={r['n_fail']}")
        # fork-only quant types (pins #61, #332): no Metal kernels, so Metal must refuse them
        for key, op, filt in (("fork_get_rows", "GET_ROWS", f"type={FORK_TYPES},"),
                              ("fork_mul_mat", "MUL_MAT", f"type_a={FORK_TYPES},"),
                              ("fork_mul_mat_id", "MUL_MAT_ID", f"type_a={FORK_TYPES},")):
            r = self.tbo(key, b, op, filt)
            seen = r["n_ok"] + r["n_fail"] + r["n_unsup"]          # a filter matching no case would pass vacuously
            self.gate(f"pins_{key}", r["rc"] == 0 and r["n_fail"] == 0 and seen > 0,
                      f"ok={r['n_ok']} fail={r['n_fail']} unsup={r['n_unsup']} rc={r['rc']}")

    def tiny_model(self, b):
        mdir = self.work / "models"
        mdir.mkdir(exist_ok=True)
        rc, _ = run([b / "test-llama-archs", "-a", "^inkling2$", "-s", "42", "-o", mdir], self.work / "archs_save.log",
                    1800, self.env)
        models = sorted(p for p in mdir.rglob("*.gguf") if "inkling2" in p.name and "loop" not in p.name)
        self.res["steps"]["tiny_model"] = {"rc": rc, "files": [p.name for p in models]}
        if not models:
            print(read(self.work / "archs_save.log")[-3000:])
            setup_fail("no inkling2 GGUF written")
        return models[0]

    def part_model(self, b, model):
        a, work = self.a, self.work
        corpus = work / "corpus.txt"
        corpus_text(corpus)
        g_kld, g_eng, table = True, True, {}
        for k, v in KVS:
            name = k if k == v else f"{k}-{v}"
            ref = work / f"ref_cpu_{name}.bin"
            common = ["-m", model, "-f", corpus, "-t", "4", "--seed", "42", "-fa", "on", "-c", "256", "-b", "256",
                      "--chunks", "4", "-ctk", k, "-ctv", v, "-lv", "5"]
            for arm, extra, score in (("cpu_write", ["-dev", "none", "-ngl", "0"], False),
                                      ("metal_vs_cpu", ["-ngl", "99"], True),
                                      ("metal_ub16_vs_cpu", ["-ngl", "99", "-ub", "16"], True)):
                log = work / f"kld_{name}_{arm}.log"
                cmd = [b / "llama-perplexity"] + common + extra + ["--kl-divergence-base", ref] + (["--kl-divergence"] if score else [])
                rc, dt = run(cmd, log, 1800, self.env)
                p = parse_ppl(read(log))
                p.update(rc=rc, seconds=dt)
                self.res["steps"][f"kld_{name}_{arm}"] = p
                print(f"{name} {arm}: rc={rc} PPL={p.get('ppl') or p.get('ppl_q')} KLD={p.get('kld_mean')} "
                      f"top1={p.get('top1')} rel_pipelines={p['n_rel_pipelines']}", flush=True)
                if score:
                    g_kld &= kld_ok(p, a.kld_tol)
                    g_eng &= p["n_rel_pipelines"] > 0
                    table.setdefault(name, {})[arm] = {x: p.get(x) for x in ("kld_mean", "kld_max", "top1", "n_rel_pipelines")}
                self.save()
            ref.unlink(missing_ok=True)
        self.res["kld_table"] = table
        self.gate("model_kld_metal_vs_cpu", g_kld, f"tol {a.kld_tol}")
        self.gate("model_banded_metal_engaged", g_eng)

    def part_bench(self, b, model):
        bench = {}
        for k, v in BENCH_KVS:
            log = self.work / f"bench_{k}-{v}.log"
            rc, _ = run([b / "llama-bench", "-m", model, "-t", "4", "-fa", "1", "-ngl", "99", "-p", "512", "-n", "64",
                         "-d", "0,4096", "-r", "3", "-ctk", k, "-ctv", v, "-o", "json"], log, 1800, self.env)
            rows = parse_bench(log)
            bench[f"{k}-{v}"] = {"rc": rc, "rows": rows}
            print(f"bench {k}-{v}: rc={rc} " + "; ".join(
                f"{'pp' + str(r['n_prompt']) if r['n_prompt'] else 'tg' + str(r['n_gen'])}@d{r['n_depth']} "
                f"{r['avg_ts']:.1f}+-{r['stddev_ts']:.1f} t/s" for r in rows if r.get("avg_ts") is not None), flush=True)
            self.save()
        self.res["bench"] = bench
        self.gate("bench_ran", all(x["rc"] == 0 and x["rows"] for x in bench.values()))

    def part_mvid(self, b):
        a = self.a
        new = self.tbo("mvid_eval", b, "MUL_MAT_ID")
        old = self.tbo("mvid_eval_kill", b, "MUL_MAT_ID", a.mvid_eval_filter if a.quick else None, MVID_KILL)
        self.gate("mvid_eval", new["rc"] == 0 and new["n_fail"] == 0 and new["n_ok"] > 0,
                  f"ok={new['n_ok']} fail={new['n_fail']} rc={new['rc']}")
        self.gate("mvid_engaged", len(new["ext_pipelines"]) > 0, f"{len(new['ext_pipelines'])} ext pipelines")
        self.gate("mvid_eval_killswitch", old["rc"] == 0 and old["n_fail"] == 0 and old["n_ok"] > 0
                  and not old["ext_pipelines"], f"ok={old['n_ok']} fail={old['n_fail']} ext={len(old['ext_pipelines'])}")
        times = {}
        arms = (("new", {}), ("old", MVID_KILL))
        for rnd in range(a.mvid_rounds):
            for arm, env in (arms if rnd % 2 == 0 else arms[::-1]):
                log = self.work / f"mvid_perf_r{rnd}_{arm}.log"
                run([b / "test-backend-ops", "perf", "-o", "MUL_MAT_ID", "-b", "MTL0", "-p", a.mvid_filter], log, 3600,
                    {**self.env, **env})
                for m in PERF_RE.finditer(read(log)):
                    times.setdefault(m.group(1), {}).setdefault(arm, []).append(float(m.group(2)))
        cases = {}
        for case, by in sorted(times.items()):
            if len(by.get("old", [])) != a.mvid_rounds or len(by.get("new", [])) != a.mvid_rounds:
                continue
            mo, mn = statistics.median(by["old"]), statistics.median(by["new"])
            cases[case] = {"old_us": mo, "new_us": mn, "old_over_new": round(mo / mn, 3) if mn else None,
                           "old_spread": round((max(by["old"]) - min(by["old"])) / mo, 4) if mo else None,
                           "rounds": by}
        sp = sorted(c["old_over_new"] for c in cases.values() if c["old_over_new"])
        slower = [c for c, x in cases.items() if x["old_over_new"] and x["old_over_new"] < 1 - max(0.03, x["old_spread"])]
        self.res["mvid_perf"] = {"filter": a.mvid_filter, "rounds": a.mvid_rounds, "n_cases": len(cases),
                                 "old_over_new_median": sp[len(sp) // 2] if sp else None, "slower_than_old": slower,
                                 "cases": cases, "timing_is_real": not a.ci_hooks}
        self.gate("mvid_perf_ran", len(cases) > 0, f"{len(cases)} cases")

    # ---------------------------------------------------------------- summary
    def summary(self):
        r, L = self.res, []
        L.append(f"mac_metal_check  host={r.get('host')}  tag={self.a.tag}  pins={len(r.get('pins', []))}  "
                 f"pinset={r.get('pinset', {}).get('commit', '')[:10]}  ci_hooks={self.a.ci_hooks}")
        if "mvid_tree" in r:
            L.append(f"mvid tree: {r['mvid_tree']['base']} ({r['mvid_tree'].get('merge') or r['mvid_tree'].get('merge_conflicts')})")
        L.append("")
        L.append("| gate | verdict |")
        L.append("|---|---|")
        L += [f"| {g} | {'PASS' if v else 'FAIL'} |" for g, v in r["gates"].items()]
        if r.get("kld_table"):
            L += ["", "| K/V | Metal vs CPU KLD | max | top1 | ub16 KLD | banded pipelines |", "|---|---|---|---|---|---|"]
            for kv, t in r["kld_table"].items():
                m, u = t.get("metal_vs_cpu", {}), t.get("metal_ub16_vs_cpu", {})
                L.append(f"| {kv} | {m.get('kld_mean')} | {m.get('kld_max')} | {m.get('top1')} | {u.get('kld_mean')} | "
                         f"{m.get('n_rel_pipelines')} |")
        if r.get("bench"):
            L += ["", "| llama-bench (tiny Inkling 2, Metal) | t/s |", "|---|---|"]
            for kv, x in r["bench"].items():
                for row in x["rows"]:
                    t = f"pp{row['n_prompt']}" if row["n_prompt"] else f"tg{row['n_gen']}"
                    L.append(f"| {kv} {t} @d{row['n_depth']} | {row['avg_ts']:.1f} +- {row['stddev_ts']:.1f} |")
        p = r.get("mvid_perf")
        if p and p["cases"]:
            L += ["", f"MUL_MAT_ID perf, median of {p['rounds']} interleaved rounds (old = kill switch on); "
                      f"old/new median {p['old_over_new_median']}" + ("" if p["timing_is_real"] else " (CI runner: not real timing)"),
                  "n outside 2..8 = control rows (the new kernel does not engage there, old/new should be ~1)",
                  "| case | old us | new us | old/new | old spread |", "|---|---|---|---|---|"]
            for c, x in p["cases"].items():
                n = re.search(r",n=(\d+),", c)
                ctl = " (control)" if n and not 2 <= int(n.group(1)) <= 8 else ""
                L.append(f"| {short_case(c)}{ctl} | {x['old_us']:.1f} | {x['new_us']:.1f} | {x['old_over_new']} | "
                         f"{x['old_spread']} |")
        L += ["", f"OVERALL: {'PASS' if r['ok'] else 'FAIL'}"]
        txt = "\n".join(L) + "\n"
        (self.work / "summary.md").write_text(txt)
        print("\n" + txt, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="b12722", help="upstream base tag of the prebuilt")
    ap.add_argument("--pinset", help="dir of an unslothai/llama.cpp checkout (default: clone master into --work)")
    ap.add_argument("--pinset-ref", default="master")
    ap.add_argument("--add-commit", help="https://github.com/<owner>/llama.cpp@<sha40>: merged onto the mix as mvid")
    ap.add_argument("--parts", help="comma list of pins,model,bench,mvid (default: all that apply)")
    ap.add_argument("--quick", action="store_true", help="skip the full FLASH_ATTN_EXT run, kill-switch eval on the new cases only")
    ap.add_argument("--kld-tol", type=float, default=5e-3)
    ap.add_argument("--mvid-rounds", type=int, default=3)
    ap.add_argument("--mvid-filter", default="n_mats=128,n_used=8,b=[01],m=2304,")
    ap.add_argument("--mvid-eval-filter", default="n_mats=128,n_used=8,")
    ap.add_argument("--ci-hooks", action="store_true", default=os.environ.get("MAC_METAL_CI_HOOKS") == "1",
                    help="patch + set the paravirtual-GPU force hooks (GitHub macOS runners only)")
    ap.add_argument("--work", default="mac_metal_check_work")
    ap.add_argument("--out", default=None, help="result JSON (default: <work>/mac_metal_check.json)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--prepare-only", action="store_true", help="build the merged trees, then stop")
    ap.add_argument("--build-only", action="store_true", help="trees + builds, no tests (Linux/CPU smoke)")
    a = ap.parse_args()
    parts = (a.parts.split(",") if a.parts else ["pins", "model", "bench"] + (["mvid"] if a.add_commit else []))
    bad = set(parts) - {"pins", "model", "bench", "mvid"}
    if bad:
        ap.error(f"unknown parts {bad}")
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    out = Path(a.out).resolve() if a.out else work / "mac_metal_check.json"
    for old in work.glob("*.log"):              # logs of an earlier run would land in this run's bundle
        if not old.name.startswith("run_"):
            old.unlink()
    job = Job(a, work, out)
    try:
        if "mvid" in parts:
            job.add = parse_add_commit(a.add_commit)
        try:
            host = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], capture_output=True,
                                  text=True, check=False).stdout.split()
            sw = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True, check=False).stdout.strip()
            job.res["host"] = " ".join(host[:-1]) + (f" {int(host[-1]) >> 30} GB" if host and host[-1].isdigit() else "") + f" macOS {sw}"
        except (FileNotFoundError, ValueError):
            job.res["host"] = f"{sys.platform} {os.uname().machine}"
        job.res["parts"] = parts
        job.save()
        job.trees(parts)
        if a.prepare_only:
            print(json.dumps({k: job.res[k] for k in ("steps", "mix_metal_diffstat", "mvid_tree") if k in job.res}, indent=1))
            return 0
        bins = {}
        mix_targets = (["test-backend-ops"] if "pins" in parts else []) + (
            ["test-llama-archs", "llama-perplexity", "llama-bench"] if {"model", "bench"} & set(parts) else [])
        if mix_targets:
            bins["mix"], job.res["steps"]["build_mix_seconds"] = cmake_build(job.trees_["mix"], work / "build_mix",
                                                                             mix_targets, work, a.jobs)
        if "mvid" in parts:
            bins["mvid"], job.res["steps"]["build_mvid_seconds"] = cmake_build(job.trees_["mvid"], work / "build_mvid",
                                                                               ["test-backend-ops"], work, a.jobs)
        job.save()
        if a.build_only:
            print(json.dumps(job.res["steps"], indent=1, default=str))
            return 0
        if "pins" in parts:
            job.part_pins(bins["mix"])
        if {"model", "bench"} & set(parts):
            model = job.tiny_model(bins["mix"])
            if "model" in parts:
                job.part_model(bins["mix"], model)
            if "bench" in parts:
                job.part_bench(bins["mix"], model)
        if "mvid" in parts:
            job.part_mvid(bins["mvid"])
    except SetupFail as e:
        print(f"SETUP FAIL: {e}", flush=True)
        job.res["setup_fail"] = str(e)
        job.save()
        return 2
    job.res["ok"] = bool(job.res["gates"]) and all(job.res["gates"].values())
    job.save()
    job.summary()
    return 0 if job.res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
