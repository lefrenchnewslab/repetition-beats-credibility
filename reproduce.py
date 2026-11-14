#!/usr/bin/env python3
"""Rebuild every table, figure and number in the paper from the committed outputs (offline, CPU, about 1-2 minutes).

    python3 reproduce.py                 # writes reproduced/ and prints the headline numbers
    python3 reproduce.py --check-data    # also regenerates the prompt files and checks them byte for byte (+30 s)

What it does:
 1. copies scripts/, data/, results/ and paper/main.tex into a temporary directory and decompresses every *.gz there
    (the repository itself is never modified; committed numbers_*.json are NOT copied, so they are regenerated);
 2. runs the analysis scripts in order: lexical_ngram -> analyze_v2 (Block A) -> analyze_b (Block B, + Llama and Gemma
    replications) -> figs_b -> analyze_d (Block D) -> analyze_f + posthoc_f1_templates (Block F) -> polish_figs + fig_intro (print
    figures) -> make_appendix (appendix tables) -> number_check (every hand-typed number in the main text);
 3. compares the regenerated numbers_A/B/D/F.json, numbers_F_posthoc.json and appendix tables with the committed ones;
 4. copies numbers, summaries, appendix .tex, figures and NUMBER_CHECK.md to reproduced/.
Exit code 0 only if every step ran, the regenerated numbers equal the committed ones and number_check passes.
Requirements: Python >= 3.10 with numpy, scipy, scikit-learn, matplotlib (requirements.txt). No GPU, no network.
"""
import gzip, hashlib, json, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "reproduced")
QWEN = "Qwen2.5-7B-Instruct"
NUMS = ("A", "B", "D", "F", "F_posthoc")


def sh(tmp, *args):
    t = time.time()
    r = subprocess.run([sys.executable, *args], cwd=tmp, capture_output=True, text=True,
                       env={**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": os.path.join(tmp, ".mpl")})
    print(f"  {' '.join(args):<72} {time.time() - t:5.1f} s", flush=True)
    if r.returncode != 0 and "number_check" not in args[0]:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n"); sys.exit(f"step failed: {' '.join(args)}")
    return r


def copytree_gunzip(src, dst):
    for d, _, files in os.walk(src):
        rel = os.path.relpath(d, src); os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            s = os.path.join(d, f)
            if f.endswith(".gz"):
                with gzip.open(s, "rb") as fi, open(os.path.join(dst, rel, f[:-3]), "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            else:
                shutil.copy2(s, os.path.join(dst, rel, f))


def compare_numbers(tmp):
    bad = 0
    for n in NUMS:
        new, old = (json.load(open(os.path.join(p, f"numbers_{n}.json"))) for p in (tmp, ROOT))
        diff = sorted(set(new) ^ set(old))
        for k in set(new) & set(old):
            for f in ("value", "ci"):
                x, y = new[k].get(f), old[k].get(f)
                xs, ys = (x if isinstance(x, list) else [x]), (y if isinstance(y, list) else [y])
                if len(xs) != len(ys) or any((p is None) != (q is None) or (p is not None and abs(p - q) > 1e-9) for p, q in zip(xs, ys)):
                    diff.append(k); break
        bad += len(diff)
        print(f"  numbers_{n}.json: {len(new)} keys regenerated, {len(diff)} differ from the committed file" + (f" (e.g. {diff[:3]})" if diff else ""))
    for f in ("app_prereg.tex", "app_tables.tex"):
        same = open(os.path.join(tmp, "paper", f)).read() == open(os.path.join(ROOT, "paper", f)).read()
        bad += not same; print(f"  paper/{f}: {'identical to' if same else 'DIFFERS from'} the committed file")
    return bad


def check_data(tmp):
    d = os.path.join(tmp, "data_rebuilt"); os.makedirs(d)
    sh(tmp, "scripts/build_data_v2.py", d, os.path.join(tmp, "data", "pilot_pairs.jsonl"))
    bad = 0
    for f in sorted(os.listdir(d)):
        a = hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
        b = hashlib.sha256(open(os.path.join(tmp, "data", f), "rb").read()).hexdigest()
        bad += a != b; print(f"  data/{f}: {'same SHA-256' if a == b else 'DIFFERENT SHA-256'}")
    man = json.load(open(os.path.join(ROOT, "data", "manifest_v2.json")))["prompts_sha256"]
    got = hashlib.sha256(open(os.path.join(d, "prompts_v2.jsonl"), "rb").read()).hexdigest()
    bad += man != got; print(f"  prompts_v2.jsonl SHA-256 {'matches' if man == got else 'does NOT match'} manifest_v2.json")
    # Block F prompts: build_data_f.py asserts C1/C2k4/C3k4/C7k4 are byte-identical to prompts_v2.jsonl, then rewrites data/prompts_f.jsonl
    sh(tmp, "scripts/build_data_f.py")
    man = json.load(open(os.path.join(ROOT, "data", "manifest_f.json")))["sha256"]
    got = hashlib.sha256(open(os.path.join(tmp, "data", "prompts_f.jsonl"), "rb").read()).hexdigest()
    bad += man != got; print(f"  prompts_f.jsonl SHA-256 {'matches' if man == got else 'does NOT match'} manifest_f.json")
    return bad


def headline(tmp):
    J = lambda f: json.load(open(os.path.join(tmp, f)))
    A, B, D, F = J("numbers_A.json"), J("numbers_B.json"), J("numbers_D.json"), J("numbers_F.json")
    v = lambda X, k, s=1: X[k]["value"] * s
    ci = lambda X, k, s=1: "[" + ", ".join(f"{c * s:.1f}" for c in X[k]["ci"]) + "]"
    rows = [
        ("Abstract, Sec. 4.2", "Qwen2.5-32B, documents: items flipped by 4 paraphrased copies (%)", f"{v(A, 'Qwen-32B.F3.test.C7k4.flip_pct'):.1f}"),
        ("Abstract, Sec. 4.2", "Qwen2.5-32B, documents: items flipped by 4 exact copies (%)", f"{v(A, 'Qwen-32B.F3.test.C2k4.flip_pct'):.2f}"),
        ("Sec. 4.2", "Qwen2.5-32B, documents: paraphrased minus exact (points)",
         f"{v(A, 'Qwen-32B.F3.test.C7k4_minus_C2k4_pts'):.1f} {ci(A, 'Qwen-32B.F3.test.C7k4_minus_C2k4_pts')}"),
        *[("Sec. 4.3, Fig. 4", f"{m}: top-10 heads recover, tables (%)",
           f"{v(A, f'{m}.F1.patch.top10.recovery_pct'):.1f} {ci(A, f'{m}.F1.patch.top10.recovery_pct')}") for m in ("Llama-8B", "Qwen-7B", "Gemma-12B")],
        *[("Sec. 4.2 (Block F)", f"{m}, documents: paraphrase gain minus exact-copy gain D (points)",
           f"{v(F, f'F1.{m}.F3.D_pts'):.1f} {ci(F, f'F1.{m}.F3.D_pts')}") for m in ("Qwen-32B", "Qwen-7B")],
        ("Sec. 4.2 (Block F)", "Qwen2.5-32B, documents: one paraphrased copy, wording effect W (points)", f"{v(F, 'F1.Qwen-32B.F3.W_pts'):.1f} {ci(F, 'F1.Qwen-32B.F3.W_pts')}"),
        *[("Sec. 4.3 (Block F)", f"{m}: top-10 heads recover, tables, C1->C6 government repeated (%)",
           f"{v(F, f'F2.{m}.C1k1->C6k4.recovery_top10_pct'):.1f} {ci(F, f'F2.{m}.C1k1->C6k4.recovery_top10_pct')}") for m in ("Llama-8B", "Qwen-7B", "Gemma-12B")],
        ("Sec. 4.5", "Head damping (Qwen2.5-7B, tables): exact-copy effect removed (%)", f"{v(B, 'B3.F1.F1.BR_C2k4', 100):.1f} {ci(B, 'B3.F1.F1.BR_C2k4', 100)}"),
        ("Sec. 4.5", "Head damping: source preference kept (%)", f"{v(B, 'B3.F1.F1.pref_kept', 100):.1f} {ci(B, 'B3.F1.F1.pref_kept', 100)}"),
        ("Sec. 4.5, Fig. 6", "LoRA (Qwen2.5-7B, documents): paraphrase effect removed (%)", f"{v(D, 'D.F3.BR_C7k4', 100):.1f} {ci(D, 'D.F3.BR_C7k4', 100)}"),
        ("Sec. 4.5", "LoRA, documents: independent-user effect kept (%)", f"{v(D, 'D.F3.corrob_kept', 100):.1f} {ci(D, 'D.F3.corrob_kept', 100)}"),
    ]
    print("\nHeadline numbers (should match README 'Quickstart'):")
    for where, what, val in rows:
        print(f"  {where:<20} {what:<70} {val}")


def main():
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="rbc_reproduce_")
    print(f"[1/4] staging committed outputs in {tmp}")
    for d in ("scripts", "data", "results"):
        copytree_gunzip(os.path.join(ROOT, d), os.path.join(tmp, d))
    os.makedirs(os.path.join(tmp, "paper")); os.makedirs(os.path.join(tmp, "figs"))
    shutil.copy2(os.path.join(ROOT, "paper", "main.tex"), os.path.join(tmp, "paper", "main.tex"))
    print("[2/4] running the analysis scripts")
    sh(tmp, "scripts/lexical_ngram.py", "data/prompts_v2.jsonl", "results/A")
    sh(tmp, "scripts/analyze_v2.py", "results/A")
    sh(tmp, "scripts/analyze_b.py", "results/B", "results/A")
    sh(tmp, "scripts/analyze_b.py", "results/B/llama", "results/A", "Llama-3.1-8B-Instruct")
    sh(tmp, "scripts/analyze_b.py", "results/B/gemma", "results/A", "gemma-3-12b-it")
    sh(tmp, "scripts/figs_b.py", "results/B")
    sh(tmp, "scripts/analyze_d.py", "results/D")
    sh(tmp, "scripts/analyze_f.py")
    sh(tmp, "scripts/posthoc_f1_templates.py")
    sh(tmp, "scripts/polish_figs.py")
    sh(tmp, "scripts/fig_intro.py", ".")
    sh(tmp, "scripts/make_appendix.py")
    nc = sh(tmp, "scripts/number_check.py", "--out", os.path.join(tmp, "NUMBER_CHECK.md"))
    print("[3/4] comparing with the committed outputs")
    bad = compare_numbers(tmp)
    if "--check-data" in sys.argv:
        bad += check_data(tmp)
    print(f"[4/4] writing {os.path.relpath(OUT, ROOT)}/")
    shutil.rmtree(OUT, ignore_errors=True)
    for sub in ("figs", "paper", "summaries"):
        os.makedirs(os.path.join(OUT, sub))
    for n in NUMS:
        shutil.copy2(os.path.join(tmp, f"numbers_{n}.json"), OUT)
    shutil.copy2(os.path.join(tmp, "NUMBER_CHECK.md"), OUT)
    for f in os.listdir(os.path.join(tmp, "figs")):
        shutil.copy2(os.path.join(tmp, "figs", f), os.path.join(OUT, "figs"))
    for f in ("app_prereg.tex", "app_tables.tex"):
        shutil.copy2(os.path.join(tmp, "paper", f), os.path.join(OUT, "paper"))
    for f in ("results/A/summary_A.md", "results/B/summary_B.md", "results/D/summary_D.md",
              "results/B/llama/summary_B_Llama-3.1-8B-Instruct.md", "results/B/gemma/summary_B_gemma-3-12b-it.md", "results/F/summary_F.md"):
        shutil.copy2(os.path.join(tmp, f), os.path.join(OUT, "summaries"))
    headline(tmp)
    if "--keep" not in sys.argv:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + nc.stdout.strip())
    ok = bad == 0 and nc.returncode == 0
    print(f"\nreproduce.py: {'OK' if ok else 'FAILED'} in {time.time() - t0:.0f} s "
          f"({'all regenerated numbers equal the committed ones' if bad == 0 else f'{bad} differences'}; "
          f"number_check {'passed' if nc.returncode == 0 else 'FAILED'})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
