"""SPRINT Block F (14 Nov 2026) - reviewer controls F1: C7 k=1 per template, C8 k=4 (different users, paraphrased), C5p. New file.
Reuses build_data_v2's renderers (run in a throw-away namespace), mirrors its build() and asserts that C1, C2k4, C3k4, C7k4 rebuilt
here are byte-identical to data/prompts_v2.jsonl (FM H2). Test split only. Writes data/prompts_f.jsonl + data/manifest_f.json.
  python3 scripts/build_data_f.py
"""
import json, os, sys, runpy, tempfile, itertools, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "data")
sys.argv = ["build_data_v2.py", tempfile.mkdtemp()]
NS = runpy.run_path(os.path.join(HERE, "build_data_v2.py"), run_name="v2")
INTRO, BLOCK = NS["INTRO"], NS["BLOCK"]; pairs = [p for p in NS["pairs"] if p["split"] == "test"]

def build_f(p, fmt, cond, k, lowidx, order, ans, t=0):
    ent, cf = p["ent"], p["cfield"]
    vlow, vhigh = p["values"][lowidx], p["values"][1 - lowidx]
    H = ("H", p["gov"], vhigh, ent, 0); L1 = ("L", p["users"][0], vlow, ent, 0)
    blocks_high, blocks_low = [H], []
    if cond == "C1": blocks_low = [L1]
    elif cond == "C2": blocks_low = [L1] * k
    elif cond == "C3": blocks_low = [("L", p["users"][i], vlow, ent, 0) for i in range(k)]
    elif cond == "C7": blocks_low = [("L", p["users"][0], vlow, ent, v) for v in range(k)]
    elif cond == "C7k1": blocks_low = [("L", p["users"][0], vlow, ent, t)]                       # one paraphrased copy, template t
    elif cond == "C8": blocks_low = [("L", p["users"][i], vlow, ent, i) for i in range(k)]         # different users, paraphrased
    elif cond == "C5p": blocks_low = [("F", None, None, f, v) for f, v in zip(p["fillers"][:3], (1, 2, 3))] + [L1]   # filler in templates 1-3
    else: raise ValueError(cond)
    if cond == "C5p" and order == "LF": blocks = blocks_low + blocks_high
    elif cond == "C5p": blocks = blocks_high + blocks_low
    else: blocks = blocks_high + blocks_low if order == "HF" else blocks_low + blocks_high
    s, spans = INTRO[fmt], []; lcopy = 0
    for i, (role, lab, val, e, var) in enumerate(blocks):
        t0 = len(s)
        if cond == "C5p" and fmt == "F3" and role == "F":   # F3 filler: 4 attribute sentences (no source line), to match C7k4 length
            order_ = list(e["attrs"])[:4]; order_ = order_[var % 4:] + order_[:var % 4]
            txt = f"[Document {i + 1}]\nText: " + " ".join(NS["SENT"][(e["cls"], f_)][var].replace("{n}", e["name"]).replace("{v}", NS["cap"](e["attrs"][f_])) for f_ in order_); sp = []
        else:
            txt, sp = BLOCK[fmt](i, role, lab, val, e, cf, var, "Source")
        for kind, a, b in sp: spans.append({"kind": kind, "role": role, "idx": i, "copy": lcopy if role == "L" else None, "start": t0 + a, "end": t0 + b})
        s += txt; spans.append({"kind": "block", "role": role, "idx": i, "copy": lcopy if role == "L" else None, "start": t0, "end": len(s)})
        if role == "L": lcopy += 1
        s += "\n\n"
    q = f"What is {ent['name']}'s {cf}?"; opts = (vhigh, vlow) if ans == "AH" else (vlow, vhigh)
    s += (f"Question: {q}" if fmt == "F3" else f"\n{q}") + f"\n(A) {opts[0]}\n(B) {opts[1]}"
    for sp in spans:
        if sp["kind"] == "value": assert s[sp["start"]:sp["end"]] == (vlow if sp["role"] == "L" else vhigh)
    return s, spans, ("B" if ans == "AH" else "A")

# ---- H2: byte-identity of shared conditions with prompts_v2.jsonl ----
ref = {}
for l in open(os.path.join(D, "prompts_v2.jsonl")):
    r = json.loads(l)
    if r["split"] == "test" and r["fmt"] in ("F1", "F3") and r["ck"] in ("C1k1", "C2k4", "C3k4", "C7k4"):
        ref[(r["fmt"], r["ck"], r["pair_id"], r["lowidx"], r["order"], r["ans"])] = r["user"]
n_ok = 0
for (fmt, ck, pid, lowidx, order, ans), u in ref.items():
    p = next(q for q in pairs if q["pair_id"] == pid); cond, k = ck.split("k"); assert build_f(p, fmt, cond, int(k), lowidx, order, ans)[0] == u, (fmt, ck, pid); n_ok += 1
# C7k1 template 0 must equal C1 (FM H1)
p0 = pairs[0]; assert build_f(p0, "F3", "C7k1", 1, 0, "HF", "AH", t=0)[0] == build_f(p0, "F3", "C1", 1, 0, "HF", "AH")[0]
# ---- write ----
PLAN = [("C7k1", 1, t, f"C7k1t{t}") for t in (1, 2, 3)] + [("C8", 4, 0, "C8k4"), ("C5p", 4, 0, "C5pk4")]
n = 0; counts = {}
with open(os.path.join(D, "prompts_f.jsonl"), "w") as f:
    for fmt in ("F3", "F1"):
        for p in pairs:
            for (cond, k, t, ck), lowidx, order, ans in itertools.product(PLAN, [0, 1], ["HF", "LF"], ["AH", "AL"]):
                u, spans, ll = build_f(p, fmt, cond, k, lowidx, order, ans, t)
                f.write(json.dumps({"id": 5_000_000 + n, "fmt": fmt, "pair_id": p["pair_id"], "split": "test", "cls": p["ent"]["cls"], "cond": cond, "k": k,
                                    "ck": ck, "template": t, "lowidx": lowidx, "order": order, "ans": ans, "letter_low": ll,
                                    "system": NS["SYSTEM"][fmt], "user": u, "spans": spans}) + "\n"); n += 1; counts[f"{fmt}|{ck}"] = counts.get(f"{fmt}|{ck}", 0) + 1
man = {"n_prompts": n, "counts": counts, "byte_identical_rebuilds_checked": n_ok, "C7k1_t0_equals_C1": True,
       "sha256": hashlib.sha256(open(os.path.join(D, "prompts_f.jsonl"), "rb").read()).hexdigest()}
json.dump(man, open(os.path.join(D, "manifest_f.json"), "w"), indent=1); print(json.dumps(man, indent=1)[:600])
