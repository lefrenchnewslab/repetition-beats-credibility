"""SPRINT Block B3 (13 Nov 2026) — training-free fix by scaling repetition heads. New file (encoding/scoring follow run_v2.py).
  python run_b3.py select MODEL OUTDIR --sets F1,F3_C7k4 [--pairs 200]
      grid k in {5,10,20} x gamma in {0,.25,.5,.75} (+ base) on SELECTION; writes b3_select_<tag>.json with the chosen (k, gamma) per set
  python run_b3.py test MODEL OUTDIR --set F1 [--fmts F1,F3,F2] [--cap]
      chosen setting on TEST (streamed per condition), optional capability (MMLU 500, WikiText 200) base + fixed
Head sets: Block A selection rankings in runs/A/patch_<set>_<tag>.json (`order_selection`).
"""
import json, os, sys, time, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
mode, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; args = sys.argv[4:]
def arg(n, d): return type(d)(args[args.index(n) + 1]) if n in args else d
HERE = os.path.dirname(os.path.abspath(__file__)); RA = arg("--runsA", "runs/A")
tag = MODEL.split("/")[-1]; dev = "cuda"; t0 = time.time(); os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
if "--lora" in args:   # Block D: evaluate a trained LoRA adapter with the identical harness (merged; no head scaling)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, arg("--lora", "")).merge_and_unload().eval()
    print("loaded + merged LoRA", arg("--lora", ""), flush=True)
mm = model.model.language_model if hasattr(model.model, "language_model") else model.model
LAYERS = mm.layers; cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
def lid(L):
    s = set()
    for x in (L, " " + L):
        e = tok.encode(x, add_special_tokens=False)
        if len(e) == 1: s.add(e[0])
    return sorted(s)
LID = {c: lid(c) for c in "ABCD"}
# ---- head scaling hook ----
SCALE = {}  # layer -> list of (head, gamma)
def pre(li):
    def h(mod, inp):
        if li not in SCALE: return None
        x = inp[0].clone()
        for hh, g in SCALE[li]: x[..., hh * HD:(hh + 1) * HD] *= g
        return (x,)
    return h
for li, l in enumerate(LAYERS): l.self_attn.o_proj.register_forward_pre_hook(pre(li))
def set_fix(heads, gamma):
    SCALE.clear()
    for l, h in heads: SCALE.setdefault(l, []).append((h, gamma))
def headset(name):
    return [tuple(x) for x in json.load(open(f"{RA}/patch_{name}_{tag}.json"))["order_selection"]]
# ---- prompts ----
def load(paths, keep):
    R = []
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if keep(r): R.append(r)
    return R
def render(r):
    return tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True)
MAXTOK = arg("--maxtok", 48000)
def run_rows(R, maxtok=None, fout=None, tag_extra=None):
    maxtok = maxtok or MAXTOK
    texts = [render(r) for r in R]; lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    order = list(np.argsort(lens)[::-1]); res = [None] * len(R); i = 0
    while i < len(order):
        bs = max(1, maxtok // lens[order[i]]); idx = order[i:i + bs]; i += len(idx)
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        m = enc["attention_mask"].to(dev); pos = (m.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            lg = model(input_ids=enc["input_ids"].to(dev), attention_mask=m, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        lp = torch.log_softmax(lg, -1); pa = lp[:, LID["A"]].exp().sum(-1); pb = lp[:, LID["B"]].exp().sum(-1)
        for b, j in enumerate(idx):
            r = R[j]; pl = (pb[b] if r["letter_low"] == "B" else pa[b]).item(); ms = (pa[b] + pb[b]).item()
            res[j] = {"id": r["id"], "fmt": r["fmt"], "pair_id": r["pair_id"], "ck": r["ck"], "order": r["order"], "ans": r["ans"],
                      "p_low": pl / ms, "mass": ms, **(tag_extra or {})}
            if fout: fout.write(json.dumps(res[j]) + "\n")
        if fout: fout.flush()
    return res
def pmean(res, fmt, ck):
    by = {}
    for x in res:
        if x["fmt"] == fmt and x["ck"] == ck: by.setdefault(x["pair_id"], []).append(x["p_low"])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None
GRID = [(k, g) for k in (5, 10, 20) for g in (0.0, 0.25, 0.5, 0.75)]
DATA = os.path.join(HERE, "data", "prompts_v2.jsonl"); X7 = os.path.join(HERE, "data", "prompts_f3selC7.jsonl")

# =========================================== SELECT ===========================================
if mode == "select":
    npairs = arg("--pairs", 200); sets = arg("--sets", "F1,F3_C7k4").split(",")
    sel_p = sorted({json.loads(l)["pair_id"] for l in open(DATA) if '"selection"' in l})[:npairs]; sp = set(sel_p)
    R1 = load([DATA], lambda r: r["fmt"] == "F1" and r["split"] == "selection" and r["pair_id"] in sp and r["ck"] in ("C0k1", "C1k1", "C2k4"))
    R3 = load([DATA, X7], lambda r: r["fmt"] == "F3" and r["split"] == "selection" and r["pair_id"] in sp and r["ck"] in ("C1k1", "C7k4"))
    out = {"model": MODEL, "pairs": len(sel_p), "grid": GRID, "sets": {}, "job": os.environ.get("SLURM_JOB_ID")}
    prev_f = f"{OUT}/b3_select_{tag}.json"
    if os.path.exists(prev_f):   # merge with an earlier selection run (B3a2 appends Feeders to B3a's file)
        old = json.load(open(prev_f)); out["sets"] = old.get("sets", {}); out["jobs_before"] = [old.get("job")] + old.get("jobs_before", [])
    def metrics(rf1, rf3, base):
        d = {"P": {f"F1_{c}": pmean(rf1, "F1", c) for c in ("C0k1", "C1k1", "C2k4")}}
        if rf3: d["P"].update({f"F3_{c}": pmean(rf3, "F3", c) for c in ("C1k1", "C7k4")})
        if base:
            B = base["P"]; P = d["P"]
            d["SP_F1"] = (P["F1_C0k1"] - P["F1_C1k1"]) / (B["F1_C0k1"] - B["F1_C1k1"])
            d["BR_F1_C2k4"] = 1 - (P["F1_C2k4"] - P["F1_C1k1"]) / (B["F1_C2k4"] - B["F1_C1k1"])
            if rf3: d["BR_F3_C7k4"] = 1 - (P["F3_C7k4"] - P["F3_C1k1"]) / (B["F3_C7k4"] - B["F3_C1k1"])
        return d
    SCALE.clear()
    base = metrics(run_rows(R1), run_rows(R3), None); out["base"] = base
    print("base", base, f"{time.time()-t0:.0f}s", flush=True)
    def heads_for(s, k):  # amendment (B3-U): union of the two rankings' top-k; B3-Feeders: fixed F3/C7-only top-10 heads
        if s == "Feeders":
            hf1 = headset("F1")[:10]; return [h for h in headset("F3_C7k4")[:10] if h not in hf1]
        if s != "U": return headset(s)[:k]
        u = []
        for h in headset("F1")[:k] + headset("F3_C7k4")[:k]:
            if h not in u: u.append(h)
        return u
    for s in sets:
        obj = "BR_F1_C2k4" if s == "F1" else ("min_BR" if s == "U" else "BR_F3_C7k4"); rows = {}
        for k, g in (GRID if s != "Feeders" else [(10, g) for g in (0.0, 0.25, 0.5, 0.75)]):
            set_fix(heads_for(s, k), g)
            d = metrics(run_rows(R1), run_rows(R3) if s != "F1" else [], base)
            if s == "U": d["min_BR"] = min(d["BR_F1_C2k4"], d["BR_F3_C7k4"])
            rows[f"k{k}_g{g}"] = d
            print(s, k, g, {kk: round(v, 3) for kk, v in d.items() if kk != "P"}, f"{time.time()-t0:.0f}s", flush=True)
        SCALE.clear()
        ok = [(kg, d) for kg, d in rows.items() if d["SP_F1"] >= 0.80]
        if ok:   # max bias reduction; ties -> larger gamma, then smaller k
            best = max(ok, key=lambda t: (round(t[1][obj], 3), float(t[0].split("_g")[1]), -int(t[0][1:].split("_")[0])))
        else:
            best = max(rows.items(), key=lambda t: t[1]["SP_F1"])
        k, g = int(best[0][1:].split("_")[0]), float(best[0].split("_g")[1])
        out["sets"][s] = {"objective": obj, "rows": rows, "chosen": {"k": k, "gamma": g, "heads": heads_for(s, k), "constraint_met": bool(ok)}}
        print("CHOSEN", s, k, g, "constraint met", bool(ok), flush=True)
        json.dump(out, open(f"{OUT}/b3_select_{tag}.json", "w"), indent=1)   # incremental: a timeout keeps finished sets
    out["seconds"] = time.time() - t0
    json.dump(out, open(f"{OUT}/b3_select_{tag}.json", "w"), indent=1)
    print("DONE select", time.time() - t0)

# =========================================== CURVE (B3e) ===========================================
if mode == "curve":
    sets = arg("--sets", "U,F3_C7k4,Feeders").split(","); SELJ = json.load(open(f"{OUT}/b3_select_{tag}.json"))["sets"]
    R = load([DATA], lambda r: r["split"] == "test" and r["fmt"] == "F3" and r["ck"] in ("C0k1", "C1k1", "C7k4"))
    def pts(res): return {c: pmean(res, "F3", c) for c in ("C0k1", "C1k1", "C7k4")}
    out = {"model": MODEL, "job": os.environ.get("SLURM_JOB_ID"), "curves": {}}
    SCALE.clear(); out["base"] = pts(run_rows(R)); print("base", out["base"], flush=True)
    B = out["base"]
    for s in sets:
        heads = [tuple(h) for h in SELJ[s]["chosen"]["heads"]]; cur = {}
        for g in (0.0, 0.25, 0.5, 0.75):
            set_fix(heads, g); P = pts(run_rows(R))
            cur[g] = {"P": P, "BR_C7k4": 1 - (P["C7k4"] - P["C1k1"]) / (B["C7k4"] - B["C1k1"]),
                      "SP_F3": (P["C0k1"] - P["C1k1"]) / (B["C0k1"] - B["C1k1"])}
            print("curve", s, g, {k: round(v, 3) for k, v in cur[g].items() if k != "P"}, f"{time.time()-t0:.0f}s", flush=True)
        SCALE.clear(); out["curves"][s] = {"k": SELJ[s]["chosen"]["k"], "heads": heads, "points": cur}
        json.dump(out, open(f"{OUT}/b3_curve_{tag}.json", "w"), indent=1)
    print("DONE curve", time.time() - t0)

# =========================================== TEST ===========================================
if mode == "test":
    s = arg("--set", "F1"); fmts = arg("--fmts", "F1,F3,F2").split(",")
    if s == "LORA": C = {"k": 0, "gamma": 1.0, "heads": [], "adapter": arg("--lora", "")}
    else: C = json.load(open(f"{OUT}/b3_select_{tag}.json"))["sets"][s]["chosen"]
    heads = [tuple(h) for h in C["heads"]]
    PRI = {"F1": ["C0k1", "C1k1", "C2k4", "C3k4", "C7k4", "C2k2", "C2k8", "C5k4"], "F3": ["C0k1", "C1k1", "C7k4", "C3k4", "C2k4", "C2k2", "C5k4"],
           "F2": ["C0k1", "C1k1", "C7k4", "C3k4", "C2k4", "C2k2", "C5k4"]}
    if s == "LORA": PRI["F1"] += ["C6k4", "C6k2", "C0rk4"]   # held-out side-effect conditions (FM G3)
    allR = load([DATA], lambda r: r["split"] == "test" and r["fmt"] in fmts)
    if arg("--maxpairs", 0):   # smoke tests only
        keep = sorted({r["pair_id"] for r in allR})[:arg("--maxpairs", 0)]; allR = [r for r in allR if r["pair_id"] in set(keep)]
    set_fix(heads, C["gamma"])
    fout = open(f"{OUT}/b3_test_{s}_{tag}.jsonl", "w")
    for f in fmts:
        for ck in PRI[f]:
            R = [r for r in allR if r["fmt"] == f and r["ck"] == ck]
            run_rows(R, fout=fout, tag_extra={"set": s, "k": C["k"], "gamma": C["gamma"]})
            print("done", f, ck, len(R), f"{time.time()-t0:.0f}s", flush=True)
    fout.close()
    if "--cap" in args:   # capability: base and fixed
        from datasets import load_from_disk
        mm_ = load_from_disk(arg("--mmlu", "data_cap/mmlu_test")); idx = sorted(random.Random(0).sample(range(len(mm_)), 500))
        wk = load_from_disk(arg("--wiki", "data_cap/wikitext103_test"))
        paras = [t["text"].strip() for t in wk if len(t["text"].strip()) > 0 and not t["text"].strip().startswith("=")]
        paras = [p for p in paras if len(tok(p, add_special_tokens=False)["input_ids"]) >= 50][:200]
        def mmlu():
            correct = []
            for i in idx:
                q = mm_[i]; u = q["question"] + "\n" + "\n".join(f"{L}. {c}" for L, c in zip("ABCD", q["choices"])) + "\nAnswer:"
                t = tok.apply_chat_template([{"role": "system", "content": "Answer the multiple choice question with the letter only."}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
                e = tok(t, return_tensors="pt", add_special_tokens=False).to(dev)
                with torch.no_grad(): lg = model(**e, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                sc = [max(lg[j].item() for j in LID[L]) for L in "ABCD"]; correct.append(int(int(np.argmax(sc)) == q["answer"]))
            return correct
        def ppl():
            nll = []
            for p in paras:
                e = tok(p, return_tensors="pt", truncation=True, max_length=256).to(dev)
                with torch.no_grad(): o = model(**e, labels=e["input_ids"], use_cache=False)
                nll.append(o.loss.item())
            return nll
        cap = {}
        if s == "LORA":   # base = B3's measurement on identical items (same base model); the loaded model here is the LoRA
            bc = json.load(open(f"{OUT}/b3_cap_F1_{tag}.json")); assert bc["mmlu_idx"] == idx; cap["base"] = {"mmlu": bc["base"]["mmlu"], "nll": bc["base"]["nll"]}
        else:
            SCALE.clear(); cap["base"] = {"mmlu": mmlu(), "nll": ppl()}
        set_fix(heads, C["gamma"]); cap["fixed"] = {"mmlu": mmlu(), "nll": ppl()}; SCALE.clear()
        for k_ in ("base", "fixed"): print("cap", k_, "MMLU acc", np.mean(cap[k_]["mmlu"]), "ppl", float(np.exp(np.mean(cap[k_]["nll"]))), flush=True)
        json.dump({"set": s, "chosen": C, "mmlu_idx": idx, **cap}, open(f"{OUT}/b3_cap_{s}_{tag}.json", "w"))
    print("DONE test", s, time.time() - t0)
