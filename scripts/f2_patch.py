"""SPRINT Block F2 (14 Nov 2026) - patch the FIXED Block A head sets on four clean->corrupt contrasts (F1, test). New file.
Heads: top-10 of the Block A selection ranking and the same 20 random 10-head sets (results of patch_F1_<model>.json); no re-ranking.
Patching: clean head outputs into the corrupt run on the shared token suffix (Block A method). Items: gap > 1 logit in the expected direction.
  python f2_patch.py MODEL OUTDIR [--nmax 160]
"""
import json, os, sys, time, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
MODEL, OUT = sys.argv[1], sys.argv[2]; args = sys.argv[3:]
def arg(n, d): return type(d)(args[args.index(n) + 1]) if n in args else d
HERE = os.path.dirname(os.path.abspath(__file__)); tag = MODEL.split("/")[-1]; dev = "cuda"; t0 = time.time(); os.makedirs(OUT, exist_ok=True); NMAX = arg("--nmax", 160)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
mm = model.model.language_model if hasattr(model.model, "language_model") else model.model
LAYERS = mm.layers; cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; NL = len(LAYERS)
def lid(L):
    s = set()
    for x in (L, " " + L):
        e = tok.encode(x, add_special_tokens=False)
        if len(e) == 1: s.add(e[0])
    return sorted(s)[0]
LID = {"A": lid("A"), "B": lid("B")}
ST = {"S": 0, "mode": None, "set": {}}; CZ = {}
def hook(li):
    def h(mod, inp):
        x = inp[0]; S = ST["S"]
        if ST["mode"] == "cache": CZ[li] = x[0, -S:].clone(); return None
        if ST["mode"] == "patch" and li in ST["set"]:
            x = x.clone()
            for hh in ST["set"][li]: x[0, -S:, hh * HD:(hh + 1) * HD] = CZ[li][:, hh * HD:(hh + 1) * HD]
            return (x,)
    return h
for li, l in enumerate(LAYERS): l.self_attn.o_proj.register_forward_pre_hook(hook(li))
def enc(r):
    t = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True)
    return tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
def LD(ids, r):
    lg = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    lo = LID[r["letter_low"]]; hi = LID["A" if r["letter_low"] == "B" else "B"]; return (lg[lo] - lg[hi]).item()
P = json.load(open(f"runs/A/patch_F1_{tag}.json"))
SETS = {"top10": [tuple(h) for h in P["top20"][:10]]}
SETS.update({k: [tuple(h) for h in v] for k, v in P["hsets"].items() if k.startswith("rand10_")})
assert len([k for k in SETS if k.startswith("rand10_")]) == 20
beh = {}
for l in open(f"runs/A/behaviour_{tag}.jsonl"): x = json.loads(l); beh[x["id"]] = x
rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "prompts_v2.jsonl"))]
rows = [r for r in rows if r["fmt"] == "F1" and r["split"] == "test"]
key = lambda r: (r["pair_id"], r["lowidx"], r["order"], r["ans"]); by = {}
for r in rows: by.setdefault(r["ck"], {})[key(r)] = r
CONTRASTS = [("C1k1", "C2k4", +1), ("C0k1", "C1k1", -1), ("C0k1", "C0rk4", +1), ("C1k1", "C6k4", -1)]
res = {"model": MODEL, "job": os.environ.get("SLURM_JOB_ID"), "sets": {k: [list(h) for h in v] for k, v in SETS.items()}, "contrasts": {}}
for clean, corrupt, sgn in CONTRASTS:
    ks = [k for k in sorted(by[clean]) if k in by[corrupt] and sgn * (beh[by[corrupt][k]["id"]]["ld"] - beh[by[clean][k]["id"]]["ld"]) > 1.0]
    n_elig = len(ks); random.Random(0).shuffle(ks); ks = ks[:NMAX]; items = []
    for n, k in enumerate(ks):
        rc, rk = by[clean][k], by[corrupt][k]; ic, ik = enc(rc), enc(rk)
        a, b = ic[0].tolist(), ik[0].tolist(); S = 0
        while S < min(len(a), len(b)) and a[-1 - S] == b[-1 - S]: S += 1
        ST["S"] = S; ST["mode"] = "cache"; CZ.clear(); ldc = LD(ic, rc); ST["mode"] = None; ldk = LD(ik, rk)
        it = {"pair_id": k[0], "S": S, "ld_clean": ldc, "ld_corrupt": ldk}
        for name, hs in SETS.items():
            m = {}
            for l, h in hs: m.setdefault(l, []).append(h)
            ST["mode"] = "patch"; ST["set"] = m; it[name] = LD(ik, rk); ST["mode"] = None
        items.append(it)
        if n % 40 == 0: print(f"{clean}->{corrupt} {n}/{len(ks)} {time.time()-t0:.0f}s", flush=True)
    res["contrasts"][f"{clean}->{corrupt}"] = {"expected_sign": sgn, "n_eligible": n_elig, "items": items}
    json.dump(res, open(f"{OUT}/f2_patch_{tag}.json", "w"))
    print("done", clean, corrupt, "eligible", n_elig, "used", len(items), f"{time.time()-t0:.0f}s", flush=True)
print("DONE f2", tag, time.time() - t0)
