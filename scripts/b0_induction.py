"""SPRINT Block B0 (13 Nov 2026) — induction-head check for the Block A repetition heads. New file.
Standard recipe (Olsson et al. 2022): BOS + T random tokens repeated twice. For every head:
  prefix-matching score = mean attention from each second-copy token to the token AFTER its previous occurrence;
  previous-token score  = mean attention to the immediately preceding position (reported as a control, FM F2);
  copying score         = direct logit attribution of the head's output at second-copy positions to the correct next token,
                          minus its mean over the vocabulary, through the final norm's per-position scale (FM F3).
All heads are scored; the top-10 Block A heads (F1/C2 selection ranking; for Qwen-7B also F3/C2 and F3/C7) and 20 random heads
are extracted. Output: RUNDIR/b0_induction_<tag>.json
  python b0_induction.py MODEL RUNDIR --patchdir runs/A [--T 64 --n 48]
"""
import json, os, sys, time, random, glob
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, OUT = sys.argv[1], sys.argv[2]; args = sys.argv[3:]
def arg(n, d): return type(d)(args[args.index(n) + 1]) if n in args else d
PD = arg("--patchdir", "runs/A"); T = arg("--T", 64); N = arg("--n", 48)
tag = MODEL.split("/")[-1]; dev = "cuda"; t0 = time.time(); os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)  # job 165344 OOM'd: weights require grad by default, so the DLA loop built an autograd graph
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
m = model.model.language_model if hasattr(model.model, "language_model") else model.model
LAYERS = m.layers; cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; NL = len(LAYERS)
WU = model.get_output_embeddings().weight.float()            # [V, d]
WUmean = WU.mean(0)
norm = m.norm
with torch.no_grad(): w_eff = norm(torch.ones(1, 1, cfg.hidden_size, device=dev, dtype=torch.bfloat16)).float()[0, 0]  # RMSNorm effective weight
rng = np.random.default_rng(0)
lo, hi = 1000, min(tok.vocab_size, 30000)
seqs = []
for _ in range(N):
    r = rng.integers(lo, hi, T).tolist()
    seqs.append([tok.bos_token_id if tok.bos_token_id is not None else r[0]] + r + r)
ids = torch.tensor(seqs, device=dev)                       # [N, 1+2T]
z = {}
hooks = [l.self_attn.o_proj.register_forward_pre_hook(lambda mod, inp, li=li: z.__setitem__(li, inp[0].detach())) for li, l in enumerate(LAYERS)]
prefix = torch.zeros(NL, NH); prev = torch.zeros(NL, NH); copy = torch.zeros(NL, NH)
B = 8
for s in range(0, N, B):
    x = ids[s:s + B]
    with torch.no_grad():
        out = model(input_ids=x, output_attentions=True, output_hidden_states=True, use_cache=False)
    P = torch.arange(T + 1, 2 * T + 1, device=dev)           # second-copy positions (BOS at 0)
    tgt = P - T + 1                                          # token after the previous occurrence
    for li in range(NL):
        A = out.attentions[li].float()                       # [b, NH, L, L]
        prefix[li] += A[:, :, P, tgt].mean((0, 2)).cpu() * x.shape[0]
        prev[li] += A[:, :, P, P - 1].mean((0, 2)).cpu() * x.shape[0]
    # copying via DLA at positions P[:-1] (the next token is known)
    xf = out.hidden_states[-1].float()                       # final residual before the norm
    Pc = P[:-1]; nxt = x[:, Pc + 1]                          # [b, T-1]
    rms = xf[:, Pc].pow(2).mean(-1, keepdim=True).add(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6))).rsqrt()  # [b, T-1, 1]
    for li in range(NL):
        Wo = LAYERS[li].self_attn.o_proj.weight.float()      # [d, NH*HD]
        zz = z[li][:, Pc].float().view(x.shape[0], T - 1, NH, HD)
        for h in range(NH):
            v = zz[:, :, h] @ Wo[:, h * HD:(h + 1) * HD].T   # [b, T-1, d]
            v = v * rms * w_eff
            lt = (v * WU[nxt]).sum(-1)                       # logit on the correct next token
            lm = v @ WUmean                                  # mean logit over the vocabulary
            copy[li, h] += (lt - lm).mean().cpu() * x.shape[0]
    del out; torch.cuda.empty_cache()
for hk in hooks: hk.remove()
prefix /= N; prev /= N; copy /= N
# Block A head sets
sets = {}
for f in sorted(glob.glob(f"{PD}/patch_*_{tag}.json")):
    R = json.load(open(f)); name = os.path.basename(f)[len("patch_"):-len(f"_{tag}.json")]
    sets[name] = [tuple(h) for h in R["top20"][:10]]
allh = [(l, h) for l in range(NL) for h in range(NH)]
sets["random20"] = random.Random(7).sample(allh, 20)
def pct(M, l, h): return float((M.flatten() < M[l, h]).float().mean())   # percentile among all heads
res = {"model": MODEL, "T": T, "N": N, "layers": NL, "heads": NH, "seconds": time.time() - t0, "job": os.environ.get("SLURM_JOB_ID"),
       "all": {"prefix": prefix.tolist(), "prev": prev.tolist(), "copy": copy.tolist()},
       "copy_p90_all": float(np.percentile(copy.numpy(), 90)), "sets": {}}
for name, hs in sets.items():
    res["sets"][name] = [{"head": [l, h], "prefix": float(prefix[l, h]), "prev": float(prev[l, h]), "copy": float(copy[l, h]),
                          "prefix_pct": pct(prefix, l, h), "copy_pct": pct(copy, l, h),
                          "induction": bool(prefix[l, h] >= .3 and copy[l, h] > res["copy_p90_all"])} for l, h in hs]
# null for the top-10 mean prefix score: 1000 random 10-head sets
flat = prefix.flatten().numpy(); rs = np.random.default_rng(1)
res["null_mean_prefix_10"] = {"p95": float(np.percentile([flat[rs.choice(len(flat), 10, replace=False)].mean() for _ in range(1000)], 95))}
json.dump(res, open(f"{OUT}/b0_induction_{tag}.json", "w"))
for name, L in res["sets"].items():
    print(name, "n_induction", sum(x["induction"] for x in L), "mean prefix", round(np.mean([x["prefix"] for x in L]), 3), flush=True)
print("DONE b0", tag, time.time() - t0)
