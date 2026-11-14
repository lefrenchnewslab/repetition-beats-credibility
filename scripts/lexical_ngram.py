"""SPRINT Block A (13 Nov 2026), amendment 2 — model-free lexical baseline for the credibility probe.
TF-IDF word 1-3-grams of the user prompt -> L2 logistic regression, same label ("option A is gov-sourced"), same
splits as cred_all (fit on selection F1 labelled conditions, C chosen on the selection-internal fold, AUC on test).
  python3 lexical_ngram.py data/prompts_v2.jsonl RESULTS_DIR   -> RESULTS_DIR/lexical_ngram.json
"""
import json, sys, os, glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
DATA, RD = sys.argv[1], sys.argv[2]
LAB = ["C1k1", "C2k2", "C2k4", "C2k8", "C3k2", "C3k4", "C7k2", "C7k4"]
R = [r for r in map(json.loads, open(DATA)) if r["fmt"] == "F1" and r["ck"] in LAB + ["C0k1"]]
y = np.array([r["letter_low"] == "B" for r in R], dtype=int)
sel = np.array([r["split"] == "selection" for r in R]); lab = np.array([r["ck"] in LAB for r in R])
sp = sorted({r["pair_id"] for r in R if r["split"] == "selection"}); vp = set(sp[int(.75 * len(sp)):])
val = np.array([r["pair_id"] in vp for r in R])
vec = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=200000, token_pattern=r"[^\s]+")
X = vec.fit_transform([r["user"] for r in R])  # unsupervised vocabulary; labels only from selection
tr, va = sel & lab & ~val, sel & lab & val
best = max([0.01, 0.1, 1, 10], key=lambda C: roc_auc_score(y[va], LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr]).decision_function(X[va])))
clf = LogisticRegression(C=best, max_iter=2000).fit(X[sel & lab], y[sel & lab])
out = {"C": best, "n_features": X.shape[1]}
for ck in ["C1k1", "C2k4", "C2k8", "C0k1"]:
    m = ~sel & np.array([r["ck"] == ck for r in R])
    out[ck] = float(roc_auc_score(y[m], clf.decision_function(X[m]))); out[ck + "_n"] = int(m.sum())
for f in glob.glob(os.path.join(RD, "behaviour_*.jsonl")):
    tag = f.split("behaviour_")[1][:-6]
    if "32B" in tag: continue
    flip = {x["id"] for x in map(json.loads, open(f)) if x["p_low"] > .5}
    for ck in ["C2k4", "C2k8"]:
        m = ~sel & np.array([r["ck"] == ck and r["id"] in flip for r in R])
        out[f"{tag}.{ck}_flipped"] = float(roc_auc_score(y[m], clf.decision_function(X[m]))) if len(set(y[m])) == 2 else None
        out[f"{tag}.{ck}_flipped_n"] = int(m.sum())
json.dump(out, open(os.path.join(RD, "lexical_ngram.json"), "w"), indent=1); print(out)
