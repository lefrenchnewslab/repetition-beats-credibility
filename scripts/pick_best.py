"""SPRINT Block A: pick the model for the F3 head analysis (FM E19): highest F1 held-out top-10 recovery. Prints the HF model id."""
import json, glob, sys
import numpy as np
RD = sys.argv[1]; best = None
for f in glob.glob(f"{RD}/patch_F1_*.json"):
    R = json.load(open(f)); V = R["verify"]
    lc = np.mean([v["ld_clean"] for v in V]); lk = np.mean([v["ld_corrupt"] for v in V]); pv = np.mean([v["top10"] for v in V])
    rec = (lk - pv) / (lk - lc) * 100
    print(f"{R['model']} top10 recovery {rec:.1f}", file=sys.stderr)
    if best is None or rec > best[0]: best = (rec, R["model"])
print(best[1])
