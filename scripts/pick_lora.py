"""SPRINT Block D: pick (lr, checkpoint) by validation loss on held-out TRAINING entities (FM G2). Prints the adapter path."""
import json, glob, sys
best = None
for f in glob.glob(f"{sys.argv[1]}/lr*/train_log.json"):
    L = json.load(open(f)); d = f.rsplit("/", 1)[0]
    for ck, v in L["checkpoints"].items():
        if ck == "base": continue
        print(d, ck, round(v["loss"], 5), file=sys.stderr)
        if best is None or v["loss"] < best[0]: best = (v["loss"], f"{d}/ckpt_{ck}")
print(best[1])
