"""SPRINT Block D (14 Nov 2026) — LoRA training data: NEW fictional entities, disjoint from v2 (selection+test) and the pilot. New file.
Reuses build_data_v2's generators (run in a throw-away namespace so its own outputs are untouched), then switches to a new seed
and new place pools. Writes data/train_d.jsonl, data/val_d.jsonl, data/manifest_d.json.
  python3 scripts/build_data_d.py
"""
import json, os, sys, random, runpy, tempfile, itertools, hashlib, re
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "data")
tmp = tempfile.mkdtemp()
sys.argv = ["build_data_v2.py", tmp]
NS = runpy.run_path(os.path.join(HERE, "build_data_v2.py"), run_name="v2")   # rebuilds v2 into tmp (deterministic), fills NS["used"]
v2pairs = NS["pairs"]; used_before = set(NS["used"])
NS["R"].seed(20261114)                     # new seed for everything generated below
word, entity, gov, user, build = NS["word"], NS["entity"], NS["gov"], NS["user"], NS["build"]
NS["POOLS"]["train"] = {"towns": [word(3) for _ in range(20)], "gov": [word(3) for _ in range(20)]}
R = NS["R"]; CONFLICT, fields, cap = NS["CONFLICT"], NS["fields"], NS["cap"]
pairs = []
for j in range(300):
    cls = ["person", "company", "town"][j % 3]
    ent = entity(cls, "train"); cf = R.choice(CONFLICT[cls]); gen = fields(cls, "train")[cf]
    v1 = ent["attrs"][cf]; v2 = gen()
    while v2 == v1: v2 = gen()
    pairs.append({"pair_id": 10_000 + j, "split": "train_d" if j < 250 else "val_d", "ent": ent, "cfield": cf, "values": [cap(v1), cap(v2)],
                  "gov": gov(cls, "train"), "users": [user() for _ in range(8)], "fillers": [entity(cls, "train") for _ in range(3)]})
# ---- G1: overlap checks against v2 (selection + test) and the pilot ----
WR = re.compile(r"[A-Z][a-z]+")
words = lambda p: set(WR.findall(json.dumps([p["ent"], p["fillers"], p["gov"]])))
names = lambda P: {p["ent"]["name"] for p in P} | {f["name"] for p in P for f in p["fillers"]}
generic = set(WR.findall(json.dumps(NS["GOV"]))) | {"Holdings", "Works", "Group", "Industries", "Of"} | {w.capitalize() for w in NS["OCC"] + NS["IND"]}
new_words = set().union(*(words(p) for p in pairs)) - generic
old_words = (set().union(*(words(p) for p in v2pairs)) | NS["pilot_words"]) - generic
assert not (names(pairs) & (names(v2pairs) | NS["pilot_names"])), "entity-name overlap"
assert not ({u for p in pairs for u in p["users"]} & ({u for p in v2pairs for u in p["users"]} | NS["pilot_users"])), "username overlap"
assert not (new_words & old_words), f"word overlap: {sorted(new_words & old_words)[:10]}"
# ---- items: 6 per entity from {C2,C7} x {k2,k4} x {F1,F3} x 8 variants, each with its matched C1 prompt ----
rs = random.Random(7); combos = list(itertools.product([("C2", 2), ("C2", 4), ("C7", 2), ("C7", 4)], ["F1", "F3"], [0, 1], ["HF", "LF"], ["AH", "AL"]))
out = {"train_d": [], "val_d": []}
for p in pairs:
    for (cond, k), fmt, lowidx, order, ans in rs.sample(combos, 6):
        u_rep, _, ll = build(p, fmt, cond, k, lowidx, order, ans); u_c1, _, ll1 = build(p, fmt, "C1", 1, lowidx, order, ans)
        assert ll == ll1
        out[p["split"]].append({"pair_id": p["pair_id"], "fmt": fmt, "cond": cond, "k": k, "lowidx": lowidx, "order": order, "ans": ans,
                                "letter_low": ll, "system": NS["SYSTEM"][fmt], "user_c1": u_c1, "user_rep": u_rep})
for s, rows in out.items():
    with open(os.path.join(D, f"{s}.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
man = {"seed": 20261114, "entities": {"train_d": 250, "val_d": 50}, "items": {s: len(r) for s, r in out.items()},
       "overlap_with_v2_and_pilot": {"entity_names": 0, "usernames": 0, "words": 0},
       "sha256": {s: hashlib.sha256(open(os.path.join(D, f"{s}.jsonl"), "rb").read()).hexdigest() for s in out}}
json.dump(man, open(os.path.join(D, "manifest_d.json"), "w"), indent=1)
# v2 must be unchanged by this run (G1 / provenance)
v2sha = hashlib.sha256(open(os.path.join(tmp, "prompts_v2.jsonl"), "rb").read()).hexdigest()
assert v2sha == json.load(open(os.path.join(D, "manifest_v2.json")))["prompts_sha256"], "v2 regeneration differs"
print("train/val items", man["items"], "; v2 reproduced exactly:", v2sha[:12])
