"""SPRINT Block A (13 Nov 2026) — data v2. PILOT-DERIVED from pilot_illusory/scripts/build_data.py (11 Nov), modified in sprint.
Changes vs pilot: new seed; 600 pairs split BY ENTITY into 200 selection / 400 test; pilot words excluded (FM E1);
disjoint place pools per split (E2); new condition C7 (paraphrased copies, same social source, deterministic hand-written
templates, no LLM); three formats F1 table (pilot/Schuster), F2 news excerpt, F3 retrieved documents; matched attribution
frames in F2/F3 (E7); manifest with counts, overlap checks and copy-similarity (E6).
Usage: python3 build_data_v2.py OUTDIR [PILOT_PAIRS_JSONL]   (stdlib only)
"""
import json, random, os, sys, itertools, re, hashlib, collections

SEED = 20261113
R = random.Random(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
PILOT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "data", "pilot_pairs.jsonl")   # release: pilot pairs (word exclusion only)
os.makedirs(OUT, exist_ok=True)

SYSTEM = {
    "F1": "The following are multiple choice questions. You are provided tabular data from sources. "
          "Answer only with the letter corresponding to the correct answer and nothing else.",
    "F2": "The following are multiple choice questions. You are provided news excerpts from sources. "
          "Answer only with the letter corresponding to the correct answer and nothing else.",
    "F3": "The following are multiple choice questions. You are provided retrieved documents from sources. "
          "Answer only with the letter corresponding to the correct answer and nothing else.",
}

# ---------------- E1: exclude every pilot word ----------------
pilot_pairs = [json.loads(l) for l in open(PILOT)]
WORDRE = re.compile(r"[A-Z][a-z]+")
def words_in(obj):
    s = json.dumps(obj); return set(WORDRE.findall(s))
pilot_words = set()
for p in pilot_pairs:
    pilot_words |= words_in(p["ent"]) | words_in(p["fillers"]) | words_in(p["gov"])
pilot_names = {p["ent"]["name"] for p in pilot_pairs} | {f["name"] for p in pilot_pairs for f in p["fillers"]}
pilot_users = {u for p in pilot_pairs for u in p["users"]}

ON = ["dr", "v", "k", "t", "m", "br", "s", "l", "h", "q", "z", "f", "gr", "n", "p", "th"]
VW = ["a", "e", "i", "o", "u", "ae", "ei", "y"]
CO = ["l", "n", "r", "th", "v", "sk", "nd", "rk", "s", "m", "x", "lt"]
used = set(pilot_words)
def word(nsyl):
    while True:
        w = "".join(R.choice(ON) + R.choice(VW) + (R.choice(CO) if i == nsyl - 1 or R.random() < .3 else "") for i in range(nsyl))
        w = w.capitalize()
        if w not in used and 4 <= len(w) <= 10:
            used.add(w); return w

SPLITS = {"selection": 200, "test": 400}
POOLS = {s: {"towns": [word(3) for _ in range(20)], "gov": [word(3) for _ in range(20)]} for s in SPLITS}  # E2 disjoint

ADJ = ["Bright", "Silent", "Rapid", "Lucky", "Cosmic", "Velvet", "Hidden", "Golden", "Frosty", "Clever", "Lazy", "Wild", "Brave", "Misty", "Sunny", "Rusty"]
NOUN = ["Falcon", "Otter", "Pebble", "Comet", "Willow", "Badger", "Lantern", "Cactus", "Harbor", "Maple", "Raven", "Biscuit", "Thistle", "Anchor", "Fox", "Meadow"]
used_users = set(pilot_users)
def user():
    while True:
        u = f"User @{R.choice(ADJ)}{R.choice(NOUN)}{R.randint(100, 9999)}"
        if u not in used_users: used_users.add(u); return u

GOV = {"person": ["Civil Registry Office of {P}", "Ministry of Interior of {P}", "Population Records Office of {P}"],
       "company": ["Companies Registry of {P}", "Ministry of Commerce of {P}", "Trade Register Office of {P}"],
       "town": ["National Statistics Office of {P}", "Regional Government of {P}", "Census Bureau of {P}"]}
def gov(cls, split):
    return R.choice(GOV[cls]).replace("{P}", R.choice(POOLS[split]["gov"])).title()

OCC = ["architect", "pharmacist", "carpenter", "journalist", "veterinarian", "accountant", "librarian", "electrician", "translator", "geologist", "baker", "nurse"]
IND = ["textiles", "shipping", "software", "brewing", "mining", "furniture", "insurance", "ceramics", "dairy", "printing"]
def fields(cls, split):
    T = POOLS[split]["towns"]
    if cls == "person":
        return {"birth year": lambda: str(R.randint(1950, 1995)), "birthplace": lambda: R.choice(T),
                "occupation": lambda: R.choice(OCC), "height": lambda: f"{R.randint(155, 195)} cm",
                "number of children": lambda: str(R.randint(0, 5))}
    if cls == "company":
        return {"founding year": lambda: str(R.randint(1890, 2010)), "headquarters": lambda: R.choice(T),
                "industry": lambda: R.choice(IND), "number of employees": lambda: f"{R.randint(20, 9000):,}",
                "chief executive": lambda: f"{word(2)} {word(2)}"}
    return {"population": lambda: f"{R.randint(3000, 95000):,}", "founding year": lambda: str(R.randint(1100, 1900)),
            "main industry": lambda: R.choice(IND), "mayor": lambda: f"{word(2)} {word(2)}",
            "elevation": lambda: f"{R.randint(5, 1400)} m"}
CONFLICT = {"person": ["birth year", "birthplace", "occupation", "height"],
            "company": ["founding year", "headquarters", "industry", "number of employees"],
            "town": ["population", "founding year", "main industry", "mayor"]}

# ---- C7 paraphrase material (hand-written, deterministic; variant 0 = canonical) ----
LAB = {  # F1 table field labels
    ("person", "birth year"): ["Birth Year", "Year Of Birth", "Year Born", "Born (Year)"],
    ("person", "birthplace"): ["Birthplace", "Place Of Birth", "Native Town", "Born (Place)"],
    ("person", "occupation"): ["Occupation", "Profession", "Job", "Line Of Work"],
    ("person", "height"): ["Height", "Stature", "Body Height", "Measured Height"],
    ("person", "number of children"): ["Number Of Children", "Children", "Kids", "Offspring Count"],
    ("company", "founding year"): ["Founding Year", "Year Founded", "Established", "Founded In"],
    ("company", "headquarters"): ["Headquarters", "Head Office", "Main Office Location", "HQ"],
    ("company", "industry"): ["Industry", "Sector", "Business Area", "Line Of Business"],
    ("company", "number of employees"): ["Number Of Employees", "Employees", "Staff Count", "Workforce"],
    ("company", "chief executive"): ["Chief Executive", "CEO", "Head Of Company", "Managing Director"],
    ("town", "population"): ["Population", "Inhabitants", "Number Of Residents", "Resident Count"],
    ("town", "founding year"): ["Founding Year", "Year Founded", "Established", "Founded In"],
    ("town", "main industry"): ["Main Industry", "Primary Industry", "Leading Sector", "Key Industry"],
    ("town", "mayor"): ["Mayor", "Current Mayor", "Head Of Council", "Town Leader"],
    ("town", "elevation"): ["Elevation", "Altitude", "Height Above Sea Level", "Elevation Above Sea"],
}
NAMELAB = ["Name", "Entity", "Subject", "Name Of Record"]
HEAD = [("Field", "Value"), ("Attribute", "Entry"), ("Property", "Detail"), ("Item", "Information")]
SENT = {  # F2/F3 sentence verbalisations, one {v}
    ("person", "birth year"): ["{n} was born in {v}.", "{n}'s year of birth is {v}.", "{n} came into the world in {v}.", "The birth year of {n} is {v}."],
    ("person", "birthplace"): ["{n} was born in {v}.", "{n}'s birthplace is {v}.", "{n} is a native of {v}.", "The place of birth of {n} is {v}."],
    ("person", "occupation"): ["{n}'s occupation is {v}.", "{n} is employed in the profession of {v}.", "By profession, {n} is listed as {v}.", "The occupation of {n} is recorded as {v}."],
    ("person", "height"): ["{n}'s height is {v}.", "{n} stands {v} tall.", "{n} measures {v} in height.", "The height of {n} is {v}."],
    ("person", "number of children"): ["{n}'s number of children is {v}.", "{n} has {v} children.", "The number of children {n} has is {v}.", "{n} is a parent of {v} children."],
    ("company", "founding year"): ["{n} was founded in {v}.", "{n}'s founding year is {v}.", "{n} was established in {v}.", "The company {n} dates back to {v}."],
    ("company", "headquarters"): ["{n} is headquartered in {v}.", "{n}'s headquarters are in {v}.", "The head office of {n} is located in {v}.", "{n} runs its main office from {v}."],
    ("company", "industry"): ["{n}'s industry is {v}.", "{n} operates in the {v} sector.", "The business sector of {n} is {v}.", "{n} is active in {v}."],
    ("company", "number of employees"): ["{n}'s number of employees is {v}.", "{n} has {v} employees.", "The workforce of {n} numbers {v}.", "{n} employs {v} people."],
    ("company", "chief executive"): ["{n}'s chief executive is {v}.", "{n} is led by chief executive {v}.", "The CEO of {n} is {v}.", "{v} serves as chief executive of {n}."],
    ("town", "population"): ["{n}'s population is {v}.", "{n} has a population of {v}.", "The number of residents of {n} is {v}.", "{n} is home to {v} inhabitants."],
    ("town", "founding year"): ["{n} was founded in {v}.", "{n}'s founding year is {v}.", "{n} was established in {v}.", "The town of {n} dates back to {v}."],
    ("town", "main industry"): ["{n}'s main industry is {v}.", "The leading sector in {n} is {v}.", "The economy of {n} relies mainly on {v}.", "{v} is the primary industry of {n}."],
    ("town", "mayor"): ["{n}'s mayor is {v}.", "{n} is governed by mayor {v}.", "The current mayor of {n} is {v}.", "{v} serves as mayor of {n}."],
    ("town", "elevation"): ["{n}'s elevation is {v}.", "{n} lies at an elevation of {v}.", "The altitude of {n} is {v}.", "{n} sits {v} above sea level."],
}
FRAME2 = ["According to {s}, ", "In a post, {s} wrote that ", "{s} stated that ", "As reported by {s}, "]  # F2 frames; [0] used for BOTH sources (E7)

def entity(cls, split):
    if cls == "person": name = f"{word(2)} {word(3)}"
    elif cls == "company": name = f"{word(3)} {R.choice(['Holdings', 'Works', 'Group', 'Industries'])}"
    else: name = word(3)
    return {"cls": cls, "name": name, "attrs": {f: g() for f, g in fields(cls, split).items()}}
def cap(v): return v[0].upper() + v[1:]

# ---------------- pairs ----------------
pairs = []
pid = 0
for split, n in SPLITS.items():
    for j in range(n):
        cls = ["person", "company", "town"][pid % 3]
        ent = entity(cls, split)
        cf = R.choice(CONFLICT[cls])
        gen = fields(cls, split)[cf]
        v1 = ent["attrs"][cf]; v2 = gen()
        while v2 == v1: v2 = gen()
        fillers = [entity(cls, split) for _ in range(3)]
        pairs.append({"pair_id": pid, "split": split, "ent": ent, "cfield": cf, "values": [cap(v1), cap(v2)],
                      "gov": gov(cls, split), "users": [user() for _ in range(8)], "fillers": fillers})
        pid += 1

# ---------------- block renderers: return text, [(kind, start, end)] relative ----------------
def rows_of(e, cf, val):
    return [(f, cap(val if f == cf else v)) for f, v in e["attrs"].items()]

def block_F1(i, role, lab, val, e, cf, var, kw):
    sp = []
    s = f"Table {chr(65 + i)}"
    if lab is None: s += ":\n"
    else:
        s += f" ({kw}: "; sp.append(("label", len(s), len(s) + len(lab))); s += lab + "):\n"
    h1, h2 = HEAD[var]
    s += f"| {h1} | {h2} |\n|{'-' * (len(h1) + 2)}|{'-' * (len(h2) + 2)}|\n"
    rows = rows_of(e, cf if role != "F" else None, val)
    if var:  # deterministic row permutation for paraphrased copies
        rows = rows[var:] + rows[:var]
    s += f"| {NAMELAB[var]} | {e['name']} |"
    for f, v in rows:
        s += f"\n| {LAB[(e['cls'], f)][var]} | "
        if role != "F" and f == cf: sp.append(("value", len(s), len(s) + len(v)))
        s += v + " |"
    return s, sp

def sentences(e, cf, val, var, role):
    """conflict sentence + 2 context sentences; returns text, value span (rel) or None"""
    others = [f for f in e["attrs"] if f != cf][:2]
    order = [cf] + others if role != "F" else list(e["attrs"])[:3]
    if var: order = order[var % 3:] + order[:var % 3]
    s, vsp = "", None
    for f in order:
        v = cap(val if (role != "F" and f == cf) else e["attrs"][f])
        t = SENT[(e["cls"], f)][var].replace("{n}", e["name"])
        a = t.index("{v}")
        if s: s += " "
        if role != "F" and f == cf: vsp = (len(s) + a, len(s) + a + len(v))
        s += t.replace("{v}", v)
    return s, vsp

def lower_first(t):
    return t[0].lower() + t[1:] if t.split()[0] in ("The", "By") else t

def block_F2(i, role, lab, val, e, cf, var, kw):
    sp = []
    s = f"Excerpt {i + 1}. "
    body, vsp = sentences(e, cf, val, var, role)
    if lab is not None:
        art, core = ("the ", lab) if role == "H" else ("", "social media " + lab.replace("User ", "user "))
        fr = FRAME2[var]; a = fr.index("{s}"); pre = fr[:a]
        if pre == "":  # frame starts with the source: capitalise its first character
            if art: art = "The "
            else: core = cap(core)
        s += pre + art
        sp.append(("label", len(s), len(s) + len(core)))  # label span = source string without the article
        s += core + fr[a + 3:]
        off = len(s); body = lower_first(body)
    else:
        off = len(s)
    if vsp: sp.append(("value", off + vsp[0], off + vsp[1]))
    s += body
    return s, sp

def block_F3(i, role, lab, val, e, cf, var, kw):
    sp = []
    s = f"[Document {i + 1}]\n"
    if lab is not None:
        src = lab if role == "H" else "Social media post by " + lab.replace("User ", "user ")
        s += "Source: "; sp.append(("label", len(s), len(s) + len(src))); s += src + "\n"
    s += "Text: "
    body, vsp = sentences(e, cf, val, var, role)
    if vsp: sp.append(("value", len(s) + vsp[0], len(s) + vsp[1]))
    s += body
    return s, sp

BLOCK = {"F1": block_F1, "F2": block_F2, "F3": block_F3}
INTRO = {"F1": "Context:\n", "F2": "News excerpts:\n\n", "F3": "Retrieved documents:\n\n"}

def build(p, fmt, cond, k, lowidx, order, ans):
    ent, cf = p["ent"], p["cfield"]
    vlow, vhigh = p["values"][lowidx], p["values"][1 - lowidx]
    H = ("H", p["gov"], vhigh, ent, 0)
    L1 = ("L", p["users"][0], vlow, ent, 0)
    blocks_high, blocks_low = [H], []
    if cond == "C0":  blocks_high = [("H", None, vhigh, ent, 0)]; blocks_low = [("L", None, vlow, ent, 0)]
    elif cond == "C0r": blocks_high = [("H", None, vhigh, ent, 0)]; blocks_low = [("L", None, vlow, ent, 0)] * k
    elif cond == "C1": blocks_low = [L1]
    elif cond == "C2": blocks_low = [L1] * k
    elif cond == "C3": blocks_low = [("L", p["users"][i], vlow, ent, 0) for i in range(k)]
    elif cond == "C4": blocks_low = [("L", ", ".join(p["users"][:k]), vlow, ent, 0)]
    elif cond == "C5": blocks_low = [("F", None, None, f, 0) for f in p["fillers"][:k - 1]] + [L1]
    elif cond == "C6": blocks_high = [H] * k; blocks_low = [L1]
    elif cond == "C7": blocks_low = [("L", p["users"][0], vlow, ent, v) for v in range(k)]
    if cond == "C5" and order == "LF": blocks = blocks_low + blocks_high
    elif cond == "C5": blocks = blocks_high + blocks_low
    else: blocks = blocks_high + blocks_low if order == "HF" else blocks_low + blocks_high
    kw = "Sources" if cond == "C4" else "Source"
    s, spans = INTRO[fmt], []
    lcopy = 0
    for i, (role, lab, val, e, var) in enumerate(blocks):
        t0 = len(s)
        txt, sp = BLOCK[fmt](i, role, lab, val, e, cf, var, kw)
        for kind, a, b in sp:
            spans.append({"kind": kind, "role": role, "idx": i, "copy": lcopy if role == "L" else None, "start": t0 + a, "end": t0 + b})
        s += txt
        spans.append({"kind": "block", "role": role, "idx": i, "copy": lcopy if role == "L" else None, "start": t0, "end": len(s)})
        if role == "L": lcopy += 1
        s += "\n\n"
    q = f"What is {ent['name']}'s {cf}?"
    opts = (vhigh, vlow) if ans == "AH" else (vlow, vhigh)
    s += (f"Question: {q}" if fmt == "F3" else f"\n{q}") + f"\n(A) {opts[0]}\n(B) {opts[1]}"
    # E5 assertions: every value span holds exactly the intended value
    for sp in spans:
        if sp["kind"] == "value":
            assert s[sp["start"]:sp["end"]] == (vlow if sp["role"] == "L" else vhigh), (fmt, cond, s[sp["start"]:sp["end"]])
        if sp["kind"] == "label":
            assert len(s[sp["start"]:sp["end"]]) > 3
    return s, spans, ("B" if ans == "AH" else "A")

COND_F1 = [("C0", 1), ("C0r", 4), ("C1", 1), ("C2", 2), ("C2", 4), ("C2", 8), ("C3", 2), ("C3", 4),
           ("C4", 2), ("C4", 4), ("C5", 4), ("C6", 2), ("C6", 4), ("C7", 2), ("C7", 4)]
COND_F23_TEST = [("C0", 1), ("C1", 1), ("C2", 2), ("C2", 4), ("C3", 4), ("C5", 4), ("C7", 4)]
COND_F3_SEL = [("C1", 1), ("C2", 4)]  # for F3 head ranking on the selection split
def plan(p, fmt):
    if fmt == "F1": return COND_F1
    if p["split"] == "test": return COND_F23_TEST
    return COND_F3_SEL if fmt == "F3" else []

# ---------------- write ----------------
n = 0
counts = collections.Counter()
examples = {}
with open(os.path.join(OUT, "prompts_v2.jsonl"), "w") as f:
    for fmt in ["F1", "F2", "F3"]:
        for p in pairs:
            for (cond, k), lowidx, order, ans in itertools.product(plan(p, fmt), [0, 1], ["HF", "LF"], ["AH", "AL"]):
                u, spans, ll = build(p, fmt, cond, k, lowidx, order, ans)
                r = {"id": n, "fmt": fmt, "pair_id": p["pair_id"], "split": p["split"], "cls": p["ent"]["cls"],
                     "cond": cond, "k": k, "ck": f"{cond}k{k}", "lowidx": lowidx, "order": order, "ans": ans,
                     "letter_low": ll, "system": SYSTEM[fmt], "user": u, "spans": spans}
                f.write(json.dumps(r) + "\n"); n += 1
                counts[(p["split"], fmt, f"{cond}k{k}")] += 1
                key = (fmt, f"{cond}k{k}")
                if key not in examples or (len(examples[key]) < 2 and p["pair_id"] == examples[key][0]["pair_id"] and order == "LF"):
                    examples.setdefault(key, []).append({"pair_id": p["pair_id"], "order": order, "ans": ans, "user": u})
with open(os.path.join(OUT, "pairs_v2.jsonl"), "w") as f:
    for p in pairs: f.write(json.dumps(p) + "\n")
with open(os.path.join(OUT, "examples_v2.md"), "w") as f:  # D3: full examples per format x condition
    for (fmt, ck), exs in sorted(examples.items()):
        for ex in exs[:2]:
            f.write(f"## {fmt} {ck} pair {ex['pair_id']} order {ex['order']} ans {ex['ans']}\n```\n{ex['user']}\n```\n\n")

# ---------------- checks for the manifest ----------------
names = [p["ent"]["name"] for p in pairs] + [fl["name"] for p in pairs for fl in p["fillers"]]
assert len(set(names)) == len(names), "duplicate entity names"
split_words = {s: set() for s in SPLITS}
for p in pairs: split_words[p["split"]] |= words_in(p["ent"]) | words_in(p["fillers"]) | words_in(p["gov"])
generic = words_in(GOV) | {"Holdings", "Works", "Group", "Industries"} | set(w.capitalize() for w in OCC + IND) | {"Of"}
for fld in ["birth year", "birthplace", "occupation", "height", "number of children", "founding year", "headquarters", "industry",
            "number of employees", "chief executive", "population", "main industry", "mayor", "elevation", "cls", "name", "attrs",
            "person", "company", "town"]:
    generic |= set(WORDRE.findall(fld.title()))
cross = (split_words["selection"] & split_words["test"]) - generic
pil = (split_words["selection"] | split_words["test"]) & pilot_words - generic
users_all = [u for p in pairs for u in p["users"]]
assert len(set(users_all)) == len(users_all)
assert not (set(users_all) & pilot_users)
assert not (set(names) & pilot_names)
assert not pil, f"pilot word overlap: {sorted(pil)[:20]}"
assert not cross, f"selection/test word overlap: {sorted(cross)[:20]}"

# E6: mean pairwise token-Jaccard between low-credibility blocks (test split, first 60 pairs, variant lowidx0 HF AH)
def jacc(a, b):
    A, B = set(a.split()), set(b.split()); return len(A & B) / len(A | B)
sim = {}
for fmt in ["F1", "F2", "F3"]:
    for ck in [("C2", 4), ("C7", 4), ("C3", 4)]:
        vals = []
        for p in [q for q in pairs if q["split"] == "test"][:60]:
            u, spans, _ = build(p, fmt, ck[0], ck[1], 0, "HF", "AH")
            bl = [u[s_["start"]:s_["end"]] for s_ in spans if s_["kind"] == "block" and s_["role"] == "L"]
            bl = [re.sub(r"^(Table [A-Z]|Excerpt \d+\.|\[Document \d+\])", "", b) for b in bl]  # drop block index
            vals += [jacc(bl[i], bl[j]) for i in range(len(bl)) for j in range(i + 1, len(bl))]
        sim[f"{fmt}_{ck[0]}k{ck[1]}"] = round(sum(vals) / len(vals), 3)

sha = hashlib.sha256(open(os.path.join(OUT, "prompts_v2.jsonl"), "rb").read()).hexdigest()
man = {"seed": SEED, "n_prompts": n, "n_pairs": len(pairs), "pairs_per_split": SPLITS, "prompts_sha256": sha,
       "paraphrase_method": "deterministic hand-written templates (field-label synonyms, header variants, row rotation for F1; sentence templates, frames and sentence rotation for F2/F3); no LLM",
       "overlap_with_pilot": {"entity_names": 0, "usernames": 0, "name_words": 0, "pilot_words_excluded": len(pilot_words)},
       "overlap_selection_test_words": 0,
       "copy_block_token_jaccard": sim,
       "counts": {f"{s}|{fmt}|{ck}": c for (s, fmt, ck), c in sorted(counts.items())},
       "totals": {f"{s}|{fmt}": sum(c for (s2, f2, _), c in counts.items() if s2 == s and f2 == fmt) for s in SPLITS for fmt in ["F1", "F2", "F3"]}}
json.dump(man, open(os.path.join(OUT, "manifest_v2.json"), "w"), indent=1)
print("prompts", n, "pairs", len(pairs), "sha", sha[:12]); print(json.dumps(man["totals"])); print("jaccard", sim)

# ---------------- graded-credibility set (added 13 Nov, Amr's amendment 2; generated AFTER the main set, so prompts_v2 is unchanged) ----------------
# Schuster's middle levels: newspaper ("The {Place} Gazette") and individual person (full name). Source X ∈ {gov, newspaper, person, social}
# holds one value; the partner block is always a social-media user (users[0]) with the other value. F1, C1-shaped, first 100 TEST pairs, 8 variants.
NEWS_T = ["The {P} Gazette", "The {P} Tribune", "The {P} Herald", "The {P} Courier", "The {P} Chronicle"]
G_PLACES = [word(3) for _ in range(20)]  # fresh words, not in any split pool
LEVELS = [("gov", 3), ("newspaper", 2), ("person", 1), ("social", 0)]
g_pairs = [p for p in pairs if p["split"] == "test"][:100]
ng = 0
with open(os.path.join(OUT, "prompts_graded.jsonl"), "w") as f:
    for p in g_pairs:
        srcs = {"gov": p["gov"], "newspaper": R.choice(NEWS_T).replace("{P}", R.choice(G_PLACES)),
                "person": f"{word(2)} {word(3)}", "social": p["users"][1]}
        for (lev, rank), lowidx, order, ans in itertools.product(LEVELS, [0, 1], ["HF", "LF"], ["AH", "AL"]):
            q = dict(p); q["gov"] = srcs[lev]  # X takes the gov slot (role H); partner = users[0] (role L)
            u, spans, ll = build(q, "F1", "C1", 1, lowidx, order, ans)
            f.write(json.dumps({"id": 1_000_000 + ng, "fmt": "F1", "pair_id": p["pair_id"], "split": "graded", "cls": p["ent"]["cls"],
                                "cond": "G", "k": 1, "ck": f"G_{lev}", "level": lev, "level_rank": rank, "source_X": srcs[lev],
                                "lowidx": lowidx, "order": order, "ans": ans, "letter_low": ll, "system": SYSTEM["F1"],
                                "user": u, "spans": spans}) + "\n"); ng += 1
# amendment 4: 5th, exploratory level X = neutral tag "Blue" vs the same social partner. Appended in its own id range, so the
# four-level rows above are unchanged; no RNG calls, so the neutral binding set below is unchanged too.
with open(os.path.join(OUT, "prompts_graded.jsonl"), "a") as f:
    for p in g_pairs:
        for j, (lowidx, order, ans) in enumerate(itertools.product([0, 1], ["HF", "LF"], ["AH", "AL"])):
            q = dict(p); q["gov"] = "Blue"
            u, spans, ll = build(q, "F1", "C1", 1, lowidx, order, ans)
            f.write(json.dumps({"id": 1_500_000 + p["pair_id"] * 8 + j, "fmt": "F1", "pair_id": p["pair_id"], "split": "graded", "cls": p["ent"]["cls"],
                                "cond": "G", "k": 1, "ck": "G_neutral", "level": "neutral", "level_rank": None, "source_X": "Blue",
                                "lowidx": lowidx, "order": order, "ans": ans, "letter_low": ll, "system": SYSTEM["F1"],
                                "user": u, "spans": spans}) + "\n"); ng += 1
print("graded prompts", ng)

# ---------------- neutral-label binding control (amendment 3, 13 Nov) ----------------
# C1-shaped F1 prompts with tags "Blue"/"Green" instead of sources; 16 variants = tag swap x value swap x order x answer order.
nn = 0
with open(os.path.join(OUT, "prompts_neutral.jsonl"), "w") as f:
    for split, ps in [("neutral_sel", [p for p in pairs if p["split"] == "selection"][:100]),
                      ("neutral_test", [p for p in pairs if p["split"] == "test"][:100])]:
        for p in ps:
            for tagH, lowidx, order, ans in itertools.product(["Blue", "Green"], [0, 1], ["HF", "LF"], ["AH", "AL"]):
                tagL = "Green" if tagH == "Blue" else "Blue"
                q = dict(p); q["gov"] = tagH; q["users"] = [tagL] + p["users"][1:]
                u, spans, ll = build(q, "F1", "C1", 1, lowidx, order, ans)
                assert f"(Source: {tagH})" in u and f"(Source: {tagL})" in u
                hA = ll == "B"  # the H-role block's value is option A
                f.write(json.dumps({"id": 2_000_000 + nn, "fmt": "F1", "pair_id": p["pair_id"], "split": split, "cls": p["ent"]["cls"],
                                    "cond": "N", "k": 1, "ck": "N_C1", "tagH": tagH, "blue_is_A": float((tagH == "Blue") == hA),
                                    "lowidx": lowidx, "order": order, "ans": ans, "letter_low": ll, "system": SYSTEM["F1"],
                                    "user": u, "spans": spans}) + "\n"); nn += 1
print("neutral prompts", nn)

# ---------------- amendment 5 (13 Nov): F3 selection-split C7k4 prompts, for ranking heads with paraphrased copies as the corrupt run ----------------
# Deterministic (no RNG), own file and id range, so every earlier file is unchanged.
n7 = 0
with open(os.path.join(OUT, "prompts_f3selC7.jsonl"), "w") as f:
    for p in [q for q in pairs if q["split"] == "selection"]:
        for lowidx, order, ans in itertools.product([0, 1], ["HF", "LF"], ["AH", "AL"]):
            u, spans, ll = build(p, "F3", "C7", 4, lowidx, order, ans)
            f.write(json.dumps({"id": 3_000_000 + n7, "fmt": "F3", "pair_id": p["pair_id"], "split": "selection", "cls": p["ent"]["cls"],
                                "cond": "C7", "k": 4, "ck": "C7k4", "lowidx": lowidx, "order": order, "ans": ans,
                                "letter_low": ll, "system": SYSTEM["F3"], "user": u, "spans": spans}) + "\n"); n7 += 1
print("F3 selection C7k4 prompts", n7)

# ---------------- amendment 6 (13 Nov): C6 test set for the C1-only probe. C6k4 = text-identical to the main F1 test C6k4 rows
# (verified), new ids; C6k8 = new. All ids 4,000,000+. Deterministic; earlier files unchanged.
n6 = 0
with open(os.path.join(OUT, "prompts_c6.jsonl"), "w") as f:
    for k in (4, 8):
        for p in [q for q in pairs if q["split"] == "test"]:
            for lowidx, order, ans in itertools.product([0, 1], ["HF", "LF"], ["AH", "AL"]):
                u, spans, ll = build(p, "F1", "C6", k, lowidx, order, ans)
                f.write(json.dumps({"id": 4_000_000 + n6, "fmt": "F1", "pair_id": p["pair_id"], "split": "test", "cls": p["ent"]["cls"],
                                    "cond": "C6", "k": k, "ck": f"C6k{k}", "lowidx": lowidx, "order": order, "ans": ans,
                                    "letter_low": ll, "system": SYSTEM["F1"], "user": u, "spans": spans}) + "\n"); n6 += 1
print("C6 test prompts", n6)
