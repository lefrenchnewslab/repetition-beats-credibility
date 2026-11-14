<script lang="ts">
  import { Expander } from "@stdlogits/ledger";
  type V = "met" | "not met" | "–" | string;
  const ROWS: { id: string; name: string; v: [V, V, V] }[] = [
    { id: "H1", name: "Top-10 heads recover ≥40%, above random", v: ["met", "met", "met"] },
    { id: "H3", name: "Graded credibility: government > newspaper > person > social", v: ["not met", "met", "met"] },
    { id: "H4", name: "Source identity survives repetition, reverses at output", v: ["not met (expl.)", "not met", "not met"] },
    { id: "H5", name: "Source readable at output while the answer flips", v: ["met (expl.)", "not met", "not met"] },
    { id: "B0", name: "Top heads are induction heads", v: ["not met", "not met", "not met"] },
    { id: "B1", name: "≤10 upstream parts mediate ≥40%", v: ["–", "met (marginal)", "–"] },
    { id: "B2", name: "Credibility direction steers the answer", v: ["–", "not met", "–"] },
    { id: "B3-S1", name: "Damping fix, exact copies, tables", v: ["not met", "met", "not met"] },
    { id: "B3-S2", name: "Damping fix, paraphrases, documents", v: ["–", "not met", "–"] },
    { id: "D-S2", name: "Trained adapter, paraphrases, documents", v: ["–", "met", "–"] },
    { id: "D-sel", name: "Adapter keeps ≥50% of the distinct-user effect", v: ["–", "not met", "–"] },
    { id: "P1", name: "Paraphrase gain exceeds exact-copy gain", v: ["met (expl.)", "met", "met (expl.)"] },
    { id: "P2", name: "Top heads specific to the conflict", v: ["not met", "not met", "not met"] },
  ];
  const cls = (v: V) => (v.startsWith("met") ? "met" : v.startsWith("not") ? "no" : "na");
  let open = $state(false);
</script>

<section class="part" id="record">
  <p class="kicker">The record</p>
  <h2>Written down before the data existed, including what would fail</h2>
  <p class="lede">Every criterion was fixed in advance with the outcome that would count against it. Here is how each came out. The failures are part of the result.</p>
  <div class="surface figure">
    <div class="tbl" role="table">
      <div class="tr th" role="row"><span role="columnheader">Criterion</span><span role="columnheader">Llama 8B</span><span role="columnheader">Qwen 7B</span><span role="columnheader">Gemma 12B</span></div>
      {#each ROWS as r, i}
        <div class="tr rise" role="row" style:--i={i}>
          <span role="cell" class="crit"><b>{r.id}</b> {r.name}</span>
          {#each r.v as v}<span role="cell"><i class="pill {cls(v)}">{v}</i></span>{/each}
        </div>
      {/each}
    </div>
    <p class="note">P1 is also met, confirmatory, in Qwen 32B. “expl.” marks exploratory tests: Llama's probe layers were seen before the layer rule was fixed.</p>
  </div>
  <Expander label="What this does not show" openLabel="Hide" bind:open>
    <ul class="lim">
      <li>Entities are fictional, each claim is one attribute in a two-option question, and paraphrases come from four templates, which may make exact copies easier to spot than natural text.</li>
      <li>Models run from 7B to 32B; the circuit work stops at 12B, and patching covers the shared question tokens only.</li>
      <li>Steering edited the final token only, up to 32 standard deviations of the probe projection.</li>
      <li>The adapter used one seed and never saw distinct users in training. Counting sources instead of passages, source-counting prompts and span-level attention edits were not tested.</li>
    </ul>
  </Expander>
</section>

<style>
  .tbl { display: grid; min-width: 640px; }
  .tr { display: grid; grid-template-columns: minmax(0, 2.4fr) repeat(3, minmax(0, 1fr)); gap: var(--s3); align-items: center; padding: var(--s2) var(--s3); border-radius: var(--r-col); }
  .tr:nth-child(even) { background: var(--paper); }
  .th { color: var(--ink3); font: var(--type-head); }
  .crit { font: var(--type-small); color: var(--ink2); }
  .crit b { color: var(--ink); font-weight: 600; margin-right: var(--s1); }
  .pill { display: inline-block; padding: 2px 10px; border-radius: var(--r-pill); font: var(--type-meta); font-style: normal; white-space: nowrap; }
  .pill.met { background: var(--tint); color: var(--paper); font-weight: 600; }
  .pill.no { background: var(--wash2); color: var(--ink2); }
  .pill.na { color: var(--ink3); }
  .lim { display: grid; gap: var(--s2); padding-left: var(--s5); color: var(--ink2); font: var(--type-body); max-width: 72ch; margin-top: var(--s3); }
</style>
