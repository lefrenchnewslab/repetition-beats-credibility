<script lang="ts">
  import { MorphText, Segmented } from "@stdlogits/ledger";
  import { D, depth, headId } from "../lib";

  // Block F, criterion P2: the fixed top-10 heads of H1 (tables), patched for four contrasts on held-out test runs.
  const P2: Record<string, { c: string; name: string; tip: string; rec: number; ci: [number, number]; p95: number; n: number }[]> = {
    llama: [
      { c: "C1→C2 k=4", name: "A post copied four times", tip: "The conflict itself", rec: 92.1, ci: [90.5, 93.7], p95: 31.1, n: 2845 },
      { c: "C1→C6 k=4", name: "The government repeated", tip: "Repetition that agrees with the credible source", rec: 92.8, ci: [91.5, 94.0], p95: 34.1, n: 3021 },
      { c: "C0→C0r k=4", name: "Repetition with no labels", tip: "No sources named at all", rec: 90.7, ci: [89.4, 91.9], p95: 28.0, n: 3118 },
      { c: "C0→C1", name: "Credibility alone", tip: "Adding the source labels, no repetition", rec: 92.6, ci: [91.3, 93.9], p95: 24.3, n: 1297 },
    ],
    qwen7: [
      { c: "C1→C2 k=4", name: "A post copied four times", tip: "The conflict itself", rec: 95.4, ci: [92.5, 98.6], p95: 18.7, n: 2429 },
      { c: "C1→C6 k=4", name: "The government repeated", tip: "Repetition that agrees with the credible source", rec: 96.6, ci: [92.3, 101.5], p95: 16.6, n: 2576 },
      { c: "C0→C0r k=4", name: "Repetition with no labels", tip: "No sources named at all", rec: 90.2, ci: [86.5, 94.3], p95: 14.0, n: 2856 },
      { c: "C0→C1", name: "Credibility alone", tip: "Adding the source labels, no repetition", rec: 72.3, ci: [69.4, 75.2], p95: 22.7, n: 2599 },
    ],
    gemma: [
      { c: "C1→C2 k=4", name: "A post copied four times", tip: "The conflict itself", rec: 93.2, ci: [91.4, 95.3], p95: 13.6, n: 2269 },
      { c: "C1→C6 k=4", name: "The government repeated", tip: "Repetition that agrees with the credible source", rec: 93.7, ci: [91.7, 95.7], p95: 11.7, n: 2040 },
      { c: "C0→C0r k=4", name: "Repetition with no labels", tip: "No sources named at all", rec: 90.7, ci: [87.8, 93.9], p95: 14.7, n: 2851 },
      { c: "C0→C1", name: "Credibility alone", tip: "Adding the source labels, no repetition", rec: 92.2, ci: [89.3, 95.1], p95: 12.8, n: 1387 },
    ],
  };
  let model = $state<"qwen7" | "llama" | "gemma">("qwen7");
  const A = $derived(D.attr[`${model}_F1C2`]);
  const x = (v: number) => `${Math.min(100, v)}%`;
</script>

<section class="part">
  <p class="kicker">Not a repetition circuit</p>
  <h2>The same ten heads settle every answer change</h2>
  <p class="lede">Patch only those ten heads and you restore most of the gap for repetition, for credibility, for unlabelled repetition and for a repeated government source. They are where the answer gets decided, so damping them trades one signal for the other.</p>
  <div class="controls"><div class="ctl"><span class="label">Model</span><Segmented label="Model" value={model} onchange={(v) => (model = v)} options={[{ value: "qwen7", label: "Qwen 7B" }, { value: "llama", label: "Llama 8B" }, { value: "gemma", label: "Gemma 12B" }]} /></div></div>
  <div class="surface jobs">
    <div class="heads">
      {#each A.top as [l, h], i (headId(l, h))}
        <span class="hd rise" style:--i={i} style:background={depth(l, A.L)} data-tip-title={headId(l, h)} data-tip="Layer {l} of {A.L}">{headId(l, h)}</span>
      {/each}
    </div>
    {#each P2[model] as r, i (r.c)}
      <div class="job" data-tip-title={r.name} data-tip="{r.tip} · {r.c} · 95% CI {r.ci[0]} to {r.ci[1]} · {r.n} eligible runs (gap above 1 logit)">
        <span class="nm">{r.name}</span>
        <span class="track">
          <span class="fill" style:width={x(r.rec)}></span>
          <span class="rand" style:left={x(r.p95)} data-tip="95th percentile of 20 random sets of ten heads: {r.p95}%"></span>
        </span>
        <span class="v num"><MorphText text={`${Math.round(r.rec)}%`} /></span>
      </div>
    {/each}
    <p class="note">Share of each logit gap restored by patching the ten heads chosen on the conflict in tables. The thin mark is what random sets of ten heads reach (95th percentile).</p>
  </div>
</section>

<style>
  .jobs { display: grid; gap: var(--s4); }
  .heads { display: flex; flex-wrap: wrap; gap: var(--s1); }
  .hd { padding: 2px 8px; border-radius: var(--r-pill); color: var(--paper); font: var(--type-meta); font-weight: 600; }
  .job { display: grid; grid-template-columns: 220px minmax(0, 1fr) 56px; gap: var(--s4); align-items: center; }
  @media (max-width: 640px) { .job { grid-template-columns: minmax(0, 1fr) 56px; } .job .track { grid-column: 1 / -1; grid-row: 2; } }
  .nm { font: var(--type-label); }
  .track { position: relative; height: 18px; border-radius: 999px; background: var(--paper); }
  .fill { position: absolute; inset: 0 auto 0 0; border-radius: 999px; background: var(--tint); transition: width var(--glide) var(--ease); }
  .rand { position: absolute; top: -4px; bottom: -4px; width: 3px; margin-left: -1.5px; border-radius: 2px; background: var(--ink); transition: left var(--glide) var(--ease); }
  .v { font: 600 20px/24px var(--display); text-align: right; }
</style>
