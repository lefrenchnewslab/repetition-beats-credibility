<script lang="ts">
  import { Meter, MorphText, Segmented } from "@stdlogits/ledger";
  import { D } from "../lib";

  type SetK = "F3_C7k4" | "U" | "Feeders";
  let set = $state<SetK>("F3_C7k4");
  let g = $state("0.5");
  const SETS = [
    { value: "F3_C7k4" as SetK, label: "Paraphrase heads (20)", hint: "Top heads ranked on reworded copies in documents" },
    { value: "U" as SetK, label: "Both sets (28)", hint: "Union of the copies and paraphrase top-20 lists" },
    { value: "Feeders" as SetK, label: "Feeders only (5)", hint: "Paraphrase top-10 heads that are not in the copies top-10" },
  ];
  const curve = (s: SetK) => [["1", [0, 1]], ...Object.entries(D.curves[s]).sort((a, b) => Number(b[0]) - Number(a[0]))] as [string, [number, number]][];
  const pt = $derived<[number, number]>(g === "1" ? [0, 1] : D.curves[set][g === "0" ? "0.0" : g]);
  const W = 560, H = 420, L = 56, R = 24, T = 20, B = 48;
  const x = (v: number) => L + Math.max(0, Math.min(1.05, v)) / 1.05 * (W - L - R);
  const y = (v: number) => T + (1 - Math.min(1.05, v) / 1.05) * (H - T - B);
  const LORA = { removed: 0.95, kept: 1.02, users: 0.02 };
</script>

<section class="part" id="fixes">
  <p class="kicker">Fixes</p>
  <h2>Turn the heads down and you lose the ministry too</h2>
  <p class="lede">Scaling the top heads' output removes reworded repetition only by removing source preference at the same rate. A trained adapter does better on both, but by ignoring every extra low-credibility block, including four genuinely different users.</p>
  <div class="controls">
    <div class="ctl"><span class="label">Heads damped</span><Segmented label="Heads damped" value={set} onchange={(v) => (set = v)} options={SETS} /></div>
    <div class="ctl"><span class="label">Output kept (γ)</span><Segmented label="Output kept" value={g} onchange={(v) => (g = v)} options={[{ value: "1", label: "100%" }, { value: "0.75", label: "75%" }, { value: "0.5", label: "50%" }, { value: "0.25", label: "25%" }, { value: "0", label: "0%" }]} /></div>
  </div>
  <div class="fx">
    <div class="surface figure">
      <svg viewBox="0 0 {W} {H}" role="img" aria-label="Paraphrase effect removed against source preference kept">
        <rect x={x(0.5)} y={y(1.05)} width={x(1.05) - x(0.5)} height={y(0.7) - y(1.05)} fill="color-mix(in oklch, var(--tint) 12%, var(--wash))" />
        <text class="axis" x={x(1.03)} y={y(0.72)} text-anchor="end">pre-registered target</text>
        {#each [0, 0.25, 0.5, 0.75, 1] as t}
          <line class="gridline" x1={L} x2={W - R} y1={y(t)} y2={y(t)} />
          <text class="axis" x={L - 8} y={y(t) + 4} text-anchor="end">{Math.round(t * 100)}%</text>
          <text class="axis" x={x(t)} y={H - 24} text-anchor="middle">{Math.round(t * 100)}%</text>
        {/each}
        <text class="axis" x={(L + W - R) / 2} y={H - 6} text-anchor="middle">reworded-copy effect removed (documents)</text>
        <text class="axis" x="12" y={T + 4}>source preference kept</text>
        {#each SETS as s (s.value)}
          {@const c = curve(s.value)}
          <path d={c.map(([, v], i) => `${i ? "L" : "M"}${x(v[0])},${y(v[1])}`).join(" ")} fill="none" stroke={s.value === set ? "var(--tint)" : "var(--wash2)"} stroke-width={s.value === set ? 3 : 2} style="transition: stroke var(--glide) var(--ease)" />
          {#each c as [gg, v]}<circle cx={x(v[0])} cy={y(v[1])} r={s.value === set ? 4 : 3} fill={s.value === set ? "var(--tint)" : "var(--wash2)"} />{/each}
        {/each}
        <circle cx={x(pt[0])} cy={y(pt[1])} r="10" fill="none" stroke="var(--ink)" stroke-width="2" style="transition: cx var(--glide) var(--ease), cy var(--glide) var(--ease)" />
        <g transform="translate({x(LORA.removed)}, {y(LORA.kept)})" data-tip-title="Trained LoRA adapter" data-tip="95% removed, 102% of source preference kept, but only 2% of the distinct-user effect">
          <rect x="-8" y="-8" width="16" height="16" transform="rotate(45)" fill="var(--ink)" />
          <text x="-14" y="5" text-anchor="end" class="ln">adapter</text>
        </g>
      </svg>
    </div>
    <div class="cards">
      <div class="surface card">
        <p class="gh">Head damping at {g === "1" ? "full output" : `${Math.round(Number(g) * 100)}% output`}</p>
        <div class="m"><span>Reworded effect removed</span><Meter value={Math.max(0, pt[0])} label="removed" width={64} /><b class="num"><MorphText text={`${Math.round(pt[0] * 100)}%`} /></b></div>
        <div class="m"><span>Source preference kept</span><Meter value={Math.min(1, pt[1])} label="kept" width={64} /><b class="num"><MorphText text={`${Math.round(pt[1] * 100)}%`} /></b></div>
        <p class="note">Damping scales the chosen heads' output at every position. Qwen 7B, retrieved documents, held-out test entities.</p>
      </div>
      <div class="surface card">
        <p class="gh">Trained adapter (rank-16 LoRA)</p>
        <div class="m"><span>Reworded effect removed</span><Meter value={0.95} label="95%" width={64} /><b class="num">95%</b></div>
        <div class="m"><span>Source preference kept</span><Meter value={1} label="102%" width={64} /><b class="num">102%</b></div>
        <div class="m"><span data-tip="Four different users posting the same claim: a signal a careful reader should still count">Distinct-user effect kept</span><Meter value={0.02} label="2%" width={64} /><b class="num">2%</b></div>
        <p class="note">It never saw distinct users in training, so the simplest rule that fits is to ignore every extra block. MMLU −0.6 points.</p>
      </div>
      <div class="surface card">
        <p class="gh">For comparison: exact copies in tables</p>
        <p class="note">Damping ten heads to half output removes 54% of the exact-copy effect, keeps 80% of source preference and 60% of the distinct-user effect, at −1.8 MMLU points (95% CI −3.6 to 0.0). It passed its bar in Qwen 7B only.</p>
      </div>
    </div>
  </div>
</section>

<style>
  .fx { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: var(--s5); align-items: start; }
  @media (max-width: 900px) { .fx { grid-template-columns: minmax(0, 1fr); } }
  svg { min-width: 420px; }
  .cards { display: grid; gap: var(--s4); }
  .card { display: grid; gap: var(--s3); }
  .gh { color: var(--ink3); font: var(--type-head); }
  .m { display: grid; grid-template-columns: minmax(0, 1fr) auto 48px; gap: var(--s3); align-items: center; font: var(--type-small); }
  .m b { font: 600 16px/20px var(--display); text-align: right; }
  .ln { fill: var(--ink); font: var(--type-meta); font-weight: 600; }
</style>
