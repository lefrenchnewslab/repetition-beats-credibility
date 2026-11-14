<script lang="ts">
  import { Segmented } from "@stdlogits/ledger";
  import { FORMATS, MODELS, p, type Fmt, type ModelKey } from "../lib";

  let model = $state<ModelKey>("qwen7");
  let fmt = $state<Fmt>("F1");

  const SERIES = [
    { c: "C2", name: "One user, copied", dash: "" , w: 3, o: 1 },
    { c: "C7", name: "One user, reworded", dash: "", w: 3, o: 0.55 },
    { c: "C3", name: "Many users, same words", dash: "6 5", w: 2, o: 1 },
    { c: "C4", name: "Many users in one block", dash: "2 4", w: 2, o: 0.55 },
  ];
  const KS = [1, 2, 4, 8];
  const W = 760, H = 360, L = 56, R = 190, T = 20, B = 44;
  const x = (k: number) => L + (KS.indexOf(k) / (KS.length - 1)) * (W - L - R);
  const y = (v: number) => T + (1 - v) * (H - T - B);

  const lines = $derived(
    SERIES.map((s) => {
      const pts = KS.map((k) => ({ k, v: p(model, fmt, k === 1 ? "C1k1" : `${s.c}k${k}`) })).filter((d) => d.v) as { k: number; v: [number, number | null, number | null] }[];
      return { ...s, pts };
    }).filter((s) => s.pts.length > 1),
  );
  const filler = $derived(p(model, fmt, "C5k4"));
  const gov = $derived(p(model, fmt, "C6k4"));
  const tint = (o: number) => `color-mix(in oklch, var(--tint) ${Math.round(o * 100)}%, var(--wash2))`;
  const modelOpts = $derived(MODELS.filter((m) => p(m.key, fmt, "C1k1")).map((m) => ({ value: m.key, label: m.short })));
  $effect(() => { if (!p(model, fmt, "C1k1")) model = "qwen7"; });
</script>

<section class="part" id="copies">
  <p class="kicker">Dose</p>
  <h2>More copies, more pull, until the format pushes back</h2>
  <p class="lede">In tables every model gives in as the post repeats. In news snippets and documents, exact copies stall near a coin flip while reworded copies keep climbing.</p>
  <div class="controls">
    <div class="ctl"><span class="label">Model</span><Segmented label="Model" value={model} onchange={(v) => (model = v)} options={modelOpts} /></div>
    <div class="ctl"><span class="label">Format</span><Segmented label="Format" value={fmt} onchange={(v) => (fmt = v)} options={FORMATS} /></div>
  </div>
  <div class="surface figure">
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Probability of the social-media value by number of copies">
      {#each [0, 0.25, 0.5, 0.75, 1] as g}
        <line class="gridline" x1={L} x2={W - R} y1={y(g)} y2={y(g)} style:stroke={g === 0.5 ? "var(--ink3)" : undefined} style:stroke-dasharray={g === 0.5 ? "3 4" : undefined} />
        <text class="axis" x={L - 10} y={y(g) + 4} text-anchor="end">{g.toFixed(2)}</text>
      {/each}
      {#each KS as k}<text class="axis" x={x(k)} y={H - 18} text-anchor="middle">{k === 1 ? "1 post" : `${k} copies`}</text>{/each}
      {#each lines as s (s.c)}
        <path d={s.pts.map((d, i) => `${i ? "L" : "M"}${x(d.k)},${y(d.v[0])}`).join(" ")} fill="none" stroke={tint(s.o)} stroke-width={s.w} stroke-dasharray={s.dash} stroke-linecap="round" style="transition: d var(--glide) var(--ease)" />
        {#each s.pts as d}
          <circle cx={x(d.k)} cy={y(d.v[0])} r="5" fill={tint(s.o)} data-tip-title="{s.name}, {d.k === 1 ? 'one post' : d.k + ' copies'}" data-tip="P = {d.v[0].toFixed(3)}{d.v[1] != null ? `, 95% CI ${d.v[1].toFixed(3)} to ${d.v[2]?.toFixed(3)}` : ''}" style="transition: cy var(--glide) var(--ease)" />
        {/each}
        {@const last = s.pts[s.pts.length - 1]}
        <text x={x(last.k) + 12} y={y(last.v[0]) + 4} fill="var(--ink2)" font-size="13">{s.name}</text>
      {/each}
      {#if filler}
        <rect x={x(4) - 6} y={y(filler[0]) - 6} width="12" height="12" rx="2" fill="var(--ink3)" data-tip-title="Length control (C5)" data-tip="One post padded with blocks about other entities to the length of four copies: P = {filler[0].toFixed(3)}" />
      {/if}
      {#if gov}
        <rect x={x(4) - 6} y={y(gov[0]) - 6} width="12" height="12" rx="6" fill="var(--ink)" data-tip-title="Government repeated (C6)" data-tip="The government block four times instead: P = {gov[0].toFixed(3)}" />
      {/if}
      <text class="axis" x={L} y={12}>P(social-media value)</text>
    </svg>
    <p class="legendline"><i class="sq"></i> filler of the same length <i class="ci"></i> government block repeated instead · dashed line: chance</p>
  </div>
</section>

<style>
  .legendline { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2); color: var(--ink3); font: var(--type-meta); margin-top: var(--s2); }
  .sq, .ci { display: inline-block; width: 10px; height: 10px; background: var(--ink3); border-radius: 2px; }
  .ci { background: var(--ink); border-radius: 999px; margin-left: var(--s3); }
  svg { min-width: 560px; }
</style>
