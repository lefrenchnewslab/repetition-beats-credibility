<script lang="ts">
  import { MorphText, Segmented } from "@stdlogits/ledger";
  import { D, depth } from "../lib";

  let model = $state<"qwen7" | "llama" | "gemma">("qwen7");
  const P = $derived(D.probe[model]);
  const layers = $derived<number[]>(Object.keys(P.layers).map(Number).sort((a, b) => a - b));
  const NL = $derived(layers[layers.length - 1]);
  const Lstar = $derived<number>(P.H4?.Lstar ?? layers[Math.floor(layers.length / 2)]);
  let at = $state(18);
  $effect(() => { at = Lstar; });
  const cur = $derived(P.layers[String(layers.reduce((b, l) => (Math.abs(l - at) < Math.abs(b - at) ? l : b), layers[0]))]);

  const W = 760, H = 300, L = 48, R = 20, T = 16, B = 40;
  const x = (l: number) => L + (l / NL) * (W - L - R);
  const y = (v: number) => T + (1 - v) * (H - T - B);
  const line = (k: "flip" | "c0r" | "c1") => layers.map((l, i) => `${i ? "L" : "M"}${x(l)},${y(P.layers[String(l)][k])}`).join(" ");
  const SER = [
    { k: "c1" as const, name: "One post vs one ministry (C1)", o: 0.35, w: 2, dash: "" },
    { k: "flip" as const, name: "Eight copies, answer flipped", o: 1, w: 3, dash: "" },
    { k: "c0r" as const, name: "Repetition with no labels (C0r)", o: 0.6, w: 2, dash: "5 4" },
  ];
  // Table 11: graded set, projection onto the all-conditions probe direction at the H3 layer.
  const LADDER: Record<string, { layer: number; pts: [string, number][] }> = {
    llama: { layer: 26, pts: [["Government", 5.14], ["Newspaper", 2.94], ["Person", 2.9], ["Neutral tag", 1.79], ["Social media", -0.04]] },
    qwen7: { layer: 24, pts: [["Government", 6.24], ["Newspaper", 4.09], ["Person", 3.45], ["Neutral tag", 3.21], ["Social media", 0.06]] },
    gemma: { layer: 26, pts: [["Government", 6.19], ["Newspaper", 4.21], ["Neutral tag", 3.28], ["Person", 3.02], ["Social media", 0.05]] },
  };
  const lad = $derived(LADDER[model]);
  const lx = (v: number) => 24 + ((v + 0.5) / 7) * (W - 48);
</script>

<section class="part" id="signal">
  <p class="kicker">The credibility signal</p>
  <h2>The model knows the ministry outranks the post, until the copies pile up</h2>
  <p class="lede">A linear probe trained on one-post prompts reads which option the government holds from mid-depth on. Under eight copies the same direction turns around and follows the answer instead. Nudging along it does not change answers.</p>
  <div class="controls">
    <div class="ctl"><span class="label">Model</span><Segmented label="Model" value={model} onchange={(v) => (model = v)} options={[{ value: "qwen7", label: "Qwen 7B" }, { value: "gemma", label: "Gemma 12B" }, { value: "llama", label: "Llama 8B (exploratory)" }]} /></div>
    <label class="ctl slider"><span class="label">Layer <b class="num">{at}</b></span><input id="probe-layer" type="range" min={layers[0]} max={NL} step="1" bind:value={at} style:--tint={depth(at, NL + 1)} /></label>
  </div>
  <div class="surface figure">
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Probe AUC by layer">
      {#each [0, 0.25, 0.5, 0.75, 1] as g}
        <line class="gridline" x1={L} x2={W - R} y1={y(g)} y2={y(g)} style:stroke={g === 0.5 ? "var(--ink3)" : undefined} style:stroke-dasharray={g === 0.5 ? "3 4" : undefined} />
        <text class="axis" x={L - 8} y={y(g) + 4} text-anchor="end">{g.toFixed(2)}</text>
      {/each}
      <rect x={L} y={y(0.5)} width={W - L - R} height={y(0) - y(0.5)} fill="var(--wash2)" opacity=".5" />
      <text class="axis" x={W - R - 6} y={y(0.06)} text-anchor="end">below chance: tracks the answer, not the source</text>
      {#each SER as s (s.k)}
        <path d={line(s.k)} fill="none" stroke="color-mix(in oklch, var(--tint) {Math.round(s.o * 100)}%, var(--wash2))" stroke-width={s.w} stroke-dasharray={s.dash} />
      {/each}
      <line x1={x(Lstar)} x2={x(Lstar)} y1={T} y2={H - B} stroke="var(--ink3)" stroke-dasharray="2 3" />
      <text class="axis" x={x(Lstar) + 4} y={T + 10}>L* = {Lstar}</text>
      <line x1={x(at)} x2={x(at)} y1={T} y2={H - B} stroke="var(--ink)" stroke-width="1.5" style="transition: x1 var(--fast) var(--ease), x2 var(--fast) var(--ease)" />
      {#each layers.filter((_, i) => i % 2 === 0) as l}<text class="axis" x={x(l)} y={H - 18} text-anchor="middle">{l}</text>{/each}
    </svg>
    <div class="readout">
      {#each SER as s}
        <div><dt>{s.name}</dt><dd class="num"><MorphText text={cur[s.k].toFixed(2)} /></dd></div>
      {/each}
    </div>
  </div>

  <div class="surface figure">
    <p class="gh">Source ladder at layer {lad.layer}: projection onto the probe direction, one block per source</p>
    <svg viewBox="0 0 {W} 110" role="img" aria-label="Sources ordered by credibility">
      <line x1="24" x2={W - 24} y1="56" y2="56" stroke="var(--wash2)" stroke-width="6" stroke-linecap="round" />
      {#each lad.pts as [n, v], i (n)}
        <g style:transform="translate({lx(v)}px, 56px)" style="transition: transform var(--glide) var(--ease)">
          <circle r="9" fill={n === "Neutral tag" ? "var(--ink3)" : `color-mix(in oklch, var(--deep) ${Math.round((v / 6.3) * 100)}%, var(--late))`} />
          <text y={i % 2 ? 32 : -18} text-anchor="middle" class="ln">{n}</text>
          <text y={i % 2 ? 46 : -32} text-anchor="middle" class="axis">{v >= 0 ? "+" : ""}{v.toFixed(2)}</text>
        </g>
      {/each}
    </svg>
    <p class="note">Government above newspaper above person above social media, each step with a 95% CI above zero in Qwen 7B and Gemma. This ladder uses the probe trained on all labelled conditions, not the one-post probe above, and the neutral tag “Blue” lands near the person level, so part of the axis just separates social media from everything else.</p>
  </div>
</section>

<style>
  .slider { min-width: 240px; }
  .slider b { color: var(--ink); }
  input[type="range"] { width: 260px; accent-color: var(--tint); }
  .readout { display: flex; flex-wrap: wrap; gap: var(--s3) var(--s6); margin-top: var(--s3); }
  dt { color: var(--ink3); font: var(--type-meta); }
  dd { margin: 0; font: 600 22px/28px var(--display); }
  .gh { color: var(--ink3); font: var(--type-head); margin-bottom: var(--s2); }
  .ln { fill: var(--ink); font: var(--type-meta); font-weight: 600; }
  svg { min-width: 520px; }
</style>
