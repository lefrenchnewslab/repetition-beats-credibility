<script lang="ts">
  import { MorphText, Segmented } from "@stdlogits/ledger";
  import { MODELS, p, type Fmt, type ModelKey } from "../lib";

  let model = $state<ModelKey>("qwen32");
  let fmt = $state<Fmt>("F3");
  const cell = (c: string) => p(model, fmt, `${c}k4`)?.[0] ?? null;
  const grid = $derived({ C2: cell("C2"), C3: cell("C3"), C7: cell("C7"), C8: cell("C8") });
  const discount = $derived(grid.C2 != null && grid.C3 != null ? grid.C3 - grid.C2 : null);
  const left = $derived(grid.C7 != null && grid.C8 != null ? grid.C8 - grid.C7 : null);
  const opts = $derived(MODELS.filter((m) => p(m.key, fmt, "C8k4")).map((m) => ({ value: m.key, label: m.short })));
  $effect(() => { if (!p(model, fmt, "C8k4")) model = "qwen7"; });
  const tiles = [
    { c: "C2", row: "Same words", col: "One user", tip: "Four exact copies of one post" },
    { c: "C3", row: "Same words", col: "Four users", tip: "Four usernames posting identical text" },
    { c: "C7", row: "Reworded", col: "One user", tip: "One user's claim in four templates" },
    { c: "C8", row: "Reworded", col: "Four users", tip: "Four usernames, each in a different template" },
  ] as const;
</script>

<section class="part">
  <p class="kicker">The duplicate discount</p>
  <h2>Models discount copies, and rewording erases the discount</h2>
  <p class="lede">Four copies from one account count for less than four separate accounts: a sensible duplicate discount. Rewrite the copies and the gap mostly closes, so a single account passes for a crowd.</p>
  <div class="controls">
    <div class="ctl"><span class="label">Model</span><Segmented label="Model" value={model} onchange={(v) => (model = v)} options={opts} /></div>
    <div class="ctl"><span class="label">Format</span><Segmented label="Format" value={fmt} onchange={(v) => (fmt = v)} options={[{ value: "F1", label: "Tables" }, { value: "F3", label: "Retrieved documents" }]} /></div>
  </div>
  <div class="wrap">
    <div class="grid">
      <span></span><span class="ch">One user</span><span class="ch">Four users</span>
      {#each ["Same words", "Reworded"] as r}
        <span class="rh">{r}</span>
        {#each tiles.filter((t) => t.row === r) as t (t.c)}
          {@const v = grid[t.c]}
          <div class="tile" data-tip={t.tip} data-tip-title="{t.c} k=4">
            <span class="fill" style:height={v != null ? `${v * 100}%` : "0%"}></span>
            <span class="big num">{#if v != null}<MorphText text={v.toFixed(2)} />{:else}–{/if}</span>
          </div>
        {/each}
      {/each}
    </div>
    <div class="facts">
      <div><dt>Duplicate discount (four users minus four copies)</dt><dd class="num">{discount != null ? `${discount >= 0 ? "+" : ""}${(discount * 100).toFixed(1)} points` : "–"}</dd></div>
      <div><dt>Left after rewording (reworded four users minus reworded one user)</dt><dd class="num">{left != null ? `${left >= 0 ? "+" : ""}${(left * 100).toFixed(1)} points` : "–"}</dd></div>
      {#if model === "gemma" && fmt === "F3"}
        <p class="note">Gemma, documents, exploratory: its small paraphrase advantage reverses against its strongest single rewording template (−4.4 points, 95% CI −6.6 to −2.2). Treat this cell with care.</p>
      {/if}
      <p class="note">Mean probability of the social-media value at four blocks, 400 test entities. Higher fill means the post wins more often.</p>
    </div>
  </div>
</section>

<style>
  .wrap { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: var(--s6); align-items: center; }
  @media (max-width: 820px) { .wrap { grid-template-columns: minmax(0, 1fr); } }
  .grid { display: grid; grid-template-columns: 96px repeat(2, minmax(0, 1fr)); gap: var(--s3); align-items: center; }
  .ch { color: var(--ink3); font: var(--type-head); text-align: center; }
  .rh { color: var(--ink2); font: var(--type-control); }
  .tile { position: relative; aspect-ratio: 1.3; max-width: 100%; background: var(--wash); border-radius: var(--r-box); overflow: hidden; display: grid; place-items: center; }
  .fill { position: absolute; left: 0; right: 0; bottom: 0; background: color-mix(in oklch, var(--tint) 40%, var(--wash)); transition: height var(--glide) var(--ease); }
  .big { position: relative; font: 600 32px/36px var(--display); letter-spacing: -.02em; }
  .facts { display: grid; gap: var(--s4); }
  dt { color: var(--ink3); font: var(--type-meta); }
  dd { margin: 0; font: 600 24px/32px var(--display); }
</style>
