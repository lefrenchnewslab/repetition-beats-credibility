<script lang="ts">
  import { MorphText } from "@stdlogits/ledger";
  import { onMount } from "svelte";

  // Qwen2.5-32B, retrieved documents, 400 held-out entities: entities whose P_low, averaged over 8 variants, exceeds 0.5.
  const cases = [
    { label: "One post", sub: "C1", n: 0, copies: 0, kind: "one", tip: "One government passage against one social-media post" },
    { label: "Four exact copies", sub: "C2 k=4", n: 11, copies: 4, kind: "copy", tip: "The same post pasted four times, word for word" },
    { label: "Four users, same words", sub: "C3 k=4", n: 206, copies: 4, kind: "users", tip: "Four different usernames posting identical text" },
    { label: "Four reworded copies", sub: "C7 k=4", n: 342, copies: 4, kind: "reword", tip: "One user's claim rewritten four ways" },
  ];
  let shown = $state(false);
  onMount(() => requestAnimationFrame(() => (shown = true)));
</script>

<section class="hero part">
  <p class="kicker">Apart AI x Epistemics sprint · open-weight models, 7B to 32B</p>
  <h1>Repetition beats credibility</h1>
  <p class="byline"><span class="au">Amr Sobhy</span><span class="sep" aria-hidden="true">·</span><span>Le French News Lab</span></p>
  <p class="lede">
    Reword one social-media post four times and Qwen2.5-32B sides with it against a government source on
    <b>86%</b> of items. Paste the same post four times and it holds firm: <b>3%</b>.
    The fight is settled in a handful of attention heads that weigh both signals at once.
  </p>

  <div class="stacks" style:--tint="color-mix(in oklch, var(--deep) 35%, var(--late))">
    {#each cases as c, i}
      <div class="stack rise" style:--i={i} data-tip="{c.tip} · {c.n} of 400 entities" data-tip-title={c.label}>
        <div class="glyphs" aria-hidden="true">
          <i class="gov"></i>
          {#each Array(Math.max(1, c.copies)) as _, j}
            <i class="soc {c.kind}" style:--j={j}></i>
          {/each}
        </div>
        <div class="bar"><span style:height={shown ? `${(c.n / 400) * 100}%` : "0%"}></span></div>
        <div class="val num"><MorphText text={shown ? `${Math.round((c.n / 400) * 100)}%` : "0%"} /></div>
        <div class="lab">{c.label}</div>
      </div>
    {/each}
  </div>
  <p class="note">Share of 400 held-out fictional entities on which Qwen2.5-32B picks the social-media value, retrieved-document format. Every number on this page comes from the paper's saved results.</p>
</section>

<style>
  .hero { padding-top: var(--s7); }
  h1 { font: var(--type-title); letter-spacing: -.045em; max-width: 14ch; text-wrap: balance; }
  .byline { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s2); margin-top: calc(var(--s3) * -1); color: var(--ink2); font: var(--type-lead); }
  .byline .au { color: var(--ink); font-weight: 600; }
  .byline .sep { color: var(--ink3); }
  .lede b { color: var(--ink); font-weight: 600; }
  .stacks { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--s4); margin-top: var(--s4); }
  .stack { display: grid; gap: var(--s3); align-content: end; padding: var(--s5) var(--s4) var(--s4); background: var(--wash); border-radius: var(--r-box); }
  .glyphs { display: flex; flex-wrap: wrap; gap: 4px; min-height: 30px; align-items: end; }
  .glyphs i { display: block; width: 22px; height: 12px; border-radius: 4px; }
  .gov { background: var(--ink); }
  .soc { background: color-mix(in oklch, var(--tint) 55%, var(--wash)); }
  .soc.reword:nth-child(odd) { width: 16px; }
  .soc.reword:nth-child(3n) { width: 26px; }
  .soc.users { border-radius: 999px; width: 12px; }
  .bar { height: 150px; display: flex; align-items: end; background: var(--paper); border-radius: var(--r-col); overflow: hidden; }
  .bar span { display: block; width: 100%; background: var(--tint); border-radius: var(--r-col); transition: height 900ms var(--ease); }
  .val { font: 600 32px/36px var(--display); letter-spacing: -.02em; }
  .lab { color: var(--ink2); font: var(--type-control); }
  @media (max-width: 720px) { .stacks { grid-template-columns: repeat(2, minmax(0, 1fr)); } .bar { height: 90px; } h1 { font: 600 40px/1.05 var(--display); } }
</style>
