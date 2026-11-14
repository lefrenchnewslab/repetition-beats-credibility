<script lang="ts">
  const BIB = `@misc{sobhy2026repetition,
  title  = {Repetition Beats Credibility: How Language Models Let Reworded Claims Outvote Official Sources},
  author = {Sobhy, Amr},
  year   = {2026},
  note   = {AI x Epistemics Research Sprint, Apart Research},
  url    = {https://github.com/lefrenchnewslab/repetition-beats-credibility}
}`;
  const TEXT = "Amr Sobhy. 2026. Repetition Beats Credibility: How Language Models Let Reworded Claims Outvote Official Sources. AI x Epistemics Research Sprint, Apart Research.";
  let copied = $state<"" | "bib" | "text">("");
  let pre: HTMLPreElement;
  async function copy(which: "bib" | "text") {
    try { await navigator.clipboard.writeText(which === "bib" ? BIB : TEXT); copied = which; setTimeout(() => (copied = ""), 1600); }
    catch { const r = document.createRange(); r.selectNodeContents(pre); const s = getSelection(); s?.removeAllRanges(); s?.addRange(r); }
  }
</script>

<section class="part" id="cite">
  <p class="kicker">Cite</p>
  <h2>Cite this work</h2>
  <p class="cite-text">{TEXT}</p>
  <div class="surface box">
    <pre bind:this={pre}>{BIB}</pre>
    <div class="actions">
      <button type="button" onclick={() => copy("bib")}>{copied === "bib" ? "Copied" : "Copy BibTeX"}</button>
      <button type="button" onclick={() => copy("text")}>{copied === "text" ? "Copied" : "Copy citation"}</button>
    </div>
  </div>
</section>

<style>
  .cite-text { color: var(--ink2); font: var(--type-body); max-width: 72ch; }
  .box { display: grid; gap: var(--s4); }
  pre { margin: 0; overflow-x: auto; white-space: pre; color: var(--ink2); font: 400 13px/20px var(--text); font-variant-numeric: tabular-nums; }
  .actions { display: flex; flex-wrap: wrap; gap: var(--s2); }
  button { height: 36px; padding: 0 var(--s4); border: 0; border-radius: var(--r-pill); background: var(--paper); color: var(--ink); font: var(--type-control); cursor: pointer; transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  button:active { scale: .97; }
  @media (hover: hover) { button:hover { background: var(--wash2); } }
  button:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
</style>
