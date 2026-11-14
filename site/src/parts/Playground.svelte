<script lang="ts">
  import { MorphText, Segmented, Swap } from "@stdlogits/ledger";
  import { D, FORMATS, MODELS, PATTERNS, blocks, ck, ks, p, values, type Fmt, type Pattern } from "../lib";

  let fmt = $state<Fmt>("F3");
  let pattern = $state<Pattern>("C7");
  let k = $state(4);

  const pat = $derived<Pattern>(ks(fmt, pattern).length > 1 ? pattern : "C7");
  const avail = $derived(ks(fmt, pat));
  const kk = $derived(avail.includes(k) ? k : avail[avail.length - 1]);
  const key = $derived(ck(pat, kk));
  const prompt = $derived<string>(D.prompts[`${fmt}|${key}`] ?? D.prompts[`${fmt}|C1k1`]);
  const bl = $derived(blocks(prompt));
  const vals = $derived(values(prompt));

  // Marks the two answer values inside a block: the government's in ink, the post's in the tint.
  function marked(text: string) {
    const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const parts: { t: string; m?: "hi" | "lo" }[] = [];
    if (!vals.high || !vals.low) return [{ t: text }];
    const re = new RegExp(`(${esc(vals.high)}|${esc(vals.low)})`, "g");
    let last = 0;
    for (const m of text.matchAll(re)) {
      if (m.index! > last) parts.push({ t: text.slice(last, m.index) });
      parts.push({ t: m[0], m: m[0] === vals.high ? "hi" : "lo" });
      last = m.index! + m[0].length;
    }
    if (last < text.length) parts.push({ t: text.slice(last) });
    return parts;
  }
  // A block's text as runs of prose and markdown tables, so tables render as tables.
  type Piece = { kind: "text"; text: string } | { kind: "table"; head: string[]; rows: string[][] };
  function pieces(text: string): Piece[] {
    const out: Piece[] = [];
    const cells = (ln: string) => ln.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    let tbl: string[][] | null = null;
    let txt: string[] = [];
    const flushText = () => { if (txt.length) out.push({ kind: "text", text: txt.join("\n") }); txt = []; };
    const flushTable = () => { if (tbl && tbl.length) out.push({ kind: "table", head: tbl[0], rows: tbl.slice(1) }); tbl = null; };
    for (const ln of text.replace(/^Context:\n/, "").split("\n")) {
      if (ln.trim().startsWith("|")) {
        flushText();
        if (/^\|[\s:|-]+\|?$/.test(ln.trim())) continue;
        (tbl ??= []).push(cells(ln));
      } else { flushTable(); txt.push(ln); }
    }
    flushTable(); flushText();
    return out;
  }
  const label = { gov: "Government source", social: "Social media", filler: "Filler", intro: "", question: "Question" } as const;
  const rows = $derived(MODELS.map((m) => ({ m, v: p(m.key, fmt, key) })));
</script>

<section class="part" id="try">
  <p class="kicker">Try it</p>
  <h2>Stack copies of one post and watch four models change their minds</h2>
  <p class="lede">These are the real prompts. Each item states one attribute of a fictional entity twice: once from a government source and once from a social-media user. Add copies and see which value each model picks.</p>

  <div class="controls">
    <div class="ctl"><span class="label">Format</span><Segmented label="Format" value={fmt} onchange={(v) => (fmt = v)} options={FORMATS} /></div>
    <div class="ctl"><span class="label">Repeated as</span><Segmented label="Repeated as" value={pat} onchange={(v) => (pattern = v)} options={PATTERNS.filter((x) => ks(fmt, x.value).length > 1)} /></div>
    <div class="ctl">
      <span class="label">Copies</span>
      <Segmented label="Copies" value={String(kk)} onchange={(v) => (k = Number(v))} options={avail.map((n) => ({ value: String(n), label: n === 1 ? "1 (one post)" : String(n) }))} />
    </div>
  </div>

  <div class="play">
    <div class="prompt surface" aria-label="The prompt the models read">
      <Swap key={fmt + key}>
        <div class="blocks">
          {#each bl as b, i}
            {#if b.kind !== "intro"}
              <div class="blk {b.kind} rise" style:--i={i}>
                {#if label[b.kind]}<span class="tag">{label[b.kind]}</span>{/if}
                {#each pieces(b.text) as pc}
                  {#if pc.kind === "text"}
                    <pre>{#each marked(pc.text) as s}{#if s.m}<mark class={s.m}>{s.t}</mark>{:else}{s.t}{/if}{/each}</pre>
                  {:else}
                    <table>
                      <thead><tr>{#each pc.head as c}<th>{c}</th>{/each}</tr></thead>
                      <tbody>{#each pc.rows as r}<tr>{#each r as c, ci}<td class:key={ci === 0}>{#each marked(c) as s}{#if s.m}<mark class={s.m}>{s.t}</mark>{:else}{s.t}{/if}{/each}</td>{/each}</tr>{/each}</tbody>
                    </table>
                  {/if}
                {/each}
              </div>
            {/if}
          {/each}
        </div>
      </Swap>
    </div>

    <div class="verdicts">
      <p class="vh">Probability of the social-media value</p>
      {#each rows as r (r.m.key)}
        <div class="row" data-tip-title={r.m.name} data-tip={r.v ? (r.v[1] != null ? `95% CI ${r.v[1].toFixed(3)} to ${r.v[2]!.toFixed(3)}, 400 test entities × 8 variants` : "Mean over 400 test entities × 8 variants") : "Not run in this format"}>
          <span class="name">{r.m.short}</span>
          <span class="track">
            <i class="half"></i>
            <span class="fill" class:over={r.v && r.v[0] > 0.5} style:width={r.v ? `${r.v[0] * 100}%` : "0%"}></span>
          </span>
          <span class="v num">{#if r.v}<MorphText text={r.v[0].toFixed(2)} />{:else}<span class="na">not run</span>{/if}</span>
          <span class="who">{#if r.v}{r.v[0] > 0.5 ? "sides with the post" : "sides with the government"}{/if}</span>
        </div>
      {/each}
      <p class="note">0.5 is a coin flip. Values average 8 variants per entity that swap block order, answer letter and which value is repeated, so position cannot explain them.</p>
    </div>
  </div>
</section>

<style>
  .play { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: var(--s5); align-items: start; }
  @media (max-width: 900px) { .play { grid-template-columns: minmax(0, 1fr); } }
  .prompt { max-height: 640px; overflow: auto; }
  .blocks { display: grid; gap: var(--s2); }
  .blk { display: grid; gap: var(--s1); padding: var(--s3) var(--s4); border-radius: var(--r-col); background: var(--paper); }
  .blk.social { background: color-mix(in oklch, var(--tint) 9%, var(--paper)); }
  .blk.filler { opacity: .7; }
  .blk.question { background: transparent; padding-inline: 0; }
  .tag { color: var(--ink3); font: var(--type-head); }
  table { border-collapse: separate; border-spacing: 0; width: 100%; max-width: 420px; font: 400 13px/18px var(--text); color: var(--ink2); font-variant-numeric: tabular-nums; }
  th { text-align: left; color: var(--ink3); font: var(--type-meta); padding: 2px 10px 4px 0; }
  td { padding: 3px 10px 3px 0; }
  tbody tr:nth-child(odd) td { background: color-mix(in oklab, var(--wash) 60%, transparent); }
  td:first-child, th:first-child { padding-left: 8px; border-radius: 6px 0 0 6px; }
  td:last-child { border-radius: 0 6px 6px 0; }
  td.key { color: var(--ink3); width: 45%; }
  pre { margin: 0; white-space: pre-wrap; word-break: break-word; font: 400 13px/19px var(--text); color: var(--ink2); }
  mark { border-radius: 4px; padding: 0 2px; color: var(--ink); font-weight: 600; }
  mark.lo { background: color-mix(in oklch, var(--tint) 26%, transparent); }
  mark.hi { background: var(--wash2); }
  .verdicts { display: grid; gap: var(--s3); position: sticky; top: 80px; }
  .vh { color: var(--ink3); font: var(--type-head); }
  .row { display: grid; grid-template-columns: 84px minmax(0, 1fr) 48px; grid-template-rows: auto auto; column-gap: var(--s3); align-items: center; padding: var(--s3) var(--s4); background: var(--wash); border-radius: var(--r-row); }
  .name { font: var(--type-label); }
  .track { position: relative; height: 10px; background: var(--paper); border-radius: 999px; overflow: hidden; }
  .half { position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: var(--wash2); z-index: 1; }
  .fill { position: absolute; left: 0; top: 0; bottom: 0; background: color-mix(in oklch, var(--tint) 45%, var(--wash2)); border-radius: 999px; transition: width var(--glide) var(--ease), background-color var(--glide) var(--ease); }
  .fill.over { background: var(--tint); }
  .v { font: 600 18px/24px var(--display); text-align: right; }
  .who { grid-column: 2 / 4; color: var(--ink3); font: var(--type-meta); min-height: 16px; }
  .na { color: var(--ink3); font: var(--type-meta); }
</style>
