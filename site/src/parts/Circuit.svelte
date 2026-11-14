<script lang="ts">
  import { Meter, MorphText, Segmented, Swap, Token, glide } from "@stdlogits/ledger";
  import { D, depth, headId } from "../lib";

  type Cond = "qwen7_F1C2" | "qwen7_F3C2" | "qwen7_F3C7" | "llama_F1C2" | "gemma_F1C2";
  let model = $state<"qwen7" | "llama" | "gemma">("qwen7");
  let qcond = $state<"F1C2" | "F3C2" | "F3C7">("F1C2");
  const cond = $derived<Cond>(model === "qwen7" ? (`qwen7_${qcond}` as Cond) : (`${model}_F1C2` as Cond));
  const A = $derived(D.attr[cond]);
  const NL = $derived<number>(A.L), NH = $derived<number>(A.H);
  const maxv = $derived(Math.max(...A.map.flat().map((v: number) => Math.max(0, v))));
  const top = $derived<[number, number][]>(A.top);
  const topSet = $derived(new Set(top.map(([l, h]) => headId(l, h))));
  const path = $derived(model === "qwen7" && qcond !== "F3C2" ? D.path[qcond === "F1C2" ? "F1C2" : "F3C7"] : null);

  let sel = $state<[number, number]>([18, 4]);
  $effect(() => { if (!topSet.has(headId(...sel))) sel = top[0]; });
  const selId = $derived(headId(...sel));

  // Graph geometry: depth down the page, a band per layer.
  const GW = 720, GH = 860, GT = 36, GB = 740;
  const gy = (l: number, kind = "h") => GT + ((l + (kind === "m" ? 0.5 : 0)) / NL) * (GB - GT);
  const recNodes = $derived.by(() => {
    const ws = top.map(([l, h]) => Math.max(0, A.map[l][h]));
    const mw = Math.max(...ws);
    const sorted = top.map(([l, h], i) => ({ l, h, w: ws[i] / mw })).sort((a, b) => a.l - b.l || a.h - b.h);
    const xs = [-120, -20, 80, 180];
    return sorted.map((n, i) => ({ ...n, id: headId(n.l, n.h), x: 420 + xs[i % 4] + (Math.floor(i / 4) % 2 ? 40 : 0), y: gy(n.l) }));
  });
  const sndNodes = $derived.by(() => {
    if (!path) return [];
    const pool = new Map(path.pool.map((p: any) => [`${p[0]}${p[1]}_${p[2]}`, p[3]]));
    const list = path.top.map((s: any) => ({ kind: s[0], l: s[1], h: s[2], v: pool.get(`${s[0]}${s[1]}_${s[2]}`) as number }));
    const mv = Math.max(...list.map((s: any) => s.v));
    const byRow = new Map<string, any[]>();
    for (const s of list) { const k = s.kind + s.l; byRow.set(k, [...(byRow.get(k) ?? []), s]); }
    const out: any[] = [];
    for (const [, row] of byRow) row.forEach((s, i) => out.push({ ...s, w: s.v / mv, id: s.kind === "m" ? `MLP${s.l}` : headId(s.l, s.h), x: (s.kind === "m" ? 120 : 250) + i * 90, y: gy(s.l, s.kind) }));
    return out;
  });
  const links = $derived.by(() => {
    const out: { id: string; d: string; w: number; c: string }[] = [];
    for (const s of sndNodes) {
      const later = recNodes.filter((r) => r.l * 2 > s.l * 2 + (s.kind === "m" ? 1 : 0));
      for (const r of later) out.push({ id: `${s.id}>${r.id}`, d: `M${s.x},${s.y} C${s.x + 140},${s.y} ${r.x - 120},${r.y - 30} ${r.x},${r.y}`, w: 0.6 + (2.6 * s.w) / Math.sqrt(later.length), c: depth(s.l, NL) });
    }
    return out;
  });
  const ans = { x: 470, y: 800 };

  // The selected head's evidence.
  const shares = $derived.by(() => {
    const conds: [string, string][] = model === "qwen7" ? [["qwen7_F1C2", "Copies · tables"], ["qwen7_F3C2", "Copies · documents"], ["qwen7_F3C7", "Reworded · documents"]] : [[cond, "Copies · tables"]];
    return conds.map(([c, name]) => {
      const M = D.attr[c];
      const all = M.map.flat().map((v: number) => Math.max(0, v)).sort((a: number, b: number) => b - a);
      const v = Math.max(0, M.map[sel[0]]?.[sel[1]] ?? 0);
      return { name, v, rank: all.indexOf(v) + 1, max: all[0], inTop: M.top.some(([l, h]: number[]) => l === sel[0] && h === sel[1]) };
    });
  });
  const att = $derived.by(() => {
    for (const c of [cond, "qwen7_F1C2", "qwen7_F3C7", "qwen7_F3C2"]) { const a = D.attr[c]?.att?.[selId]; if (a && (c === cond || model === "qwen7")) return { a, c }; }
    return null;
  });
  const SPANS = [
    { k: ["value_L"], name: "The post's value", strong: 2 },
    { k: ["label_L"], name: "The post's source label", strong: 1 },
    { k: ["block_L"], name: "Rest of the posts", strong: 0.5 },
    { k: ["value_H"], name: "The government's value", strong: -2 },
    { k: ["label_H"], name: "The government's label", strong: -1 },
    { k: ["block_H"], name: "Rest of the government block", strong: -0.5 },
    { k: ["opt_low", "opt_high"], name: "The two answer options", strong: 0 },
    { k: ["question"], name: "The question", strong: 0 },
    { k: ["bos_system", "chat_tail"], name: "System prompt and template", strong: 0 },
  ];
  const spans = $derived(att ? SPANS.map((s) => ({ ...s, v: s.k.reduce((a, k) => a + (att!.a[k] ?? 0), 0) })) : []);
  const fix = $derived(model === "qwen7" ? Object.entries({ "Copies fix (tables)": D.fixsets.F1, "Paraphrase heads": D.fixsets.F3C7, "Union of both": D.fixsets.U, "Paraphrase-only feeders": D.fixsets.Feeders }).filter(([, hs]) => (hs as number[][]).some(([l, h]) => l === sel[0] && h === sel[1])).map(([n]) => n) : []);
  const both = new Set(D.attr.qwen7_F1C2.top.map(([l, h]: number[]) => headId(l, h)).filter((id: string) => D.attr.qwen7_F3C7.top.some(([l, h]: number[]) => headId(l, h) === id)));
  const colTint = (l: number) => depth(l, NL);
</script>

<section class="part" id="circuit">
  <p class="kicker">Circuit</p>
  <h2>Where the answer flips: ten heads in the middle of the stack</h2>
  <p class="lede">Every attention head of the model, one column per layer, lit by how much of the repetition effect it carries. Ten heads, mid-to-late, carry most of it. None of them is an induction head.</p>
  <div class="controls">
    <div class="ctl"><span class="label">Model</span><Segmented label="Model" value={model} onchange={(v) => (model = v)} options={[{ value: "qwen7", label: "Qwen 7B" }, { value: "llama", label: "Llama 8B" }, { value: "gemma", label: "Gemma 12B" }]} /></div>
    {#if model === "qwen7"}
      <div class="ctl"><span class="label">Conflict</span><Segmented label="Conflict" value={qcond} onchange={(v) => (qcond = v)} options={[{ value: "F1C2", label: "Copies · tables" }, { value: "F3C2", label: "Copies · documents" }, { value: "F3C7", label: "Reworded · documents" }]} /></div>
    {/if}
  </div>

  <div class="atlas surface" style:--count={NL} use:glide={".col[aria-current='true']"}>
    <div class="plot" style:--count={NL}>
      {#each Array(NL) as _, l (l + "-" + NL)}
        <button class="col" type="button" aria-current={l === sel[0] ? "true" : undefined} aria-label="Layer {l}">
          {#each Array(NH) as _, h}
            {@const v = Math.max(0, A.map[l][h])}
            {@const id = headId(l, h)}
            <i
              role="button" tabindex="-1"
              class:top={topSet.has(id)} class:on={id === selId}
              style:--c={colTint(l)} style:--o={Math.max(0.07, Math.sqrt(v / maxv))} style:--l={l}
              data-tip-title={id} data-tip={topSet.has(id) ? `One of the top ten · attribution ${(v * 100).toFixed(1)}% of the gap` : `Attribution ${(v * 100).toFixed(1)}% of the gap`}
              onclick={() => topSet.has(id) && (sel = [l, h])}
            ></i>
          {/each}
        </button>
      {/each}
    </div>
    <div class="nums" aria-hidden="true">{#each Array(NL) as _, l}<span class:shown={l % 4 === 0 || l === sel[0]}>{l}</span>{/each}</div>
    <p class="axisnote">layer → · each bar is one head · click a lit head with an outline to open it</p>
  </div>

  <div class="panes">
    <div class="graph surface figure">
      <svg viewBox="0 0 {GW} {GH}" role="img" aria-label="Decision heads and the parts that feed them">
        {#each Array(Math.ceil(NL / 4)) as _, i}
          {@const l = i * 4}
          <line class="gridline" x1="30" x2={GW - 10} y1={gy(l)} y2={gy(l)} />
          <text class="axis" x="4" y={gy(l) + 4}>L{l}</text>
        {/each}
        <g class="links">
          {#each links as e (e.id)}<path d={e.d} stroke={e.c} stroke-opacity=".35" stroke-width={e.w} fill="none" />{/each}
          {#each recNodes as r (r.id)}
            <path d="M{r.x},{r.y} C{r.x},{r.y + 110} {ans.x},{ans.y - 110} {ans.x},{ans.y}" stroke={depth(r.l, NL)} stroke-opacity=".55" stroke-width={1 + 8 * r.w} fill="none" />
          {/each}
        </g>
        {#each sndNodes as s (s.id)}
          <g class="node" style:transform="translate({s.x}px, {s.y}px)" data-tip-title={s.id} data-tip="Upstream {s.kind === 'm' ? 'MLP block' : 'head'} · mediates {s.v.toFixed(2)} logits into the decision heads (selection split)">
            {#if s.kind === "m"}<circle r={4 + 6 * Math.sqrt(s.w)} fill={depth(s.l, NL)} />{:else}<rect x={-(6 + 10 * s.w)} y="-4" width={12 + 20 * s.w} height="8" rx="4" fill={depth(s.l, NL)} />{/if}
            <text x={s.kind === "m" ? -14 : -(10 + 12 * s.w)} y="4" text-anchor="end" class="nl">{s.id}</text>
          </g>
        {/each}
        {#each recNodes as r (r.id)}
          <g class="node rec" class:on={r.id === selId} style:transform="translate({r.x}px, {r.y}px)" role="button" tabindex="0"
             onclick={() => (sel = [r.l, r.h])} onkeydown={(e) => e.key === "Enter" && (sel = [r.l, r.h])}
             data-tip-title={r.id} data-tip="Decision head{model === 'qwen7' && both.has(r.id) ? ' · top ten for copies and for rewording' : ''}">
            {#if r.id === selId}<rect x={-(12 + 18 * r.w) - 5} y="-11" width={2 * (12 + 18 * r.w) + 10} height="22" rx="11" fill="var(--wash2)" />{/if}
            <rect x={-(12 + 18 * r.w)} y="-6" width={2 * (12 + 18 * r.w)} height="12" rx="6" fill={depth(r.l, NL)} />
            {#if model === "qwen7" && both.has(r.id)}<rect x={-(12 + 18 * r.w) - 3} y="-9" width={2 * (12 + 18 * r.w) + 6} height="18" rx="9" fill="none" stroke="var(--ink)" stroke-width="1.2" />{/if}
            <text y="24" text-anchor="middle" class="nl b">{r.id}</text>
          </g>
        {/each}
        <g transform="translate({ans.x}, {ans.y})">
          <rect x="-150" y="0" width="300" height="28" rx="14" fill="var(--paper)" />
          <text y="18" text-anchor="middle" class="nl">answer: the government's value or the post's</text>
        </g>
        {#if path}
          <text x="30" y="14" class="axis">Ten upstream parts carry {Math.round(path.mediated * 100)}% of the decision heads' effect (random sets of ten: at most {Math.round(path.p95 * 100)}%)</text>
        {:else}
          <text x="30" y="14" class="axis">{model === "qwen7" ? "Upstream path patching was run for copies in tables and rewording in documents" : "Upstream path patching was run on Qwen 7B only"}</text>
        {/if}
      </svg>
    </div>

    <aside class="panel surface" style:--tint={depth(sel[0], NL)}>
      <Swap key={cond + selId}>
        <div class="pin">
          <div class="title"><h3 class="num"><MorphText text={selId} /></h3><span class="sw"></span></div>
          <p class="meta">Layer {sel[0]} of {NL} · head {sel[1]}{model === "qwen7" && both.has(selId) ? " · in both circuits" : ""}</p>

          <div class="grp">
            <p class="gh" data-tip="Attribution patching on held-out test pairs: this head's share of the logit gap between one post and four copies. Shares overlap, so they can add up past 100%.">Share of the repetition effect</p>
            {#each shares as s}
              <div class="mrow">
                <span>{s.name}</span>
                <Meter value={s.v / s.max} label="{(s.v * 100).toFixed(1)}% of the gap" width={48} />
                <span class="num">{(s.v * 100).toFixed(1)}%</span>
                <span class="rk num">#{s.rank}</span>
              </div>
            {/each}
          </div>

          {#if att}
            <div class="grp">
              <p class="gh" data-tip="Attention from the answer position, averaged over 24 held-out prompts with four copies of the post">Where it looks from the answer position</p>
              <div class="look">
                {#each spans.filter((s) => s.v > 0.004) as s}
                  <span class="seg" class:lo={s.strong > 0} class:hi={s.strong < 0} style:flex-grow={s.v} data-tip-title={s.name} data-tip="{(s.v * 100).toFixed(1)}% of its attention"></span>
                {/each}
              </div>
              <div class="tokens">
                {#each spans.filter((s) => s.v > 0.02).sort((a, b) => b.v - a.v).slice(0, 5) as s}
                  <Token token={s.name} value={s.v} />
                {/each}
              </div>
            </div>
          {/if}

          {#if fix.length}
            <div class="grp">
              <p class="gh" data-tip="Head sets the training-free fix damps (section Fixes)">Damped by</p>
              <div class="tokens">{#each fix as f}<Token token={f} />{/each}</div>
            </div>
          {/if}
        </div>
      </Swap>
    </aside>
  </div>
  <p class="note">Attribution patching ranks heads on 200 selection entities; every number shown is on held-out test entities. Patching covers the shared question tokens only, so it shows where the decision is read out, not where copies are first compared inside the documents.</p>
</section>

<style>
  .atlas { --glide-bg: var(--wash2); --glide-radius: var(--r-col); position: relative; display: grid; gap: var(--s2); }
  .plot { display: grid; grid-template-columns: repeat(var(--count), minmax(0, 1fr)); gap: 2px; }
  .col { display: grid; gap: 2px; padding: 4px 1px; border: 0; background: none; border-radius: var(--r-col); cursor: default; position: relative; z-index: 1; }
  .col i { display: block; height: 5px; border-radius: 3px; background: color-mix(in oklch, var(--c) calc(var(--o) * 100%), var(--wash)); animation: arrive 400ms var(--ease) both; animation-delay: calc(var(--l) * 16ms); }
  .col i.top { outline: 1.5px solid var(--ink); outline-offset: 0; cursor: pointer; }
  .col i.on { outline-width: 2.5px; }
  @keyframes arrive { from { opacity: 0; transform: translateY(4px); } }
  @media (prefers-reduced-motion: reduce) { .col i { animation: none; } }
  .nums { display: grid; grid-template-columns: repeat(var(--count), minmax(0, 1fr)); color: var(--ink3); font: var(--type-meta); text-align: center; }
  .nums span { visibility: hidden; } .nums span.shown { visibility: visible; }
  .axisnote { color: var(--ink3); font: var(--type-meta); }
  .panes { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr); gap: var(--s5); align-items: start; }
  @media (max-width: 900px) { .panes { grid-template-columns: minmax(0, 1fr); } }
  .graph svg { min-width: 560px; }
  .node { transition: transform 420ms var(--ease); }
  .node.rec { cursor: pointer; }
  .nl { fill: var(--ink2); font: var(--type-meta); }
  .nl.b { fill: var(--ink); font-weight: 600; }
  .panel { position: sticky; top: 80px; }
  .pin { display: grid; gap: var(--s5); }
  .title { display: flex; align-items: center; gap: var(--s3); }
  h3 { font: var(--type-heading); letter-spacing: -.02em; }
  .sw { width: 28px; height: 10px; border-radius: 5px; background: var(--tint); }
  .meta { color: var(--ink3); font: var(--type-small); margin-top: calc(var(--s4) * -1); }
  .grp { display: grid; gap: var(--s2); }
  .gh { color: var(--ink3); font: var(--type-head); }
  .mrow { display: grid; grid-template-columns: minmax(0, 1fr) auto 52px 32px; align-items: center; gap: var(--s2); font: var(--type-small); }
  .rk { color: var(--ink3); font: var(--type-meta); text-align: right; }
  .look { display: flex; gap: 2px; height: 14px; }
  .seg { min-width: 3px; border-radius: 4px; background: var(--wash2); }
  .seg.lo { background: var(--tint); }
  .seg.hi { background: var(--ink2); }
  .tokens { display: flex; flex-wrap: wrap; gap: var(--s1); }
</style>
