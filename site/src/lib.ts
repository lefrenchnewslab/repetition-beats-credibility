// Everything the page draws comes from data.json, built from the sprint's saved results (results/, numbers_*.json).
import raw from "./data.json";

export const D = raw as any;

export type ModelKey = "qwen7" | "llama" | "gemma" | "qwen32";
export type Fmt = "F1" | "F2" | "F3";

export const MODELS: { key: ModelKey; name: string; short: string; layers: number }[] = [
  { key: "qwen32", name: "Qwen2.5-32B-Instruct", short: "Qwen 32B", layers: 64 },
  { key: "gemma", name: "Gemma-3-12B-it", short: "Gemma 12B", layers: 48 },
  { key: "llama", name: "Llama-3.1-8B-Instruct", short: "Llama 8B", layers: 32 },
  { key: "qwen7", name: "Qwen2.5-7B-Instruct", short: "Qwen 7B", layers: 28 },
];
export const modelName = (k: ModelKey) => MODELS.find((m) => m.key === k)!.short;

export const FORMATS: { value: Fmt; label: string; hint: string }[] = [
  { value: "F1", label: "Tables", hint: "Each source as a small table, after Schuster et al." },
  { value: "F2", label: "News snippets", hint: "Each source as a news excerpt: “According to …”" },
  { value: "F3", label: "Retrieved documents", hint: "Each source as a retrieved passage, as in a RAG prompt" },
];

/** Depth ramp, Ledger's way: --deep at the first layer, --late at the last, mixed in oklch. */
export const depth = (layer: number, layers: number) =>
  `color-mix(in oklch, var(--deep) ${Math.round((1 - layer / Math.max(1, layers - 1)) * 100)}%, var(--late))`;

/** P_low with its 95% CI, or null when that condition was not run. */
export function p(model: ModelKey, fmt: Fmt, ck: string): [number, number | null, number | null] | null {
  return D.beh?.[model]?.[fmt]?.[ck] ?? null;
}

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

export type Pattern = "C2" | "C3" | "C7" | "C8" | "C5";
export const PATTERNS: { value: Pattern; label: string; hint: string }[] = [
  { value: "C2", label: "One user, copied", hint: "The same post pasted again, word for word" },
  { value: "C7", label: "One user, reworded", hint: "The same user's claim rewritten in a new template each time" },
  { value: "C3", label: "Many users, same words", hint: "Different usernames posting identical text" },
  { value: "C8", label: "Many users, reworded", hint: "Different usernames, each in a different template" },
  { value: "C5", label: "Filler", hint: "One post, padded with blocks about other entities to the same length (length control)" },
];

/** Which copy counts a pattern was run at, per format. k = 1 is the single conflict, C1. */
export function ks(fmt: Fmt, pattern: Pattern): number[] {
  const out = [1];
  for (const k of [2, 4, 8]) {
    const has = MODELS.some((m) => p(m.key, fmt, `${pattern}k${k}`));
    const prompt = D.prompts[`${fmt}|${pattern}k${k}`];
    if (has && prompt) out.push(k);
  }
  return out;
}

export const ck = (pattern: Pattern, k: number) => (k === 1 ? "C1k1" : `${pattern}k${k}`);

export interface Block { kind: "gov" | "social" | "filler" | "intro" | "question"; text: string }

/** Splits a stored prompt into its blocks and names each by what it holds. */
export function blocks(prompt: string): Block[] {
  const parts = prompt.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean);
  return parts.map((text) => {
    if (/^(What|Question)/.test(text) || /\(A\)/.test(text)) return { kind: "question", text };
    if (/^(Context:|News excerpts:|Retrieved documents:)$/.test(text)) return { kind: "intro", text };
    if (/User @|social media/i.test(text)) return { kind: "social", text };
    if (/Ministry|Office|Bureau|Agency|Department|Statistics|Registry|Authority/.test(text)) return { kind: "gov", text };
    return { kind: "filler", text };
  });
}

/** The two answer values from a prompt's options: (A) is the government's in every stored example shown here. */
export function values(prompt: string): { high: string; low: string } {
  const a = prompt.match(/\(A\) (.+)/)?.[1]?.trim() ?? "";
  const b = prompt.match(/\(B\) (.+)/)?.[1]?.trim() ?? "";
  return { high: a, low: b };
}

export const headId = (l: number, h: number) => `L${l}H${h}`;
