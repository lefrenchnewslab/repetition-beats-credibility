import { defineConfig, type Plugin } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { viteSingleFile } from "vite-plugin-singlefile";
import { readFileSync } from "node:fs";

// Inline Ledger's pre-paint theme script so the built page is one file that opens from disk.
function inlinePreflight(): Plugin {
  return {
    name: "inline-preflight",
    apply: "build",
    transformIndexHtml(html) {
      const js = readFileSync("public/theme-preflight.js", "utf8");
      return html.replace(
        /<script src="\/theme-preflight\.js"([^>]*)><\/script>/,
        (_m, attrs) => `<script${attrs}>${js}</script>`,
      );
    },
  };
}

export default defineConfig({
  plugins: [svelte(), inlinePreflight(), viteSingleFile()],
  // Ledger ships .svelte.ts runes modules: serve it as source, don't pre-bundle it.
  optimizeDeps: { exclude: ["@stdlogits/ledger"] },
  build: { target: "es2022", assetsInlineLimit: 100_000_000, copyPublicDir: true },
  server: { port: 5199 },
});
