# Repetition Beats Credibility: interactive companion

A one-page Svelte 5 site drawn in Ledger (`@stdlogits/ledger`, the design system of neuron-lens.com).
Every number comes from `src/data.json`, built from the sprint's saved results (`results/`, `numbers_*.json`, the paper's appendix tables).

    npm install
    npm run dev      # http://localhost:5199
    npm run build    # dist/index.html, one self-contained file

Ledger is AGPL-3.0-or-later: a publicly deployed page must publish its source (this folder).

This folder is AGPL-3.0-or-later because it bundles @stdlogits/ledger; the rest of the repo keeps its own licence.
