<script lang="ts">
  import { onMount } from "svelte";
  import { ThemeButton, createTheme, tips } from "@stdlogits/ledger";
  import Hero from "./parts/Hero.svelte";
  import Playground from "./parts/Playground.svelte";
  import Dose from "./parts/Dose.svelte";
  import Discount from "./parts/Discount.svelte";
  import Circuit from "./parts/Circuit.svelte";
  import Jobs from "./parts/Jobs.svelte";
  import Probe from "./parts/Probe.svelte";
  import Fixes from "./parts/Fixes.svelte";
  import Prereg from "./parts/Prereg.svelte";

  const theme = createTheme({ key: "rbc.theme" });
  onMount(() => theme.start());
</script>

<div class="site" use:tips>
  <header>
    <a class="mark" href="#top">Repetition Beats Credibility</a>
    <nav>
      <a href="#try">Try it</a>
      <a href="#copies">Copies</a>
      <a href="#circuit">Circuit</a>
      <a href="#signal">Signal</a>
      <a href="#fixes">Fixes</a>
      <a href="#record">Record</a>
      <ThemeButton {theme} />
    </nav>
  </header>

  <main id="top">
    <Hero />
    <Playground />
    <Dose />
    <Discount />
    <Circuit />
    <Jobs />
    <Probe />
    <Fixes />
    <Prereg />
  </main>

  <footer>
    <span>Amr Sobhy · Le French News Lab · Apart AI x Epistemics sprint, November 2026</span>
    <a href="https://github.com/lefrenchnewslab/repetition-beats-credibility">Code and data</a>
    <span data-tip="Built on Ledger, the design language of neuron-lens.com">Drawn in Ledger</span>
  </footer>
</div>
