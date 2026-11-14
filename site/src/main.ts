import "@stdlogits/ledger/ledger.css";
import "./site.css";
import { mount } from "svelte";
import App from "./App.svelte";
mount(App, { target: document.getElementById("app")! });
