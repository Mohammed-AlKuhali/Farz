// Checks that reports_v2/localcal_spec.md runs as written in the app's runtime: onnxruntime-web (wasm, 1 thread, from
// app/node_modules, read only) on the demo calibration crops -> prototypes, leave-one-out margins, m_hi, exactly per the
// spec, compared with the Python values in models_v2/localcal_demo/manifest.json (computed with Python onnxruntime).
//   node src/beans_v2/localcal_spec_check.mjs
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const require = createRequire(path.join(ROOT, "app/package.json"));
const ort = require("onnxruntime-web");
const { PNG } = require("pngjs");
const DEMO = path.join(ROOT, "models_v2/localcal_demo");
const man = JSON.parse(readFileSync(path.join(DEMO, "manifest.json"), "utf8"));
const S = 128, N = man.calibration.length;
const x = new Float32Array(N * 3 * S * S);
man.calibration.forEach((it, n) => {
  const png = PNG.sync.read(readFileSync(path.join(DEMO, it.file)));
  for (let y = 0; y < S; y++) for (let xx = 0; xx < S; xx++) {
    const i = (y * S + xx) * 4;
    for (let c = 0; c < 3; c++) x[n * 3 * S * S + c * S * S + y * S + xx] = png.data[i + c] / 255;
  }
});
ort.env.wasm.numThreads = 1;
const sess = await ort.InferenceSession.create(readFileSync(path.join(ROOT, man.head_from_shipped_model.model)), { executionProviders: ["wasm"] });
const out = await sess.run({ image: new ort.Tensor("float32", x, [N, 3, S, S]) });
const D = out.embed.dims[1], E = out.embed.data;
const emb = (i) => Array.from(E.slice(i * D, (i + 1) * D));
const norm = (v) => { const n = Math.sqrt(v.reduce((a, b) => a + b * b, 0)); return v.map((t) => t / n); };
const dot = (a, b) => a.reduce((s, t, k) => s + t * b[k], 0);
const lab = man.calibration.map((c) => (c.label === "defect" ? 1 : 0));
const Sg = new Array(D).fill(0), Sd = new Array(D).fill(0);
for (let i = 0; i < N; i++) { const e = emb(i); const T = lab[i] ? Sd : Sg; for (let k = 0; k < D; k++) T[k] += e[k]; }
const muG = norm(Sg), muD = norm(Sd);
const loo = [], wrong = [];
for (let i = 0; i < N; i++) {
  const e = emb(i);
  const g = lab[i] ? muG : norm(Sg.map((t, k) => t - e[k]));
  const d = lab[i] ? norm(Sd.map((t, k) => t - e[k])) : muD;
  const s = dot(e, d) - dot(e, g); loo.push(s); wrong.push((s >= 0 ? 1 : 0) !== lab[i]);
}
const abs = loo.map(Math.abs).sort((a, b) => a - b);
const median = abs.length % 2 ? abs[(abs.length - 1) / 2] : (abs[abs.length / 2 - 1] + abs[abs.length / 2]) / 2;
const maxWrong = Math.max(0, ...loo.filter((_, i) => wrong[i]).map(Math.abs));
const mHi = Math.max(maxWrong, median);
const py = man.head_from_shipped_model;
console.log(JSON.stringify({ runtime: `onnxruntime-web ${ort.env.versions?.web ?? "?"} wasm, Node ${process.version}`, n: N, embed_dim: D,
  m_hi_js: mHi, m_hi_python: py.m_hi, abs_diff_m_hi: Math.abs(mHi - py.m_hi),
  loo_errors_js: wrong.filter(Boolean).length, loo_errors_python: py.loo_errors,
  cosine_mu_good_js_vs_python: dot(muG, norm(py.mu_good)), cosine_mu_defect_js_vs_python: dot(muD, norm(py.mu_defect)) }));
