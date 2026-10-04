// Parity of the clean two-output ONNX files in onnxruntime-web (wasm, 1 thread; the package version the app ships,
// read from app/node_modules, never modified) against Python onnxruntime on the SAME float32 inputs.
//   node src/beans_v2/clean_wasm_parity.mjs <dir> <model.onnx> <tag>
// <dir> holds x.bin (float32 [N,3,128,128]), meta.json {n, d}, and py_<tag>_probs.bin / py_<tag>_embed.bin (float32).
// Prints one JSON line: max abs diff of probs and embed, argmax agreement, 3-way call agreement, min embed cosine.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const require = createRequire(path.join(ROOT, "app/package.json"));
const ort = require("onnxruntime-web");
const [dir, model, tag] = process.argv.slice(2);
const meta = JSON.parse(readFileSync(path.join(dir, "meta.json"), "utf8"));
const f32 = (f) => { const b = readFileSync(path.join(dir, f)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const x = f32("x.bin"), pyP = f32(`py_${tag}_probs.bin`), pyE = f32(`py_${tag}_embed.bin`);
const N = meta.n, D = meta.d, C = 6;
ort.env.wasm.numThreads = 1;
const t0 = Date.now();
const session = await ort.InferenceSession.create(readFileSync(model), { executionProviders: ["wasm"] });
const out = await session.run({ image: new ort.Tensor("float32", x, [N, 3, 128, 128]) });
const ms = Date.now() - t0;
const P = out.probs.data, E = out.embed.data;
const call = (p, tg, td) => (p[0] >= tg ? 0 : 1 - p[0] >= td ? 1 : 2);
let maxP = 0, maxE = 0, agree = 0, agree3 = 0, minCos = 1;
for (let n = 0; n < N; n++) {
  const a = Array.from(P.slice(n * C, n * C + C)), b = Array.from(pyP.slice(n * C, n * C + C));
  a.forEach((v, k) => { maxP = Math.max(maxP, Math.abs(v - b[k])); });
  if (a.indexOf(Math.max(...a)) === b.indexOf(Math.max(...b))) agree++;
  if (call(a, meta.t_good, meta.t_defect) === call(b, meta.t_good, meta.t_defect)) agree3++;
  let dot = 0, na = 0, nb = 0;
  for (let k = 0; k < D; k++) {
    const u = E[n * D + k], v = pyE[n * D + k];
    maxE = Math.max(maxE, Math.abs(u - v)); dot += u * v; na += u * u; nb += v * v;
  }
  minCos = Math.min(minCos, dot / Math.sqrt(na * nb));
}
console.log(JSON.stringify({ runtime: `onnxruntime-web ${ort.env.versions?.web ?? "?"} wasm numThreads=1, Node ${process.version}`,
  model: path.relative(ROOT, model), tag, n: N, embed_dim: out.embed.dims[1], output_names: session.outputNames,
  max_abs_prob_diff_vs_python_ort: maxP, max_abs_embed_diff_vs_python_ort: maxE, min_embed_cosine_vs_python_ort: minCos,
  argmax_agreement: agree / N, call3_agreement_at_pair: agree3 / N, wall_ms_load_and_run: ms }));
