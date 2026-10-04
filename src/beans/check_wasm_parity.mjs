// Parity check: onnxruntime-web (WASM backend, the same package version the app ships) vs Python onnxruntime.
// Reads tests/fixtures/expected_probs.json + crops/*.png, runs app/public/models/farz_beans.onnx with numThreads=1,
// reports the max absolute probability difference and argmax agreement. Run from anywhere:
//   node src/beans/check_wasm_parity.mjs
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const require = createRequire(path.join(ROOT, "app/package.json"));
const ort = require("onnxruntime-web");
const { PNG } = require("pngjs");

// optional: node check_wasm_parity.mjs <expected_probs.json> <model.onnx>  (defaults: the shipped fixtures + model)
const fx = JSON.parse(readFileSync(process.argv[2] ?? path.join(ROOT, "tests/fixtures/expected_probs.json"), "utf8"));
if (process.argv[3]) fx.model = process.argv[3];
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(readFileSync(path.isAbsolute(fx.model) ? fx.model : path.join(ROOT, fx.model)), { executionProviders: ["wasm"] });

const N = fx.items.length, S = 128;
const data = new Float32Array(N * 3 * S * S);
fx.items.forEach((it, n) => {
  const png = PNG.sync.read(readFileSync(path.join(ROOT, "tests/fixtures", it.file)));
  if (png.width !== S || png.height !== S) throw new Error(`${it.file} is ${png.width}x${png.height}`);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const i = (y * S + x) * 4;                      // RGBA
    for (let c = 0; c < 3; c++) data[n * 3 * S * S + c * S * S + y * S + x] = png.data[i + c] / 255;
  }
});
const out = await session.run({ image: new ort.Tensor("float32", data, [N, 3, S, S]) });
const probs = out.probs.data;
let maxDiff = 0, agree = 0;
fx.items.forEach((it, n) => {
  const row = Array.from(probs.slice(n * 5, n * 5 + 5));
  row.forEach((v, k) => { maxDiff = Math.max(maxDiff, Math.abs(v - it.probs[k])); });
  if (fx.classes[row.indexOf(Math.max(...row))] === it.argmax) agree++;
});
const result = { runtime: `onnxruntime-web ${ort.env.versions?.web ?? "?"} wasm, numThreads=1, Node ${process.version}`,
  model: fx.model, n: N, max_abs_prob_diff_vs_python: maxDiff, argmax_agreement: agree / N, tolerance: fx.suggested_tolerance_abs,
  pass: maxDiff <= fx.suggested_tolerance_abs && agree === N };
console.log(JSON.stringify(result));
process.exit(result.pass ? 0 : 1);
