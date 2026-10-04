// Copy the onnxruntime-web wasm runtime from node_modules into public/ort/ so it is served (and precached)
// from this app's own origin — no CDN. Runs before dev and build (package.json "predev"/"prebuild").
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const src = path.join(app, "node_modules/onnxruntime-web/dist");
const dst = path.join(app, "public/ort");
fs.mkdirSync(dst, { recursive: true });
for (const f of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) {
  fs.copyFileSync(path.join(src, f), path.join(dst, f));
  console.log(`copied ${f} (${(fs.statSync(path.join(dst, f)).size / 1e6).toFixed(2)} MB) -> public/ort/`);
}
