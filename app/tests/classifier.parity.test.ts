/**
 * Classifier parity, v2 licence-stated fp16 model: our preprocessing + onnxruntime-web (wasm, 1 thread) vs Python
 * onnxruntime on the SAME inputs (fixtures: tests/fixtures/classifier_v2, written by tests/make_classifier_fixtures.py).
 *  - 40 crops (20 J4ckDev, 20 loja_yolo calibration crops): every prob within TOL, same top class, embedding cosine
 *    >= 0.999, same two-threshold call unless P(good) sits within BORDER of a threshold (reported, not hidden).
 *  - 3 whole photos end to end: the app's TypeScript bean finder gives the Python boxes exactly; probs within TOL;
 *    the same call counts (borderline beans reported).
 * The test checks the model's sha256 against the fixtures, so it fails loudly if the model changes without new fixtures.
 */
import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { PNG } from "pngjs";
import * as ort from "onnxruntime-web/wasm";
import { cropsToTensorData } from "../src/lib/classify";
import { findBeans } from "../src/lib/beanfinder";
import { callBean, CLASSES, V2_THRESHOLDS, type BeanCall } from "../src/lib/rules";

const APP = path.resolve(__dirname, "..");
const FX = path.join(APP, "tests/fixtures/classifier_v2");
const MODEL = path.join(APP, "public/models/farz_beans.onnx");
const TOL = 0.02;
const BORDER = 0.005;
const have = fs.existsSync(path.join(FX, "expected.json")) && fs.existsSync(MODEL);

function readPngRGB(file: string) {
  const png = PNG.sync.read(fs.readFileSync(file));
  const rgb = new Uint8Array(png.width * png.height * 3);
  for (let i = 0, j = 0; i < png.data.length; i += 4, j += 3) {
    rgb[j] = png.data[i]; rgb[j + 1] = png.data[i + 1]; rgb[j + 2] = png.data[i + 2];
  }
  return { width: png.width, height: png.height, data: rgb };
}
const b64f32 = (s: string) => new Float32Array(Uint8Array.from(Buffer.from(s, "base64")).buffer);

ort.env.wasm.numThreads = 1;
let session: ort.InferenceSession;
async function run(crops: Uint8Array[]) {
  if (!session) session = await ort.InferenceSession.create(fs.readFileSync(MODEL), { executionProviders: ["wasm"] });
  const probs: Float32Array[] = [], embeds: Float32Array[] = [];
  for (let i = 0; i < crops.length; i += 16) {
    const b = crops.slice(i, i + 16);
    const out = await session.run({ image: new ort.Tensor("float32", cropsToTensorData(b), [b.length, 3, 128, 128]) });
    const p = out.probs.data as Float32Array, e = out.embed.data as Float32Array, D = out.embed.dims[1];
    for (let j = 0; j < b.length; j++) { probs.push(p.slice(j * 6, j * 6 + 6)); embeds.push(e.slice(j * D, j * D + D)); }
  }
  return { probs, embeds };
}

describe.skipIf(!have)("classifier parity with Python onnxruntime (v2 fp16, probs + embed)", () => {
  const fx = have ? JSON.parse(fs.readFileSync(path.join(FX, "expected.json"), "utf8")) : null;
  const thr = fx?.thresholds;
  const border = (p: ArrayLike<number>) => Math.abs(p[0] - thr.good) < BORDER || Math.abs(1 - p[0] - thr.defect) < BORDER;

  it("model file is the one the fixtures were made with; labels say 6 classes and the shipped two-threshold rule", () => {
    const sha = crypto.createHash("sha256").update(fs.readFileSync(MODEL)).digest("hex");
    expect(sha).toBe(fx.model_sha256);
    expect(fx.classes).toEqual([...CLASSES]);
    expect(thr).toEqual(V2_THRESHOLDS);
  });

  it(`40 crops: probs within ${TOL}, same top class, embedding cosine >= 0.999, same calls (borderline reported)`, async () => {
    const crops = fx.crops.map((c: { file: string; base: string }) => readPngRGB(c.base === "app" ? path.join(APP, c.file) : path.join(FX, c.file)).data);
    const { probs, embeds } = await run(crops);
    let maxDiff = 0, argmaxAgree = 0, minCos = 1, callAgree = 0;
    const borderline: string[] = [];
    fx.crops.forEach((c: { file: string; probs: number[]; call: BeanCall; embed_b64: string }, i: number) => {
      const row = probs[i];
      row.forEach((p, j) => (maxDiff = Math.max(maxDiff, Math.abs(p - c.probs[j]))));
      if (row.indexOf(Math.max(...row)) === c.probs.indexOf(Math.max(...c.probs))) argmaxAgree++;
      const ref = b64f32(c.embed_b64);
      let dot = 0, na = 0, nb = 0;
      for (let k = 0; k < ref.length; k++) { dot += ref[k] * embeds[i][k]; na += ref[k] ** 2; nb += embeds[i][k] ** 2; }
      minCos = Math.min(minCos, dot / Math.sqrt(na * nb));
      const call = callBean(row, false, thr);
      if (call === c.call) callAgree++;
      else if (border(c.probs)) borderline.push(c.file);
      else expect(call, c.file).toBe(c.call);
    });
    fs.writeFileSync(path.join(APP, "reports/classifier_parity_report.json"), JSON.stringify({
      model_sha256: fx.model_sha256, n: fx.crops.length, maxAbsProbDiff: maxDiff, argmaxAgree, minEmbedCosine: minCos, callAgree, borderline,
    }, null, 1));
    expect(argmaxAgree).toBe(fx.crops.length);
    expect(maxDiff).toBeLessThan(TOL);
    expect(minCos).toBeGreaterThanOrEqual(0.999);
  });

  it("3 fixture photos end to end: identical boxes, probs within tolerance, same call counts", async () => {
    const rep: Record<string, unknown>[] = [];
    for (const ph of fx.photos) {
      const r = findBeans(readPngRGB(path.join(FX, ph.file)));
      expect(r.beans.map((b) => [b.x0, b.y0, b.x1, b.y1])).toEqual(ph.beans.map((b: { box: number[] }) => b.box));
      const { probs } = await run(r.beans.map((b) => b.crop!));
      let maxDiff = 0;
      const counts: Record<string, number> = Object.fromEntries([...CLASSES, "unsure"].map((k) => [k, 0]));
      let borderline = 0;
      ph.beans.forEach((b: { probs: number[]; touching: boolean; call: BeanCall }, i: number) => {
        probs[i].forEach((p, j) => (maxDiff = Math.max(maxDiff, Math.abs(p - b.probs[j]))));
        const c = callBean(probs[i], b.touching, thr);
        counts[c]++;
        if (c !== b.call) { expect(border(b.probs), `${ph.file} bean ${i}`).toBe(true); borderline++; }
      });
      rep.push({ photo: ph.file, beans: r.beans.length, maxAbsDiff: maxDiff, counts, expected: ph.call_counts, borderline });
      expect(maxDiff).toBeLessThan(TOL);
      if (!borderline) expect(counts).toEqual(ph.call_counts);
    }
    fs.writeFileSync(path.join(APP, "reports/pipeline_parity_report.json"), JSON.stringify(rep, null, 1));
  });
});
