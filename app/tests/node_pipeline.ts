/**
 * The app pipeline in Node for measurement tests: jpeg-js decode + EXIF orientation (as the browser applies it) ->
 * beanfinder.ts -> touching.ts -> colourgate.ts -> gate() -> onnxruntime-web classifier -> callBean() -> decide().
 * Same code the worker runs (src/worker/core.ts); only the JPEG decoder differs (jpeg-js vs the browser's).
 */
import fs from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-web/wasm";
import type { FindResult } from "../src/lib/beanfinder";
import { markTouching } from "../src/lib/touching";
import { colourGate } from "../src/lib/colourgate";
import { cropsToTensorData, labelsThresholds } from "../src/lib/classify";
import { callBean, decide, gate, type BeanCall, type Outcome, type Retake, type Thresholds } from "../src/lib/rules";
import { MAX_BEANS, MIN_BEANS, type Checks } from "../src/lib/beanfinder";
import type { ColourGate } from "../src/lib/colourgate";
// @ts-expect-error plain .mjs helper
import { decodeJpegRGB } from "./decode_jpeg.mjs";
import { exifOrientation } from "./exif";

export const APP = path.resolve(__dirname, "..");
export const ROOT = path.resolve(APP, "..");
/** The published repo ships without data/raw (licences): tests that need it skip cleanly when it is absent. */
export const HAVE_RAW = (rel: string) => fs.existsSync(path.join(ROOT, "data/raw", rel));

/** Apply orientation 3 / 6 / 8 to packed RGB. */
export function orient(im: { width: number; height: number; data: Uint8Array }, o: number) {
  if (o !== 3 && o !== 6 && o !== 8) return im;
  const { width: w, height: h, data: d } = im;
  const W = o === 3 ? w : h, H = o === 3 ? h : w;
  const out = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let X: number, Y: number;
    if (o === 3) { X = w - 1 - x; Y = h - 1 - y; } else if (o === 6) { X = h - 1 - y; Y = x; } else { X = y; Y = w - 1 - x; }
    const s = (y * w + x) * 3, t = (Y * W + X) * 3;
    out[t] = d[s]; out[t + 1] = d[s + 1]; out[t + 2] = d[s + 2];
  }
  return { width: W, height: H, data: out };
}

export function loadImage(f: string) {
  const buf = fs.readFileSync(f);
  const raw = decodeJpegRGB(buf);
  return orient({ width: raw.width, height: raw.height, data: new Uint8Array(raw.data) }, exifOrientation(buf));
}

export async function makeClassifier() {
  const labels = JSON.parse(fs.readFileSync(path.join(APP, "public/models/farz_beans_labels.json"), "utf8"));
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(fs.readFileSync(path.join(APP, "public/models/farz_beans.onnx")), { executionProviders: ["wasm"] });
  const hasEmbed = session.outputNames.includes("embed");
  /** probs per crop (6 classes, T baked in) and, when the model has it, the L2-normalised 1,024-d embedding */
  const run = async (crops: Uint8Array[]) => {
    const probs: Float32Array[] = [];
    const embeds: Float32Array[] = [];
    for (let i = 0; i < crops.length; i += 16) {
      const b = crops.slice(i, i + 16);
      const r = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", cropsToTensorData(b), [b.length, 3, 128, 128]) });
      const p = r.probs ?? r[session.outputNames[0]];
      const k = p.dims[1];
      const pd = p.data as Float32Array;
      for (let j = 0; j < b.length; j++) probs.push(pd.slice(j * k, j * k + k));
      if (hasEmbed) {
        const D = r.embed.dims[1], ed = r.embed.data as Float32Array;
        for (let j = 0; j < b.length; j++) embeds.push(ed.slice(j * D, j * D + D));
      }
    }
    return { probs, embeds: hasEmbed ? embeds : null };
  };
  const classify = async (crops: Uint8Array[]) => (await run(crops)).probs;
  return { classify, run, thresholds: labelsThresholds(labels), modelSha: labels.model_sha256 as string };
}

export type Classifier = Awaited<ReturnType<typeof makeClassifier>>;

/** gate() exactly as shipped until 3 Oct 2026 23:59 (dark -> blur -> not green -> count -> touching), for "before". */
export function gateLegacy(checks: Checks, colour: ColourGate): Retake | null {
  const clip = { dark: "c03_dark", blur: "c02_blur", not_green: "c06_not_green", count: "c05_count", spread: "c04_spread" } as const;
  let reason: keyof typeof clip | null = null;
  if (checks.too_dark) reason = "dark";
  else if (checks.too_blurry) reason = "blur";
  else if (colour.notGreen) reason = "not_green";
  else if (checks.count < MIN_BEANS || checks.count > MAX_BEANS) reason = "count";
  else if (checks.too_many_touching) reason = "spread";
  return reason ? { kind: "retake", reason, clip: clip[reason] } : null;
}

/**
 * One photo through the rules. `touchRule` = true is the shipped app (touching.ts + the new gate order); false =
 * the build before 4 Oct 2026 (Python-parity touching flag only, legacy gate order). Probabilities are reused.
 */
export function outcomeOf(r: FindResult, probs: (Float32Array | null)[] | null, thr: Thresholds, touchRule: boolean): { outcome: Outcome; calls: BeanCall[] | null } {
  const rr = touchRule ? markTouching(r) : r;
  const g = (touchRule ? gate : gateLegacy)(rr.checks, colourGate(rr as FindResult));
  if (g) return { outcome: g, calls: null };
  const calls = rr.beans.map((b, i) => callBean(probs ? probs[i] : null, b.touching, thr));
  return { outcome: decide(calls), calls };
}
