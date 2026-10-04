/**
 * The analysis pipeline (bean finder rules + onnxruntime-web classifier). Normally runs inside farz.worker.ts;
 * `?mainthread=1` runs it on the page thread instead — only so DevTools/CDP CPU throttling (which Chromium
 * refuses to apply to workers) can be measured. The wasm runtime is served from this app's own /ort/ folder.
 * Model: v2 licence-stated fp16 ONNX, outputs `probs` [N,6] (T baked in) and `embed` [N,1024] (L2-normalised).
 */
import * as ort from "onnxruntime-web/wasm";
import { findBeans } from "../lib/beanfinder";
import { markTouching } from "../lib/touching";
import { colourGate } from "../lib/colourgate";
import { cropsToTensorData, labelsThresholds, type Labels } from "../lib/classify";
import { CLASSES, V2_THRESHOLDS, screen } from "../lib/rules";
import { darkLot } from "../lib/darklot";
import type { FromWorker, ToWorker, WorkerBean } from "./protocol";

const DEFAULT_LABELS: Labels = { classes: [...CLASSES], good_at_or_above: V2_THRESHOLDS.good, defect_at_or_above: V2_THRESHOLDS.defect, input: 128 };
let session: ort.InferenceSession | null = null;
let labels: Labels = DEFAULT_LABELS;
let stub = true;
let embedDim = 0;
const BATCH = 16;

async function sha256Hex(buf: Uint8Array): Promise<string | null> {
  try {
    if (!globalThis.crypto?.subtle) return null; // not a secure context (plain http on a LAN IP)
    const d = await crypto.subtle.digest("SHA-256", buf as unknown as ArrayBuffer);
    return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
  } catch {
    return null;
  }
}

export async function init(base: string): Promise<Extract<FromWorker, { type: "ready" }>> {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = `${base}ort/`;
  let modelBytes = 0;
  let modelSha = "";
  let error: string | undefined;
  try {
    const lr = await fetch(`${base}models/farz_beans_labels.json`);
    if (lr.ok) labels = { ...DEFAULT_LABELS, ...(await lr.json()) };
    const mr = await fetch(`${base}models/farz_beans.onnx`);
    if (!mr.ok) throw new Error(`model HTTP ${mr.status}`);
    const buf = new Uint8Array(await mr.arrayBuffer());
    // an HTML fallback page is not a model
    if (buf.length < 1024 || buf[0] === 0x3c) throw new Error("model file missing");
    modelBytes = buf.length;
    const real = await sha256Hex(buf);
    if (real && labels.model_sha256 && real !== labels.model_sha256) console.warn("[Farz] model file sha256 differs from the labels json");
    modelSha = real ?? labels.model_sha256 ?? "";
    session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    embedDim = session.outputNames.includes("embed") ? 1024 : 0;
    stub = false;
  } catch (e) {
    error = String((e as Error)?.message ?? e);
    console.warn(`[Farz] STUB CLASSIFIER IN USE — uniform probabilities, every bean will be "unsure". Reason: ${error}`);
    session = null;
    stub = true;
  }
  return { type: "ready", stub, classes: labels.classes, thresholds: labelsThresholds(labels), modelBytes, modelSha, embedDim, error };
}

async function classify(crops: Uint8Array[]): Promise<{ probs: number[][]; embeds: Float32Array | null }> {
  const k = labels.classes.length;
  if (!session) return { probs: crops.map(() => Array(k).fill(1 / k)), embeds: null };
  const probs: number[][] = [];
  const embeds = embedDim ? new Float32Array(crops.length * embedDim) : null;
  for (let i = 0; i < crops.length; i += BATCH) {
    const batch = crops.slice(i, i + BATCH);
    const t = new ort.Tensor("float32", cropsToTensorData(batch), [batch.length, 3, 128, 128]);
    const res = await session.run({ [session.inputNames[0]]: t });
    const p = res.probs ?? res[session.outputNames[0]];
    const pd = p.data as Float32Array;
    const kk = p.dims[1] ?? k;
    for (let b = 0; b < batch.length; b++) probs.push(Array.from(pd.subarray(b * kk, b * kk + kk)));
    if (embeds && res.embed) {
      const ed = res.embed.data as Float32Array;
      embeds.set(ed.subarray(0, batch.length * embedDim), i * embedDim);
    }
    t.dispose?.();
  }
  return { probs, embeds };
}

export async function analyze(msg: Extract<ToWorker, { type: "analyze" }>): Promise<Extract<FromWorker, { type: "result" }>> {
  const t0 = performance.now();
  const rgba = new Uint8Array(msg.rgba);
  const n = msg.width * msg.height;
  const rgb = new Uint8Array(n * 3);
  for (let i = 0, j = 0; i < n * 4; i += 4, j += 3) {
    rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2];
  }
  // bean finder (Python parity) + the touching rule: merged/touching blobs are never classified as one bean
  const r = markTouching(findBeans({ width: msg.width, height: msg.height, data: rgb }));
  const colour = colourGate(r);
  colour.darkLot = darkLot(r, colour);
  const t1 = performance.now();
  // retake gate + "dark lot" (both skip the classifier)
  const gateFail = screen(r.checks, colour, { minBeans: msg.minBeans });
  let probs: (number[] | null)[] = r.beans.map(() => null);
  let embeds: Float32Array | null = null;
  let classified = false;
  if (!gateFail || !msg.skipModelIfGateFails) {
    const c = await classify(r.beans.map((b) => b.crop!));
    probs = c.probs;
    embeds = c.embeds;
    classified = true;
  }
  const t2 = performance.now();
  const beans: WorkerBean[] = r.beans.map((b, i) => ({ x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1, touching: b.touching, probs: probs[i] }));
  return {
    type: "result", id: msg.id, workW: r.work.width, workH: r.work.height, scale: r.scale, checks: r.checks, colour,
    beans, embeds, embedDim: embeds ? embedDim : 0, classified, stub, ms: { finder: t1 - t0, model: t2 - t1, total: t2 - t0 },
  };
}

/** Ready-made crops (n x 128 x 128 x 3 RGB bytes) -> probs + embeddings. */
export async function embedCrops(msg: Extract<ToWorker, { type: "embed" }>): Promise<Extract<FromWorker, { type: "embedded" }>> {
  const all = new Uint8Array(msg.crops);
  const size = 128 * 128 * 3;
  const crops = Array.from({ length: msg.n }, (_, i) => all.subarray(i * size, (i + 1) * size));
  const c = await classify(crops);
  return { type: "embedded", id: msg.id, probs: c.probs, embeds: c.embeds, embedDim: c.embeds ? embedDim : 0 };
}
