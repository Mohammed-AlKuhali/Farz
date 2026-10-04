/** Main-thread side: decode the photo (EXIF orientation applied), hand pixels to the worker, apply the rules. */
import type { FromWorker, ToWorker, WorkerBean } from "../worker/protocol";
import type { Checks } from "./beanfinder";
import type { ColourGate } from "./colourgate";
import { callBean, decide, screen, type BeanCall, type Outcome, type Thresholds } from "./rules";
import { callBeanLocal, type Head } from "./localcal";

export interface ReadyInfo {
  stub: boolean;
  classes: string[];
  thresholds: Thresholds;
  modelBytes: number;
  modelSha: string;
  embedDim: number;
  error?: string;
}

export interface Analysis {
  outcome: Outcome;
  beans: (WorkerBean & { call: BeanCall | null })[];
  /** per-bean L2-normalised embeddings (beans x embedDim), kept in memory only for the grader's labelling screen */
  embeds: Float32Array | null;
  embedDim: number;
  /** the cooperative's local head decided the bean calls (calibration pilot) */
  calibrated: boolean;
  checks: Checks;
  colour: ColourGate;
  workW: number;
  workH: number;
  stub: boolean;
  ms: { decode: number; finder: number; model: number; total: number };
}

export interface AnalyzeOptions {
  /** lower bean minimum (grader calibration photos, the labelled calibration demo) */
  minBeans?: number;
  /** ignore the local head (the grader's own calibration photos are judged by the base model) */
  noHead?: boolean;
}

/** iOS Safari canvas area limit is ~16.7 MP; above it we pre-shrink with the browser's own resampler. */
const MAX_PIXELS = 16_000_000;

async function decode(blob: Blob): Promise<{ rgba: ArrayBuffer; width: number; height: number }> {
  let src: ImageBitmap | HTMLImageElement;
  let w: number, h: number;
  try {
    src = await createImageBitmap(blob, { imageOrientation: "from-image" });
    w = src.width; h = src.height;
  } catch {
    const url = URL.createObjectURL(blob);
    try {
      const img = new Image();
      img.decoding = "async";
      img.src = url;
      await img.decode();
      src = img;
      w = img.naturalWidth; h = img.naturalHeight;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  let dw = w, dh = h;
  if (w * h > MAX_PIXELS) {
    const k = Math.sqrt(MAX_PIXELS / (w * h));
    dw = Math.floor(w * k); dh = Math.floor(h * k);
  }
  const canvas = document.createElement("canvas");
  canvas.width = dw; canvas.height = dh;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, 0, 0, dw, dh);
  if ("close" in src) src.close();
  const data = ctx.getImageData(0, 0, dw, dh).data;
  canvas.width = 1; canvas.height = 1; // release canvas memory early (iOS)
  return { rgba: data.buffer as ArrayBuffer, width: dw, height: dh };
}

/** If the inference worker has not said "ready" this long after the page started it, the app shows a retry message. */
export const START_TIMEOUT_MS = 15_000;

/** The inference worker did not start in time (or failed outright): the UI offers "Try again" instead of "Counting…". */
export class EngineStartError extends Error {
  constructor(message = `inference worker did not start within ${START_TIMEOUT_MS / 1000} s`) {
    super(message);
    this.name = "EngineStartError";
  }
}

/** Debug/measurement only: `?mainthread=1` runs the pipeline on the page thread (see worker/core.ts). */
const MAIN_THREAD = typeof location !== "undefined" && new URLSearchParams(location.search).has("mainthread");

export class Engine {
  private worker: Worker | null = null;
  private core: Promise<typeof import("../worker/core")> | null = null;
  private nextId = 1;
  private pending = new Map<number, { resolve: (m: FromWorker) => void; reject: (e: Error) => void }>();
  readonly ready: Promise<ReadyInfo>;
  /** rejects with EngineStartError when the worker is not ready within START_TIMEOUT_MS (resolves like `ready` otherwise) */
  readonly started: Promise<ReadyInfo>;
  private info: ReadyInfo | null = null;
  /** the active cooperative head (calibration pilot), or null = the base model's own two thresholds */
  head: Head | null = null;
  private startTimer: ReturnType<typeof setTimeout> | undefined;
  readonly mainThread = MAIN_THREAD;

  constructor(startTimeoutMs = START_TIMEOUT_MS) {
    let resolveReady!: (r: ReadyInfo) => void;
    this.ready = new Promise((r) => (resolveReady = r));
    this.started = new Promise<ReadyInfo>((resolve, reject) => {
      this.ready.then(resolve);
      this.startTimer = setTimeout(() => { if (!this.info) reject(new EngineStartError()); }, startTimeoutMs);
    });
    this.started.catch(() => { /* handled by callers; never an unhandled rejection */ });
    const onReady = (m: Extract<FromWorker, { type: "ready" }>) => {
      this.info = { stub: m.stub, classes: m.classes, thresholds: m.thresholds, modelBytes: m.modelBytes, modelSha: m.modelSha, embedDim: m.embedDim, error: m.error };
      clearTimeout(this.startTimer);
      resolveReady(this.info);
    };
    // absolute base: inside the worker, relative URLs would resolve against assets/
    const base = new URL(import.meta.env.BASE_URL, document.baseURI).href;
    if (MAIN_THREAD) {
      this.core = import("../worker/core");
      this.core.then((c) => c.init(base)).then(onReady);
      return;
    }
    this.worker = new Worker(new URL("../worker/farz.worker.ts", import.meta.url), { type: "module" });
    // if the module worker cannot start (very old browser), fall back to running the same pipeline on the page
    this.worker.onerror = (ev) => {
      console.warn("[Farz] worker failed, running on the main thread instead", ev.message);
      this.worker?.terminate();
      this.worker = null;
      this.pending.forEach((p) => p.reject(new Error("worker failed")));
      this.pending.clear();
      if (!this.core) {
        this.core = import("../worker/core");
        this.core.then((c) => c.init(base)).then(onReady);
      }
    };
    this.worker.onmessage = (ev: MessageEvent<FromWorker>) => {
      const m = ev.data;
      if (m.type === "ready") return onReady(m);
      const id = m.id;
      if (id !== undefined && this.pending.has(id)) {
        const p = this.pending.get(id)!;
        this.pending.delete(id);
        if (m.type === "error") p.reject(new Error(m.message));
        else p.resolve(m);
      }
    };
    const init: ToWorker = { type: "init", base };
    this.worker.postMessage(init);
  }

  /** Stop the worker (used before "Try again" builds a new Engine). */
  dispose() {
    clearTimeout(this.startTimer);
    this.worker?.terminate();
    this.worker = null;
    this.pending.forEach((p) => p.reject(new EngineStartError("engine disposed")));
    this.pending.clear();
  }

  private send(msg: ToWorker & { id: number }, transfer: Transferable[]): Promise<FromWorker> {
    return new Promise<FromWorker>((resolve, reject) => {
      this.pending.set(msg.id, { resolve, reject });
      this.worker!.postMessage(msg, transfer);
    });
  }

  /** Ready-made 128x128 RGB crops -> probs + embeddings (the bundled calibration demo set). */
  async embedCrops(crops: Uint8Array[]): Promise<{ probs: number[][]; embeds: Float32Array | null; embedDim: number }> {
    await (this.info ? Promise.resolve(this.info) : this.started);
    const size = 128 * 128 * 3;
    const all = new Uint8Array(crops.length * size);
    crops.forEach((c, i) => all.set(c, i * size));
    const msg: Extract<ToWorker, { type: "embed" }> = { type: "embed", id: this.nextId++, n: crops.length, crops: all.buffer };
    const res = this.core ? await (await this.core).embedCrops(msg) : await this.send(msg, [all.buffer]);
    if (res.type !== "embedded") throw new Error("unexpected worker reply");
    return { probs: res.probs, embeds: res.embeds, embedDim: res.embedDim };
  }

  async analyze(blob: Blob, opts: AnalyzeOptions = {}): Promise<Analysis> {
    // never wait forever: a worker that is not ready within START_TIMEOUT_MS throws EngineStartError (retry message)
    const info = this.info ?? (await this.started);
    const t0 = performance.now();
    const { rgba, width, height } = await decode(blob);
    const tDecode = performance.now() - t0;
    const id = this.nextId++;
    const msg: Extract<ToWorker, { type: "analyze" }> = { type: "analyze", id, width, height, rgba, skipModelIfGateFails: true, minBeans: opts.minBeans };
    const res = this.core ? await (await this.core).analyze(msg) : await this.send(msg, [rgba]);
    if (res.type !== "result") throw new Error("unexpected worker reply");
    const g = screen(res.checks, res.colour, { minBeans: opts.minBeans });
    const head = opts.noHead || !res.embeds ? null : this.head;
    const D = res.embedDim;
    const beans = res.beans.map((b, i) => ({
      ...b,
      call: !res.classified ? null
        : head ? callBeanLocal(b.probs, res.embeds!.subarray(i * D, (i + 1) * D), b.touching, head)
        : callBean(b.probs, b.touching, info.thresholds),
    }));
    const outcome: Outcome = g ?? decide(beans.map((b) => b.call as BeanCall));
    return {
      outcome, beans, embeds: res.embeds, embedDim: D, calibrated: !!head && !g, checks: res.checks, colour: res.colour,
      workW: res.workW, workH: res.workH, stub: res.stub,
      ms: { decode: tDecode, finder: res.ms.finder, model: res.ms.model, total: performance.now() - t0 },
    };
  }
}
