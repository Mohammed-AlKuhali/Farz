/**
 * "Calibrate for your cooperative" (PILOT): the prototype head of reports_v2/localcal_spec.md, ported line for line.
 * Pre-registered study: reports_v2/localcal.md — GO on ONE source (loja_yolo, K = 20); the same method gave clean
 * trays called "many" on two other sources (lojano, Mindforge). So it is a clearly labelled pilot, off until a grader
 * builds a head, and every photo-level fail-safe of rules.ts still applies unchanged.
 *
 * Calibration (§2): n = 10 good + 10 defect embeddings (touching beans never used; refuse with fewer than 10 of either).
 *   muG = Σ good / ||Σ good||, muD = Σ defect / ||Σ defect|| (float64)
 *   leave-one-out margin s_i = dot(e_i, d) - dot(e_i, g) with the bean's own class prototype recomputed without it
 *   mHi = max(max |s_i| over LOO-wrong beans (0 if none), median |s_i|)
 * Per bean (§3): touching -> unsure; s = dot(e, muD) - dot(e, muG); local = s >= 0 ? defect : good;
 *   base = P(good) >= 0.5 ? good : defect; agree -> local; else |s| >= mHi -> local; else unsure.
 * Nothing here ever leaves the phone: the head is stored in IndexedDB ("farz-localcal"), the photos never are.
 */
import { likelyDefect, type BeanCall } from "./rules";

export const LOCALCAL_FORMAT = "farz-localcal-1";
/** K = 20 is the measured setting (10 good + 10 defect). */
export const N_PER_CLASS = 10;
/**
 * Extra guard (our decision, 4 Oct 2026, NOT part of the pre-registered rule): refuse a head when more than 10% of its
 * calibration beans are mis-sorted in leave-one-out. The study (localcal_spec.md §5, exploratory) found this kept 13 of
 * 15 loja_yolo heads and rejected every Mindforge and USK head; it is not sufficient on its own. Refusing is safe.
 */
export const MAX_LOO_ERROR_FRAC = 0.1;

export type Label = "good" | "defect";

export interface StoredHead {
  format: typeof LOCALCAL_FORMAT;
  model_sha256: string;
  embed_dim: number;
  n_good: number;
  n_defect: number;
  mu_good: string; // base64 of Float32Array(embed_dim)
  mu_defect: string;
  m_hi: number;
  loo_errors: number;
  created: string;
  /** the cooperative / set-up name the grader typed (shown as "calibrated for: <name>") */
  name: string;
  /** beans the grader labelled in total (the head uses N_PER_CLASS of each) */
  labelled_good: number;
  labelled_defect: number;
  trays: number;
  /** built from the bundled demo set (loja_yolo, Ecuador), not from a grader's own photos */
  demo?: boolean;
}

export interface Head {
  muG: Float64Array;
  muD: Float64Array;
  mHi: number;
}

export interface FitReport {
  ok: boolean;
  /** why the head was refused: too few labels of a class, or too many leave-one-out errors */
  refused?: "too_few" | "loo_errors";
  nGood: number;
  nDefect: number;
  looErrors: number;
  looN: number;
  mHi: number;
  muG?: Float64Array;
  muD?: Float64Array;
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += a[k] * b[k];
  return s;
};
function unit(v: Float64Array): Float64Array {
  const n = Math.sqrt(dot(v, v));
  const o = new Float64Array(v.length);
  for (let k = 0; k < v.length; k++) o[k] = v[k] / n;
  return o;
}
const median = (xs: number[]) => {
  const a = [...xs].sort((p, q) => p - q);
  return a.length % 2 ? a[(a.length - 1) / 2] : (a[a.length / 2 - 1] + a[a.length / 2]) / 2;
};

/** Deterministic PRNG (mulberry32) so the same labels always give the same head. */
function rng(seed: number) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
/** n items drawn uniformly at random without replacement (all of them, in order, if there are exactly n). */
function draw<T>(xs: T[], n: number, rand: () => number): T[] {
  if (xs.length <= n) return xs.slice();
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}

/** §2: build the prototype head from labelled embeddings (each L2-normalised, from the model's `embed` output). */
export function fitHead(items: { embed: ArrayLike<number>; label: Label }[], seed = 1): FitReport {
  const good = items.filter((i) => i.label === "good");
  const bad = items.filter((i) => i.label === "defect");
  if (good.length < N_PER_CLASS || bad.length < N_PER_CLASS)
    return { ok: false, refused: "too_few", nGood: good.length, nDefect: bad.length, looErrors: 0, looN: 0, mHi: NaN };
  const rand = rng(seed);
  const g = draw(good, N_PER_CLASS, rand), d = draw(bad, N_PER_CLASS, rand);
  const D = g[0].embed.length;
  const Sg = new Float64Array(D), Sd = new Float64Array(D);
  for (const it of g) for (let k = 0; k < D; k++) Sg[k] += it.embed[k];
  for (const it of d) for (let k = 0; k < D; k++) Sd[k] += it.embed[k];
  const muG = unit(Sg), muD = unit(Sd);
  const abs: number[] = [];
  let maxWrong = 0, errors = 0;
  const loo = (e: ArrayLike<number>, isDefect: boolean) => {
    const rest = new Float64Array(D);
    const S = isDefect ? Sd : Sg;
    for (let k = 0; k < D; k++) rest[k] = S[k] - e[k];
    const own = unit(rest);
    const s = isDefect ? dot(e, own) - dot(e, muG) : dot(e, muD) - dot(e, own);
    const wrong = (s >= 0) !== isDefect;
    abs.push(Math.abs(s));
    if (wrong) { errors++; maxWrong = Math.max(maxWrong, Math.abs(s)); }
  };
  for (const it of g) loo(it.embed, false);
  for (const it of d) loo(it.embed, true);
  const mHi = Math.max(maxWrong, median(abs));
  const looN = g.length + d.length;
  const ok = errors / looN <= MAX_LOO_ERROR_FRAC;
  return { ok, refused: ok ? undefined : "loo_errors", nGood: good.length, nDefect: bad.length, looErrors: errors, looN, mHi, muG, muD };
}

/** §3: one bean's call while a local head is active (replaces callBean). Defect type is display-only ("possible …"). */
export function callBeanLocal(probs: ArrayLike<number> | null, embed: ArrayLike<number> | null, touching: boolean, h: Head): BeanCall {
  if (touching || !probs || !embed) return "unsure";
  const s = dot(embed, h.muD) - dot(embed, h.muG);
  const local: Label = s >= 0 ? "defect" : "good";
  const base: Label = probs[0] >= 0.5 ? "good" : "defect";
  let call: Label | "unsure";
  if (local === base) call = local;
  else if (Math.abs(s) >= h.mHi) call = local;
  else call = "unsure";
  if (call === "defect") return likelyDefect(probs);
  return call;
}

/* ---------- storage: one head (the active cooperative set-up), base64 Float32 as in the spec ---------- */

function toB64(v: Float64Array): string {
  const f = new Float32Array(v);
  const bytes = new Uint8Array(f.buffer);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
function fromB64(b: string): Float64Array {
  const s = atob(b);
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return Float64Array.from(new Float32Array(bytes.buffer));
}

export function toStored(r: FitReport, meta: { modelSha: string; name: string; trays: number; demo?: boolean }, now = new Date()): StoredHead {
  if (!r.ok || !r.muG || !r.muD) throw new Error("head was refused");
  return {
    format: LOCALCAL_FORMAT, model_sha256: meta.modelSha, embed_dim: r.muG.length, n_good: N_PER_CLASS, n_defect: N_PER_CLASS,
    mu_good: toB64(r.muG), mu_defect: toB64(r.muD), m_hi: r.mHi, loo_errors: r.looErrors, created: now.toISOString(),
    name: meta.name, labelled_good: r.nGood, labelled_defect: r.nDefect, trays: meta.trays, demo: meta.demo,
  };
}

/** The runtime head, or null when the stored head is missing, malformed or made with a different model file. */
export function toHead(s: StoredHead | null, modelSha: string | undefined): Head | null {
  if (!s || s.format !== LOCALCAL_FORMAT || !modelSha || s.model_sha256 !== modelSha) return null;
  const muG = fromB64(s.mu_good), muD = fromB64(s.mu_defect);
  if (muG.length !== s.embed_dim || muD.length !== s.embed_dim || !Number.isFinite(s.m_hi)) return null;
  return { muG, muD, mHi: s.m_hi };
}

const DB = "farz-localcal";
const STORE = "head";
const KEY = "active";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T> | void): Promise<T | undefined> {
  return open().then((db) => new Promise<T | undefined>((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const r = fn(t.objectStore(STORE));
    t.oncomplete = () => { db.close(); resolve(r ? (r.result as T) : undefined); };
    t.onerror = () => { db.close(); reject(t.error); };
  }));
}

export async function loadStoredHead(): Promise<StoredHead | null> {
  try { return (await tx<StoredHead>("readonly", (s) => s.get(KEY))) ?? null; } catch { return null; }
}
export async function saveStoredHead(h: StoredHead): Promise<void> {
  await tx("readwrite", (s) => s.put(h, KEY));
}
/** "Reset calibration": deletes the head; Farz goes back to the base model's own two thresholds. */
export async function resetStoredHead(): Promise<void> {
  await tx("readwrite", (s) => s.delete(KEY));
}
