/** Last 5 checks in IndexedDB — counts only, never photos. One call wipes everything. */
import type { BeanCall, Result } from "./rules";
import type { Band } from "./wilson";

export interface HistoryRecord {
  id?: number;
  ts: number;
  total: number;
  /** per-class counts; NEVER stored for a "not sure" result (REGRESSION_A I-1) — only the total is kept */
  counts?: Record<BeanCall, number>;
  band: Band;
  /** 95% range of the defect share; not stored for a "not sure" result */
  lo?: number;
  hi?: number;
  about?: boolean; // the 95% range crossed a band edge
  handfuls?: number; // photos of the same coffee pooled into this record
  demo?: "synthetic" | "real"; // a demo tray (synthetic composite or a real dataset photo), not a farmer's check
}

/**
 * What History keeps for one result. PRIVACY (REGRESSION_A I-1): when Farz says "not sure" (c07, any reason) it
 * keeps only the date, the total bean count and the "not sure" band — no per-class or defect counts, no range,
 * the same as the slip. Records written by older builds are still shown without their counts (see App.tsx).
 */
export function historyRecordFor(o: Result, ts: number, demo?: HistoryRecord["demo"]): HistoryRecord {
  const base: HistoryRecord = { ts, total: o.total, band: o.band, handfuls: o.handfuls, demo };
  if (o.band === "unsure") return base;
  return { ...base, counts: { ...o.counts }, lo: o.interval.lo, hi: o.interval.hi, about: o.about };
}

/** Per-class counts History may show for a record: none for "not sure" (also for records saved by older builds). */
export function shownCounts(h: HistoryRecord): Record<BeanCall, number> | null {
  return h.band === "unsure" || !h.counts ? null : h.counts;
}

const DB = "farz";
const STORE = "checks";
export const KEEP = 5;

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T> | void): Promise<T | undefined> {
  return open().then(
    (db) =>
      new Promise<T | undefined>((resolve, reject) => {
        const t = db.transaction(STORE, mode);
        const r = fn(t.objectStore(STORE));
        t.oncomplete = () => { db.close(); resolve(r ? (r.result as T) : undefined); };
        t.onerror = () => { db.close(); reject(t.error); };
      }),
  );
}

export async function listHistory(): Promise<HistoryRecord[]> {
  const all = (await tx<HistoryRecord[]>("readonly", (s) => s.getAll())) ?? [];
  return all.sort((a, b) => b.ts - a.ts).slice(0, KEEP);
}

/** Adds a record (or replaces it when `rec.id` is set — a lot that got another handful). Returns its id. */
export async function addHistory(rec: HistoryRecord): Promise<number | undefined> {
  const id = await tx<IDBValidKey>("readwrite", (s) => (rec.id === undefined ? s.add(rec) : s.put(rec)));
  const all = (await tx<HistoryRecord[]>("readonly", (s) => s.getAll())) ?? [];
  const old = all.sort((a, b) => b.ts - a.ts).slice(KEEP);
  if (old.length) await tx("readwrite", (s) => { old.forEach((o) => s.delete(o.id!)); });
  return typeof id === "number" ? id : undefined;
}

export async function wipeHistory(): Promise<void> {
  await tx("readwrite", (s) => s.clear());
}
