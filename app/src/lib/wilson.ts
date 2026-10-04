/**
 * 95% Wilson score interval for a defect rate, and the band rule.
 *
 * Bands are OUR rule of thumb (not an official grade): clean < 5%, some 5–20%, many > 20%.
 * Boundaries: clean = [0, 5%), some = [5%, 20%], many = (20%, 100%].
 *
 * The band is picked from the POINT estimate (k / n). The 95% interval is always shown next to it; when the interval
 * crosses a band edge the answer is marked "about" (sampling uncertainty — fix it by counting more beans, not by
 * sending the farmer away). Example (docs/PLAN.md §3): 12 defects in 100 beans -> 7.0%–19.8% -> "some", not "about".
 */
export const Z95 = 1.96;
export const CLEAN_MAX = 0.05;
export const MANY_MIN = 0.2;

export interface Interval {
  k: number;
  n: number;
  p: number;
  lo: number;
  hi: number;
}

export function wilson(k: number, n: number, z = Z95): Interval {
  if (!Number.isFinite(k) || !Number.isFinite(n) || n <= 0 || k < 0 || k > n) {
    return { k, n, p: NaN, lo: 0, hi: 1 };
  }
  const p = k / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const centre = (p + z2 / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  return { k, n, p, lo: Math.max(0, centre - half), hi: Math.min(1, centre + half) };
}

export type Band = "clean" | "some" | "many" | "unsure";

/** Band of a single rate (same edges as above). NaN -> "unsure". */
export function bandOfRate(p: number): Band {
  if (!Number.isFinite(p)) return "unsure";
  if (p < CLEAN_MAX) return "clean";
  if (p <= MANY_MIN) return "some";
  return "many";
}

/** Band of the whole interval: a band only if lo and hi fall in the same band (the old, stricter rule). */
export function bandOf(iv: Pick<Interval, "lo" | "hi">): Band {
  const a = bandOfRate(iv.lo), b = bandOfRate(iv.hi);
  return a === b ? a : "unsure";
}

/** True when the 95% interval reaches outside the band of its point estimate. */
export function crossesBandEdge(iv: Interval): boolean {
  const b = bandOfRate(iv.p);
  if (b === "unsure") return true;
  return bandOfRate(iv.lo) !== b || bandOfRate(iv.hi) !== b;
}
