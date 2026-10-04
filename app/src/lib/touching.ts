/**
 * Touching / merged beans — a fixed shape rule (NOT AI), app-only, run right after the bean finder.
 *
 * Why: the Python-parity bean finder (beanfinder.ts) calls a blob "touching" only when its area is > 1.8 x the
 * MEDIAN blob area. When beans are pushed together, many blobs are merged pairs, the median itself grows, and a
 * merged pair can pass as one bean. The classifier then sees two beans in one crop and answers with confidence.
 *
 * Rule: a blob is treated as touching (yellow "?", never classified) when ANY of these holds
 *   1. the parity rule: area > 1.8 x median blob area                                   (beanfinder.ts)
 *   2. area > AREA_X x a robust ONE-bean area: the median area of the blobs whose shape looks like a single bean
 *      (convex and ellipse-like); falls back to the plain median when fewer than MIN_SINGLES such blobs exist.
 *      Catches beans pushed together in pairs, where the plain median is itself a pair.
 *   3. a shape no single bean has: solidity (area / convex-hull area) < MIN_SOLIDITY, or ellipse fill (area / area
 *      of the ellipse with the same second moments) < MIN_FILL — clumps of 3+ beans.
 *   The shape limits sit below every single bean measured (synthetic trays and real CBD photos), because a broken
 *   or insect-damaged bean is irregular too and must stay a defect, not become a "?".
 * The spread-out retake (c04) uses the same 8% threshold as before, counted two ways (either one triggers it):
 *   - touching BLOBS > max(2, 8% of blobs)                                   (the existing rule)
 *   - beans HIDDEN in CLUMPS > max(2, 8% of all beans): a clump is a touching blob of bean colour (pale or dark,
 *     colourgate.ts) holding round(area / one-bean area) >= 3 beans — so one clump of 30 beans counts as 30 hidden
 *     beans, not as 1 blob. A finger or a hand (skin colour) is not a clump; pairs still count as one blob each.
 * Thresholds are fixed numbers measured on known single beans and on known merged blobs
 * (app/reports/touching_measure.json, tests/touching.measure.test.ts).
 */
import type { Bean, Checks, FindResult } from "./beanfinder";
import { classifyColour } from "./colourgate";

export const TOUCH_RULE = {
  AREA_X: 1.6,
  MIN_SOLIDITY: 0.8,
  MIN_FILL: 0.85,
  /** single-looking blobs only (reference area): beans lying end to end are longer than this */
  MAX_ELONG: 2.2,
  /** a blob counts as "single-looking" for the reference area when solidity >= this and fill >= this */
  SINGLE_SOLIDITY: 0.93,
  SINGLE_FILL: 0.93,
  /** need at least this many single-looking blobs to trust their median; else the plain median is used */
  MIN_SINGLES: 5,
};

export interface Shape {
  area: number;
  solidity: number;
  fill: number;
  elong: number;
}

/** Shape of one labelled component (pixels with value `label` inside its bbox). */
export function shapeOf(labels: Int32Array, w: number, label: number, x0: number, y0: number, x1: number, y1: number): Shape {
  let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  // hull points: the outer corners of the leftmost and rightmost pixel of every row
  const pts: number[][] = [];
  for (let y = y0; y < y1; y++) {
    let lo = -1, hi = -1;
    const row = y * w;
    for (let x = x0; x < x1; x++) {
      if (labels[row + x] !== label) continue;
      if (lo < 0) lo = x;
      hi = x;
      n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
    }
    if (lo >= 0) pts.push([lo, y], [lo, y + 1], [hi + 1, y], [hi + 1, y + 1]);
  }
  if (n === 0) return { area: 0, solidity: 1, fill: 1, elong: 1 };
  const mx = sx / n, my = sy / n;
  const cxx = sxx / n - mx * mx + 1 / 12, cyy = syy / n - my * my + 1 / 12, cxy = sxy / n - mx * my;
  const tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(1e-9, tr / 2 - disc);
  const fill = n / (4 * Math.PI * Math.sqrt(l1 * l2));
  const elong = Math.sqrt(l1 / l2);
  const hull = hullArea(pts);
  return { area: n, solidity: hull > 0 ? Math.min(1, n / hull) : 1, fill, elong };
}

/** Area of the convex hull (Andrew's monotone chain + shoelace). */
export function hullArea(points: number[][]): number {
  if (points.length < 3) return 0;
  const p = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: number[][] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper: number[][] = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  const h = lower.slice(0, -1).concat(upper.slice(0, -1));
  let a = 0;
  for (let i = 0; i < h.length; i++) {
    const [x1, y1] = h[i], [x2, y2] = h[(i + 1) % h.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

export type TouchWhy = "" | "area_median" | "area_single" | "solidity" | "fill";

export interface TouchBean extends Bean {
  shape: Shape;
  /** the first rule that fired ("" = looks like one bean) */
  touchWhy: TouchWhy;
  /** the Python-parity flag alone (rule 1), kept for measurement */
  touchingParity: boolean;
}

export interface TouchChecks extends Checks {
  /** touching blobs by the parity rule alone (beanfinder.ts) */
  touching_parity: number;
  /** robust one-bean area used by rule 2 */
  single_area: number;
  /** beans hidden in clumps: bean-coloured touching blobs of >= 3 beans (round(area / single_area)) */
  hidden_beans: number;
  /** beans in the photo, counting each clump as the beans it holds instead of one blob */
  bean_equiv: number;
}

export interface TouchResult extends Omit<FindResult, "beans" | "checks"> {
  beans: TouchBean[];
  checks: TouchChecks;
}

export function npMedianOf(v: number[]): number {
  if (!v.length) return 0;
  const a = v.slice().sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/** Colour class (colourgate.ts) of the mean white-balanced colour of one blob's pixels. */
function blobColour(r: FindResult, b: Bean) {
  const w = r.work.width, d = r.work.data, lab = b.comp + 1;
  let sr = 0, sg = 0, sb = 0, n = 0;
  for (let y = b.y0; y < b.y1; y++) for (let x = b.x0; x < b.x1; x++) {
    const i = y * w + x;
    if (r.labels[i] !== lab) continue;
    sr += d[i * 3]; sg += d[i * 3 + 1]; sb += d[i * 3 + 2]; n++;
  }
  return n ? classifyColour(sr / n, sg / n, sb / n) : "off";
}

/**
 * Apply the touching rule to a bean-finder result: every merged/touching blob gets touching = true (so callBean()
 * never classifies it), and checks.touching / too_many_touching are recomputed with the unchanged threshold.
 */
export function markTouching(r: FindResult): TouchResult {
  const w = r.work.width;
  const shapes = r.beans.map((b) => shapeOf(r.labels, w, b.comp + 1, b.x0, b.y0, b.x1, b.y1));
  const singles = shapes.filter((s) => s.solidity >= TOUCH_RULE.SINGLE_SOLIDITY && s.fill >= TOUCH_RULE.SINGLE_FILL && s.elong <= TOUCH_RULE.MAX_ELONG).map((s) => s.area);
  const singleArea = singles.length >= TOUCH_RULE.MIN_SINGLES ? npMedianOf(singles) : r.median_area;
  const beans: TouchBean[] = r.beans.map((b, i) => {
    const s = shapes[i];
    let why: TouchWhy = "";
    if (b.touching) why = "area_median";
    else if (singleArea > 0 && s.area > TOUCH_RULE.AREA_X * singleArea) why = "area_single";
    else if (s.solidity < TOUCH_RULE.MIN_SOLIDITY) why = "solidity";
    else if (s.fill < TOUCH_RULE.MIN_FILL) why = "fill";
    return { ...b, touching: why !== "", touchingParity: b.touching, touchWhy: why, shape: s };
  });
  const n = beans.length;
  const nTouch = beans.reduce((a, b) => a + (b.touching ? 1 : 0), 0);
  let hidden = 0, clumps = 0;
  for (const b of beans) {
    if (!b.touching || !(singleArea > 0)) continue;
    const eq = Math.round(b.shape.area / singleArea);
    if (eq < 3) continue;
    const c = blobColour(r, b);
    if (c !== "pale" && c !== "dark") continue; // skin, roast, leaves, paper: not beans in a clump
    hidden += eq;
    clumps++;
  }
  const equiv = n - clumps + hidden;
  const checks: TouchChecks = {
    ...r.checks,
    touching: nTouch,
    too_many_touching: nTouch > Math.max(2, 0.08 * Math.max(n, 1)) || hidden > Math.max(2, 0.08 * Math.max(equiv, 1)),
    touching_parity: r.checks.touching,
    single_area: singleArea,
    hidden_beans: hidden,
    bean_equiv: equiv,
  };
  return { ...r, beans, checks };
}
