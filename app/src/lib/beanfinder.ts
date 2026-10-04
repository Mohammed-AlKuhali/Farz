/**
 * Farz bean finder — deterministic image processing, NO machine learning.
 *
 * Line-for-line TypeScript port of src/beans/beanfinder.py (the frozen reference). Same constants,
 * same steps, same order:
 *   downscale (Pillow BILINEAR) -> sheet colour from a 4% border band (median) -> white balance ->
 *   "difference from the sheet" = max(darkness, 1.5 x chroma) -> Otsu threshold (floor 18) ->
 *   3x3 open then close -> 4-connected components -> speck / frame-edge filter -> touching flag ->
 *   square padded crop resized to CROP px (Pillow BILINEAR).
 *
 * numpy arithmetic on float32 arrays is reproduced with Math.fround at every step (NumPy 2 "weak"
 * Python scalars => float32 maths), so on identical input pixels the results are bit-identical
 * except `blur_var`, which numpy sums pairwise in float32 (we sum in float64; difference ~1e-5
 * relative, only compared against the 15.0 threshold). See tests/beanfinder.parity.test.ts.
 */
import { pillowResizeBilinear, type RGBImage } from "./pillowResize";

export const WORK_MAX = 1200;
export const SHEET_TARGET = 235.0;
export const PAD = 0.18;
export const CROP = 128;
export const MIN_BEANS = 20;
export const MAX_BEANS = 160;

const f = Math.fround;
const F_0299 = f(0.299);
const F_0587 = f(0.587);
const F_0114 = f(0.114);

export interface Bean {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  area: number;
  touching: boolean;
  /** index into FindResult.components (its pixels carry label comp + 1 in FindResult.labels) */
  comp: number;
  /** CROP x CROP x 3 uint8 RGB (white-balanced), the classifier input before /255 */
  crop?: Uint8Array;
}

export interface Checks {
  sheet_luma: number;
  too_dark: boolean;
  blur_var: number;
  too_blurry: boolean;
  count: number;
  count_ok: boolean;
  touching: number;
  too_many_touching: boolean;
  threshold: number;
}

export interface FindResult {
  beans: Bean[];
  /** white-balanced working image, uint8 RGB */
  work: RGBImage;
  /** work px per original px */
  scale: number;
  checks: Checks;
  median_area: number;
  /** foreground mask after morphology (1 = not sheet), work-image sized; used by the colour gate */
  mask: Uint8Array;
  /** all connected components before filtering: [x0,y0,x1,y1,area] */
  components: number[][];
  /** per work-image pixel: 0 = sheet, k + 1 = components[k] (app-only, used by touching.ts) */
  labels: Int32Array;
  /** sheet colour (per channel, before white balance) */
  sheet: [number, number, number];
}

/** Python 3 round(): round half to even. */
export function pyRound(x: number): number {
  const r = Math.round(x); // half up
  if (Math.abs(x % 1) === 0.5) {
    // exactly half: choose even
    const fl = Math.floor(x);
    return fl % 2 === 0 ? fl : fl + 1;
  }
  return r;
}

/** numpy.percentile(values, q) with the default 'linear' method (values need not be sorted). */
export function npPercentileLinear(values: number[], q: number): number {
  const arr = values.slice().sort((a, b) => a - b);
  const n = arr.length;
  const qq = q / 100;
  const v = (n - 1) * qq;
  let prev = Math.floor(v);
  let next = prev + 1;
  if (v >= n - 1) {
    prev = n - 1;
    next = n - 1;
  }
  if (v < 0) {
    prev = 0;
    next = 0;
  }
  const a = arr[prev];
  const b = arr[next];
  const gamma = v >= n - 1 ? v + 1 : v - prev; // numpy computes gamma against the adjusted index
  const diff = b - a;
  return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
}

/** numpy.median for a list of numbers (float64 result). */
export function npMedian(values: number[]): number {
  const arr = values.slice().sort((a, b) => a - b);
  const n = arr.length;
  if (n === 0) return NaN;
  const m = n >> 1;
  return n % 2 ? arr[m] : (arr[m - 1] + arr[m]) / 2;
}

export function otsu(values: Float32Array): number {
  const hist = new Float64Array(256);
  for (let i = 0; i < values.length; i++) {
    let v = values[i];
    if (v < 0) v = 0;
    else if (v > 255) v = 255;
    hist[Math.trunc(v)] += 1;
  }
  let total = 0;
  let sumAll = 0;
  for (let t = 0; t < 256; t++) {
    total += hist[t];
    sumAll += t * hist[t];
  }
  let wB = 0;
  let sumB = 0;
  let best = -1.0;
  let thr = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const d = mB - mF;
    const between = wB * wF * (d * d);
    if (between > best) {
      best = between;
      thr = t;
    }
  }
  return thr;
}

function erode(m: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 1;
      for (let dy = -1; dy <= 1 && v; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) {
          v = 0;
          break;
        }
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w || !m[yy * w + xx]) {
            v = 0;
            break;
          }
        }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

function dilate(m: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let dy = -1; dy <= 1 && !v; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx >= 0 && xx < w && m[yy * w + xx]) {
            v = 1;
            break;
          }
        }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

/**
 * 4-connected labelling, iterative flood fill (same as the Python reference). Returns [x0,y0,x1,y1,area].
 * `labelsOut` (optional, w*h) receives the label image: components[k] has label k + 1.
 */
export function components(mask: Uint8Array, w: number, h: number, labelsOut?: Int32Array): number[][] {
  const labels = labelsOut ?? new Int32Array(w * h);
  const comps: number[][] = [];
  let nxt = 1;
  const stack = new Int32Array(w * h);
  for (let start = 0; start < w * h; start++) {
    if (!mask[start] || labels[start]) continue;
    let sp = 0;
    stack[sp++] = start;
    labels[start] = nxt;
    let xs0 = w, ys0 = h, xs1 = -1, ys1 = -1, area = 0;
    while (sp > 0) {
      const i = stack[--sp];
      const y = (i / w) | 0;
      const x = i - y * w;
      area++;
      if (x < xs0) xs0 = x;
      if (x > xs1) xs1 = x;
      if (y < ys0) ys0 = y;
      if (y > ys1) ys1 = y;
      if (x > 0) {
        const j = i - 1;
        if (mask[j] && !labels[j]) { labels[j] = nxt; stack[sp++] = j; }
      }
      if (x < w - 1) {
        const j = i + 1;
        if (mask[j] && !labels[j]) { labels[j] = nxt; stack[sp++] = j; }
      }
      if (y > 0) {
        const j = i - w;
        if (mask[j] && !labels[j]) { labels[j] = nxt; stack[sp++] = j; }
      }
      if (y < h - 1) {
        const j = i + w;
        if (mask[j] && !labels[j]) { labels[j] = nxt; stack[sp++] = j; }
      }
    }
    comps.push([xs0, ys0, xs1 + 1, ys1 + 1, area]);
    nxt++;
  }
  return comps;
}

function laplacianVar(gray: Float32Array, w: number, h: number): number {
  // lap = -4*c + up + down + left + right, float32 per step (as numpy); variance in float64
  const n = (w - 2) * (h - 2);
  if (n <= 0) return 0;
  const lap = new Float32Array(n);
  let k = 0;
  let sum = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      let v = f(f(-4) * gray[i]);
      v = f(v + gray[i - w]);
      v = f(v + gray[i + w]);
      v = f(v + gray[i - 1]);
      v = f(v + gray[i + 1]);
      lap[k++] = v;
      sum += v;
    }
  }
  const mean = sum / n;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const d = lap[i] - mean;
    acc += d * d;
  }
  return acc / n;
}

/** Median of the N/2-th and N/2+1-th order statistics from a 256-bin histogram (numpy median, float32 inputs). */
function histMedian(hist: Float64Array, n: number): number {
  const order = (k: number) => {
    // k is 0-based order statistic
    let c = 0;
    for (let v = 0; v < 256; v++) {
      c += hist[v];
      if (c > k) return v;
    }
    return 255;
  };
  if (n % 2) return order((n - 1) / 2);
  return f(f(order(n / 2 - 1) + order(n / 2)) / 2);
}

export interface FindOptions {
  /** produce 128 px crops (default true) */
  crops?: boolean;
}

/**
 * Run the bean finder on an RGB image (original resolution, already EXIF-oriented by the caller if desired).
 */
export function findBeans(input: RGBImage, opts: FindOptions = {}): FindResult {
  const wantCrops = opts.crops !== false;
  const s = Math.min(1.0, WORK_MAX / Math.max(input.width, input.height));
  let img = input;
  if (s < 1.0) {
    img = pillowResizeBilinear(input, pyRound(input.width * s), pyRound(input.height * s));
  }
  const w = img.width;
  const h = img.height;
  const a = img.data; // uint8, exact as float32

  // 1) sheet colour = median of a 4% border band (rows top/bottom, columns left/right; corners counted twice)
  const b = Math.max(2, Math.trunc(0.04 * Math.min(h, w)));
  const hists = [new Float64Array(256), new Float64Array(256), new Float64Array(256)];
  let nB = 0;
  const addPx = (x: number, y: number) => {
    const p = (y * w + x) * 3;
    hists[0][a[p]]++;
    hists[1][a[p + 1]]++;
    hists[2][a[p + 2]]++;
    nB++;
  };
  for (let y = 0; y < b; y++) for (let x = 0; x < w; x++) addPx(x, y);
  for (let y = h - b; y < h; y++) for (let x = 0; x < w; x++) addPx(x, y);
  for (let y = 0; y < h; y++) for (let x = 0; x < b; x++) addPx(x, y);
  for (let y = 0; y < h; y++) for (let x = w - b; x < w; x++) addPx(x, y);
  const sheet: [number, number, number] = [
    histMedian(hists[0], nB),
    histMedian(hists[1], nB),
    histMedian(hists[2], nB),
  ];

  // 2) white balance from the sheet (float32)
  const fac = sheet.map((v) => f(SHEET_TARGET / f(Math.max(v, 1.0))));
  const npx = w * h;
  const wb = new Float32Array(npx * 3);
  for (let i = 0; i < npx * 3; i++) {
    let v = f(a[i] * fac[i % 3]);
    if (v < 0) v = 0;
    else if (v > 255) v = 255;
    wb[i] = v;
  }

  // 3) difference from the sheet
  const luma = new Float32Array(npx);
  const diff = new Float32Array(npx);
  for (let i = 0; i < npx; i++) {
    const r = wb[i * 3], g = wb[i * 3 + 1], bl = wb[i * 3 + 2];
    const l = f(f(f(F_0299 * r) + f(F_0587 * g)) + f(F_0114 * bl));
    luma[i] = l;
    const mx = Math.max(r, g, bl);
    const mn = Math.min(r, g, bl);
    const chroma = f(mx - mn);
    const d1 = f(SHEET_TARGET - l);
    const d2 = f(chroma * 1.5);
    diff[i] = d1 > d2 ? d1 : d2;
  }
  const thr = Math.max(otsu(diff), 18.0);
  let mask: Uint8Array = new Uint8Array(npx);
  for (let i = 0; i < npx; i++) mask[i] = diff[i] > thr ? 1 : 0;

  // 4) open then close
  mask = dilate(erode(mask, w, h), w, h);
  mask = erode(dilate(mask, w, h), w, h);
  const labels = new Int32Array(npx);
  const comps = components(mask, w, h, labels);

  let med = 0;
  if (comps.length) {
    const areas = comps.map((c) => c[4]);
    const p90 = npPercentileLinear(areas, 90);
    const cut = Math.max(30, 0.15 * p90);
    const big = areas.filter((ar) => ar >= cut);
    med = big.length ? npMedian(big) : 0.0;
  }

  // uint8 work image (truncation, as numpy astype(uint8))
  const work = new Uint8Array(npx * 3);
  for (let i = 0; i < npx * 3; i++) work[i] = Math.trunc(wb[i]);

  const beans: Bean[] = [];
  for (let ci = 0; ci < comps.length; ci++) {
    const [x0, y0, x1, y1, ar] = comps[ci];
    if (med === 0 || ar < 0.25 * med) continue;
    if (x0 <= 0 || y0 <= 0 || x1 >= w || y1 >= h) continue;
    const bean: Bean = { x0, y0, x1, y1, area: ar, touching: ar > 1.8 * med, comp: ci };
    if (wantCrops) {
      const side = Math.max(x1 - x0, y1 - y0);
      const pad = Math.trunc(pyRound(PAD * side));
      const half = Math.floor(side / 2) + pad;
      const cx = Math.floor((x0 + x1) / 2);
      const cy = Math.floor((y0 + y1) / 2);
      const S = 2 * half;
      const canvas = new Uint8Array(S * S * 3).fill(SHEET_TARGET); // 235.0 -> uint8 235
      const sx0 = Math.max(0, cx - half), sy0 = Math.max(0, cy - half);
      const sx1 = Math.min(w, cx + half), sy1 = Math.min(h, cy + half);
      const ox = cx - half, oy = cy - half;
      for (let y = sy0; y < sy1; y++) {
        const src = (y * w + sx0) * 3;
        const dst = ((y - oy) * S + (sx0 - ox)) * 3;
        canvas.set(work.subarray(src, src + (sx1 - sx0) * 3), dst);
      }
      bean.crop = pillowResizeBilinear({ width: S, height: S, data: canvas }, CROP, CROP).data;
    }
    beans.push(bean);
  }

  const sheetLuma = f(f(f(F_0299 * sheet[0]) + f(F_0587 * sheet[1])) + f(F_0114 * sheet[2]));
  const blurVar = laplacianVar(luma, w, h);
  const n = beans.length;
  const nTouch = beans.reduce((acc, bb) => acc + (bb.touching ? 1 : 0), 0);
  const checks: Checks = {
    sheet_luma: sheetLuma,
    too_dark: sheetLuma < 90,
    blur_var: blurVar,
    too_blurry: blurVar < 15.0,
    count: n,
    count_ok: MIN_BEANS <= n && n <= MAX_BEANS,
    touching: nTouch,
    too_many_touching: nTouch > Math.max(2, 0.08 * Math.max(n, 1)),
    threshold: thr,
  };
  return {
    beans,
    work: { width: w, height: h, data: work },
    scale: s,
    checks,
    median_area: med,
    mask,
    components: comps,
    labels,
    sheet,
  };
}
