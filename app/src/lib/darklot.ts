/**
 * "A lot of very dark BEANS" — a fixed shape + colour rule (NOT AI), app-only.
 *
 * Why: the colour gate (colourgate.ts) refuses a photo as "not green coffee" (c06) when more than 60% of the blob area is
 * dark (V < 0.30). That is right for dark-roasted coffee, but a real lot of black (defective) green beans — J4ckDev
 * "Negros", 100 black beans, the worst kind of lot — was told "this isn't green coffee" (JUDGE_B_worldbank). Such a lot
 * must go to a person: c07 "not sure, take the sample to the cooperative".
 *
 * Rule: a "too dark beans" photo is a DARK LOT (c07) instead of "not green" (c06) when ALL hold
 *   1. the blobs have bean SHAPE: >= MIN_BEAN_SHAPED of them look like one bean (not touching by touching.ts, and
 *      elongation between ELONG_MIN and ELONG_MAX — beans are ellipses; peas, dried cherries and lentils are rounder);
 *   2. the blobs are alike in SIZE: coefficient of variation of their areas <= MAX_AREA_CV;
 *   3. the dark blobs have the colour of black GREEN coffee, not of roast: median hue inside the colour gate's
 *      green-coffee window (26°–62°) and median saturation <= MAX_DARK_SAT.
 * The bean count range and the touching retake are checked by rules.ts screen() as for any photo.
 *
 * Measured (app/tests/darklot.measure.test.ts, reports_v2/threshold_ship.md): J4ckDev Negros (real, 100 black beans)
 * dark blobs median hue 31.8°, saturation 0.26, elongation 1.53; synthetic dark-roast trays (REGRESSION_B recolour
 * 62,38,24) hue 21.6°, saturation 0.58. The colour limits are set between those two; ONE real photo of black beans and
 * SYNTHETIC roast only — unverified on real roasted coffee. Either outcome is safe: c06 and c07 both give no band.
 */
import type { ColourGate } from "./colourgate";
import { rgbToHsv } from "./colourgate";
import type { TouchResult } from "./touching";

export const DARK_LOT_RULE = {
  MIN_BEAN_SHAPED: 0.8,
  ELONG_MIN: 1.15,
  ELONG_MAX: 2.2,
  MAX_AREA_CV: 0.6,
  DARK_V: 0.3,
  HUE_MIN: 26,
  HUE_MAX: 62,
  MAX_DARK_SAT: 0.42,
};

export interface DarkLot {
  isDarkLot: boolean;
  /** first condition that failed ("" when it is a dark lot; "not_dark" when the colour gate did not say too_dark_beans) */
  why: "" | "not_dark" | "no_blobs" | "shape" | "size" | "colour";
  beanShaped: number;
  areaCV: number;
  darkHue: number;
  darkSat: number;
  medElong: number;
}

const median = (v: number[]) => {
  if (!v.length) return NaN;
  const a = v.slice().sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
};

export function darkLot(r: Pick<TouchResult, "beans" | "work" | "labels">, colour: ColourGate): DarkLot {
  const R = DARK_LOT_RULE;
  const out: DarkLot = { isDarkLot: false, why: "not_dark", beanShaped: 0, areaCV: NaN, darkHue: NaN, darkSat: NaN, medElong: NaN };
  if (colour.reason !== "too_dark_beans") return out;
  const beans = r.beans;
  if (!beans.length) return { ...out, why: "no_blobs" };
  const w = r.work.width, d = r.work.data;
  const hs: number[] = [], ss: number[] = [];
  for (const b of beans) {
    let sr = 0, sg = 0, sb = 0, n = 0;
    for (let y = b.y0; y < b.y1; y++) for (let x = b.x0; x < b.x1; x++) {
      const i = y * w + x;
      if (r.labels[i] !== b.comp + 1) continue;
      sr += d[i * 3]; sg += d[i * 3 + 1]; sb += d[i * 3 + 2]; n++;
    }
    if (!n) continue;
    const [h, s, v] = rgbToHsv(sr / n, sg / n, sb / n);
    if (v < R.DARK_V) { hs.push(h); ss.push(s); }
  }
  const elongs = beans.map((b) => b.shape.elong);
  const shaped = beans.filter((b) => !b.touching && b.shape.elong >= R.ELONG_MIN && b.shape.elong <= R.ELONG_MAX).length / beans.length;
  const areas = beans.map((b) => b.shape.area);
  const mean = areas.reduce((a, x) => a + x, 0) / areas.length;
  const sd = Math.sqrt(areas.reduce((a, x) => a + (x - mean) ** 2, 0) / areas.length);
  const res: DarkLot = { ...out, why: "", beanShaped: shaped, areaCV: mean ? sd / mean : NaN, darkHue: median(hs), darkSat: median(ss), medElong: median(elongs) };
  if (shaped < R.MIN_BEAN_SHAPED) return { ...res, why: "shape" };
  if (!(res.areaCV <= R.MAX_AREA_CV)) return { ...res, why: "size" };
  if (!(res.darkHue >= R.HUE_MIN && res.darkHue <= R.HUE_MAX && res.darkSat <= R.MAX_DARK_SAT)) return { ...res, why: "colour" };
  return { ...res, isDarkLot: true };
}
