/**
 * "Is this green coffee at all?" — a fixed colour rule (NOT AI), run before the classifier.
 *
 * Works on the bean finder's white-balanced work image: every foreground component bigger than a speck
 * (including ones cut by the frame edge, so a hand reaching in from the side counts) gets the mean colour of
 * its mask pixels, classified in HSV as:
 *   dark  : V < 0.30                       (black beans, dark roast)
 *   white : S < 0.15 and V >= 0.60         (rice, paper, salt...)
 *   off   : H < 26° or H > 62° or S > 0.66 (skin, medium roast, red lentils, leaves...)
 *   pale  : everything else                (what green coffee looks like after white balance)
 * The sample is "not green coffee" when, by area, off+white > 50%, or dark > 60%, or one blob covers >= 8% of
 * the photo (beans are each well under 2.5% of the frame in every test photo).
 *
 * Tuned on real green-coffee photos only: all 464 CBD photos and 9 of 11 J4ckDev class photos pass; the two
 * J4ckDev single-class photos that are 94–100% black or dried-cherry are refused (dark > 60%). Real hands,
 * roasted beans and leaves were NOT available tonight — synthetic checks only (tests/rules.test.ts). UNVERIFIED
 * on real negatives until real not-a-bean photos are run.
 */
import type { FindResult } from "./beanfinder";
import type { DarkLot } from "./darklot";

export const COLOUR_RULE = {
  darkV: 0.3,
  whiteS: 0.15,
  whiteV: 0.6,
  hueMin: 26,
  hueMax: 62,
  satMax: 0.66,
  maxOffFrac: 0.5,
  maxDarkFrac: 0.6,
  maxBlobFrac: 0.08,
};

export type ColourClass = "pale" | "dark" | "white" | "off";

export interface ColourGate {
  notGreen: boolean;
  reason: "" | "off_colour" | "too_dark_beans" | "big_blob";
  frac: Record<ColourClass, number>;
  largestBlobFrac: number;
  componentsUsed: number;
  /** "too dark" blobs that have bean shape and the colour of black green coffee (darklot.ts); set by the worker */
  darkLot?: DarkLot;
}

export function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  // same as Python colorsys.rgb_to_hsv on [0,1] inputs, hue in degrees
  r /= 255; g /= 255; b /= 255;
  const maxc = Math.max(r, g, b);
  const minc = Math.min(r, g, b);
  const v = maxc;
  if (minc === maxc) return [0, 0, v];
  const s = (maxc - minc) / maxc;
  const rc = (maxc - r) / (maxc - minc);
  const gc = (maxc - g) / (maxc - minc);
  const bc = (maxc - b) / (maxc - minc);
  let h: number;
  if (r === maxc) h = bc - gc;
  else if (g === maxc) h = 2.0 + rc - bc;
  else h = 4.0 + gc - rc;
  h = ((h / 6.0) % 1.0 + 1.0) % 1.0;
  return [h * 360, s, v];
}

export function classifyColour(r: number, g: number, b: number): ColourClass {
  const [h, s, v] = rgbToHsv(r, g, b);
  const R = COLOUR_RULE;
  if (v < R.darkV) return "dark";
  if (s < R.whiteS && v >= R.whiteV) return "white";
  if (h < R.hueMin || h > R.hueMax || s > R.satMax) return "off";
  return "pale";
}

export function colourGate(res: Pick<FindResult, "work" | "mask" | "components" | "median_area">): ColourGate {
  const { work, mask, components, median_area } = res;
  const w = work.width;
  const h = work.height;
  const minArea = Math.max(30, 0.25 * median_area);
  const tot: Record<ColourClass, number> = { pale: 0, dark: 0, white: 0, off: 0 };
  let largest = 0;
  let used = 0;
  for (const [x0, y0, x1, y1, area] of components) {
    if (area < minArea) continue;
    let sr = 0, sg = 0, sb = 0, n = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = y * w + x;
        if (!mask[i]) continue;
        sr += work.data[i * 3];
        sg += work.data[i * 3 + 1];
        sb += work.data[i * 3 + 2];
        n++;
      }
    }
    if (!n) continue;
    tot[classifyColour(sr / n, sg / n, sb / n)] += area;
    largest = Math.max(largest, area);
    used++;
  }
  const A = tot.pale + tot.dark + tot.white + tot.off;
  const frac: Record<ColourClass, number> = A
    ? { pale: tot.pale / A, dark: tot.dark / A, white: tot.white / A, off: tot.off / A }
    : { pale: 0, dark: 0, white: 0, off: 0 };
  const largestBlobFrac = largest / (w * h);
  const R = COLOUR_RULE;
  let reason: ColourGate["reason"] = "";
  if (A > 0) {
    if (largestBlobFrac >= R.maxBlobFrac) reason = "big_blob";
    else if (frac.off + frac.white > R.maxOffFrac) reason = "off_colour";
    else if (frac.dark > R.maxDarkFrac) reason = "too_dark_beans";
  }
  return { notGreen: reason !== "", reason, frac, largestBlobFrac, componentsUsed: used };
}
