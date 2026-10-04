/**
 * "Dark lot" (src/lib/darklot.ts + rules.ts screen()): a lot of bean-shaped, very dark beans gets c07 "not sure, take the
 * sample to the cooperative" instead of c06 "not green coffee"; dark-ROAST coloured beans still get c06.
 * Synthetic trays are made here from the app's own SYNTHETIC demo tray (J4ckDev crops on a grey sheet) by recolouring
 * every bean pixel, keeping its luminance as texture: black green beans (low saturation, olive hue) and dark roast
 * (REGRESSION_B's recolour 62,38,24). The real J4ckDev "Negros" photo is used when data/raw is present.
 */
import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { findBeans, type Checks } from "../src/lib/beanfinder";
import { colourGate, type ColourGate } from "../src/lib/colourgate";
import { darkLot, DARK_LOT_RULE } from "../src/lib/darklot";
import { gate, screen } from "../src/lib/rules";
import { markTouching } from "../src/lib/touching";
import { APP, ROOT, loadImage } from "./node_pipeline";

const okChecks: Checks = { sheet_luma: 180, too_dark: false, blur_var: 50, too_blurry: false, count: 100, count_ok: true, touching: 0, too_many_touching: false, threshold: 60 };
const darkColour = (isDarkLot: boolean): ColourGate => ({
  notGreen: true, reason: "too_dark_beans", frac: { pale: 0.05, dark: 0.95, white: 0, off: 0 }, largestBlobFrac: 0.002, componentsUsed: 100,
  darkLot: { isDarkLot, why: isDarkLot ? "" : "colour", beanShaped: 1, areaCV: 0.2, darkHue: 32, darkSat: 0.26, medElong: 1.5 },
});

describe("screen(): dark lot -> c07 not sure, everything else as the gate", () => {
  it("a dark lot of bean-shaped beans is 'not sure (dark_lot)', never a band and never counts", () => {
    const o = screen(okChecks, darkColour(true));
    expect(o?.kind).toBe("result");
    if (o?.kind !== "result") return;
    expect(o.band).toBe("unsure");
    expect(o.unsureReason).toBe("dark_lot");
    expect(o.clips).toEqual(["c07_unsure"]);
    expect(o.total).toBe(100);
    expect(o.answered).toBe(0);
  });
  it("too dark beans that are NOT a dark lot (roast colour) stay c06 not green", () => {
    expect(screen(okChecks, darkColour(false))).toEqual(gate(okChecks, darkColour(false)));
    expect((screen(okChecks, darkColour(false)) as { clip: string }).clip).toBe("c06_not_green");
  });
  it("photo checks first: a dark photo is c03 and a blurred one c02, even for a dark lot", () => {
    expect((screen({ ...okChecks, too_dark: true }, darkColour(true)) as { clip: string }).clip).toBe("c03_dark");
    expect((screen({ ...okChecks, too_blurry: true }, darkColour(true)) as { clip: string }).clip).toBe("c02_blur");
  });
  it("outside the bean count or with beans touching it is NOT called a dark lot (stays c06)", () => {
    expect((screen({ ...okChecks, count: 9 }, darkColour(true)) as { clip: string }).clip).toBe("c06_not_green");
    expect((screen({ ...okChecks, count: 170 }, darkColour(true)) as { clip: string }).clip).toBe("c06_not_green");
    expect((screen({ ...okChecks, too_many_touching: true, touching: 20 }, darkColour(true)) as { clip: string }).clip).toBe("c06_not_green");
  });
  it("other not-green reasons are untouched (a hand, leaves: big blob / off colour)", () => {
    const hand: ColourGate = { ...darkColour(true), reason: "off_colour" };
    expect((screen(okChecks, hand) as { clip: string }).clip).toBe("c06_not_green");
  });
});

/** Recolour every bean pixel of a demo tray: out = clip(L / L90 * rgb), L = luma of the original pixel. */
function recolour(file: string, rgb: [number, number, number]) {
  const im = loadImage(file);
  const d = im.data;
  const lum: number[] = [];
  for (let i = 0; i < d.length; i += 3) { const L = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; if (L < 200) lum.push(L); }
  lum.sort((a, b) => a - b);
  const L90 = lum[Math.floor(lum.length * 0.9)];
  const out = new Uint8Array(d.length);
  for (let i = 0; i < d.length; i += 3) {
    const L = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    if (L >= 200) { out[i] = d[i]; out[i + 1] = d[i + 1]; out[i + 2] = d[i + 2]; continue; }
    const k = L / L90;
    out[i] = Math.min(255, k * rgb[0]); out[i + 1] = Math.min(255, k * rgb[1]); out[i + 2] = Math.min(255, k * rgb[2]);
  }
  return { width: im.width, height: im.height, data: out };
}
const run = (img: { width: number; height: number; data: Uint8Array }) => {
  const r = markTouching(findBeans(img));
  const c = colourGate(r);
  c.darkLot = darkLot(r, c);
  return { c, o: screen(r.checks, c) };
};

const TRAY = path.join(APP, "public/demo/tray_synthetic_00.jpg");
describe.skipIf(!fs.existsSync(TRAY))("SYNTHETIC recoloured trays through the real finder, touching rule and colour gate", () => {
  it("black green beans (olive-grey, low saturation) -> dark lot -> c07", () => {
    const { c, o } = run(recolour(TRAY, [58, 56, 44]));
    expect(c.reason).toBe("too_dark_beans");
    expect(c.darkLot?.isDarkLot).toBe(true);
    expect(o?.kind === "result" && o.unsureReason).toBe("dark_lot");
  });
  it("dark roast (62,38,24: REGRESSION_B's recolour) -> still c06 not green coffee", () => {
    const { c, o } = run(recolour(TRAY, [62, 38, 24]));
    expect(c.reason).toBe("too_dark_beans");
    expect(c.darkLot?.isDarkLot).toBe(false);
    expect(c.darkLot?.why).toBe("colour");
    expect(o?.kind === "retake" && o.clip).toBe("c06_not_green");
  });
  it("medium roast (120,72,40) -> c06 (off colour, not a dark-lot question)", () => {
    const { c, o } = run(recolour(TRAY, [120, 72, 40]));
    expect(c.reason).toBe("off_colour");
    expect(o?.kind === "retake" && o.clip).toBe("c06_not_green");
  });
  it(`the rule's colour limits sit between the measured black beans and roast (sat <= ${DARK_LOT_RULE.MAX_DARK_SAT})`, () => {
    expect(DARK_LOT_RULE.MAX_DARK_SAT).toBeGreaterThan(0.26);
    expect(DARK_LOT_RULE.MAX_DARK_SAT).toBeLessThan(0.58);
  });
});

const NEGROS = path.join(ROOT, "data/raw/green/ImageDataset/Negros.jpg");
const CEREZA = path.join(ROOT, "data/raw/green/ImageDataset/CerezaSeca.jpg");
describe.skipIf(!fs.existsSync(NEGROS))("real J4ckDev photos (data/raw, not in the published repo)", () => {
  it("Negros (100 black beans, a real very bad lot) -> c07 not sure, no longer c06 'not green coffee'", () => {
    const { c, o } = run(loadImage(NEGROS));
    expect(c.darkLot?.isDarkLot).toBe(true);
    expect(o?.kind === "result" && o.band).toBe("unsure");
    expect(o?.kind === "result" && o.unsureReason).toBe("dark_lot");
  });
  it("CerezaSeca (9 dried cherries, not hulled) stays c06", () => {
    const { o } = run(loadImage(CEREZA));
    expect(o?.kind === "retake" && o.clip).toBe("c06_not_green");
  });
});
