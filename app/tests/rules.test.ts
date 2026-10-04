import { describe, expect, it } from "vitest";
import { callBean, decide, decideLot, gate, isDefect, IMPLAUSIBLE_DEFECT_FRAC, MAX_HANDFULS, V2_THRESHOLDS, type BeanCall } from "../src/lib/rules";
import type { Checks } from "../src/lib/beanfinder";
import { findBeans } from "../src/lib/beanfinder";
import { colourGate, classifyColour, type ColourGate } from "../src/lib/colourgate";
import fs from "node:fs";
import path from "node:path";
// @ts-expect-error plain .mjs helper
import { decodeJpegRGB } from "./decode_jpeg.mjs";

const okChecks: Checks = { sheet_luma: 180, too_dark: false, blur_var: 50, too_blurry: false, count: 100, count_ok: true, touching: 0, too_many_touching: false, threshold: 60 };
const okColour: ColourGate = { notGreen: false, reason: "", frac: { pale: 1, dark: 0, white: 0, off: 0 }, largestBlobFrac: 0.001, componentsUsed: 100 };
const calls = (o: Partial<Record<BeanCall, number>>): BeanCall[] => Object.entries(o).flatMap(([k, v]) => Array(v).fill(k as BeanCall));

describe("fail-safe gate (fixed rules, before the model)", () => {
  it("passes a good photo", () => expect(gate(okChecks, okColour)).toBeNull());
  it("dark -> c03, blur -> c02", () => {
    expect(gate({ ...okChecks, too_dark: true, too_blurry: true }, okColour)?.clip).toBe("c03_dark");
    expect(gate({ ...okChecks, too_blurry: true }, okColour)?.clip).toBe("c02_blur");
  });
  it("not green coffee -> c06, before the count check", () => {
    expect(gate({ ...okChecks, count: 3 }, { ...okColour, notGreen: true, reason: "big_blob" })?.clip).toBe("c06_not_green");
  });
  it("count <20 or >160 -> c05; 20 and 160 are fine", () => {
    expect(gate({ ...okChecks, count: 19 }, okColour)?.clip).toBe("c05_count");
    expect(gate({ ...okChecks, count: 161 }, okColour)?.clip).toBe("c05_count");
    expect(gate({ ...okChecks, count: 20 }, okColour)).toBeNull();
    expect(gate({ ...okChecks, count: 160 }, okColour)).toBeNull();
  });
  it("too many touching -> c04", () => {
    expect(gate({ ...okChecks, too_many_touching: true }, okColour)?.clip).toBe("c04_spread");
  });
  it("empty or near-empty sheet -> c05 «use about 100 beans», not c02 blurred (the blur measure needs beans)", () => {
    // what the bean finder reports for a blank sheet (01_blank_white: 0 beans, blur_var < 15) and 11 beans
    expect(gate({ ...okChecks, count: 0, count_ok: false, too_blurry: true, blur_var: 0.4 }, okColour)?.clip).toBe("c05_count");
    expect(gate({ ...okChecks, count: 11, count_ok: false, too_blurry: true, blur_var: 9 }, okColour)?.clip).toBe("c05_count");
    // dark still wins; with 20+ beans a blurred photo is still c02
    expect(gate({ ...okChecks, count: 0, too_dark: true, too_blurry: true }, okColour)?.clip).toBe("c03_dark");
    expect(gate({ ...okChecks, count: 20, too_blurry: true }, okColour)?.clip).toBe("c02_blur");
    // few blobs that are mostly merged beans: not "use 100 beans" but blurred / spread out
    expect(gate({ ...okChecks, count: 10, too_blurry: true, touching: 6, too_many_touching: true }, okColour)?.clip).toBe("c02_blur");
    expect(gate({ ...okChecks, count: 10, touching: 6, too_many_touching: true }, okColour)?.clip).toBe("c04_spread");
  });
  it("touching beats the count range (pushed-together beans are counted as fewer, bigger blobs)", () => {
    expect(gate({ ...okChecks, count: 12, touching: 8, too_many_touching: true }, okColour)?.clip).toBe("c04_spread");
    expect(gate({ ...okChecks, count: 170, touching: 20, too_many_touching: true }, okColour)?.clip).toBe("c04_spread");
  });
  it("a bean-coloured big blob while many blobs touch is a clump of beans -> c04, not «not green coffee»", () => {
    const clump: ColourGate = { ...okColour, notGreen: true, reason: "big_blob", largestBlobFrac: 0.12 };
    expect(gate({ ...okChecks, count: 10, touching: 6, too_many_touching: true }, clump)?.clip).toBe("c04_spread");
    // a skin-coloured big blob (a hand) stays c06, touching or not
    const hand: ColourGate = { ...clump, frac: { pale: 0.3, dark: 0, white: 0, off: 0.7 } };
    expect(gate({ ...okChecks, count: 10, touching: 6, too_many_touching: true }, hand)?.clip).toBe("c06_not_green");
    expect(gate({ ...okChecks, count: 3 }, clump)?.clip).toBe("c06_not_green");
  });
});

describe("per-bean call (v2 two-threshold rule: good if P(good) >= 0.55, defect if 1 - P(good) >= 0.91)", () => {
  const thr = V2_THRESHOLDS;
  it("thresholds: defect 0.91 from reports_v2/clean_model.json, good 0.55 from the pre-registered sweep", () => expect(thr).toEqual({ good: 0.55, defect: 0.91 }));
  it("good at P(good) >= 0.55 even when a defect class is the single largest", () => {
    expect(callBean([0.55, 0.09, 0.09, 0.09, 0.09, 0.09], false, thr)).toBe("good");
    expect(callBean([0.9, 0.02, 0.02, 0.02, 0.02, 0.02], false, thr)).toBe("good");
  });
  it("P(good) 0.50-0.55 is no longer sound: no abstention margin at 0.50 (threshold_sweep.md)", () => {
    expect(callBean([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], false, thr)).toBe("unsure");
    expect(callBean([0.549, 0.451, 0, 0, 0, 0], false, thr)).toBe("unsure");
  });
  it("unsure between the thresholds (0.09 < P(good) < 0.55)", () => {
    expect(callBean([0.49, 0.51, 0, 0, 0, 0], false, thr)).toBe("unsure");
    expect(callBean([0.1, 0.0, 0.0, 0.9, 0, 0], false, thr)).toBe("unsure");
  });
  it("defect when P(defect) >= 0.91; the TYPE is the largest defect class (shown only as 'possible')", () => {
    expect(callBean([0.09, 0.0, 0.0, 0.91, 0, 0], false, thr)).toBe("broken");
    expect(callBean([0.01, 0.3, 0.2, 0.2, 0.1, 0.19], false, thr)).toBe("dark");
    expect(callBean([0.0, 0.1, 0.1, 0.1, 0.1, 0.6], false, thr)).toBe("other_defect");
    expect(isDefect("other_defect")).toBe(true);
  });
  it("touching blobs and missing probabilities are never trusted", () => {
    expect(callBean([1, 0, 0, 0, 0, 0], true, thr)).toBe("unsure");
    expect(callBean(null, false, thr)).toBe("unsure");
  });
  it("the stub (uniform 1/6) is always unsure", () => expect(callBean(Array(6).fill(1 / 6), false, thr)).toBe("unsure"));
});

describe("result decision: band of the point estimate, 'about' for sampling uncertainty", () => {
  it("12/100 defects -> some (c09), not 'about' (7.0%-19.8% sits inside 5-20%); defect TYPES are never spoken", () => {
    const r = decide(calls({ good: 88, dark: 7, insect: 3, broken: 2 }));
    expect(r.band).toBe("some");
    expect(r.about).toBe(false);
    expect(r.canAddHandful).toBe(false);
    expect(r.clips).toEqual(["c09_some"]);
  });
  it("other_defect counts as a defect for the band", () => {
    const r = decide(calls({ good: 88, other_defect: 12 }));
    expect(r.defects).toBe(12);
    expect(r.band).toBe("some");
  });
  it("0/100 -> clean (c08), not about", () => {
    const r = decide(calls({ good: 100 }));
    expect(r.band).toBe("clean");
    expect(r.about).toBe(false);
    expect(r.clips).toEqual(["c08_clean"]);
  });
  it("40/100 -> many (c10), not about", () => {
    const r = decide(calls({ good: 60, broken: 40 }));
    expect(r.clips[0]).toBe("c10_many");
    expect(r.about).toBe(false);
  });
  it("3/100 (the ~3% demo tray) -> 'about clean' + c18 'photograph another handful', never c07", () => {
    const r = decide(calls({ good: 97, dark: 1, insect: 1, broken: 1 }));
    expect(r.band).toBe("clean");
    expect(r.about).toBe(true);
    expect(r.unsureReason).toBe("");
    expect(r.canAddHandful).toBe(true);
    expect(r.clips).toEqual(["c08_clean", "c18_another"]);
  });
  it("10% of 50 -> 'about some' (4.3%-21.4% crosses both edges), with c18 and no type clip", () => {
    const r = decide(calls({ good: 45, dark: 5 }));
    expect(r.band).toBe("some");
    expect(r.about).toBe(true);
    expect(r.clips).toEqual(["c09_some", "c18_another"]);
  });
  it("25/97 (the ~30% demo tray) -> 'about many'", () => {
    const r = decide(calls({ good: 72, dark: 11, insect: 1, broken: 8, unhulled: 5, unsure: 3 }));
    expect(r.band).toBe("many");
    expect(r.about).toBe(true);
    expect(r.clips.slice(0, 2)).toEqual(["c10_many", "c18_another"]);
  });
  it("band edges follow the point estimate: 5% is some, 20% is some, 20.x% is many, 4.x% is clean", () => {
    expect(decide(calls({ good: 95, dark: 5 })).band).toBe("some");
    expect(decide(calls({ good: 80, dark: 20 })).band).toBe("some");
    expect(decide(calls({ good: 79, dark: 21 })).band).toBe("many");
    expect(decide(calls({ good: 96, dark: 4 })).band).toBe("clean");
  });
  it("c07 is never used for sampling uncertainty: no band-spanning interval gives c07", () => {
    for (let n = 20; n <= 160; n += 7)
      for (let k = 0; k <= Math.floor(n * 0.6); k++) {
        const r = decide(calls({ good: n - k, broken: k }));
        expect(r.band).not.toBe("unsure");
        expect(r.clips).not.toContain("c07_unsure");
      }
  });
  it(">15% unsure -> c07 even if the rest looks clean (model uncertainty)", () => {
    const r = decide(calls({ good: 84, unsure: 16 }));
    expect(r.band).toBe("unsure");
    expect(r.unsureReason).toBe("too_many_unsure");
    expect(r.clips).toEqual(["c07_unsure"]);
    expect(r.canAddHandful).toBe(false);
  });
  it("exactly 15% unsure is still allowed", () => expect(decide(calls({ good: 85, unsure: 15 })).unsureReason).not.toBe("too_many_unsure"));
  it("nothing answered -> c07", () => expect(decide(calls({ unsure: 30 })).clips).toEqual(["c07_unsure"]));
});

describe("implausible-sample fail-safe (> 60% defects among answered beans -> c07)", () => {
  it(`threshold is ${IMPLAUSIBLE_DEFECT_FRAC}`, () => expect(IMPLAUSIBLE_DEFECT_FRAC).toBe(0.6));
  it("60% exactly is still a band (many); 61% is not sure", () => {
    expect(decide(calls({ good: 40, broken: 60 })).band).toBe("many");
    const r = decide(calls({ good: 39, broken: 61 }));
    expect(r.band).toBe("unsure");
    expect(r.unsureReason).toBe("implausible");
    expect(r.clips).toEqual(["c07_unsure"]);
  });
  it("the real CBD AAA demo photo as the app counts it (0 good, 39 broken, 8 husk, 3 ?) -> not sure, implausible", () => {
    const r = decide(calls({ broken: 39, unhulled: 8, unsure: 3 }));
    expect(r.band).toBe("unsure");
    expect(r.unsureReason).toBe("implausible");
  });
  it("implausible wins over too-many-unsure (CBD Bits: 29 answered, all defects, 21 unsure)", () => {
    const r = decide(calls({ dark: 5, insect: 1, broken: 14, unhulled: 9, unsure: 21 }));
    expect(r.unsureReason).toBe("implausible");
  });
  it("the ratio uses ANSWERED beans only (unsure beans do not dilute it)", () => {
    const r = decide(calls({ good: 10, broken: 20, unsure: 4 })); // 20/30 = 67% of answered
    expect(r.unsureReason).toBe("implausible");
  });
});

describe("pooling handfuls of the same coffee (decideLot)", () => {
  const h3 = calls({ good: 97, dark: 1, insect: 1, broken: 1 });
  it("one handful = decide()", () => expect(decideLot([h3])).toEqual(decide(h3)));
  it("pools counts and the interval; interval narrows", () => {
    const one = decide(h3);
    const two = decideLot([h3, calls({ good: 98, broken: 2 })]);
    expect(two.total).toBe(200);
    expect(two.defects).toBe(5);
    expect(two.handfuls).toBe(2);
    expect(two.interval.hi - two.interval.lo).toBeLessThan(one.interval.hi - one.interval.lo);
  });
  it(`offers another handful only up to ${MAX_HANDFULS} photos`, () => {
    const lot = [h3, h3, h3];
    const r = decideLot(lot);
    expect(r.about).toBe(true);
    expect(r.canAddHandful).toBe(false);
    expect(r.clips).not.toContain("c18_another");
    expect(decideLot([h3, h3]).canAddHandful).toBe(true);
  });
  it("a not-sure handful makes the whole lot not sure (pooling never dilutes the fail-safe)", () => {
    const bad = calls({ good: 30, broken: 70 }); // implausible on its own
    const r = decideLot([calls({ good: 100 }), calls({ good: 100 }), bad]); // pooled 70/300 = 23% would be 'many'
    expect(r.band).toBe("unsure");
    expect(r.unsureReason).toBe("implausible");
    expect(r.clips).toEqual(["c07_unsure"]);
  });
  it("can turn 'about clean' into a firm 'clean' with more beans", () => {
    const r = decideLot([calls({ good: 99, dark: 1 }), calls({ good: 100 }), calls({ good: 100 })]); // 1/300
    expect(r.band).toBe("clean");
    expect(r.about).toBe(false);
  });
});

describe("no clip ever mentions a price, pest species or pesticide (fixed list)", () => {
  it("every clip is one of the fixed ids", () => {
    const all = new Set<string>();
    for (const o of [{ good: 100 }, { good: 50, dark: 50 }, { good: 80, insect: 20 }, { unsure: 50 }, { good: 97, broken: 3 }]) decide(calls(o)).clips.forEach((c) => all.add(c));
    for (const c of all) expect(c).toMatch(/^c\d\d_/);
    for (const c of ["c11_dark_beans", "c12_insect", "c13_broken", "c14_unhulled"]) expect(all.has(c)).toBe(false);
  });
});

describe("colour gate (fixed rule, synthetic checks)", () => {
  const ROOT = path.resolve(__dirname, "../..");
  const load = (rel: string) => {
    const im = decodeJpegRGB(fs.readFileSync(path.join(ROOT, rel)));
    return { width: im.width, height: im.height, data: new Uint8Array(im.data) };
  };
  it("classifies typical colours", () => {
    expect(classifyColour(150, 130, 95)).toBe("pale"); // green coffee after white balance (hue ~38°)
    expect(classifyColour(60, 40, 30)).toBe("dark"); // dark roast
    expect(classifyColour(150, 95, 60)).toBe("off"); // medium roast / skin (hue ~23°)
    expect(classifyColour(90, 160, 70)).toBe("off"); // leaf green
    expect(classifyColour(230, 228, 220)).toBe("white"); // rice
  });
  const NORMALES = "data/raw/green/ImageDataset/Normales.jpg";
  const AAA1 = "data/raw/cbd/CBD_Coffee Bean Dataset/CBD_Coffee Bean Dataset/AAA/1.jpg";
  const have = (rel: string) => fs.existsSync(path.join(ROOT, rel));
  // the published repo has no data/raw (dataset licences): these skip cleanly when the photos are absent
  it.skipIf(!have(NORMALES) || !have(AAA1))("real green coffee passes (J4ckDev Normales, CBD AAA/1)", () => {
    for (const f of ["data/raw/green/ImageDataset/Normales.jpg", "data/raw/cbd/CBD_Coffee Bean Dataset/CBD_Coffee Bean Dataset/AAA/1.jpg"]) {
      const r = findBeans(load(f), { crops: false });
      expect(colourGate(r).notGreen).toBe(false);
    }
  });
  it.skipIf(!have(NORMALES))("SYNTHETIC roasted tray (Normales beans recoloured brown) is refused", () => {
    const img = load("data/raw/green/ImageDataset/Normales.jpg");
    const d = img.data;
    for (let i = 0; i < d.length; i += 3) {
      const l = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      if (l < 150) { // bean pixels (sheet is light grey)
        d[i] = Math.round(l * 0.75); d[i + 1] = Math.round(l * 0.5); d[i + 2] = Math.round(l * 0.32);
      }
    }
    const g = colourGate(findBeans(img, { crops: false }));
    expect(g.notGreen).toBe(true);
  });
  it.skipIf(!have(AAA1))("SYNTHETIC hand (skin-coloured blob from the frame edge) is refused as not green coffee", () => {
    const img = load("data/raw/cbd/CBD_Coffee Bean Dataset/CBD_Coffee Bean Dataset/AAA/1.jpg");
    const { width: w, height: h, data: d } = img;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const dx = (x - w * 0.15) / (w * 0.3), dy = (y - h * 0.5) / (h * 0.22);
      if (dx * dx + dy * dy < 1) { const i = (y * w + x) * 3; d[i] = 224; d[i + 1] = 172; d[i + 2] = 140; }
    }
    const r = findBeans(img, { crops: false });
    const g = colourGate(r);
    expect(g.notGreen).toBe(true);
    expect(gate(r.checks, g)?.clip).toBe("c06_not_green");
  });
});
