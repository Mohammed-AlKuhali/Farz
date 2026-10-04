/**
 * The touching rule (src/lib/touching.ts): merged/touching blobs are never classified as one bean, and when there
 * are more of them than max(2, 8% of beans) the photo gets the spread-out retake (c04) before any verdict.
 */
import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { components, findBeans } from "../src/lib/beanfinder";
import { hullArea, markTouching, shapeOf, TOUCH_RULE } from "../src/lib/touching";
import { colourGate } from "../src/lib/colourgate";
import { callBean, gate, V2_THRESHOLDS } from "../src/lib/rules";
import { loadImage, APP } from "./node_pipeline";

/** a filled ellipse (or union of ellipses) in a w x h mask */
function ellipses(w: number, h: number, es: [number, number, number, number][]): Uint8Array {
  const m = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    for (const [cx, cy, a, b] of es) if (((x + 0.5 - cx) / a) ** 2 + ((y + 0.5 - cy) / b) ** 2 <= 1) m[y * w + x] = 1;
  return m;
}
function shapeOfMask(m: Uint8Array, w: number, h: number) {
  const labels = new Int32Array(w * h);
  const [c] = components(m, w, h, labels);
  return shapeOf(labels, w, 1, c[0], c[1], c[2], c[3]);
}

describe("shape features", () => {
  it("hull area of a square and a triangle", () => {
    expect(hullArea([[0, 0], [4, 0], [4, 4], [0, 4], [2, 2]])).toBe(16);
    expect(hullArea([[0, 0], [4, 0], [0, 4]])).toBe(8);
  });
  it("one ellipse (a bean) is solid and fills its moment ellipse", () => {
    const s = shapeOfMask(ellipses(120, 100, [[60, 50, 30, 20]]), 120, 100);
    expect(s.solidity).toBeGreaterThan(0.95);
    expect(s.fill).toBeGreaterThan(0.97);
    expect(s.elong).toBeCloseTo(1.5, 1);
  });
  it("an L-shaped clump of three beans: low solidity and fill (rule 3)", () => {
    const s = shapeOfMask(ellipses(220, 180, [[50, 50, 30, 20], [109, 50, 30, 20], [50, 89, 30, 20]]), 220, 180);
    expect(s.solidity < TOUCH_RULE.MIN_SOLIDITY || s.fill < TOUCH_RULE.MIN_FILL, JSON.stringify(s)).toBe(true);
  });
  it("two beans side by side: still fairly solid — that is why rule 2 (area vs one bean) exists", () => {
    const s = shapeOfMask(ellipses(160, 120, [[60, 60, 30, 20], [60, 98, 30, 20]]), 160, 120);
    expect(s.solidity).toBeGreaterThan(TOUCH_RULE.MIN_SOLIDITY);
  });
});

describe("markTouching on SYNTHETIC trays (beans pasted, classes known; tests/touching/)", () => {
  const truth = JSON.parse(fs.readFileSync(path.join(APP, "tests/touching/truth.json"), "utf8")).trays as { file: string; beans: { cx: number; cy: number }[] }[];
  const merged = (file: string) => {
    const t = truth.find((x) => x.file === file)!;
    const r = findBeans(loadImage(path.join(APP, "tests/touching", file)));
    const m = markTouching(r);
    const nIn = m.beans.map((b) => t.beans.filter((p) => r.labels[p.cy * r.work.width + p.cx] === b.comp + 1).length);
    return { r, m, nIn };
  };
  it("beans pushed together in pairs (~12% tray): the parity rule flags none of the merged pairs, the new rule all", () => {
    const { r, m, nIn } = merged("touch_pairs_12.jpg");
    const pairs = nIn.filter((n) => n >= 2).length;
    expect(pairs).toBeGreaterThanOrEqual(40);
    expect(m.beans.filter((b, i) => nIn[i] >= 2 && b.touchingParity).length).toBe(0); // the old miss
    expect(m.beans.filter((b, i) => nIn[i] >= 2 && !b.touching).length).toBe(0); // never classified as one bean
    for (let i = 0; i < m.beans.length; i++) if (nIn[i] >= 2) expect(callBean([1, 0, 0, 0, 0, 0], m.beans[i].touching, V2_THRESHOLDS)).toBe("unsure");
    // and more touching blobs than max(2, 8%) -> spread-out retake before any verdict
    expect(r.checks.too_many_touching).toBe(false);
    expect(m.checks.too_many_touching).toBe(true);
    expect(gate(m.checks, colourGate(m))?.clip).toBe("c04_spread");
  });
  it("beans apart (control, same 12% mix): nothing flagged, no retake", () => {
    const { m, nIn } = merged("touch_spread_108_12.jpg");
    expect(nIn.every((n) => n === 1)).toBe(true);
    expect(m.checks.touching).toBe(0);
    expect(gate(m.checks, colourGate(m))).toBeNull();
  });
  it("one clump of 30 beans pushed together next to 70 spread beans: 30 hidden beans -> c04, not a verdict on the other 70", () => {
    const { r, m, nIn } = merged("touch_clump30_12.jpg");
    expect(Math.max(...nIn)).toBeGreaterThanOrEqual(20); // the clump is one blob
    expect(r.checks.too_many_touching).toBe(false); // 1 touching blob: the old rule let the other 70 beans through
    expect(m.checks.hidden_beans).toBeGreaterThanOrEqual(20);
    expect(gate(m.checks, colourGate(m))?.clip).toBe("c04_spread");
  });
  it("a dense clump (58 px grid: 100 beans in 10 blobs) -> c04 spread out, not «use 100 beans» or «not green coffee»", () => {
    const { m } = merged("touch_cell58_12.jpg");
    expect(m.checks.count).toBeLessThan(20);
    expect(gate(m.checks, colourGate(m))?.clip).toBe("c04_spread");
  });
});

describe("markTouching leaves the shipped demo trays alone", () => {
  for (const f of ["tray_synthetic_00.jpg", "tray_synthetic_03.jpg", "tray_synthetic_12.jpg", "tray_synthetic_30.jpg", "tray_synthetic_40.jpg", "cbd_aaa_2.jpg", "cbd_bits_262.jpg"]) {
    it(f, () => {
      const r = findBeans(loadImage(path.join(APP, "public/demo", f)));
      const m = markTouching(r);
      expect(m.beans.filter((b) => b.touching).length).toBe(r.beans.filter((b) => b.touching).length);
    });
  }
});
