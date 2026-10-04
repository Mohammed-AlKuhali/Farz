/**
 * Parity: app/src/lib/beanfinder.ts vs the frozen Python reference src/beans/beanfinder.py.
 * Fixtures: tests/fixtures/beanfinder.json (written by tests/make_fixtures.py).
 *  - "jpegjs" run: Python was fed the exact pixels jpeg-js decodes -> the port must match EXACTLY.
 *  - "pillow" run: Python decoded with Pillow/libjpeg; here we decode with jpeg-js (different chroma
 *    upsampling / IDCT) -> counts must be identical and every bbox within 3 px.
 */
import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { findBeans } from "../src/lib/beanfinder";
// @ts-expect-error plain .mjs helper shared with make_fixtures.py
import { decodeJpegRGB } from "./decode_jpeg.mjs";

const APP = path.resolve(__dirname, "..");
const ROOT = path.resolve(APP, "..");
const fx = JSON.parse(fs.readFileSync(path.join(APP, "tests/fixtures/beanfinder.json"), "utf8"));
const md5 = (u: Uint8Array) => crypto.createHash("md5").update(u).digest("hex");

interface FxBean { bbox: number[]; area: number; touching: boolean; crop_md5: string; crop_sum: number }

export const report: Record<string, unknown>[] = [];

describe("beanfinder parity with Python", () => {
  for (const ph of fx.photos) {
    // the published repo has no data/raw (dataset licences): skip cleanly when the photo is absent
    it.skipIf(!fs.existsSync(path.join(ROOT, ph.file)))(`${ph.set} ${path.basename(ph.file)}`, () => {
      const buf = fs.readFileSync(path.join(ROOT, ph.file));
      const img = decodeJpegRGB(buf);
      const r = findBeans({ width: img.width, height: img.height, data: new Uint8Array(img.data) });

      // --- strict: identical pixels
      const js = ph.jpegjs;
      expect([r.work.width, r.work.height]).toEqual(js.work_size);
      expect(r.checks.threshold).toBe(js.checks.threshold);
      expect(r.median_area).toBe(js.median_area);
      expect(r.checks.sheet_luma).toBe(js.checks.sheet_luma);
      expect(md5(r.work.data)).toBe(js.work_md5);
      expect(r.checks.count).toBe(js.checks.count);
      expect(r.checks.touching).toBe(js.checks.touching);
      expect(r.checks.blur_var).toBeCloseTo(js.checks.blur_var, 2);
      const strictBoxes = r.beans.map((b) => [b.x0, b.y0, b.x1, b.y1]);
      expect(strictBoxes).toEqual(js.beans.map((b: FxBean) => b.bbox));
      expect(r.beans.map((b) => b.area)).toEqual(js.beans.map((b: FxBean) => b.area));
      expect(r.beans.map((b) => md5(b.crop!))).toEqual(js.beans.map((b: FxBean) => b.crop_md5));

      // --- end to end: Python+Pillow decode vs JS+jpeg-js decode
      const pil = ph.pillow;
      expect(r.checks.count).toBe(pil.checks.count);
      let maxDev = 0;
      const used = new Set<number>();
      for (const b of r.beans) {
        let best = Infinity, bi = -1;
        pil.beans.forEach((pb: FxBean, i: number) => {
          if (used.has(i)) return;
          const d = Math.max(Math.abs(pb.bbox[0] - b.x0), Math.abs(pb.bbox[1] - b.y0), Math.abs(pb.bbox[2] - b.x1), Math.abs(pb.bbox[3] - b.y1));
          if (d < best) { best = d; bi = i; }
        });
        used.add(bi);
        maxDev = Math.max(maxDev, best);
      }
      report.push({ photo: ph.file, count: r.checks.count, pyCount: pil.checks.count, maxBboxDevPx: maxDev, thrJs: r.checks.threshold, thrPy: pil.checks.threshold });
      expect(maxDev).toBeLessThanOrEqual(3);
    });
  }
  it("summary", () => {
    const out = path.join(APP, "reports/beanfinder_parity_report.json");
    fs.writeFileSync(out, JSON.stringify(report, null, 1));
    expect(report.length).toBe(fx.photos.filter((ph: { file: string }) => fs.existsSync(path.join(ROOT, ph.file))).length);
  });
});
