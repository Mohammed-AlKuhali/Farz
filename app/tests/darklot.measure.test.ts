/**
 * Opt-in measurement (no assertions on data that is not shipped): per photo, the colour gate, the gate() outcome and,
 * for every bean-finder blob, its mean white-balanced HSV and its shape (touching.ts). Used to set the "dark lot" rule
 * (src/lib/darklot.ts) and to report it before/after on real J4ckDev photos (Negros, CerezaSeca …), the recoloured
 * roast trays and the CBD photos.
 * Run: FARZ_DARK=<dir or file>[,<dir or file>…] npx vitest run tests/darklot.measure.test.ts -> $FARZ_DARK_OUT (json)
 */
import { expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { findBeans } from "../src/lib/beanfinder";
import { colourGate, rgbToHsv } from "../src/lib/colourgate";
import { gate } from "../src/lib/rules";
import { markTouching } from "../src/lib/touching";
import { darkLot } from "../src/lib/darklot";
import { loadImage } from "./node_pipeline";

const med = (v: number[]) => { if (!v.length) return NaN; const a = v.slice().sort((x, y) => x - y); const m = a.length >> 1; return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2; };

it.skipIf(!process.env.FARZ_DARK)("dark-lot measurement", () => {
  const walk = (d: string): string[] => fs.statSync(d).isFile() ? [d] : fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : /\.jpe?g$/i.test(e.name) ? [path.join(d, e.name)] : []);
  const files = process.env.FARZ_DARK!.split(",").flatMap((d) => walk(d)).sort();
  const rows = files.map((f) => {
    const r = markTouching(findBeans(loadImage(f)));
    const cg = colourGate(r);
    const g = gate(r.checks, cg);
    const w = r.work.width, d = r.work.data;
    const blobs = r.beans.map((b) => {
      let sr = 0, sg = 0, sb = 0, n = 0;
      for (let y = b.y0; y < b.y1; y++) for (let x = b.x0; x < b.x1; x++) {
        const i = y * w + x;
        if (r.labels[i] !== b.comp + 1) continue;
        sr += d[i * 3]; sg += d[i * 3 + 1]; sb += d[i * 3 + 2]; n++;
      }
      const [h, s, v] = rgbToHsv(sr / n, sg / n, sb / n);
      return { h, s, v, ...b.shape, touching: b.touching };
    });
    const dark = blobs.filter((b) => b.v < 0.3);
    const dl = darkLot(r, cg);
    return {
      file: path.basename(path.dirname(f)) + "/" + path.basename(f), count: r.checks.count, colour: cg.reason, frac: cg.frac, gate: g?.reason ?? "",
      darkLot: dl,
      dark_blobs: dark.length,
      med_h_dark: med(dark.map((b) => b.h)), med_s_dark: med(dark.map((b) => b.s)), med_v_dark: med(dark.map((b) => b.v)),
      med_elong: med(blobs.map((b) => b.elong)), med_solidity: med(blobs.map((b) => b.solidity)), med_fill: med(blobs.map((b) => b.fill)),
      touching: r.checks.touching, too_many_touching: r.checks.too_many_touching,
    };
  });
  const out = process.env.FARZ_DARK_OUT ?? path.join(__dirname, "../reports/darklot_measure.json");
  fs.writeFileSync(out, JSON.stringify(rows, null, 1));
  for (const r of rows) console.log(`${r.file.padEnd(48)} n=${String(r.count).padStart(3)} colour=${r.colour || "-"} gate=${r.gate || "-"} darkLot=${r.darkLot.isDarkLot ? "YES" : "no"}(${r.darkLot.why} shaped=${r.darkLot.beanShaped.toFixed(2)} cv=${r.darkLot.areaCV.toFixed(2)}) dark=${r.dark_blobs} H=${r.med_h_dark.toFixed(1)} S=${r.med_s_dark.toFixed(2)} V=${r.med_v_dark.toFixed(2)} elong=${r.med_elong.toFixed(2)} sol=${r.med_solidity.toFixed(2)} fill=${r.med_fill.toFixed(2)}`);
  expect(rows.length).toBeGreaterThan(0);
}, 3_600_000);
