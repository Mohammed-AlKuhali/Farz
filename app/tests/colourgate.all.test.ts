/** Slow (~2 min): the colour gate on ALL 475 local photos (11 J4ckDev + 464 CBD). Run with FARZ_ALL=1 npm test. */
import { expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { findBeans } from "../src/lib/beanfinder";
import { colourGate } from "../src/lib/colourgate";
// @ts-expect-error plain .mjs helper
import { decodeJpegRGB } from "./decode_jpeg.mjs";

const HAVE = fs.existsSync(path.resolve(__dirname, "../../data/raw/cbd")) && fs.existsSync(path.resolve(__dirname, "../../data/raw/green/ImageDataset"));
it.skipIf(!process.env.FARZ_ALL || !HAVE)("colour gate passes real green coffee (all J4ckDev + CBD photos)", () => {
  const ROOT = path.resolve(__dirname, "../..");
  const walk = (d: string): string[] => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : e.name.endsWith(".jpg") ? [path.join(d, e.name)] : []);
  const files = [...walk(path.join(ROOT, "data/raw/green/ImageDataset")), ...walk(path.join(ROOT, "data/raw/cbd"))];
  const refused: string[] = [];
  for (const f of files) {
    const im = decodeJpegRGB(fs.readFileSync(f));
    const g = colourGate(findBeans({ width: im.width, height: im.height, data: new Uint8Array(im.data) }, { crops: false }));
    if (g.notGreen) refused.push(`${path.relative(ROOT, f)} ${g.reason}`);
  }
  fs.writeFileSync(path.resolve(__dirname, "../reports/colourgate_all_photos.json"), JSON.stringify({ photos: files.length, refused }, null, 1));
  // only the two single-class J4ckDev photos that are ~all black / dried cherry may be refused
  expect(refused.every((r) => /Negros|CerezaSeca/.test(r))).toBe(true);
}, 1_800_000);
