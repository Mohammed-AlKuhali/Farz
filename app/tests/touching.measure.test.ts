/**
 * Opt-in measurement of the touching rule (src/lib/touching.ts) — BEFORE (Python-parity flag only: area > 1.8 x
 * median) vs AFTER (the shipped rule), through the real app pipeline in Node (tests/node_pipeline.ts).
 *
 *  1. per-blob shape on known beans: the SYNTHETIC touching trays (tests/touching/, truth = pasted centres per
 *     blob), the synthetic demo trays (beans apart) and all 464 real CBD photos (India; beans spread on a light box)
 *  2. outcomes before/after on the touching trays, the 7 demo trays, all 464 CBD photos and any JPEGs in $FARZ_ADV
 *
 * FARZ_TOUCH=1 npx vitest run tests/touching.measure.test.ts   -> reports/touching_measure.json
 * FARZ_TOUCH_SHAPES=<file> also dumps every blob's shape (for choosing thresholds).
 */
import { expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { findBeans } from "../src/lib/beanfinder";
import { markTouching, TOUCH_RULE } from "../src/lib/touching";
import { DEFECTS, type Outcome } from "../src/lib/rules";
import { APP, ROOT, loadImage, makeClassifier, outcomeOf } from "./node_pipeline";

const walk = (d: string): string[] => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : /\.jpe?g$/i.test(e.name) ? [path.join(d, e.name)] : []);
const short = (o: Outcome) => o.kind === "retake" ? `retake:${o.reason}` : `${o.band}${o.about ? ":about" : ""}${o.unsureReason ? `:${o.unsureReason}` : ""}`;

it.skipIf(!process.env.FARZ_TOUCH || !fs.existsSync(path.join(ROOT, "data/raw/cbd")))("touching rule: shapes on known beans + outcomes before/after", async () => {
  const t0 = Date.now();
  const truth = JSON.parse(fs.readFileSync(path.join(APP, "tests/touching/truth.json"), "utf8")).trays as
    { file: string; layout: string; mix: string; true_counts: Record<string, number>; true_defects: number; beans: { cls: string; cx: number; cy: number }[] }[];
  const manifest = JSON.parse(fs.readFileSync(path.join(APP, "public/demo/manifest.json"), "utf8")) as
    { file: string; synthetic: boolean; expected: { band: string; about: boolean; reason: string } }[];
  const cbd = process.env.FARZ_TOUCH_NO_CBD ? [] : walk(path.join(ROOT, "data/raw/cbd")).sort();
  const adv = process.env.FARZ_ADV ? process.env.FARZ_ADV.split(":").filter((d) => fs.existsSync(d)).flatMap(walk).sort() : [];
  const clf = await makeClassifier();
  const shapes: Record<string, unknown>[] = [];

  const runOne = async (file: string, set: string, tray?: (typeof truth)[number]) => {
    const r = findBeans(loadImage(file));
    const after = markTouching(r);
    // truth per blob for the synthetic touching trays: how many pasted bean centres fall inside it
    const nIn = after.beans.map((b) => tray ? tray.beans.filter((p) => r.labels[Math.round(p.cy) * r.work.width + Math.round(p.cx)] === b.comp + 1).length : null);
    after.beans.forEach((b, i) => shapes.push({ set, file: path.basename(file), n: nIn[i], parity: b.touchingParity, why: b.touchWhy, ...b.shape,
      areaMed: +(b.shape.area / r.median_area).toFixed(3), areaSingle: +(b.shape.area / after.checks.single_area).toFixed(3) }));
    const probs = await clf.classify(r.beans.map((b) => b.crop!));
    const before = outcomeOf(r, probs, clf.thresholds, false);
    const now = outcomeOf(r, probs, clf.thresholds, true);
    const merged = nIn.filter((n) => n !== null && n >= 2).length;
    const mergedClassified = (o: typeof before) => o.calls ? nIn.filter((n, i) => n !== null && n >= 2 && o.calls![i] !== "unsure").length : 0;
    const row = {
      file: path.relative(ROOT, file).startsWith("..") ? file : path.relative(ROOT, file), set,
      blobs: r.checks.count, touching_before: r.checks.touching, touching_after: after.checks.touching,
      ...(tray ? { true_defects: tray.true_defects, merged_blobs: merged, merged_classified_before: mergedClassified(before), merged_classified_after: mergedClassified(now) } : {}),
      before: short(before.outcome), after: short(now.outcome),
      ...(before.outcome.kind === "result" ? { before_defects: `${before.outcome.defects}/${before.outcome.answered}`, before_unsure: before.outcome.counts.unsure } : {}),
      ...(now.outcome.kind === "result" ? { after_defects: `${now.outcome.defects}/${now.outcome.answered}`, after_unsure: now.outcome.counts.unsure } : {}),
    };
    return row;
  };

  const rows = { touching: [] as Awaited<ReturnType<typeof runOne>>[], demo: [] as Awaited<ReturnType<typeof runOne>>[], adv: [] as Awaited<ReturnType<typeof runOne>>[], cbd: [] as Awaited<ReturnType<typeof runOne>>[] };
  for (const t of truth) rows.touching.push(await runOne(path.join(APP, "tests/touching", t.file), "touching", t));
  for (const m of manifest) rows.demo.push({ ...(await runOne(path.join(APP, "public/demo", m.file), m.synthetic ? "demo_synthetic" : "demo_real")),
    expected: `${m.expected.band}${m.expected.about ? ":about" : ""}${m.expected.reason ? `:${m.expected.reason}` : ""}` } as never);
  for (const f of adv) { try { rows.adv.push(await runOne(f, "adv")); } catch { /* not a decodable JPEG (11_not_image.jpg) */ } }
  for (const f of cbd) rows.cbd.push(await runOne(f, "cbd"));

  const tally = (rs: { before: string; after: string }[], k: "before" | "after") => rs.reduce<Record<string, number>>((a, r) => { a[r[k]] = (a[r[k]] ?? 0) + 1; return a; }, {});
  const blobStats = (set: string) => {
    const b = shapes.filter((s) => s.set === set) as { parity: boolean; why: string; n: number | null }[];
    const single = b.filter((s) => s.n === null || s.n === 1), multi = b.filter((s) => s.n !== null && s.n >= 2);
    return {
      blobs: b.length,
      single_blobs: single.length, single_flagged_before: single.filter((s) => s.parity).length, single_flagged_after: single.filter((s) => s.why).length,
      merged_blobs: multi.length, merged_flagged_before: multi.filter((s) => s.parity).length, merged_flagged_after: multi.filter((s) => s.why).length,
      why_after: b.reduce<Record<string, number>>((a, s) => { if (s.why) a[s.why] = (a[s.why] ?? 0) + 1; return a; }, {}),
    };
  };
  const out = {
    what: "touching rule BEFORE (parity: area > 1.8 x median) vs AFTER (src/lib/touching.ts), real app pipeline in Node (jpeg-js + EXIF)",
    rule: TOUCH_RULE,
    blobs: { touching_trays: blobStats("touching"), demo_synthetic: blobStats("demo_synthetic"), demo_real: blobStats("demo_real"), cbd: blobStats("cbd"), adv: blobStats("adv") },
    outcomes: { touching: rows.touching, demo: rows.demo, adv: rows.adv,
      cbd: { photos: rows.cbd.length, before: tally(rows.cbd, "before"), after: tally(rows.cbd, "after"),
        changed: rows.cbd.filter((r) => r.before !== r.after).map((r) => ({ file: r.file, before: r.before, after: r.after, touching_before: r.touching_before, touching_after: r.touching_after })),
        given_a_band_after: rows.cbd.filter((r) => /^(clean|some|many)/.test(r.after)).length } },
    seconds: Math.round((Date.now() - t0) / 1000),
  };
  if (process.env.FARZ_TOUCH_SHAPES) fs.writeFileSync(process.env.FARZ_TOUCH_SHAPES, JSON.stringify(shapes));
  fs.writeFileSync(process.env.FARZ_TOUCH_DRY ?? path.join(APP, "reports/touching_measure.json"), JSON.stringify(out, null, 1));
  console.log(JSON.stringify({ blobs: out.blobs, touching: rows.touching.map((r) => [r.file.split("/").pop(), r.before, r.after, r.touching_before, r.touching_after]),
    demo: rows.demo.map((r) => [r.file.split("/").pop(), r.before, r.after, (r as never as { expected: string }).expected]), cbd: out.outcomes.cbd.before, cbdAfter: out.outcomes.cbd.after, adv: rows.adv.map((r) => [r.file.split("/").pop(), r.before, r.after]) }, null, 0));
  // the 7 demo trays keep their manifest outcomes
  for (const r of rows.demo) expect(r.after, r.file).toBe((r as never as { expected: string }).expected);
  void DEFECTS;
}, 3_600_000);
