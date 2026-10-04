/**
 * Slow (~3–6 min): the REAL app pipeline in Node — jpeg-js decode + EXIF orientation (as the browser applies it) ->
 * beanfinder.ts -> touching.ts -> colourgate.ts -> gate() -> onnxruntime-web classifier (v2 licence-stated fp16) ->
 * callBean() two-threshold rule -> decide() — on all 464 CBD photos (India, size-graded, no defect labels), the demo
 * trays and any extra JPEGs in $FARZ_ADV. Also reports the same photos with every retake gate IGNORED (stricter:
 * a retake is a safe outcome). Skips cleanly when data/raw/cbd is absent (the published repo has no data).
 * Run with: FARZ_ALL=1 npx vitest run tests/rules.all.test.ts   -> reports/rules_measure_node.json
 */
import { expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { findBeans } from "../src/lib/beanfinder";
import { colourGate } from "../src/lib/colourgate";
import { callBean, decide, screen } from "../src/lib/rules";
import { darkLot } from "../src/lib/darklot";
import { markTouching } from "../src/lib/touching";
import { APP, ROOT, loadImage, makeClassifier } from "./node_pipeline";

const CBD = path.join(ROOT, "data/raw/cbd");

it.skipIf(!process.env.FARZ_ALL || !fs.existsSync(CBD))("rules on all CBD photos + demo trays, real app pipeline in Node", async () => {
  const clf = await makeClassifier();
  const walk = (d: string): string[] => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : /\.jpe?g$/i.test(e.name) ? [path.join(d, e.name)] : []);
  const cbd = walk(CBD).sort();
  const demo = walk(path.join(APP, "public/demo")).sort();
  const adv = process.env.FARZ_ADV && fs.existsSync(process.env.FARZ_ADV) ? walk(process.env.FARZ_ADV).sort() : [];
  const run = async (f: string) => {
    const t0 = performance.now();
    const r = markTouching(findBeans(loadImage(f))); // as the app (src/worker/core.ts)
    const colour = colourGate(r);
    colour.darkLot = darkLot(r, colour); // as the worker
    const g = screen(r.checks, colour); // retake gate + "dark lot" (c07), as the app
    const name = f.startsWith(ROOT) ? path.relative(ROOT, f) : path.basename(f);
    const probs = await clf.classify(r.beans.map((b) => b.crop!));
    const calls = r.beans.map((b, i) => callBean(probs[i], b.touching, clf.thresholds));
    const d = decide(calls);
    const ms = Math.round(performance.now() - t0);
    const noGate = { band: d.band, about: d.about, reason: d.unsureReason };
    if (g?.kind === "retake") return { file: name, beans: r.checks.count, ms, app: { band: "retake", reason: g.reason }, gates_ignored: noGate, counts: null };
    if (g) return { file: name, beans: r.checks.count, ms, app: { band: g.band, about: false, reason: g.unsureReason }, gates_ignored: noGate, counts: null };
    return { file: name, beans: r.checks.count, ms, counts: d.counts,
      app: { band: d.band, about: d.about, reason: d.unsureReason, p: d.interval.p, lo: d.interval.lo, hi: d.interval.hi }, gates_ignored: noGate };
  };
  const t0 = Date.now();
  type Row = Awaited<ReturnType<typeof run>>;
  const rows = { cbd: [] as Row[], demo: [] as Row[], adv: [] as Row[] };
  for (const f of cbd) rows.cbd.push(await run(f));
  for (const f of demo) rows.demo.push(await run(f));
  for (const f of adv) rows.adv.push(await run(f));
  const key = (o: { band: string; reason?: string; about?: boolean }) => o.band + (o.reason ? `:${o.reason}` : "") + (o.about ? ":about" : "");
  const tally = (rs: Row[], k: "app" | "gates_ignored") => rs.reduce<Record<string, number>>((a, r) => { const s = key(r[k]); a[s] = (a[s] ?? 0) + 1; return a; }, {});
  const grade = (r: { file: string }) => path.basename(path.dirname(r.file));
  const isBand = (b: string) => ["clean", "some", "many"].includes(b);
  const aaa = rows.cbd.filter((r) => grade(r) === "AAA");
  const byGrade: Record<string, Record<string, number>> = {};
  for (const r of rows.cbd) { const g = (byGrade[grade(r)] ??= {}); const s = key(r.app); g[s] = (g[s] ?? 0) + 1; }
  const ms = rows.cbd.map((r) => r.ms).sort((a, b) => a - b);
  const summary = {
    model: "public/models/farz_beans.onnx", model_sha256: clf.modelSha, thresholds: clf.thresholds,
    photos: rows.cbd.length,
    app: tally(rows.cbd, "app"),
    gates_ignored: tally(rows.cbd, "gates_ignored"),
    given_a_band: rows.cbd.filter((r) => isBand(r.app.band)).length,
    given_a_band_gates_ignored: rows.cbd.filter((r) => isBand(r.gates_ignored.band)).length,
    aaa_photos: aaa.length,
    aaa_many: aaa.filter((r) => r.app.band === "many").length,
    aaa_many_gates_ignored: aaa.filter((r) => r.gates_ignored.band === "many").length,
    by_grade: byGrade,
    node_ms_per_photo_median: ms[Math.floor(ms.length / 2)],
    seconds: Math.round((Date.now() - t0) / 1000),
  };
  const out = { what: "real app pipeline in Node (jpeg-js + EXIF orientation applied), v2 licence-stated model, two-threshold rule", summary,
    demo: rows.demo, adv: { tally: tally(rows.adv, "app"), rows: rows.adv }, cbd: rows.cbd };
  fs.writeFileSync(path.join(APP, "reports/rules_measure_node.json"), JSON.stringify(out, null, 1));
  console.log(JSON.stringify(summary), "\n", rows.demo.map((r) => `${r.file} ${JSON.stringify(r.app)} ${JSON.stringify(r.counts)}`).join("\n"));
  expect(summary.aaa_many).toBe(0);
  expect(summary.aaa_many_gates_ignored).toBe(0);
  expect(rows.cbd.filter((r) => r.app.band === "many").length).toBe(0);
}, 3_600_000);
