// Re-measure in the BUILT app (Chromium, Pixel 7 emulation, the real worker pipeline): every CBD photo (data/raw/cbd,
// 464 real photos, India) through the farmer flow's "Saved photo" input, plus every demo tray. Records band / reason /
// tap-to-result time per photo -> reports/browser_measure.json. Needs `npm run preview` (vite preview :4180) running.
// usage: node scripts/measure-browser.mjs        (skips the CBD part when data/raw/cbd is absent)
import { chromium, devices } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const APP = path.resolve(import.meta.dirname, "..");
const CBD = path.join(APP, "../data/raw/cbd");
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : /\.jpe?g$/i.test(e.name) ? [path.join(d, e.name)] : []);
const files = fs.existsSync(CBD) ? walk(CBD).sort() : [];
const browser = await chromium.launch();
const ctx = await browser.newContext({ ...devices["Pixel 7"] });
const page = await ctx.newPage();
await page.goto(process.env.FARZ_URL ?? "http://localhost:4180/");
await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60000 });
const ready = await page.evaluate(() => window.__farz.ready);
const once = async (setFile) => {
  await page.evaluate(() => { if (window.__farz) window.__farz.last = undefined; });
  await setFile();
  await page.waitForFunction(() => window.__farz?.last?.tapToResultMs !== undefined, null, { timeout: 120000 });
  const last = await page.evaluate(() => window.__farz.last);
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  return last;
};
const rows = [];
const t0 = Date.now();
// warm-up (first inference initialises the wasm kernels)
if (files[0]) await once(() => page.getByTestId("photo-input").first().setInputFiles(files[0]));
for (const f of files) {
  const last = await once(() => page.getByTestId("photo-input").first().setInputFiles(f));
  rows.push({ file: path.relative(path.join(APP, ".."), f), grade: path.basename(path.dirname(f)), kind: last.kind, band: last.band ?? "retake", reason: last.reason ?? "",
    beans: last.total, tapToResultMs: Math.round(last.tapToResultMs), finderMs: Math.round(last.ms.finder), modelMs: Math.round(last.ms.model) });
}
const demos = JSON.parse(fs.readFileSync(path.join(APP, "public/demo/manifest.json"), "utf8"));
const demoRows = [];
for (const d of demos) {
  const last = await once(async () => { await page.getByTestId("demo-open").click(); await page.locator(`[data-testid="demo-tray"][data-file="${d.file}"]`).click(); });
  demoRows.push({ file: d.file, synthetic: d.synthetic, band: last.band, about: last.about, reason: last.reason ?? "", expected: d.expected, match: last.band === d.expected.band && !!last.about === d.expected.about && (last.reason ?? "") === d.expected.reason,
    counts: last.counts, tapToResultMs: Math.round(last.tapToResultMs) });
}
const tally = rows.reduce((a, r) => { const k = `${r.band}${r.reason ? `:${r.reason}` : ""}`; a[k] = (a[k] ?? 0) + 1; return a; }, {});
const ms = rows.map((r) => r.tapToResultMs).sort((a, b) => a - b);
const q = (p) => ms.length ? ms[Math.min(ms.length - 1, Math.floor(p * ms.length))] : null;
const summary = {
  when: new Date().toISOString(), browser: `Chromium ${browser.version()} (Playwright, Pixel 7 emulation), built app on vite preview, Web Worker`,
  model_sha256: ready.modelSha, model_bytes: ready.modelBytes, thresholds: ready.thresholds,
  cbd_photos: rows.length, cbd_outcomes: tally,
  cbd_bands_given: rows.filter((r) => ["clean", "some", "many"].includes(r.band)).length,
  cbd_aaa_photos: rows.filter((r) => r.grade === "AAA").length,
  cbd_aaa_many: rows.filter((r) => r.grade === "AAA" && r.band === "many").length,
  cbd_tap_to_result_ms: { median: q(0.5), p90: q(0.9), max: ms[ms.length - 1] ?? null },
  demo_trays: demoRows.length, demo_match_manifest: demoRows.filter((r) => r.match).length,
  seconds: Math.round((Date.now() - t0) / 1000),
};
fs.writeFileSync(path.join(APP, "reports/browser_measure.json"), JSON.stringify({ summary, demo: demoRows, cbd: rows }, null, 1));
console.log(JSON.stringify(summary, null, 1));
await browser.close();
