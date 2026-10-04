/**
 * End-to-end on the production build (vite preview :4180) in Chromium:
 *  1. first visit installs the service worker (precache), then the network is cut (setOffline) and the page reloaded;
 *     a demo tray is run fully offline and a result must render.
 *  2. timing for one demo tray, unthrottled and with CDP CPU throttling 4x.
 *  3. refusal path (dark photo -> c03), SMS slip, history + one-tap wipe.
 */
import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { PNG } from "pngjs";

type Tray = { file: string; synthetic: boolean; farz_python_counts?: Record<string, number>; expected?: { band: string; about: boolean; reason: string } };
const CAL = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, "../public/calib-demo/manifest.json"), "utf8")) as { head_measured_in_app: { m_hi: number; loo_errors: number } };
const CAL_CHECK = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, "../reports/calib_demo_check.json"), "utf8")) as { right: { loo_errors: number }; wrong: { loo_errors: number } };
const toLatin = (x: string) => x.replace(/[٠-٩]/g, (d) => String("٠١٢٣٤٥٦٧٨٩".indexOf(d)));
const DEMO = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, "../public/demo/manifest.json"), "utf8")) as Tray[];
const TRAY = Math.max(0, DEMO.findIndex((d) => d.file === "tray_synthetic_12.jpg"));
const DEMO_DIR = path.join(import.meta.dirname, "../public/demo");
const results: Record<string, unknown> = {};

async function waitForSW(page: Page) {
  await page.waitForFunction(async () => {
    const reg = await navigator.serviceWorker.ready;
    return !!reg.active && !!navigator.serviceWorker.controller;
  }, null, { timeout: 120_000 });
}

async function runDemo(page: Page, idx: number) {
  await page.getByTestId("demo-open").click();
  const t0 = Date.now();
  await page.locator(`[data-testid="demo-tray"][data-file="${DEMO[idx].file}"]`).click();
  await page.waitForSelector('[data-testid="result"], [data-testid="retake"]', { timeout: 120_000 });
  const wall = Date.now() - t0;
  await page.waitForFunction(() => window.__farz?.last?.tapToResultMs !== undefined, null, { timeout: 30_000 });
  const last = await page.evaluate(() => window.__farz?.last);
  return { wall, tapToResultMs: last?.tapToResultMs, last };
}

test("works offline after the first visit: demo tray -> result", async ({ page, context }) => {
  await page.goto("/");
  await waitForSW(page);
  await expect(page.getByTestId("offline-status")).toContainText(/بدون نت|offline/i);
  await context.setOffline(true);
  await page.reload();
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  const ready = await page.evaluate(() => window.__farz!.ready!);
  expect(ready.stub, "classifier must load from the offline cache, not the stub").toBe(false);
  const { wall, last } = await runDemo(page, TRAY);
  await expect(page.getByTestId("result")).toBeVisible();
  await expect(page.getByTestId("count-good")).toBeVisible();
  const shown: Record<string, number> = {};
  for (const k of ["defect", "unsure", "good"]) {
    const txt = await page.getByTestId(`count-${k}`).innerText();
    shown[k] = Number(txt.replace(/[٠-٩]/g, (d) => String("٠١٢٣٤٥٦٧٨٩".indexOf(d))).replace(/\D+/g, " ").trim().split(" ").pop());
  }
  const counts = last!.counts!;
  const defects = ["dark", "insect", "broken", "unhulled", "other_defect"].reduce((a, k) => a + counts[k], 0);
  expect(shown).toEqual({ defect: defects, unsure: counts.unsure, good: counts.good });
  // defect TYPES appear only under "possible …" (never as a count row of their own)
  await expect(page.getByTestId("possible-types")).toBeVisible();
  await expect(page.getByTestId("count-broken")).toHaveCount(0);
  expect(last?.band).toBe(DEMO[TRAY].expected!.band);
  const py = DEMO[TRAY].farz_python_counts!;
  // browser (Chromium JPEG decoder) vs Python (Pillow decoder): the fp16 model's calls near a threshold can flip when
  // the decoders differ by a grey level, so per-class agreement is measured and must stay within 2 beans moved.
  const moved = ["good", "dark", "insect", "broken", "unhulled", "other_defect"].reduce((a, k) => a + Math.abs(counts[k] - py[k]), 0) + Math.abs(counts.unsure - py.abstain);
  results.offline = { tray: DEMO[TRAY].file, wallMs: wall, ms: last?.ms, band: last?.band, counts, pythonCounts: py, beansMovedVsPython: moved / 2 };
  expect(moved / 2).toBeLessThanOrEqual(2);
  // the SMS slip: one segment, counts + tag
  await page.getByTestId("send-slip").click();
  const body = await page.getByTestId("slip-body").innerText();
  expect(body.length).toBeLessThanOrEqual(70);
  expect(body).toContain("فحص ذاتي مش تصنيف");
  // the sheet takes focus (WCAG 2.4.3)
  expect(await page.evaluate(() => !!document.activeElement?.closest('[role="dialog"]'))).toBe(true);
  const href = await page.getByRole("link", { name: /الرسائل|Messages/ }).getAttribute("href");
  expect(href).toMatch(/^sms:[?&]body=/);
  results.slip = { body, length: body.length, href };
  await context.setOffline(false);
});

test("timing: one demo tray, unthrottled and with CPU throttling 4x", async ({ page }) => {
  // Chromium rejects Emulation.setCPUThrottlingRate on worker targets ("Operation is only supported for pages,
  // not workers" — measured with scripts/measure-throttled.mjs), so the throttled run uses ?mainthread=1, which
  // runs the identical pipeline on the page thread where the throttle applies.
  await page.goto("/?mainthread=1");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  await runDemo(page, TRAY); // warm-up (first inference initialises the wasm kernels)
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  const base = await runDemo(page, TRAY);
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
  const slow = await runDemo(page, TRAY);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 1 });
  results.timing = { tray: DEMO[TRAY].file, mode: "mainthread", unthrottled: base, cpu4x: slow };
  expect(slow.last?.kind).toBe("result");
  expect(slow.tapToResultMs!).toBeGreaterThan(base.tapToResultMs!); // the throttle really applied
});

test("refusal path, history and one-tap wipe", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  // a very dark photo -> fixed rule c03 (no model involved)
  const png = new PNG({ width: 800, height: 600 });
  for (let i = 0; i < png.data.length; i += 4) { png.data[i] = 20; png.data[i + 1] = 18; png.data[i + 2] = 15; png.data[i + 3] = 255; }
  await page.getByTestId("photo-input").first().setInputFiles({ name: "dark.png", mimeType: "image/png", buffer: PNG.sync.write(png) });
  await expect(page.getByTestId("retake")).toHaveAttribute("data-reason", "dark");
  // an empty sheet -> c05 "use about 100 beans", not c02 "blurred" (the count gate runs before blur when beans are few)
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  const blank = new PNG({ width: 800, height: 600 });
  for (let i = 0; i < blank.data.length; i += 4) { const v = 236 + ((i * 7919) % 7) - 3; blank.data[i] = v; blank.data[i + 1] = v; blank.data[i + 2] = v - 4; blank.data[i + 3] = 255; }
  await page.getByTestId("photo-input").first().setInputFiles({ name: "blank.png", mimeType: "image/png", buffer: PNG.sync.write(blank) });
  await expect(page.getByTestId("retake")).toHaveAttribute("data-reason", "count");
  results.emptySheet = await page.evaluate(() => window.__farz?.last);
  // a real result, then history, then wipe
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  await runDemo(page, TRAY);
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  await page.getByRole("button", { name: /آخر الفحوصات|Recent checks/ }).click();
  await expect(page.getByTestId("history-list").locator("li")).not.toHaveCount(0);
  await page.getByTestId("wipe").click();
  await expect(page.getByTestId("history-list")).toHaveCount(0);
  const stored = await page.evaluate(() => new Promise<number>((res) => {
    const r = indexedDB.open("farz", 1);
    r.onsuccess = () => { const q = r.result.transaction("checks").objectStore("checks").count(); q.onsuccess = () => res(q.result); };
  }));
  expect(stored).toBe(0);
});

test("every demo tray gives the outcome recorded in its manifest (band, about, not-sure reason)", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  const seen: Record<string, unknown> = {};
  for (let i = 0; i < DEMO.length; i++) {
    const { last } = await runDemo(page, i);
    seen[DEMO[i].file] = { band: last?.band, about: last?.about, reason: last?.reason ?? "" };
    const ex = DEMO[i].expected;
    expect(ex, `${DEMO[i].file} has no recorded outcome`).toBeTruthy();
    expect(last?.band, DEMO[i].file).toBe(ex!.band);
    expect(!!last?.about, DEMO[i].file).toBe(ex!.about);
    expect(last?.reason ?? "", DEMO[i].file).toBe(ex!.reason);
    if (!DEMO[i].synthetic) {
      // real CBD photos: honest "not sure", no confident counts, a plain-language caption
      await expect(page.getByTestId("unsure-why")).toBeVisible();
      await expect(page.getByTestId("demo-note")).toBeVisible();
      await expect(page.getByTestId("count-defect")).toHaveCount(0);
      await expect(page.getByTestId("possible-types")).toHaveCount(0);
    } else {
      // SYNTHETIC trays: the known answer next to Farz's (sound / defect / not sure only)
      await expect(page.getByTestId("demo-table")).toBeVisible();
    }
    seen[DEMO[i].file] = { ...(seen[DEMO[i].file] as object), counts: last?.counts, tapToResultMs: Math.round(last?.tapToResultMs ?? 0) };
    // never a middle dot next to Arabic-Indic digits (it reads as ٠)
    const txt = await page.locator("main").innerText();
    expect(txt).not.toMatch(/[٠-٩]\s*·|·\s*[٠-٩]/u);
    await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  }
  results.demoOutcomes = seen;
});

test("about -> add another handful pools up to 3 photos into one count and one interval", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  const i3 = DEMO.findIndex((d) => d.file === "tray_synthetic_03.jpg");
  const first = (await runDemo(page, i3)).last!;
  const d1 = ["dark", "insect", "broken", "unhulled", "other_defect"].reduce((a, k) => a + first.counts![k], 0);
  const a1 = first.total - first.counts!.unsure;
  const r = page.getByTestId("result");
  await expect(r).toHaveAttribute("data-band", "clean");
  await expect(r).toHaveAttribute("data-about", "1");
  await expect(page.getByTestId("about-tag")).toBeVisible();
  const buf = fs.readFileSync(path.join(DEMO_DIR, "tray_synthetic_03.jpg"));
  for (const n of [2, 3]) {
    await page.getByTestId("add-handful-input").setInputFiles({ name: `handful${n}.jpg`, mimeType: "image/jpeg", buffer: buf });
    await expect(page.getByTestId("result")).toHaveAttribute("data-handfuls", String(n), { timeout: 60_000 });
    await expect(page.getByTestId("pooled")).toBeVisible();
  }
  const defectsOf = await page.getByTestId("defects-of").innerText();
  // 3 handfuls of the same photo: 3 x (defects of answered beans) of the first handful
  expect(defectsOf.replace(/[٠-٩]/g, (d) => String("٠١٢٣٤٥٦٧٨٩".indexOf(d)))).toMatch(new RegExp(`(^|\\D)${3 * d1}\\D+${3 * a1}(\\D|$)`));
  await expect(page.getByTestId("add-handful")).toHaveCount(0); // never more than 3
  // a new check starts a new lot
  await page.getByTestId("new-check-input").setInputFiles({ name: "new.jpg", mimeType: "image/jpeg", buffer: fs.readFileSync(path.join(DEMO_DIR, "tray_synthetic_12.jpg")) });
  await expect(page.getByTestId("result")).toHaveAttribute("data-handfuls", "1", { timeout: 60_000 });
  results.pooling = { lastHandfuls: 3, defectsOf };
});

test("PRIVACY (I-1): a not-sure result's slip, slip sheet and history carry only date, total and «مش متأكد»", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  const iAAA = DEMO.findIndex((d) => d.file === "cbd_aaa_2.jpg");
  const { last } = await runDemo(page, iAAA);
  expect(last?.band).toBe("unsure");
  await page.getByTestId("send-slip").click();
  const body = await page.getByTestId("slip-body").innerText();
  const d = new Date();
  const ar = (n: number | string) => String(n).replace(/[0-9]/g, (x) => "٠١٢٣٤٥٦٧٨٩"[+x]);
  // exactly: date, «مش متأكد», total, tag — nothing else
  expect(body).toBe(`فرز ${ar(`${d.getDate()}/${d.getMonth() + 1}/${String(d.getFullYear() % 100).padStart(2, "0")}`)}: مش متأكد، ${ar(50)} حبة. فحص ذاتي مش تصنيف`);
  for (const word of ["مكسر", "بقشر", "سود", "مخرم", "عيوب", "عيب", "؟"]) expect(body).not.toContain(word);
  await expect(page.getByTestId("slip-pictures")).toHaveCount(0);
  // the read-back for a not-sure slip is c15b (bean count + date + «مش متأكد»), not c15 (which mentions defect counts)
  await expect(page.getByRole("dialog")).toContainText("مش متأكد، فحص ذاتي مش تصنيف");
  const href = await page.getByRole("link", { name: /الرسائل|Messages/ }).getAttribute("href");
  expect(decodeURIComponent(href!.replace(/^sms:[?&]body=/, ""))).toBe(body);
  await page.keyboard.press("Escape");
  // history: the not-sure record shows the total and the band, no per-class counts (and none are stored)
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  await page.getByRole("button", { name: /آخر الفحوصات|Recent checks/ }).click();
  const rec = page.getByTestId("history-list").locator('li[data-band="unsure"]').first();
  await expect(rec).toBeVisible();
  await expect(rec.getByTestId("history-counts")).toHaveCount(0);
  const stored = await page.evaluate(() => new Promise<unknown[]>((res) => {
    const r = indexedDB.open("farz", 1);
    r.onsuccess = () => { const q = r.result.transaction("checks").objectStore("checks").getAll(); q.onsuccess = () => res(q.result); };
  }));
  const unsureRecs = (stored as { band: string; counts?: unknown; lo?: unknown }[]).filter((x) => x.band === "unsure");
  expect(unsureRecs.length).toBeGreaterThan(0);
  for (const x of unsureRecs) { expect(x.counts).toBeUndefined(); expect(x.lo).toBeUndefined(); }
  results.privacyUnsureSlip = { body, length: body.length };
});

test.describe("inference worker that never starts", () => {
  test.use({ serviceWorkers: "block" });
  test("after 15 s a retry message replaces «Counting…»", async ({ page }) => {
    // the worker script is replaced by one that never answers: the app must not show "Counting…" forever
    await page.route(/\/assets\/farz\.worker[^/]*\.js(\?.*)?$/, (route) =>
      route.fulfill({ status: 200, contentType: "text/javascript", body: "self.onmessage = () => {};" }));
    const t0 = Date.now();
    await page.goto("/");
    await page.getByTestId("demo-open").click();
    await page.locator('[data-testid="demo-tray"][data-file="tray_synthetic_12.jpg"]').click();
    await expect(page.getByText(/أعدّ الحبوب|Counting the beans/)).toBeVisible();
    await expect(page.getByTestId("start-failed")).toBeVisible({ timeout: 30_000 });
    const waited = Date.now() - t0;
    expect(waited).toBeGreaterThanOrEqual(14_000);
    await expect(page.getByText(/أعدّ الحبوب|Counting the beans/)).toHaveCount(0);
    await expect(page.getByTestId("start-retry")).toBeVisible();
    // a new photo while the worker is still dead: the message again at once, never "Counting…" forever
    await page.getByTestId("demo-open").click();
    await page.locator('[data-testid="demo-tray"][data-file="tray_synthetic_03.jpg"]').click();
    await expect(page.getByTestId("start-failed")).toBeVisible({ timeout: 5_000 });
    // "Try again" with a working worker: the classifier starts and a tray gives a result
    await page.unroute(/\/assets\/farz\.worker[^/]*\.js(\?.*)?$/);
    await page.getByTestId("start-retry").click();
    await page.waitForFunction(() => window.__farz?.ready && window.__farz.startFailed === false, null, { timeout: 30_000 });
    await expect(page.getByTestId("start-failed")).toHaveCount(0);
    const { last } = await runDemo(page, TRAY);
    expect(last?.kind).toBe("result");
    results.workerStartTimeout = { messageAfterMs: waited, retryWorked: last?.kind === "result" };
  });
});

test("SYNTHETIC beans pushed together: merged pairs / a clump get the spread-out retake (c04), beans apart get a result", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  const dir = path.join(import.meta.dirname, "../tests/touching");
  const seen: Record<string, unknown> = {};
  for (const [file, want] of [["touch_pairs_12.jpg", "spread"], ["touch_clump30_12.jpg", "spread"], ["touch_cell58_12.jpg", "spread"], ["touch_spread_108_12.jpg", "result"]] as const) {
    await page.getByTestId("photo-input").first().setInputFiles({ name: file, mimeType: "image/jpeg", buffer: fs.readFileSync(path.join(dir, file)) });
    await page.waitForSelector('[data-testid="result"], [data-testid="retake"]', { timeout: 60_000 });
    const last = await page.evaluate(() => window.__farz?.last);
    seen[file] = { kind: last?.kind, reason: last?.reason, band: last?.band, total: last?.total };
    if (want === "spread") await expect(page.getByTestId("retake")).toHaveAttribute("data-reason", "spread");
    else await expect(page.getByTestId("result")).toBeVisible();
    await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  }
  results.touchingTrays = seen;
});

test("header stays solid over scrolled content", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  await runDemo(page, TRAY);
  await page.evaluate(() => window.scrollTo({ top: 600 }));
  const bg = await page.locator("header.topbar").evaluate((el) => getComputedStyle(el).backgroundColor);
  expect(bg).toMatch(/^rgb\(/); // rgb() = fully opaque; rgba(..., <1) or transparent would let content show through
});

/** The calibration demo (quality check only): run it and return what the two checks showed. Never saves anything. */
async function runCalDemo(page: Page) {
  await page.getByTestId("calibrate-open").click();
  await page.getByTestId("cal-demo-start").click();
  await expect(page.getByTestId("cal-demo-only")).toBeVisible();
  await expect(page.getByTestId("cal-demo-crops").locator("li")).toHaveCount(20);
  await expect(page.getByTestId("pile-good")).toHaveAttribute("data-count", "10");
  await expect(page.getByTestId("pile-defect")).toHaveAttribute("data-count", "10");
  await page.getByTestId("cal-demo-check").click();
  const right = page.getByTestId("cal-demo-right");
  await expect(right).toHaveAttribute("data-ok", "1");
  await page.getByTestId("cal-demo-wrong").click();
  await expect(page.getByTestId("cal-demo-crops").locator('li[data-flipped="1"]')).toHaveCount(6);
  const wrong = page.getByTestId("cal-demo-wrong-result");
  await expect(wrong).toHaveAttribute("data-ok", "0");
  await expect(page.getByTestId("cal-demo-refused")).toBeVisible();
  await expect(page.getByTestId("cal-save")).toHaveCount(0); // the demo has no save button at all
  const out = {
    right: { looErrors: Number(await right.getAttribute("data-loo-errors")), mHi: Number(await right.getAttribute("data-mhi")) },
    wrong: { looErrors: Number(await wrong.getAttribute("data-loo-errors")) },
  };
  await page.getByTestId("cal-demo-back").click();
  return out;
}
/** Every stored calibration head (IndexedDB "farz-localcal"); [] when the database was never created. */
const headRecords = (page: Page) => page.evaluate(async () => {
  if (!(await indexedDB.databases()).some((d) => d.name === "farz-localcal")) return [] as unknown[];
  return new Promise<unknown[]>((res) => {
    const r = indexedDB.open("farz-localcal");
    r.onsuccess = () => {
      const db = r.result;
      if (!db.objectStoreNames.contains("head")) { db.close(); return res([]); }
      const q = db.transaction("head").objectStore("head").getAll();
      q.onsuccess = () => { db.close(); res(q.result); };
    };
  });
});

test("CALIBRATION DEMO (dataset labels standing in for a grader): correct labels accepted, 6 wrong labels refused, nothing saved", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  await expect(page.getByTestId("calibrated-chip")).toHaveCount(0);
  await page.getByTestId("calibrate-open").click();
  // the wording: the labels come from a public dataset, standing in for a grader (never "labelled by a grader")
  const intro = await page.getByTestId("cal-demo-intro").innerText();
  expect(intro).toMatch(/مكان تعليم المختص|standing in for/);
  expect(intro).not.toMatch(/labelled by a grader|علّمها مختص/);
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  const seen = await runCalDemo(page);
  // same numbers as the Python head and as the Node measurement (tests/localcal.test.ts -> reports/calib_demo_check.json)
  expect(seen.right.looErrors).toBe(CAL.head_measured_in_app.loo_errors);
  expect(Math.abs(seen.right.mHi - CAL.head_measured_in_app.m_hi)).toBeLessThan(1e-3);
  expect(seen.wrong.looErrors).toBe(CAL_CHECK.wrong.loo_errors);
  // ISOLATED: no head installed, nothing stored, no chip — the farmer flow is exactly as before the demo
  expect(await page.evaluate(() => window.__farz?.head ?? null)).toBeNull();
  expect(await headRecords(page)).toHaveLength(0);
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  await expect(page.getByTestId("calibrated-chip")).toHaveCount(0);
  const { last } = await runDemo(page, TRAY);
  expect(last?.calibrated).toBe(false);
  expect(last?.band).toBe(DEMO[TRAY].expected!.band);
  results.calibrationDemo = { ...seen, headAfter: null };
});

test("CALIBRATION by a grader: photograph trays, mark whole trays / tap beans, quality check, save; wrong labels are refused", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  await page.getByTestId("calibrate-open").click();
  await page.getByTestId("cal-name").fill("Test coop");
  const photo = (rel: string) => ({ name: path.basename(rel), mimeType: rel.endsWith(".png") ? "image/png" : "image/jpeg", buffer: fs.readFileSync(path.join(import.meta.dirname, "..", rel)) });
  // tray 1: SYNTHETIC 0% tray (100 sound J4ckDev beans) -> "whole tray sound"
  await page.getByTestId("cal-photo-input").setInputFiles(photo("public/demo/tray_synthetic_00.jpg"));
  await expect(page.getByTestId("label-photo")).toHaveAttribute("data-tray", "1", { timeout: 60_000 });
  // one tap labels one bean with the pen; a second tap clears it
  const bean = page.getByTestId("label-bean").first();
  await page.getByTestId("pen-defect").click();
  await bean.click();
  await expect(bean).toHaveAttribute("data-label", "defect");
  await bean.click();
  await expect(bean).toHaveAttribute("data-label", "none");
  await page.getByTestId("mark-all-good").click();
  await expect(page.getByTestId("cal-build")).toBeDisabled();
  // never a middle dot next to Arabic-Indic digits (it reads as ٠)
  expect(await page.locator("main").innerText()).not.toMatch(/[٠-٩]\s*·|·\s*[٠-٩]/u);
  // tray 2: a J4ckDev photo of 30 insect-damaged beans (dataset label: all defective) -> "whole tray defect"
  await page.getByTestId("cal-photo-more-input").setInputFiles(photo("tests/fixtures/classifier_v2/photos/j4ck_brocadoleve.png"));
  await expect(page.getByTestId("label-photo")).toHaveAttribute("data-tray", "2", { timeout: 60_000 });
  await page.getByTestId("mark-all-defect").click();
  await expect(page.getByTestId("pile-good")).toHaveAttribute("data-count", /\d+/);
  const nGood = Number(await page.getByTestId("pile-good").getAttribute("data-count"));
  const nDefect = Number(await page.getByTestId("pile-defect").getAttribute("data-count"));
  expect(nGood).toBeGreaterThanOrEqual(10);
  expect(nDefect).toBeGreaterThanOrEqual(10);
  await page.getByTestId("cal-build").click();
  const check = page.getByTestId("cal-check");
  await expect(check).toBeVisible();
  const ok = await check.getAttribute("data-ok");
  const loo = await check.getAttribute("data-loo-errors");
  results.graderCalibration = { nGood, nDefect, ok, looErrors: loo, mHi: await check.getAttribute("data-mhi") };
  expect(ok).toBe("1");
  await page.getByTestId("cal-save").click();
  await expect(page.getByTestId("cal-status")).toContainText("Test coop");
  // a grader whose labels do not separate (every bean of a mixed 40% tray marked "defect", the clean tray "sound",
  // so 6 of every 10 "defect" labels are sound beans) gets a refusal and nothing is saved
  await page.getByTestId("cal-photo-input").setInputFiles(photo("public/demo/tray_synthetic_00.jpg"));
  await expect(page.getByTestId("label-photo")).toHaveAttribute("data-tray", "1", { timeout: 60_000 });
  await page.getByTestId("mark-all-good").click();
  await page.getByTestId("cal-photo-more-input").setInputFiles(photo("public/demo/tray_synthetic_40.jpg"));
  await expect(page.getByTestId("label-photo")).toHaveAttribute("data-tray", "2", { timeout: 60_000 });
  await page.getByTestId("mark-all-defect").click();
  await page.getByTestId("cal-build").click();
  const bad = page.getByTestId("cal-check");
  results.graderCalibrationMislabelled = { ok: await bad.getAttribute("data-ok"), looErrors: await bad.getAttribute("data-loo-errors") };
  await expect(bad).toHaveAttribute("data-ok", "0");
  await expect(page.getByTestId("cal-refused")).toBeVisible();
  await expect(page.getByTestId("cal-save")).toHaveCount(0);
  // the saved head is still the good one
  expect(await page.evaluate(() => window.__farz?.head?.name)).toBe("Test coop");
  // running the calibration DEMO afterwards leaves the cooperative's saved calibration byte-for-byte unchanged
  const before = JSON.stringify(await headRecords(page));
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  await runCalDemo(page);
  expect(JSON.stringify(await headRecords(page))).toBe(before);
  expect(await page.evaluate(() => window.__farz?.head?.name)).toBe("Test coop");
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  await expect(page.getByTestId("calibrated-chip")).toContainText("Test coop");
  results.graderHeadUnchangedByDemo = true;
});

test("RESULT CLARITY: beans found = sure + not sure, sure = defect + sound; the verdict's denominator is the sure count (Arabic and English)", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  const seen: Record<string, unknown> = {};
  for (const lang of ["ar", "en"] as const) {
    if (lang === "en") await page.getByRole("button", { name: "English" }).click();
    const { last } = await runDemo(page, TRAY);
    const L = page.getByTestId("ledger");
    const n = async (a: string) => Number(await L.getAttribute(a));
    const found = await n("data-found"), sure = await n("data-sure"), defects = await n("data-defects"), sound = await n("data-sound"), unsure = await n("data-unsure");
    expect(found).toBe(last!.total);
    expect(found).toBe(sure + unsure);
    expect(sure).toBe(defects + sound);
    expect(toLatin(await page.getByTestId("count-total").innerText())).toContain(String(found));
    // one dot per bean
    expect(Number(await page.getByTestId("dots-sure").getAttribute("data-n"))).toBe(sure);
    if (unsure) expect(Number(await page.getByTestId("dots-unsure").getAttribute("data-n"))).toBe(unsure);
    // the verdict line: "<defects> … <sure> beans Farz was sure about" — never the found total
    const v = toLatin(await page.getByTestId("defects-of").innerText());
    expect(v).toMatch(new RegExp(`(^|\\D)${defects}\\D+${sure}(\\D|$)`));
    expect(v).toMatch(lang === "ar" ? /فرز متأكد منها/ : /beans Farz was sure about/);
    if (found !== sure) expect(v).not.toMatch(new RegExp(`(^|\\D)${found}(\\D|$)`));
    await expect(page.getByTestId("count-sure")).toContainText(lang === "ar" ? "فرز متأكد منها" : "Farz was sure");
    await expect(page.getByTestId("count-unsure")).toContainText(lang === "ar" ? "فرز مش متأكد منها" : "Farz was not sure");
    seen[lang] = { found, sure, defects, sound, unsure, verdict: v };
    await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  }
  await page.getByRole("button", { name: "العربية" }).click();
  results.resultClarity = seen;
});

test("DARK LOT: a tray of black (bean-shaped) beans is «not sure, take it to the cooperative», a dark-roast tray is still «not green coffee»", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  const dir = path.join(import.meta.dirname, "../tests/darklot");
  const seen: Record<string, unknown> = {};
  await page.getByTestId("photo-input").first().setInputFiles({ name: "black.jpg", mimeType: "image/jpeg", buffer: fs.readFileSync(path.join(dir, "black_beans_synthetic.jpg")) });
  await page.waitForSelector('[data-testid="result"], [data-testid="retake"]', { timeout: 60_000 });
  await expect(page.getByTestId("result")).toHaveAttribute("data-band", "unsure");
  await expect(page.getByTestId("result")).toHaveAttribute("data-reason", "dark_lot");
  await expect(page.getByTestId("unsure-why")).toContainText(/سودا|black/);
  await expect(page.locator("main")).toContainText("ودّي العينة للجمعية");
  seen.black = await page.evaluate(() => window.__farz?.last);
  // the slip of a dark lot is a plain "not sure" slip: date, total, «مش متأكد», tag
  await page.getByTestId("send-slip").click();
  const body = await page.getByTestId("slip-body").innerText();
  expect(body).toContain("مش متأكد");
  await expect(page.getByTestId("slip-pictures")).toHaveCount(0);
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  await page.getByTestId("photo-input").first().setInputFiles({ name: "roast.jpg", mimeType: "image/jpeg", buffer: fs.readFileSync(path.join(dir, "roast_dark_synthetic.jpg")) });
  await page.waitForSelector('[data-testid="result"], [data-testid="retake"]', { timeout: 60_000 });
  await expect(page.getByTestId("retake")).toHaveAttribute("data-reason", "not_green");
  seen.roast = await page.evaluate(() => window.__farz?.last);
  results.darkLot = seen;
});

test("VOICE: the default build ships no audio — no Listen buttons, a plain «voice coming» note, no clip requested", async ({ page }) => {
  test.skip(fs.existsSync(path.join(process.cwd(), "public/audio/pack.json")), "a recorded voice pack is installed, so the silent-build check does not apply");
  const audioReqs: string[] = [];
  page.on("request", (r) => { if (/\/audio\/.+\.(mp3|m4a|json)(\?|$)/.test(r.url())) audioReqs.push(r.url()); });
  await page.goto("/");
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60_000 });
  await expect(page.getByTestId("voice-coming")).toBeVisible();
  await expect(page.getByRole("button", { name: /اسمعي|Listen/ })).toHaveCount(0);
  await runDemo(page, TRAY);
  await expect(page.getByRole("button", { name: /اسمعي|Listen/ })).toHaveCount(0);
  await page.getByTestId("send-slip").click();
  await page.keyboard.press("Escape");
  // only the pack lookup (pack.json, absent) may be requested; never a clip
  expect(audioReqs.filter((u) => /\.(mp3|m4a)/.test(u))).toEqual([]);
  results.voice = { audioRequests: audioReqs };
});

test.afterAll(() => {
  fs.writeFileSync(path.join(import.meta.dirname, "../reports/e2e-measurements.json"), JSON.stringify(results, null, 1));
});

test("VOICE: with a recorded pack, the clips are precached and play offline", async ({ page, context }) => {
  test.skip(!fs.existsSync(path.join(process.cwd(), "public/audio/pack.json")), "no recorded voice pack");
  await page.goto("/"); await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  await page.waitForTimeout(3000);
  const cached = await page.evaluate(async () => { let n = 0; for (const k of await caches.keys()) for (const r of await (await caches.open(k)).keys()) if (/\/audio\/c\d+b?_[a-z_]+\.mp3/.test(r.url)) n++; return n; });
  expect(cached).toBeGreaterThanOrEqual(15);
  await context.setOffline(true);
  const ok = await page.evaluate(async () => (await fetch("/audio/c07_unsure.mp3")).ok);
  expect(ok).toBe(true);
});
