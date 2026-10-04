// README hero screenshots at 390x844 (iPhone 12-15 size, CSS px) in Arabic AND English -> app/screens/.
// Needs `npm run preview` (vite preview on :4180) running on a fresh build.
// usage: node scripts/hero-shots.mjs [out_dir=screens]
import { chromium } from "@playwright/test";
import { PNG } from "pngjs";
import jpeg from "jpeg-js";
import fs from "node:fs";
import path from "node:path";

const OUT = process.argv[2] ?? "screens";
const URL0 = process.env.FARZ_URL ?? "http://localhost:4180/";
fs.mkdirSync(OUT, { recursive: true });

// SYNTHETIC not-green input: the 12% demo tray with every bean pixel recoloured roast-brown (same transform as the
// unit test's synthetic roasted tray). Used only to show the refusal screen.
function roastedTray() {
  const j = jpeg.decode(fs.readFileSync(path.join(import.meta.dirname, "../public/demo/tray_synthetic_12.jpg")), { useTArray: true });
  const png = new PNG({ width: j.width, height: j.height });
  for (let i = 0; i < j.data.length; i += 4) {
    const [r, g, b] = [j.data[i], j.data[i + 1], j.data[i + 2]];
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    const bean = l < 200;
    png.data[i] = bean ? Math.round(l * 0.75) : r;
    png.data[i + 1] = bean ? Math.round(l * 0.5) : g;
    png.data[i + 2] = bean ? Math.round(l * 0.32) : b;
    png.data[i + 3] = 255;
  }
  return PNG.sync.write(png);
}

const browser = await chromium.launch();
const log = [];
for (const lang of ["ar", "en"]) {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, reducedMotion: "reduce" });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => log.push(`pageerror ${lang}: ${e.message}`));
  await page.addInitScript((l) => localStorage.setItem("farz.lang", l), lang);
  const shot = async (name) => {
    await page.waitForTimeout(700);
    // stop the voice (VITE_AUDIO=placeholder builds) so the button reads "Listen" in the picture
    const playingBtn = page.locator('button[aria-pressed="true"]');
    if (await playingBtn.count()) {
      const y = await page.evaluate(() => window.scrollY);
      await playingBtn.first().click();
      await page.evaluate((v) => window.scrollTo({ top: v }), y);
      await page.waitForTimeout(150);
    }
    const file = `${OUT}/${lang}_${name}.png`;
    await page.screenshot({ path: file });
    log.push(file);
  };
  const back = () => page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  const demo = async (file) => {
    await page.getByTestId("demo-open").click();
    await page.locator(`[data-testid="demo-tray"][data-file="${file}"]`).click();
    await page.waitForSelector('[data-testid="result"], [data-testid="retake"]', { timeout: 60000 });
    const st = await page.evaluate(() => window.__farz?.last);
    log.push(`${lang} ${file}: ${JSON.stringify({ band: st?.band, about: st?.about, reason: st?.reason })}`);
  };
  await page.goto(URL0);
  await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60000 });
  await page.waitForFunction(() => window.__farz?.swReady, null, { timeout: 60000 }).catch(() => {});
  await shot("01_home");
  await demo("tray_synthetic_12.jpg");
  await shot("02_result_some");
  await back();
  await demo("tray_synthetic_03.jpg");
  // scroll so the verdict, the "about" tag and the "Add another handful" button are all in view
  await page.locator("#verdict").evaluate((el) => window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 330 }));
  await shot("03_result_about_another_handful");
  await back();
  await demo("cbd_aaa_2.jpg");
  await shot("04_result_not_sure");
  await back();
  await page.getByTestId("photo-input").first().setInputFiles({ name: "roasted.png", mimeType: "image/png", buffer: roastedTray() });
  await page.waitForSelector('[data-testid="retake"], [data-testid="result"]', { timeout: 60000 });
  log.push(`${lang} roasted: ${await page.locator('[data-testid="retake"]').getAttribute("data-reason").catch(() => "NOT A RETAKE")}`);
  await shot("05_not_green_refusal");
  await back();
  await page.getByRole("button", { name: /آخر الفحوصات|Recent checks/ }).click();
  await shot("06_history");
  await back();
  // the defect TYPE is only "possible …": the counts card of the 12% tray
  await demo("tray_synthetic_12.jpg");
  await page.getByTestId("possible-types").evaluate((el) => window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 420 }));
  await shot("02b_counts_possible_types");
  await back();
  // "Calibrate for your cooperative" (pilot): start, labelling a tray, the demo quality check, a calibrated result
  await page.getByTestId("calibrate-open").click();
  await shot("07_calibrate_start");
  await page.getByTestId("cal-name").fill(lang === "ar" ? "جمعية حراز" : "Haraz cooperative");
  await page.getByTestId("cal-photo-input").setInputFiles(path.join(import.meta.dirname, "../public/demo/tray_synthetic_12.jpg"));
  await page.waitForSelector('[data-testid="label-photo"]', { timeout: 60000 });
  const beans = page.getByTestId("label-bean");
  await page.getByTestId("pen-defect").click();
  for (const i of [1, 9, 23, 30, 42, 57]) await beans.nth(i).click();
  await page.getByTestId("pen-good").click();
  for (const i of [0, 4, 12, 17]) await beans.nth(i).click();
  await page.evaluate(() => document.activeElement instanceof HTMLElement || document.activeElement instanceof SVGElement ? (document.activeElement).blur() : null);
  await page.getByTestId("pile-good").evaluate((el) => window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 70 }));
  await shot("08_calibrate_label_tray");
  await back();
  // the calibration DEMO = the quality check only (dataset labels standing in for a grader; nothing is saved)
  await page.getByTestId("calibrate-open").click();
  await page.getByTestId("cal-demo-start").click();
  await page.waitForSelector('[data-testid="cal-demo-crops"]', { timeout: 60000 });
  await page.getByTestId("cal-demo-check").click();
  await page.waitForSelector('[data-testid="cal-demo-right"]');
  await page.getByTestId("cal-demo-right").evaluate((el) => window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 470 }));
  await shot("09_calibrate_quality_check");
  await page.getByTestId("cal-demo-wrong").click();
  await page.waitForSelector('[data-testid="cal-demo-wrong-result"]');
  await page.getByTestId("cal-demo-wrong-result").evaluate((el) => window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 470 }));
  await shot("10_calibrate_demo_wrong_labels_refused");
  log.push(`${lang} demo head after the demo: ${JSON.stringify(await page.evaluate(() => window.__farz?.head ?? null))}`);
  await back();
  // a lot of black beans (SYNTHETIC recolour of the 0% demo tray): "not sure, take it to the cooperative"
  await page.getByTestId("photo-input").first().setInputFiles(path.join(import.meta.dirname, "../tests/darklot/black_beans_synthetic.jpg"));
  await page.waitForSelector('[data-testid="result"], [data-testid="retake"]', { timeout: 60000 });
  log.push(`${lang} black beans: ${JSON.stringify(await page.evaluate(() => ({ band: window.__farz?.last?.band, reason: window.__farz?.last?.reason })))}`);
  await shot("11_dark_lot_not_sure");
  await back();
  await ctx.close();
}
await browser.close();
console.log(log.join("\n"));
