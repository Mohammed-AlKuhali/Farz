// Screenshots of every screen for review / the demo video. Needs `vite preview` on :4180.
// usage: node scripts/screenshots.mjs <out_dir> [width=360] [height=760]
import { chromium } from "@playwright/test";
import { PNG } from "pngjs";
import fs from "node:fs";

const OUT = process.argv[2] ?? "screenshots";
const W = Number(process.argv[3] ?? 360), H = Number(process.argv[4] ?? 760);
fs.mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log("pageerror:", e.message));
const shot = (n, full = true) => page.screenshot({ path: `${OUT}/${n}.png`, fullPage: full });
const back = () => page.getByRole("button", { name: /رجوع|Back/ }).first().click();
async function demo(i) {
  await page.getByTestId("demo-open").click();
  await page.getByTestId("demo-tray").nth(i).click();
  await page.waitForSelector('[data-testid="result"], [data-testid="retake"]', { timeout: 60000 });
  await page.waitForTimeout(1800);
}
await page.goto("http://localhost:4180/");
await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 30000 });
await shot("01_home");
await page.getByTestId("demo-open").click(); await page.waitForTimeout(400); await shot("02_demo_sheet", false);
await page.keyboard.press("Escape");
const n = await page.evaluate(async () => (await (await fetch("demo/manifest.json")).json()).length);
for (let i = 0; i < n; i++) { await demo(i); await shot(`03_result_tray${i}`); await back(); }
await demo(1);
await page.getByTestId("send-slip").click(); await page.waitForTimeout(400); await shot("04_slip", false);
await page.keyboard.press("Escape"); await back();
const png = new PNG({ width: 800, height: 600 });
for (let i = 0; i < png.data.length; i += 4) { png.data[i] = 20; png.data[i + 1] = 18; png.data[i + 2] = 15; png.data[i + 3] = 255; }
await page.getByTestId("photo-input").first().setInputFiles({ name: "dark.png", mimeType: "image/png", buffer: PNG.sync.write(png) });
await page.waitForSelector('[data-testid="retake"]'); await page.waitForTimeout(500); await shot("05_retake_dark");
await back();
await page.getByRole("button", { name: /آخر الفحوصات|Recent checks/ }).click(); await page.waitForTimeout(300); await shot("06_history");
await back();
await page.getByRole("button", { name: /عن فرز|About Farz/ }).click(); await page.waitForTimeout(300); await shot("07_about");
await back();
await page.getByRole("button", { name: "English" }).click();
await shot("08_home_en");
await demo(2); await shot("09_result_en_tray2");
await back();
await page.getByRole("button", { name: "العربية" }).click();
await browser.close();
console.log("screenshots in", OUT);
