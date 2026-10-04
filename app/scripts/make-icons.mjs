// Render public/icons/icon.svg to the PNG sizes the PWA manifest and iOS need (uses Playwright's Chromium).
import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../public/icons");
const svg = fs.readFileSync(path.join(dir, "icon.svg"), "utf8");
const browser = await chromium.launch();
const page = await browser.newPage();
async function render(file, size, { maskable = false } = {}) {
  await page.setViewportSize({ width: size, height: size });
  // maskable: full-bleed background, artwork inside the 80% safe zone
  const inner = maskable ? Math.round(size * 0.8) : size;
  const bg = maskable ? "#2b1b14" : "transparent";
  await page.setContent(`<html><body style="margin:0;background:${bg};display:grid;place-items:center;width:${size}px;height:${size}px">
    <div style="width:${inner}px;height:${inner}px">${svg.replace("<svg ", `<svg width="${inner}" height="${inner}" `)}</div></body></html>`);
  await page.screenshot({ path: path.join(dir, file), omitBackground: !maskable, clip: { x: 0, y: 0, width: size, height: size } });
  console.log(file);
}
await render("icon-192.png", 192);
await render("icon-512.png", 512);
await render("icon-maskable-512.png", 512, { maskable: true });
await render("apple-touch-icon-180.png", 180, { maskable: true });
await browser.close();
