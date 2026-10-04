// Usage: node scripts/measure-throttled.mjs 4 [worker|mainthread]
// Measure one demo tray end to end with CPU throttling applied to BOTH the page and its Web Worker.
// Playwright's page CDP session throttles only the page's main thread, so we also attach to the worker target
// over Chromium's remote-debugging websocket and throttle it there. Needs `vite preview` on :4180.
import { chromium, devices } from "@playwright/test";

const RATE = Number(process.argv[2] ?? 4);
const MODE = process.argv[3] === "mainthread" ? "mainthread" : "worker";
const PORT = 9333;
const browser = await chromium.launch({ args: [`--remote-debugging-port=${PORT}`] });
const ctx = await browser.newContext({ ...devices["Pixel 7"] });
const page = await ctx.newPage();
await page.goto(`http://localhost:4180/${MODE === "mainthread" ? "?mainthread=1" : ""}`);
await page.waitForFunction(() => window.__farz?.ready, null, { timeout: 60000 });

async function throttleWorkers(rate) {
  const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
  const workers = targets.filter((t) => t.type === "worker" || t.type === "shared_worker" || /worker/.test(t.url));
  const out = [];
  for (const t of workers) {
    if (!t.webSocketDebuggerUrl) continue;
    const ws = new WebSocket(t.webSocketDebuggerUrl);
    await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
    const reply = await new Promise((r) => {
      ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id === 1) r(d); };
      ws.send(JSON.stringify({ id: 1, method: "Emulation.setCPUThrottlingRate", params: { rate } }));
    });
    out.push({ url: t.url, reply });
    ws.close();
  }
  return out;
}

async function run(label) {
  await page.getByTestId("demo-open").click();
  const t0 = Date.now();
  await page.getByTestId("demo-tray").nth(1).click();
  await page.waitForSelector('[data-testid="result"], [data-testid="retake"]', { timeout: 120000 });
  const wall = Date.now() - t0;
  await page.waitForFunction(() => window.__farz?.last?.tapToResultMs !== undefined, null, { timeout: 30000 });
  const last = await page.evaluate(() => window.__farz.last);
  await page.getByRole("button", { name: /رجوع|Back/ }).first().click();
  const row = { wall, tapToResult: last.tapToResultMs, ...last.ms };
  console.log(label, JSON.stringify(row));
  return row;
}

await run("warmup");
const r = { unthrottled: [], throttled: [] };
for (let i = 0; i < 3; i++) r.unthrottled.push(await run("unthrottled"));
const cdp = await ctx.newCDPSession(page);
await cdp.send("Emulation.setCPUThrottlingRate", { rate: RATE });
const w = await throttleWorkers(RATE);
console.log("worker throttle:", JSON.stringify(w).slice(0, 400));
for (let i = 0; i < 3; i++) r.throttled.push(await run(`cpu${RATE}x`));
const med = (a, k) => a.map((x) => x[k]).sort((p, q) => p - q)[Math.floor(a.length / 2)];
const summary = Object.fromEntries(Object.entries(r).map(([k, v]) => [k, Object.fromEntries(["tapToResult", "decode", "finder", "model", "total"].map((f) => [f, Math.round(med(v, f))]))]));
console.log("MEDIANS", JSON.stringify({ rate: RATE, mode: MODE, ...summary }));
import("node:fs").then((fs) => fs.writeFileSync(new URL(`../reports/e2e-throttled-${MODE}.json`, import.meta.url), JSON.stringify({ rate: RATE, mode: MODE, runs: r, ...summary }, null, 1)));
await browser.close();
