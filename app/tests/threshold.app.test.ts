/**
 * Opt-in re-measurement of the pre-registered "sound" threshold sweep (reports_v2/threshold_sweep.md) with the APP'S OWN
 * code: callBean() with the shipped V2_THRESHOLDS + decide(), on the sweep's held-out trays exported by
 * src/beans_v2/threshold_export_trays.py (P(good) per bean from each source's leave-one-source-out fold model, the 5
 * unlicensed test-only sources through the shipped model, and the auditor's 1,200 J4ckDev trays).
 * Run: FARZ_SWEEP=<trays.json> npx vitest run tests/threshold.app.test.ts -> reports/threshold_app_measure.json
 */
import { expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { callBean, decide, V2_THRESHOLDS, type BeanCall, type Thresholds } from "../src/lib/rules";

interface Unit { name: string; kind: string; pgood: number[]; trays: { size: number; truth: string; beans: number[] }[] }

function score(u: Unit, thr: Thresholds) {
  const calls: BeanCall[] = u.pgood.map((p) => callBean([p, 1 - p, 0, 0, 0, 0], false, thr));
  const t = { trays: 0, bands: 0, wrong: 0, dangerous: 0, notsure: 0 };
  for (const tr of u.trays) {
    t.trays++;
    const r = decide(tr.beans.map((i) => calls[i]));
    if (r.band === "unsure") { t.notsure++; continue; }
    t.bands++;
    if (r.band !== tr.truth) t.wrong++;
    if ((tr.truth === "clean" && r.band === "many") || (tr.truth === "many" && r.band === "clean")) t.dangerous++;
  }
  return t;
}

it.skipIf(!process.env.FARZ_SWEEP || !fs.existsSync(process.env.FARZ_SWEEP ?? ""))("sweep trays through the app's callBean + decide", () => {
  const data = JSON.parse(fs.readFileSync(process.env.FARZ_SWEEP!, "utf8")) as { units: Unit[] };
  const out: Record<string, unknown> = { thresholds_shipped: V2_THRESHOLDS, per_unit: {} };
  const tot = { shipped: { trays: 0, bands: 0, wrong: 0, dangerous: 0 }, old_050: { trays: 0, bands: 0, wrong: 0, dangerous: 0 } };
  for (const u of data.units) {
    const a = score(u, V2_THRESHOLDS), b = score(u, { ...V2_THRESHOLDS, good: 0.5 });
    (out.per_unit as Record<string, unknown>)[u.name] = { shipped: a, old_050: b };
    for (const [k, v] of [["shipped", a], ["old_050", b]] as const) for (const f of ["trays", "bands", "wrong", "dangerous"] as const) tot[k][f] += v[f];
  }
  out.totals = tot;
  fs.writeFileSync(path.join(__dirname, "../reports/threshold_app_measure.json"), JSON.stringify(out, null, 1));
  console.log(JSON.stringify(tot));
  expect(tot.shipped.dangerous).toBe(0);
}, 600_000);
