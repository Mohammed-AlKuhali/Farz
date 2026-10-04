/**
 * Cooperative calibration pilot (src/lib/localcal.ts = reports_v2/localcal_spec.md §2–3).
 *  1. unit tests of the spec on small hand-made vectors;
 *  2. spec parity in the app's runtime: onnxruntime-web on the 20 bundled calibration crops (public/calib-demo) gives
 *     the Python head's m_hi and leave-one-out errors (models_v2/localcal_demo/manifest.json, copied into the app);
 *  3. the demo walkthrough measured end to end in Node (jpeg-js decode -> bean finder -> touching -> gate -> model ->
 *     base rule vs local head -> decide), written to reports/localcal_demo_measure.json and asserted against the
 *     outcomes recorded in public/calib-demo/manifest.json.
 */
import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { PNG } from "pngjs";
import { callBeanLocal, fitHead, MAX_LOO_ERROR_FRAC, N_PER_CLASS, toHead, toStored, type Label } from "../src/lib/localcal";
import { findBeans } from "../src/lib/beanfinder";
import { markTouching } from "../src/lib/touching";
import { colourGate } from "../src/lib/colourgate";
import { callBean, decide, gate, isDefect, type BeanCall } from "../src/lib/rules";
import { APP, loadImage, makeClassifier } from "./node_pipeline";
import { WRONG_EACH, wrongDemoLabels } from "../src/components/Calibrate";

const unit = (v: number[]) => { const n = Math.hypot(...v); return v.map((x) => x / n); };

describe("local head: the spec on hand-made vectors", () => {
  // 4-d toy embeddings: good beans near e0, defect beans near e1
  const good = Array.from({ length: 12 }, (_, i) => unit([1, 0.05 * (i % 3), 0.02 * i, 0.01]));
  const bad = Array.from({ length: 11 }, (_, i) => unit([0.04 * (i % 4), 1, 0.01, 0.03 * i]));
  const items = [...good.map((e) => ({ embed: e, label: "good" as Label })), ...bad.map((e) => ({ embed: e, label: "defect" as Label }))];
  it(`refuses with fewer than ${N_PER_CLASS} of either class`, () => {
    const r = fitHead(items.filter((it, i) => it.label === "good" || i < 12 + 9));
    expect(r.ok).toBe(false);
    expect(r.refused).toBe("too_few");
  });
  it("uses exactly 10 + 10 beans; separable toy data -> 0 leave-one-out errors, mHi = median |s|", () => {
    const r = fitHead(items);
    expect(r.ok).toBe(true);
    expect(r.looN).toBe(20);
    expect(r.looErrors).toBe(0);
    expect(r.mHi).toBeGreaterThan(0);
    expect(r.nGood).toBe(12);
    expect(r.nDefect).toBe(11);
  });
  it("the same labels always give the same head (seeded draw)", () => {
    expect(fitHead(items).mHi).toBe(fitHead(items).mHi);
  });
  it(`refuses a head with more than ${MAX_LOO_ERROR_FRAC * 100}% leave-one-out errors (labels that do not separate)`, () => {
    const mixed = items.map((it, i) => ({ ...it, label: (i % 2 ? "good" : "defect") as Label }));
    const r = fitHead(mixed);
    expect(r.ok).toBe(false);
    expect(r.refused).toBe("loo_errors");
  });
  it("per-bean rule: agree -> answer; disagree -> answer only at |s| >= mHi; touching -> unsure; type is 'possible' only", () => {
    const h = { muG: Float64Array.from([1, 0, 0, 0]), muD: Float64Array.from([0, 1, 0, 0]), mHi: 0.5 };
    const pGood = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02], pBad = [0.2, 0.0, 0.7, 0.05, 0.05, 0.0];
    expect(callBeanLocal(pGood, [1, 0, 0, 0], false, h)).toBe("good"); // agree
    expect(callBeanLocal(pBad, [0, 1, 0, 0], false, h)).toBe("insect"); // agree; type = argmax of defect classes
    expect(callBeanLocal(pGood, unit([0.6, 0.8, 0, 0]), false, h)).toBe("unsure"); // disagree, |s| = 0.2 < 0.5
    expect(callBeanLocal(pGood, [0, 1, 0, 0], false, h)).toBe("dark"); // disagree, |s| = 1 >= 0.5 -> local call (ties -> first defect class)
    expect(callBeanLocal(pGood, [1, 0, 0, 0], true, h)).toBe("unsure"); // touching
    expect(callBeanLocal(pGood, null, false, h)).toBe("unsure");
  });
  it("stored head round-trips through base64 Float32 and is refused for a different model file", () => {
    const r = fitHead(items);
    const s = toStored(r, { modelSha: "abc", name: "Test coop", trays: 2 });
    expect(s.n_good + s.n_defect).toBe(20);
    const h = toHead(s, "abc")!;
    expect(h).not.toBeNull();
    expect(h.mHi).toBe(r.mHi);
    for (let k = 0; k < 4; k++) expect(h.muG[k]).toBeCloseTo(r.muG![k], 6);
    expect(toHead(s, "other-model")).toBeNull();
    expect(toHead(null, "abc")).toBeNull();
  });
});

const CAL = path.join(APP, "public/calib-demo");
const MODEL = path.join(APP, "public/models/farz_beans.onnx");
const haveDemo = fs.existsSync(path.join(CAL, "manifest.json")) && fs.existsSync(MODEL);

describe.skipIf(!haveDemo)("calibration demo set in the app's runtime (onnxruntime-web, wasm, Node)", () => {
  const man = haveDemo ? JSON.parse(fs.readFileSync(path.join(CAL, "manifest.json"), "utf8")) : null;
  const readCrop = (f: string) => {
    const png = PNG.sync.read(fs.readFileSync(path.join(CAL, f)));
    const rgb = new Uint8Array(128 * 128 * 3);
    for (let i = 0, j = 0; i < png.data.length; i += 4, j += 3) { rgb[j] = png.data[i]; rgb[j + 1] = png.data[i + 1]; rgb[j + 2] = png.data[i + 2]; }
    return rgb;
  };

  it("20 labelled crops -> the Python head (m_hi within 1e-3, same leave-one-out errors)", async () => {
    const clf = await makeClassifier();
    const crops = man.calibration.map((c: { file: string }) => readCrop(c.file));
    const { embeds } = await clf.run(crops);
    const r = fitHead(embeds!.map((e, i) => ({ embed: e, label: man.calibration[i].label as Label })));
    expect(r.ok).toBe(true);
    expect(r.looErrors).toBe(man.python_head_from_shipped_model.loo_errors);
    expect(Math.abs(r.mHi - man.python_head_from_shipped_model.m_hi)).toBeLessThan(1e-3);
  });

  it(`the in-app demo's quality check: dataset labels accepted, the same beans with ${2 * WRONG_EACH} deliberately wrong labels refused`, async () => {
    const clf = await makeClassifier();
    const crops = man.calibration.map((c: { file: string }) => readCrop(c.file));
    const { embeds } = await clf.run(crops);
    const items = embeds!.map((e, i) => ({ embed: e, label: man.calibration[i].label as Label }));
    const right = fitHead(items);
    const wrongItems = wrongDemoLabels(items);
    expect(wrongItems.filter((x) => x.flipped).length).toBe(2 * WRONG_EACH);
    const wrong = fitHead(wrongItems);
    expect(right.ok).toBe(true);
    expect(wrong.ok).toBe(false);
    expect(wrong.refused).toBe("loo_errors");
    fs.writeFileSync(path.join(APP, "reports/calib_demo_check.json"), JSON.stringify({
      what: "in-app calibration demo = the quality check only (nothing saved): loja_yolo dataset labels vs the same 20 beans with deliberately wrong labels",
      model_sha256: clf.modelSha, wrong_labels: 2 * WRONG_EACH, limit_errors: Math.floor(20 * MAX_LOO_ERROR_FRAC),
      right: { ok: right.ok, loo_errors: right.looErrors, loo_n: right.looN, m_hi: right.mHi },
      wrong: { ok: wrong.ok, loo_errors: wrong.looErrors, loo_n: wrong.looN, m_hi: wrong.mHi },
    }, null, 1));
  });

  it("demo walkthrough measured end to end: before (base model) and after (local head)", async () => {
    const clf = await makeClassifier();
    const crops = man.calibration.map((c: { file: string }) => readCrop(c.file));
    const cal = await clf.run(crops);
    const fit = fitHead(cal.embeds!.map((e, i) => ({ embed: e, label: man.calibration[i].label as Label })));
    const head = toHead(toStored(fit, { modelSha: clf.modelSha, name: "demo", trays: 3, demo: true }), clf.modelSha)!;
    const rows: Record<string, unknown>[] = [];
    for (const t of man.test as { file: string; truth: { good: number; defect: number }; recorded?: Record<string, unknown> }[]) {
      const r = markTouching(findBeans(loadImage(path.join(CAL, t.file))));
      const colour = colourGate(r);
      const appGate = gate(r.checks, colour);
      const demoGate = gate(r.checks, colour, { minBeans: 1 });
      const { probs, embeds } = await clf.run(r.beans.map((b) => b.crop!));
      const before = r.beans.map((b, i) => callBean(probs[i], b.touching, clf.thresholds));
      const after = r.beans.map((b, i) => callBeanLocal(probs[i], embeds![i], b.touching, head));
      const gd = (calls: BeanCall[]) => [calls.filter((c) => c === "good").length, calls.filter(isDefect).length, calls.filter((c) => c === "unsure").length];
      const out = (calls: BeanCall[]) => { const d = decide(calls); return { band: d.band, about: d.about, reason: d.unsureReason, good_defect_unsure: gd(calls) }; };
      const row = {
        file: t.file, beans: r.checks.count, touching: r.beans.filter((b) => b.touching).length, truth: t.truth,
        app_gate: appGate?.reason ?? null, demo_gate_min1: demoGate?.reason ?? null, blur_var: r.checks.blur_var,
        before: demoGate ? { band: "retake", reason: demoGate.reason } : out(before),
        after: demoGate ? { band: "retake", reason: demoGate.reason } : out(after),
      };
      rows.push(row);
      if (t.recorded) expect({ before: row.before, after: row.after }).toEqual(t.recorded);
    }
    const rep = { what: "calibration demo walkthrough, real app pipeline in Node", model_sha256: clf.modelSha,
      head: { m_hi: fit.mHi, loo_errors: fit.looErrors, n_good: N_PER_CLASS, n_defect: N_PER_CLASS }, rows };
    fs.writeFileSync(path.join(APP, "reports/localcal_demo_measure.json"), JSON.stringify(rep, null, 1));
    console.log(JSON.stringify(rep, null, 1));
  });
});
