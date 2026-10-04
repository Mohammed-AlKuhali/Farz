# Farz (فرز) — the offline web app

Installable PWA (Vite + React + TypeScript + Tailwind v4). Arabic-first, right-to-left, with an English toggle for judges.
One photo of ~100 green coffee beans → each bean is found (rules), classified on the phone (AI), and the sample gets an
on-screen verdict (spoken only once recordings are added; the default build has no audio), counts, and a 70-character SMS slip. After the first visit it runs with no network.

All numbers below were measured on the build machine (MacBook M5 Pro, Chromium headless via Playwright 1.63); the
model, rules, demo, timing and bundle figures were re-measured **4 Oct 2026, 05:15–05:25 BST** (t_good 0.55, dark-lot
rule, bean ledger, quality-check calibration demo, no-voice default build). They are not phone measurements.
**Tested in desktop Chromium only. iPhone test pending; Android untested.** The default build has **no voice** (see below).

## 4 Oct, 05:00–05:25: six changes before the freeze (numbers: `../reports_v2/threshold_ship.md`)

| change | before | after (measured) |
|---|---|---|
| **"Sound" threshold t_good**, chosen by a rule written before the results (`../reports_v2/threshold_sweep.md`): the smallest t_good in 0.50–0.75 with 0 dangerous trays and ≤ 5% wrong bands on every held-out source with ≥ 10 bands | 0.50 (no abstention margin): held-out trays 293 bands, **13 wrong** (samruddh 13 of 52), 0 dangerous | **0.55**: 16 bands, 1 wrong, 0 dangerous on the same 24,800 trays, re-measured with the app's own `callBean` + `decide` (`tests/threshold.app.test.ts` → `reports/threshold_app_measure.json`). That is one random draw of trays: on 20 further draws (not pre-registered, `../reports_v2/threshold_seeds.json`) samruddh got 105 bands, **26 wrong (about 1 in 4)**, each one band off, none dangerous; 0 dangerous trays in any draw. Cost: on the J4ckDev 1,200 trays bands fall 241 → 11 (only "many" trays still get one) |
| CBD, 464 real photos, built app | 392 implausible / 67 too many unsure / 5 blur; 0 bands; AAA "many" 0/50 | 421 / 38 / 5; **0 bands; AAA "many" 0/50** |
| 7 demo trays | clean, about clean, some, about many, many; AAA/2 not sure (too many unsure); Bits not sure (implausible) | same bands; AAA/2 now not sure (**implausible**); manifest updated |
| **Result clarity**: "11 of 94" sat above a "100 beans" card | two different totals side by side | a **bean ledger** (`src/components/Ledger.tsx`, `src/lib/ledger.ts`): one dot per bean; "100 beans found = 93 Farz was sure + 7 not sure", "93 = 11 defect + 82 sound"; the verdict says «١١ فيها عيب، من ٩٣ حبة فرز متأكد منها» / "11 with a defect, out of the 93 beans Farz was sure about" |
| **Dark lot**: J4ckDev Negros (100 black beans) was told "this isn't green coffee" | c06 | **c07 «مش متأكد. ودّي العينة للجمعية»** with «أغلب الحبوب سودا…» (`src/lib/darklot.ts`, `rules.ts screen()`); dark and medium roast stay c06 (4/4 REGRESSION_B trays + a SYNTHETIC test tray); CerezaSeca stays c06 |
| **Calibration demo** | "20 beans labelled by a grader", saved a DEMO head that stayed on and changed later results; before/after showed no change (many → many, not sure → not sure ×2) | the **quality-check demo**: dataset labels "stand in for a grader" → 20/20 accepted; the same beans with 6 deliberately wrong labels → 13/20, refused. Runs in memory only: it never saves or installs a head (e2e 10, 11). `calib-demo/CREDITS.md` now says the test photos were resized 6000×4000 → 2400×1600 |
| **Voice** | 19 macOS text-to-speech clips in every build | build flag `VITE_AUDIO` (`pack` default / `placeholder` / `none`); the **default build ships no audio** (57 precache entries; 21,226.13 KiB in the 06:31 rebuild after an About-text fix), hides Listen buttons and says «الصوت جاي قريب» / "Voice coming soon"; recordings + `pack.json` in `public/audio/` switch the voice on |


## 4 Oct: the v2 licence-stated model and the cooperative calibration pilot

**Model.** `public/models/farz_beans.onnx` is now `models_v2/farz_beans_v2_clean_fp16.onnx` (3,082,642 bytes, sha256
`c40c10f9…`): MobileNetV3-Small, fp16, trained ONLY on the 7 datasets that state a licence (CC0: afiyah_deteksi,
afiyah_bijikopi, vicanadya16; CC BY 4.0: loja_yolo; MIT: samruddh_grading; CC BY-NC 4.0: lojano; CC BY-NC-SA 4.0:
J4ckDev) — so the weights are a **non-commercial, ShareAlike research prototype**. The 5 datasets with no licence were
test-only and are not in the app. Outputs `probs` [N,6] (good, dark, insect, broken, unhulled, other_defect; T = 2.046
baked in) and `embed` [N,1024] (L2-normalised). Source of every model number: `reports_v2/clean_model.md`.

| what changed in the app | how |
|---|---|
| per-bean call | two thresholds from the labels json: **good if P(good) ≥ 0.55** (0.50 until about 04:55, see above), **defect if 1 − P(good) ≥ 0.91**, otherwise unsure; touching → unsure (`rules.ts callBean`) |
| `other_defect` | counted as a defect everywhere (band, slip, history) |
| defect TYPE | shown only under «نوع العيب المحتمل — مش أكيد» / "Possible defect type — not certain" as "possibly …"; the count card shows **Defect / Not sure / Sound** only; the type clips c11–c14 are **no longer played**; the SMS slip carries the bean count, the defect TOTAL and the unsure count, never a type |
| photo rules | unchanged: >60% of answered beans defective or >15% unsure → not sure; touching / dark / blur / colour / count gates |
| not-sure slip | `c15b_slip_unsure` clip added to the list, played before a not-sure slip instead of c15 (when a voice pack is present) |
| About | model, the 7 training datasets with their licences, non-commercial ShareAlike licence of the weights, the test-only unlicensed sets, measured limits, what a pilot needs (local labelled photos) |
| tests without data | every test that reads `data/raw/*` skips cleanly when it is absent (checked by running the suite from a copy of `app/` with no `data/`: 85 pass, 27 skip, 0 fail) |

**Cooperative calibration (PILOT).** The model agent's pre-registered study (`reports_v2/localcal.md`) said **GO
(prototypes, K = 20) on one source (loja_yolo)**, and the same method called clean trays "many" on lojano and Mindforge.
So it is built as a clearly labelled pilot for the cooperative's grader, off until a grader builds a head:
`src/lib/localcal.ts` ports `reports_v2/localcal_spec.md` §2–3 line for line; `src/components/Calibrate.tsx` is the flow
(photograph up to 3 trays, pick the "sound" or "defect" pen and tap beans, or mark a whole tray; the two piles fill to
10 + 10; build; quality check; save; reset). Stored: the spec's record (two 1,024-d prototypes as base64 Float32, m_hi,
leave-one-out errors, name, counts) in IndexedDB `farz-localcal`; photos and embeddings are never stored. A head made
with another model file (sha256) is ignored. The farmer flow shows «مضبوط على: <name> (N حبة)» / "Calibrated for: <name>
(N beans)" on Home and on every calibrated result. Grader photos are counted with the bean minimum relaxed to 1 and are
judged by the base model, never by an older head.
Our own extra guard (decision, recorded here): a head is **refused when more than 10% of its 20 calibration beans are
mis-sorted in leave-one-out** (the spec lists this as exploratory, not validated; refusing is the safe direction).

| calibration, measured | result |
|---|---|
| spec parity, 20 demo crops, onnxruntime-web in Node (`tests/localcal.test.ts`) | m_hi 0.63483 vs Python 0.63481, leave-one-out errors 0 vs 0 |
| same in Chromium (e2e 10, the demo's quality check) | m_hi 0.6348, 0 errors; the same 20 beans with 6 labels swapped on purpose: 7 errors → refused (`reports/calib_demo_check.json`, Node = Chromium) |
| grader flow in Chromium (e2e 11): SYNTHETIC 0% tray marked "whole tray sound" + a J4ckDev photo of 30 insect-damaged beans marked "whole tray defect" | 100 sound / 30 defect labelled; check 20/20; saved |
| same flow with wrong labels (0% tray "sound" + 40% tray all "defect") | **7 of 20 leave-one-out errors → refused, nothing saved** |

**Calibration demo = the quality check** (`public/calib-demo/`, loja_yolo, CC BY 4.0; since 05:10). 20 bean crops (10 sound,
10 defect) whose labels come from the public dataset, standing in for a cooperative grader. "Run the quality check" →
20/20 accepted; "Now try deliberately wrong labels" swaps 3 sound and 3 defect labels → 13/20, refused. The demo runs in
memory only: it never saves a head and never installs one on the engine, so a cooperative's saved calibration is
byte-for-byte unchanged (e2e 11) and the farmer flow is uncalibrated after it (e2e 10).
The earlier before/after demo on 3 test photos was removed because it showed no change on its own photos (re-measured at
t_good 0.55, `reports/localcal_demo_measure.json`: IMG_4804 many → many, IMG_4807 and IMG_4809 implausible → implausible):
the shipped model was trained on loja_yolo. The 3 resized test photos stay in `public/calib-demo/test/` (credited, resized
6000×4000 → 2400×1600) and are still measured by `tests/localcal.test.ts`, but no screen shows them.

**Decisions taken by the app owner (4 Oct), within the planner's rules:**
1. Ship the fp16 file (3.08 MB) as the model agent recommended; its browser parity was re-measured here (above).
2. Defect TYPES are never facts: not spoken (c11–c14 dropped from the result), not on the SMS slip, not in History
   (defect total only), shown on screen only as "possibly …" under a "not certain" heading.
3. Calibration is built (the study said GO) but as a labelled PILOT for the grader, entered from Home → "Calibrate for
   your cooperative" and inactive until a head is saved; K = 20 exactly (10 + 10, drawn with a fixed seed when more are
   labelled); extra refusal at > 10% leave-one-out errors; its own IndexedDB store and reset (the farmer's "Erase all"
   History button does not remove the cooperative's calibration).
4. Calibration demo (05:10): replaced by the quality-check demo above; the bean-count gate is relaxed to ≥ 1 bean only for
   the grader's own calibration photos.
5. Classifier fixtures were regenerated by our own Python onnxruntime run inside `app/tests/fixtures/classifier_v2/`
   (the old v1 fixtures in `../tests/fixtures/` were left untouched, they are outside `app/`).
6. `docs/CLIPS.md` is outside `app/` and was NOT edited: it still says c11–c14 play after a band — the docs owner should
   mark them "not played since 4 Oct".

## Run, test, build

```bash
cd app
npm install
npm run dev            # http://localhost:5173 (copies the ORT wasm into public/ort first)
npm test               # vitest: 126 tests + 5 opt-in skipped (wilson, sms, history privacy, fail-safe + result rules, v2 two-threshold rule, bean ledger, dark lot, touching rule, pooling, colour gate, beanfinder + classifier parity, calibration spec + demo quality check)
FARZ_SWEEP=<trays.json> npx vitest run tests/threshold.app.test.ts   # the threshold sweep's trays through the app's rules (export: ../src/beans_v2/threshold_export_trays.py)
FARZ_DARK=<dir>[,<dir>] npx vitest run tests/darklot.measure.test.ts   # colour + shape of every blob, dark-lot decision -> reports/darklot_measure.json
FARZ_ALL=1 npx vitest run tests/rules.all.test.ts   # slow (~3 min): real pipeline + rules on all 464 CBD photos -> reports/rules_measure_node.json
FARZ_TOUCH=1 npx vitest run tests/touching.measure.test.ts   # ~3.5 min: touching rule before/after -> reports/touching_measure.json
../.venv/bin/python scripts/make_touching_trays.py  # re-make the SYNTHETIC touching trays in tests/touching/
npx playwright install chromium   # once
npm run test:e2e       # builds, serves `vite preview --port 4180`, runs 14 Chromium tests
npm run build          # static site in dist/  (typecheck + vite build + Workbox service worker); NO voice by default
VITE_AUDIO=placeholder npm run build   # local only: adds the macOS text-to-speech placeholders from audio-placeholder/
npm run preview        # serve dist/ on :4180
```

Other scripts:

| command | what it does |
|---|---|
| `npm run fixtures` | re-runs the **Python** reference bean finder on 11 J4ckDev + 10 CBD photos → `tests/fixtures/beanfinder.json` |
| `../.venv/bin/python tests/make_classifier_fixtures.py` | Python onnxruntime on the shipped model: 40 crops (probs + embed) and 3 photos → `tests/fixtures/classifier_v2/` |
| `../.venv/bin/python scripts/demo_python_counts.py` | Python reference per-class counts of the demo trays with the shipped model → `farz_python_counts` in `public/demo/manifest.json` |
| `node scripts/measure-browser.mjs` | the BUILT app in Chromium on all 464 CBD photos + the 7 demo trays → `reports/browser_measure.json` (needs `npm run preview`) |
| `npm run icons` | renders `public/icons/icon.svg` to the PNG icons (Playwright Chromium) |
| `npm run audio:placeholders` | regenerates the 19 placeholder clips (macOS `say -v Majed` + ffmpeg) into `audio-placeholder/` (not in the public export, not in the default build); `ONLY=c18_another` for one |
| `node scripts/hero-shots.mjs screens` | README hero screenshots, 390×844, Arabic + English → `screens/` (needs `npm run preview`) |
| `../.venv/bin/python scripts/measure_rules.py [adv_dir]` | **v1 only (3 Oct, 5-class model via `src/beans/pipeline.py`)**: rules before/after → `reports/rules_measure.json`. Superseded for v2 by `tests/rules.all.test.ts` and `scripts/measure-browser.mjs` |
| `node scripts/screenshots.mjs <dir>` | screenshots of every screen at 360×760 (needs `npm run preview` running) |
| `node scripts/measure-throttled.mjs 4 mainthread` | end-to-end timing with CPU throttling 4× (needs preview running) |

Open `/?mainthread=1` to run the pipeline on the page thread instead of the Web Worker (measurement only, see below).

## How it is built

| file | role | AI? |
|---|---|---|
| `src/lib/beanfinder.ts` | line-for-line port of the frozen `src/beans/beanfinder.py`; numpy float32 maths reproduced with `Math.fround` | no |
| `src/lib/pillowResize.ts` | bit-exact port of Pillow's `Image.resize(..., BILINEAR)` (separable, antialiased, support scaled by the downscale factor, 22-bit fixed point) | no |
| `src/lib/colourgate.ts` | "is this green coffee at all?" colour rule (hands, roasted beans, leaves, rice) | no |
| `src/lib/touching.ts` | touching / merged beans (4 Oct): a blob is "touching" (yellow ?, never classified) if its area is > 1.8× the median blob (the Python-parity rule), > 1.6× a robust ONE-bean area (median of the single-looking blobs), or its shape is one no single bean has (solidity < 0.80 or ellipse fill < 0.85). Spread-out retake (c04) when touching blobs > max(2, 8% of blobs) or beans hidden in bean-coloured clumps (≥ 3 beans each) > max(2, 8% of beans) | no |
| `src/lib/rules.ts` | per bean (v2): good if P(good) ≥ 0.55, defect if 1 − P(good) ≥ 0.91 (type = argmax of the 5 defect classes, display only), else unsure; touching → unsure. Fail-safes: dark → **fewer than 20 beans (nothing touching, nothing off-colour) → c05 "use about 100 beans"** → blur → not green (a bean-coloured big blob while many blobs touch is a clump → c04) → touching → count <20/>160 → (model) → **implausible sample (>60% of answered beans defective)** → >15% unsure → band of the point estimate, "about" if the 95% interval crosses a band edge; `decideLot()` pools up to 3 handfuls; `screen()` = the gate, except a "dark lot" (below) | no |
| `src/lib/darklot.ts` | "too dark beans" photo whose blobs look like single beans (≥ 80% not touching, elongation 1.15–2.2, area CV ≤ 0.6) with the colour of black green coffee (dark blobs: median hue 26–62°, saturation ≤ 0.42) → c07 not sure instead of c06; limits set between J4ckDev Negros (31.8°, 0.26) and SYNTHETIC dark roast (21.6°, 0.58) | no |
| `src/lib/ledger.ts` + `src/components/Ledger.tsx` | the result's bean ledger: found = sure + not sure, sure = defect + sound, one dot per bean | no |
| `src/lib/wilson.ts` | 95% Wilson interval; bands <5% / 5–20% / >20% (our rule of thumb, not an official grade) | no |
| `src/worker/core.ts` + `farz.worker.ts` | onnxruntime-web 1.30.0, wasm backend, `numThreads=1`, wasm served from `/ort/` (copied from node_modules by `scripts/copy-ort.mjs`, no CDN), batches of 16 crops; returns `probs` and the `embed` rows (transferred, never stored); sha256 of the model checked against the labels json | **yes** (classifier) |
| `src/lib/localcal.ts` + `src/components/Calibrate.tsx` | cooperative calibration PILOT: prototype head of `reports_v2/localcal_spec.md` (10 + 10 labelled beans, leave-one-out m_hi), our extra >10% leave-one-out refusal, IndexedDB `farz-localcal`, reset; the grader flow and the Ecuador demo | **yes** (uses the classifier's embedding) |
| `src/lib/sms.ts` | slip ≤ 70 UTF-16 units: bean count, defect TOTAL (never a type), unsure count, date, «فحص ذاتي مش تصنيف»; **a not-sure result carries ONLY the date, the total and «مش متأكد»** (no per-class or defect counts, REGRESSION_A I-1); `sms:&body=` on iOS, `sms:?body=` elsewhere — it only opens the phone's own Messages app, nothing is sent by Farz | no |
| `src/lib/history.ts` | IndexedDB, last 5 checks, numbers only (never photos), one-tap wipe; demo trays are tagged; a not-sure result stores and shows only date, total and band (no per-class counts, no range) | no |
| `src/lib/engine.ts` | decodes the photo, talks to the worker; if the worker is not ready within **15 s** the app shows «فرز ما قدر يجهّز عدّاد الحبوب» with a "Try again" button instead of "Counting…" forever | no |
| `src/lib/audio.ts` + `src/lib/clips.ts` | one shared `<audio>`; a silent sound played from the first tap unlocks later playback on iOS; file names/extension from the audio pack `public/audio/pack.json`; no pack (or `VITE_AUDIO=none`) → no voice, Listen buttons hidden, nothing requested | no |

If `public/models/farz_beans.onnx` is missing, the worker uses a clearly flagged **stub** (uniform probabilities, a console
warning, and a yellow banner in the UI) — every bean becomes "not sure" and the verdict is c07.

The app never shows a price, a pest species or a pesticide: the only text it can show or say is the fixed list in
`src/lib/i18n.ts` (`CLIP_TEXT`, identical to `docs/CLIPS.md`).

## Measured results

**Bean finder parity** (`tests/beanfinder.parity.test.ts`, report in `reports/beanfinder_parity_report.json`), 21 photos
(11 J4ckDev + 10 CBD):

- Same input pixels (Python fed the RGB that jpeg-js decodes): **exact on 21/21** — work-image md5, Otsu threshold,
  sheet colour, median area, every bbox, every area, touching flags, and the md5 of **every 128 px crop**.
- Different decoders (Python + Pillow/libjpeg vs Node + jpeg-js): **counts identical on 21/21; max bbox deviation 1 px**
  (tolerance 3 px). Otsu thresholds differ by at most 1 grey level on 7/21 photos.
- In Chromium, end to end (browser JPEG decoder → JS finder → ORT web, v2 fp16 model) on `demo/tray_synthetic_12.jpg`:
  per-class counts **identical** to the Python pipeline's `farz_python_counts` (Pillow decoder, `scripts/demo_python_counts.py`):
  good 83, dark 4, insect 3, broken 1, unhulled 3, other 0, unsure 6. The e2e test allows at most 2 beans to move
  (an fp16 call near a threshold can flip when two decoders differ by a grey level); measured: 0 moved.

**Classifier parity, v2 fp16** (`tests/classifier.parity.test.ts`; fixtures from our own Python onnxruntime run,
`tests/make_classifier_fixtures.py` → `tests/fixtures/classifier_v2/`; report `reports/classifier_parity_report.json`):
40 crops (20 J4ckDev + the 20 loja_yolo calibration crops): max |Δprob| = 4.6e-4, top class 40/40, **embedding cosine ≥
0.999996**, two-threshold calls 40/40 (0 borderline); 3 fixture photos end to end: identical boxes, max |Δprob| ≤ 5.2e-4,
identical call counts (`reports/pipeline_parity_report.json`). Tolerance 0.02. The test checks the model's sha256 against
the fixtures, so it fails loudly if the model changes without new fixtures.

**e2e** (`e2e/offline.spec.ts`, Chromium, Pixel 7 emulation), **14/14 pass** (final run 4 Oct 05:25:44 BST, 52.3 s; also 14/14 at 05:18:49): (1) first visit → service
worker active → `context.setOffline(true)` → reload → model loads from the cache (not the stub) → 12% demo tray → the
ledger's Defect / Not sure / Sound rows equal the browser counts, the types only under "possible", per-class counts equal
the Python run (0 beans moved) → SMS slip «فرز ٤/١٠/٢٦: ١٠٠ حبة، ١١ فيها عيب، ٧ مش واضحة. فحص ذاتي مش تصنيف» (64
characters, `sms:?body=`), the slip sheet takes focus; (2) timing; (3) a dark photo gives c03, an empty sheet gives c05
(not c02), history fills, one-tap wipe leaves 0 records; (4) **every one of the 7 demo trays gives the outcome recorded in
its manifest** and no screen puts "·" next to Arabic-Indic digits; (5) "about" → "Add another handful" pools 3 photos and
stops offering more; "New check" starts a new lot; (6) **privacy: the not-sure slip for cbd_aaa_2 is exactly «فرز <date>:
مش متأكد، ٥٠ حبة. فحص ذاتي مش تصنيف»** (48 characters), the read-back is c15b, no bean pictures, no per-class counts in
History or IndexedDB; (7) a worker that never answers → the retry message after 15 s, "Try again" works; (8) SYNTHETIC
touching trays → c04, the same beans apart → a result; (9) opaque sticky header; (10) **calibration demo**: wording says
the labels stand in for a grader, quality check 0 errors (m_hi 0.6348), 6 wrong labels → 7 errors, refused, no save
button; afterwards no head, nothing in `farz-localcal`, no chip, a demo tray is uncalibrated; (11) **grader flow**:
tap-to-label, whole-tray marks, 20/20 check, save under a typed name; wrong labels → 7/20 errors → refused; then running
the demo leaves the saved head byte-identical; (12) **result clarity**, Arabic and English: found = sure + not sure,
sure = defect + sound, one dot per bean, the verdict's denominator is the sure count and says so; (13) **dark lot**: the
SYNTHETIC black-bean tray → not sure (dark_lot), plain not-sure slip; the SYNTHETIC dark-roast tray → c06; (14) **voice**:
the default build shows «voice coming», no Listen button anywhere, and requests no clip (only the absent `pack.json`).

**Timing, one demo tray (100 beans), tap → result painted** (`reports/e2e-measurements.json` of the final e2e run, 05:25, for e2e 1–2, one run each after warm-up; `reports/browser_measure.json`, 05:18, for the Web Worker rows):

| mode | unthrottled | CPU throttled 4× (CDP) |
|---|---|---|
| Web Worker (the shipped mode), built app, the 7 demo trays (100 / 50 beans), `scripts/measure-browser.mjs` (05:18) | **349–395 ms** tap → result (e2e 1 at 05:25, offline: finder 105 ms, model 236 ms) | not applicable (see below) |
| `?mainthread=1` (same pipeline on the page thread), e2e 2 (final run, 05:25) | **332 ms** tap → result (finder 96, model 211) | **1,310 ms** (finder 387, model 888) |
| Web Worker, built app, all 464 CBD photos (~50 beans each), `scripts/measure-browser.mjs` (05:18) | median **344 ms**, p90 362 ms, max 493 ms tap → result | – |

Chromium refuses `Emulation.setCPUThrottlingRate` on worker targets (measured reply: "Operation is only supported for
pages, not workers"), so the 4× figure comes from main-thread mode. 4× an M5 Pro is NOT a cheap Android phone; a
real-phone timing is still needed (UNVERIFIED).

**Bundle** (`dist/` of the default, no-voice build, re-measured 4 Oct 05:20; gzip -9 sizes; the Workbox build log says
57 precache entries, 21,226.13 KiB as rebuilt at 06:31 BST after an About-text fix; `VITE_AUDIO=placeholder` adds the 19 clips + pack.json: 77 entries, 21,865.46 KiB):

| part | raw MB | gzip MB |
|---|---|---|
| onnxruntime-web wasm runtime (`ort/`) | 14.264 | 3.662 |
| classifier model + labels (`models/`, **fp16**) | 3.084 | 2.809 |
| demo trays (`demo/`, 7 trays + manifest) | 2.134 | 2.073 |
| calibration demo (`calib-demo/`, 20 crops + 3 photos + manifest + credits) | 1.569 | 1.528 |
| voice clips (`audio/`): none in the default build (README only) | 0.003 | 0.001 |
| fonts | 0.328 | 0.326 |
| JS (app + worker + service worker) | 0.483 | 0.155 |
| icons, CSS, shell | 0.093 | 0.065 |
| **total `dist/`** | **21.961** | **10.620** |

Whether a host serves `.wasm`/`.onnx` compressed over the wire is host-dependent (UNVERIFIED for Vercel/Lovable).

## The result rules (product decision, 3 Oct 23:00)

* **"Not sure — take the sample to the cooperative" (c07)** is kept for MODEL or SAMPLE-PLAUSIBILITY uncertainty only:
  the model is unsure of more than 15% of the beans, nothing was answered, or the sample is **implausible: more than
  60% of the answered beans are called defective** (fixed rule, not AI; a fail-safe tuned on out-of-distribution
  photos). The photo-quality gates (dark, blur, not green, count, touching) stay retake screens (c02–c06).
* **Sampling uncertainty is shown, not refused:** the counts and the 95% Wilson range are always on screen; the band
  comes from the POINT estimate (clean <5%, some 5–20%, many >20% — our rule of thumb, not an official grade). When the
  95% range crosses a band edge the result says **"about"** (تقريباً), plays **c18** ("photograph another handful of the
  same coffee") and offers **"Add another handful"**, which pools up to 3 photos of the same lot into one count and one
  interval (`decideLot`). If any single handful is not sure on its own, the whole lot is not sure.
* Cost of the 60% rule: Farz never says "many defects" above 60% — those samples go to a person.

**Measured with the v2 licence-stated model, 4 Oct** (two independent runs of the same app code):

| set | Node (`tests/rules.all.test.ts`, jpeg-js decoder) → `reports/rules_measure_node.json` | **built app in Chromium** (`scripts/measure-browser.mjs`) → `reports/browser_measure.json` |
|---|---|---|
| 464 real CBD photos (India), EXIF orientation applied | **0 given a band**: 420 not sure (implausible), 39 not sure (too many unsure), 5 retake (blur) | **0 given a band**: 421 implausible, 38 too many unsure, 5 retake (blur) |
| CBD grade AAA (50 photos) | 0 "many" (47 implausible, 3 too many unsure); also 0 with every retake gate ignored | **0 "many"** (47 implausible, 3 too many unsure) |
| 7 demo trays | the outcomes in the manifest (table below) | 7/7 equal the current manifest (the file's `demo_match_manifest` says 6 because it compared against the pre-0.55 entry for `cbd_aaa_2`, "too many unsure"; the app now says "implausible", as the manifest does) |

All at t_good 0.55 (Chromium run 05:18, Node run 05:30 BST). The two columns differ on 7 CBD photos (net 1 in the totals) only in WHICH not-sure reason fires (decoder differences move fp16 calls near a
threshold); neither run gives any CBD photo a band. The v1 tables of 3 Oct are in git history (`git log -- app/README_APP.md`).

## Touching beans (4 Oct 00:10)

REGRESSION_A said a "~12%" touching tray (`05b_touching_some`) got a confident "many". Re-checked: that image is built
from 60 good + 20 dark + 20 broken crops (`/tmp/farz_verify/make_adv.py`), i.e. **40% defects**, and the bean finder
finds 100 blobs for its 100 pasted beans, none flagged touching — "many" (29/96 in Node) is the right band; unchanged. The real
gap was elsewhere: the Python-parity touching flag (area > 1.8× the MEDIAN blob) misses merged blobs when many beans
touch, because the median is then itself a merged blob. Measured with SYNTHETIC trays built exactly like the demo trays
but pushed together (`scripts/make_touching_trays.py` → `tests/touching/`, classes and centres known), through the app
pipeline in Node (`tests/touching.measure.test.ts` → `reports/touching_measure.json`; BEFORE = parity flag + old gate
order, AFTER = `touching.ts` + new gate order):

| tray (true mix) | blobs | merged blobs (≥ 2 beans) | merged blobs classified as one bean, before → after | outcome before → after |
|---|---|---|---|---|
| beans apart, 108 px grid (12%) | 100 | 0 | 0 → 0 | some 11/100 → some 11/100 |
| 72 px grid (12%) | 98 | 2 | 0 → 0 | some 11/95 → some 11/95 |
| 64 px grid (12% / 3% / 30%) | 85 / 76 / 81 | 9 / 16 / 12 | 0 → 0 | spread (c04) → spread |
| 58 px grid (12%) | 10 | 10 | 0 → 0 | **"not green coffee" (c06) → spread (c04)** |
| pairs pushed together (12%) | 54 | 46 | **34 → 0** | not sure (24% "?", **17/41 = 41% called defective**) → **spread (c04)** |
| 30 beans in one clump + 70 apart (12%) | 71 | 1 (≈30 beans) | 0 → 0 | **some 7/69 (verdict on 70 beans) → spread (c04)** |

Merged blobs flagged: **42/96 before → 93/96 after** (the 3 left are in the 58 px tray, which is refused anyway).
Single beans flagged: 0/479 (touching trays), **0/500** (5 synthetic demo trays), **0/100** (2 CBD demo photos) before and
after. All 464 real CBD photos: outcomes **unchanged** (459 not sure (implausible), 5 blur, 0 given a band); "?" blobs
36 → 56 of 23,449. **The 7 demo trays keep their manifest outcomes and exactly the same bean calls.** App-QA adversarial
images: 46/51 unchanged; `05_touching_dense` not sure → spread; 4 empty/near-empty sheets blur → count (see limits).
All trays here are SYNTHETIC; the beans are in the model's training set (they test the bean finder and the rules only).

## Demo trays (`public/demo/manifest.json`)

Every tray has an `expected` outcome measured with the v2 model at t_good 0.55 in the built app
(`scripts/measure-browser.mjs`, 05:18) and in Node (`tests/rules.all.test.ts`), and asserted in Chromium by e2e test (4).
The Chromium per-class counts below (`farz_browser_counts`) are **identical on all 7 trays** to the Python pipeline
(`farz_python_counts`, Pillow decoder, re-run at 0.55); the Node run (jpeg-js decoder, `farz_node_counts`) moves 0–2 beans
per tray (5 on `cbd_bits_262`) between classes near a threshold, never the outcome. At t_good 0.50 (03:42) the counts were 96/0/4, 97/3/0,
83/11/6, 70/25/5, 59/37/4, 11/9/30, 5/27/18; only `cbd_aaa_2` changed outcome (not sure: too many unsure → implausible).

| tray | what it is | Farz's counts (v2, Chromium) | outcome |
|---|---|---|---|
| `tray_synthetic_00` | SYNTHETIC, 100 sound beans | 96 sound, 0 defect, 4 ? | clean |
| `tray_synthetic_03` | SYNTHETIC, ~3% (3 defects) | 96 sound, 3 defect, 1 ? | about clean + another handful |
| `tray_synthetic_12` | SYNTHETIC, ~12% (12 defects) | 82 sound, 11 defect, 7 ? | some |
| `tray_synthetic_30` | SYNTHETIC, ~30% (30 defects) | 67 sound, 25 defect, 8 ? | about many + another handful |
| `tray_synthetic_40` | SYNTHETIC, 40 defects | 57 sound, 37 defect, 6 ? | many |
| `cbd_aaa_2` | real photo, India, CBD (CC BY 4.0) | 5 sound, 9 defect, 36 ? | not sure (implausible) |
| `cbd_bits_262` | real photo, India, CBD (CC BY 4.0) | 4 sound, 27 defect, 19 ? | not sure (implausible) |

SYNTHETIC trays are composites of J4ckDev bean crops the shipped model was trained on (J4ckDev, CC BY-NC-SA 4.0): they
show the flow, not accuracy; every tile and result says SYNTHETIC or real + source + licence.

## README hero screenshots (`screens/`, 390×844 CSS px, @2x, Arabic `ar_*` and English `en_*`)

Re-captured 4 Oct 05:21 on the default (no-voice) build with t_good 0.55 (24 files): `01_home` (with «voice coming»),
`02_result_some` (12% tray), `02b_counts_possible_types` (the bean ledger + the "possible type" list),
`03_result_about_another_handful` (3% tray), `04_result_not_sure` (real CBD photo), `05_not_green_refusal` (SYNTHETIC
roast-recoloured tray → c06), `06_history`, `07_calibrate_start`, `08_calibrate_label_tray` (the two piles + the pen +
tapped beans), `09_calibrate_quality_check` (demo: dataset labels accepted, 20/20), `10_calibrate_demo_wrong_labels_refused`
(the same beans with 6 wrong labels, 13/20, refused), `11_dark_lot_not_sure` (SYNTHETIC black-bean tray → c07). The old
`10_result_calibrated` / `11_home_calibrated` were removed (the demo no longer saves a head). Made by
`scripts/hero-shots.mjs` with reduced motion, in desktop Chromium (not a phone).

## Deploy

The `app/` folder is self-contained (its own `package.json`, lockfile and `.gitignore`); `dist/` is a plain static site
with relative paths (`base: "./"`), so it works at a domain root or a sub-path.

**Vercel:** import the GitHub repo, set **Root Directory = `app`**, framework preset Vite, build command
`npm run build`, output directory `dist`. No server code, no environment variables.

**Lovable (two-way GitHub sync) — steps UNVERIFIED tonight, check against Lovable's current docs:** Lovable projects are
the same stack (Vite + React + TS + Tailwind). The usual route is: create a Lovable project → connect GitHub from Lovable
(it creates a repo and keeps it in two-way sync) → replace that repo's contents with the contents of this `app/` folder
(so `package.json` sits at the repo root) and push → Lovable picks the change up. `public/ort/` is committed so a plain
`vite build` without the `prebuild` step still ships the runtime locally.

After any deploy: open the URL once online, wait for "يشتغل بدون نت / Works offline" on the home screen, then switch to
airplane mode and reload.

## Voice: off by default, an audio pack switches it on (no code)

The default build (`npm run build`, `VITE_AUDIO=pack`) ships **no audio**: every message is on screen as text and
pictures, the Listen buttons are hidden, and Home says «الصوت جاي قريب. الكلام مكتوب على الشاشة.» / "Voice coming soon.
Everything is written on screen." To switch a voice on, record the 19 clips in `docs/CLIPS.md` (c11–c14 are listed but
not played) under the **same filenames**, put them in `public/audio/` with a `pack.json` (copy
`audio-placeholder/pack.json`; set `ext`, `voice_en`, `voice_ar`, `"placeholder": false`), and rebuild; the About screen
shows the pack's own description. `VITE_AUDIO=placeholder npm run build` adds the macOS text-to-speech placeholders from
`audio-placeholder/` for local testing only (Apple's licence restricts sharing system-voice output; that folder is not in
the publish allowlist). `VITE_AUDIO=none` ships no audio even if a pack is present. See `public/audio/README.md`.

## Known limits (honest list)

- **Not tested on an iPhone or Android device.** iOS-specific code paths (audio unlock, `createImageBitmap` orientation
  option with an `<img>` fallback, `sms:&body=`) are UNVERIFIED on a device. Photos over 16 MP are pre-shrunk by the
  browser before the Pillow-exact downscale (only then does the app differ from the Python pipeline).
- **Colour gate** thresholds were tuned on real green coffee only. Measured with the JS gate on all 475 local photos
  (`FARZ_ALL=1 npm test`, `reports/colourgate_all_photos.json`): all 464 CBD photos and 9/11 J4ckDev photos pass; only
  the J4ckDev single-class photos `Negros.jpg` (all black) and `CerezaSeca.jpg` (dried cherry) are refused. Hands, roasted beans and leaves were tested only as
  SYNTHETIC images in `tests/rules.test.ts`. Run real not-a-bean photos (hands, roasted beans, leaves) before claiming it works on them.
- **Classifier out of distribution:** per bean the v2 model is weak on a farm it has never seen (leave-one-source-out
  good-vs-defect balanced accuracy 0.64 on average, 0.56–0.80; `reports_v2/clean_model.md`). On the real CBD photos the
  photo rules turn all 459 classified photos into "not sure" (table above). They cannot catch an unfamiliar sample whose
  beans are called mostly SOUND with confidence.
- **Calibration pilot:** validated on ONE source (loja_yolo) in the model agent's study; on lojano and Mindforge the same
  method called clean trays "many". Not tested on Yemeni beans, on a phone, or on re-photographed trays. The in-app demo
  shows only the quality check (labels accepted / wrong labels refused), not accuracy. A real pilot needs a grader's own labelled trays plus a check on
  trays not used for calibration — the app asks for that in words, it does not enforce it.
- No voice in the default build (text + pictures only) until recordings are added as an audio pack.
- **Dark-lot rule** (`darklot.ts`): its colour limits sit between ONE real black-bean photo (J4ckDev Negros) and SYNTHETIC
  dark roast; real roasted coffee was never photographed. Either answer (c06 or c07) gives no band.
- The consent read-back clip c15 describes the slip; it does not speak the numbers. The slip sheet now shows the
  same numbers as bean pictures under the text.
- Photos up to 16 MP are decoded at full size before the Pillow-exact downscale (memory on iPhone UNVERIFIED).
- Fixed 4 Oct: an empty or near-empty sheet now gets c05 "use about 100 beans" instead of c02 "blurred" (the count
  gate runs before blur when fewer than 20 beans are found, nothing touches and nothing is off-colour). Measured in Node
  on the app-QA images: 01_blank_white, 01b_white_noisy, 14_few_beans (11 beans), e_few_beans_masked (9) all went
  c02 → c05; in Chromium an empty sheet gives c05 (e2e 3). A blurred photo with 20+ beans is still c02.
- **Touching rule limits:** thresholds come from SYNTHETIC trays (pasted J4ckDev beans, size jitter ±12%) and the 464
  real CBD photos (size-graded beans). An unusually large single bean (> 1.6× the typical bean) becomes a "?"; on CBD
  that is 20 more "?" blobs out of 23,449 (0.09%), and no CBD photo changes outcome. Real farmers' trays with mixed sizes
  and real touching beans have NOT been photographed — UNVERIFIED until real iPhone photos are taken.
