# Shipping t_good = 0.55, and the other pre-freeze app fixes: re-measurements (Sun 4 Oct 2026, 05:00–05:25 BST)

The pre-registered sweep (`threshold_sweep.md`, block hashed before the run) chose **t_good = 0.55** (t_defect 0.91 and T 2.046 unchanged). This file records what changed when it was shipped, with before/after numbers. "Before" = the 03:42 build (t_good 0.50); "after" = the build of 05:18. Every "after" number below was measured on that build or by its own code; nothing here was fitted.

## 1. Where 0.55 now lives
- `app/src/lib/rules.ts` `V2_THRESHOLDS = { good: 0.55, defect: 0.91 }` (comment cites the sweep).
- `app/public/models/farz_beans_labels.json` `good_at_or_above: 0.55` (+ `good_at_or_above_source`). The worker reads the thresholds from this file. `models_v2/farz_beans_v2_clean_labels.json` is not mine and still lists the pooled fit (0.50).
- `app/src/worker/core.ts` fallback labels and the About screen text ("Sound if P(sound) ≥ 0.55").
- Unchanged: the calibration pilot's base binary call stays P(good) ≥ 0.5, as pre-registered in `localcal_spec.md`.

## 2. The held-out tray sets, re-measured with the APP'S code
`src/beans_v2/threshold_export_trays.py` exported the sweep's 24,800 trays (same seeds). `app/tests/threshold.app.test.ts` then ran them through the app's own `callBean()` + `decide()`. Output: `app/reports/threshold_app_measure.json`.

| | bands | wrong | dangerous |
|---|---|---|---|
| shipped 0.55 (app code) | **16** (samruddh 5, J4ckDev-1,200 11) | **1** (samruddh: a clean tray called "some") | **0** |
| old 0.50 (app code) | 293 (samruddh 52, J4ckDev-1,200 241) | 13 (all samruddh) | 0 |

Identical to the Python sweep, unit by unit.

## 3. CBD, 464 real photos (India), in the built app
`app/scripts/measure-browser.mjs`, Chromium, Pixel 7 emulation, Web Worker.

| | implausible | too many unsure | blur retake | bands | AAA "many" | tap → result median / p90 / max |
|---|---|---|---|---|---|---|
| before (0.50, 02:36Z run) | 392 | 67 | 5 | 0 | 0/50 | 344 / 362 / 495 ms |
| **after (0.55 + dark-lot rule)** | **421** | **38** | **5** | **0** | **0/50** | **344 / 362 / 493 ms** |

The Node run of the same pipeline (`tests/rules.all.test.ts`) gives 420 / 39 / 5, 0 bands, 0/50 AAA "many". The 1-photo difference is the JPEG decoder (jpeg-js vs Chromium).

## 4. The 7 demo trays (manifest updated honestly)
Counts are sound / defect / not sure. Both runs are browser runs.

| tray | before | after |
|---|---|---|
| synthetic 00 | clean, 96 / 0 / 4 | clean, 96 / 0 / 4 |
| synthetic 03 | about clean, 97 / 3 / 0 | about clean, 96 / 3 / 1 |
| synthetic 12 | some, 83 / 11 / 6 | some, 82 / 11 / 7 |
| synthetic 30 | about many, 70 / 25 / 5 | about many, 67 / 25 / 8 |
| synthetic 40 | many, 59 / 37 / 4 | many, 57 / 37 / 6 |
| CBD AAA/2 (real) | not sure (too many unsure), 11 / 9 / 30 | **not sure (implausible)**, 5 / 9 / 36 |
| CBD Bits/262 (real) | not sure (implausible), 5 / 27 / 18 | not sure (implausible), 4 / 27 / 19 |

- **Manifest.** `app/public/demo/manifest.json` now records:
  - the new `expected` for each tray;
  - for AAA/2, a `changed_4oct_t_good_055` note ("was unsure / too_many_unsure at t_good 0.50");
  - new Python reference counts (`scripts/demo_python_counts.py`, at 0.55; equal to the browser counts on all 7);
  - Node counts and browser counts.
- **e2e 4.** Asserts all 7 trays: pass.
- **Slip.** The slip of the ~12% tray is now «فرز ٤/١٠/٢٦: ١٠٠ حبة، ١١ فيها عيب، ٧ مش واضحة. فحص ذاتي مش تصنيف» (64 characters).

## 5. Dark lot (task 3): before / after
**The rule** is in `app/src/lib/darklot.ts`, applied by `rules.ts screen()`. A photo the colour gate calls "too dark beans" is answered with c07 "not sure, take it to the cooperative" instead of c06 "not green coffee" when all of these hold:
- ≥ 80% of its blobs look like single beans (not touching, elongation 1.15–2.2);
- blob areas vary by CV ≤ 0.6;
- the dark blobs have median hue 26–62° and median saturation ≤ 0.42.

Photo checks (dark, blur), the bean count range and the touching retake still come first.

**The limits come from one real photo and synthetic roast:**
- Negros: hue 31.8°, saturation 0.26.
- Synthetic dark roast: hue 21.6°, saturation 0.58.

It is unverified on real roasted coffee. Both outcomes give no band.

Built app, Chromium, 64 photos through the photo input; Node `tests/rules.all.test.ts` with `FARZ_ADV` gives identical outcomes. The REGRESSION_B trays contain test-only crops from unlicensed sets, so these two runs stay in the session scratchpad (`tw/adv_browser.json`, `tw/rules_measure_node_final.json`). The J4ckDev and SYNTHETIC rows are also in `app/reports/darklot_measure.json`.

| photo | before (03:42 build) | after |
|---|---|---|
| **J4ckDev Negros** (real, 100 black beans) | c06 "not green coffee" | **c07 not sure (dark lot)** |
| J4ckDev CerezaSeca (real, 9 dried cherries) | c06 | c06 (9 beans, and hue 24° is outside the window) |
| SYNTHETIC black-bean tray (`app/tests/darklot/black_beans_synthetic.jpg`) | (new) | c07 not sure (dark lot) |
| SYNTHETIC dark-roast tray (`app/tests/darklot/roast_dark_synthetic.jpg`, 62,38,24) | (new) | **c06** |
| REGRESSION_B dark roast ×2, medium roast ×2 | c06 ×4 | **c06 ×4** |
| REGRESSION_B light / yellow / blonde / cinnamon / city roast, 38 trays | 3 "clean" (F2), 35 not sure | **0 bands**: 19 too many unsure, 19 implausible |
| REGRESSION_B hand over tray ×3 / hand alone ×3 | c06 ×3 / 2 c06 + 1 blur | unchanged |
| CBD 464, the other 9 J4ckDev photos | no photo is "too dark beans" | unchanged (the rule cannot fire) |

**Why the 3 light-roast "clean" trays (REGRESSION_B F2) are gone.** That is the 0.55 threshold, not the dark-lot rule: the trays now have > 15% unsure beans.

## 6. Other app changes made in the same build (tasks 2, 4, 5)
- **Result clarity.**
  - The result card is now a bean ledger: one dot per bean.
  - It reads: **beans found = Farz was sure + Farz was not sure**, then **sure = defect + sound**.
  - The verdict line names its denominator: "11 with a defect, out of the 93 beans Farz was sure about" / «١١ فيها عيب، من ٩٣ حبة فرز متأكد منها».
  - Measured in both languages by e2e 12: 100 = 93 + 7; 93 = 11 + 82.
- **Calibration demo.**
  - The 3-photo before/after was replaced. It showed no change on its own photos: many → many and implausible → implausible ×2, re-measured at 0.55 (`app/reports/localcal_demo_measure.json`).
  - The replacement is the quality-check demo. The dataset labels "stand in for a grader": **20/20 accepted**; the same beans with 6 deliberately wrong labels give **13/20, refused** (`app/reports/calib_demo_check.json`, Node = Chromium).
  - It never saves and never installs a head. e2e 10/11 check this: no head after the demo, and a grader's saved head is byte-identical before and after running the demo.
- **Audio.** `VITE_AUDIO` = `pack` (default) | `placeholder` | `none`.
  - The default build ships **no audio files**: precache 57 entries, 21,223.77 KiB, against 77 entries, 21,855.66 KiB before.
  - It shows "Voice coming soon" with no Listen buttons, and requests no clip (e2e 14).
  - The 19 macOS-TTS placeholders moved to `app/audio-placeholder/`. This folder is not in the publish allowlist (`app/public/**` is), so the public export holds no Apple speech. `VITE_AUDIO=placeholder` builds them in for local use only: 77 entries.

## 7. Tests
- **vitest:** 126 passed, 5 skipped. The 5 skipped files are opt-in measurements: colourgate.all, rules.all, touching.measure, darklot.measure, threshold.app. Runs at 05:18 and again at 05:25.
- **Playwright e2e:** **14/14 passed**: 05:18:49 (52.4 s) and again in the final run at 05:25:44 (52.3 s; `app/reports/e2e-results.json`).
- **Final-run timing** (`?mainthread=1`, one 100-bean tray): 332 ms; 1,310 ms at 4× CPU throttle.

## Limits
- The same limits as the sweep apply.
- The dark-lot colour limits rest on one real black-bean photo and synthetic roast.
- Nothing was run on a phone.
