# Farz: model card

Farz has **one** machine-learning model: the per-bean classifier (§2). It has an optional on-phone calibration head for the cooperative's grader (§3, pilot). Everything around it is fixed, testable code: the bean finder (§1), the quality checks, the sample rule (§4) and the fixed list of messages (shown on screen; spoken only once recordings are added, as the default build has no audio).

That split is deliberate. The rules handle the cases where a confident wrong answer would cost Noor money. The model is asked only the one question rules cannot answer: does this bean look sound or defective?

**Which model is shipped.** It is the **v2 licence-stated model**, `app/public/models/farz_beans.onnx`, byte-identical to `models_v2/farz_beans_v2_clean_fp16.onnx` (sha256 `c40c10f9…`; `models_v2/` is not in the public repo). It replaced the first model (v1, J4ckDev only) on 4 Oct. v1's numbers are kept in §5 as history.

**Sound threshold.** Since about 04:55 BST on 4 Oct a bean counts as sound at P(good) ≥ **0.55**, chosen by a pre-registered sweep (`reports_v2/threshold_sweep.md`); before that it was 0.50, the pooled fit. Model-run numbers measured at 0.50 are marked as such. Raising t_good only moves beans from "sound" to "?" (a bean with P(good) ≥ 0.50 can never reach P(defect) ≥ 0.91), so no "not sure" measured at 0.50 can become a band at 0.55, and no defect call changes.

**Where the numbers come from.**
- Every number names the file it comes from.
- Model numbers come from `reports_v2/clean_model.json`, `clean_eval.json`, `clean_export.json` and `clean_deploy.json`, written by `src/beans_v2/clean_*.py`.
- App numbers come from `app/reports/*.json`, written by the build of 4 Oct 05:18–05:25 BST, which ships t_good 0.55 (`reports_v2/threshold_ship.md`). The exception is `app/reports/colourgate_all_photos.json`, measured on 3 Oct at 22:41; the colour gate does not use the model, so its numbers still apply.
- The v2 method was audited independently (`reports_v2/AUDIT_V2.md`). Where the audit corrected a claim, the corrected figure is used.
- Nothing here was measured on a phone. The device test results go in `docs/RESPONSIBLE_AI.md` §9; do not estimate them.

---

## 1. Bean finder (NOT machine learning)

| | |
|---|---|
| File | `src/beans/beanfinder.py`, the frozen reference. The app runs a line-for-line JavaScript port (`app/src/lib/beanfinder.ts`) |
| Method | Downscale (longest side ≤ 1,200 px) → take the sheet colour from the median of a 4% border band → white-balance to sheet = 235 → "difference from sheet" = max(darkness, 1.5 × chroma) → Otsu threshold (floor 18) → 3×3 open then close → 4-connected components |
| Rejects | Specks (< 0.25 × median bean area). The code also has a frame-edge check (`beanfinder.py` l.124), but **it never fires**: the morphology step clears the 1 px border first, so a bean cut by the photo edge is kept, counted and classified (the same applies to the JS port). On the J4ckDev photos no detected bean touches the edge, so the reported numbers are unaffected |
| Flags | "Touching" when a blob is > 1.8 × the median bean area (Python parity rule). The app adds `app/src/lib/touching.ts` (§4 step 1) |
| Quality checks it reports | Too dark (sheet luma < 90); blurry (Laplacian variance < 15); count outside 20–160; too many touching |
| Crop contract (shared with the model and the app) | Square box around each bean, padded by 18% of its longer side, filled with sheet colour, resized to 128 × 128 px (bilinear), RGB float in [0, 1], NCHW |

**Measured accuracy (no AI):**

| Test | Result | Source |
|---|---|---|
| CBD, 464 light-box photos, stated 50 beans each (India, CC BY 4.0) | exactly 50 on **280/464 (60.3%)**; within ±1 on **373 (80.4%)**; within ±2 on **406 (87.5%)**; mean absolute error **1.006 beans**; worst photo off by 15. Of the 58 photos off by more than 2, 50 are over-counts and 8 under-counts | `reports/count_cbd.json` (re-run twice by reviewers, identical) |
| CBD, by grade (mean count; min–max) | A 51.4 (49–59) · AA 50.1 (48–55) · AAA 50.0 (45–53) · AB 50.2 (48–59) · Bits 51.1 (44–65) · Bulk 50.5 (45–56) · C 50.9 (44–60) · PB-I 49.9 (40–55) · PB-II 50.7 (48–58) | `reports/count_cbd.json` |
| J4ckDev, 11 photos | matches the README count on 10/11. On `Normales.jpg` the finder counts 195 where the README says 255; no detected bean touches the frame edge and a visual check shows about 195 beans, all boxed. The gap is in the README | EVIDENCE M2 |
| Python vs the app's JavaScript port | different JPEG decoders: counts identical on 21/21, max box deviation 1 px | `app/reports/beanfinder_parity_report.json` (re-run 4 Oct 05:25) |
| loja_yolo tray photos (Ecuador) | fewer blobs than labelled beans: 3,050 annotated beans gave 1,857 blobs, of which 1,325 matched a label. Whether the shortfall is merged or missed beans was not measured | `docs/DATA_SOURCES_V2.md` §8 |

**Known weaknesses.**
- The "Bits" grade (broken pieces) has the widest error: small fragments split or merge.
- The finder expects a plain light sheet with beans away from the edges.
- On a dark board it segments shadows (lojano board photos, `DATA_SOURCES_V2.md` §8).
- A patterned cloth or a sack background is untested.

---

## 2. Per-bean classifier (the AI)

| | |
|---|---|
| Task | One bean crop → probabilities over 6 classes: **good / dark** (black, sour, fungus) **/ insect / broken** (broken, chipped, cut, immature, shell) **/ unhulled** (parchment, dried cherry, husk) **/ other defect** (untyped or rare defects). **The app uses only P(good)**; the defect type is shown on screen as "possibly …" and is never spoken or sent. History keeps the per-type counts of a banded result on the phone (`app/src/lib/history.ts`) but shows only the defect total |
| Architecture | torchvision MobileNetV3-Small, ImageNet-pretrained (`IMAGENET1K_V1`), 6-class head. The full model has 2,542,856 parameters with its 1,000-class head (EVIDENCE E50) |
| Input / outputs | `image`: float32 [N, 3, 128, 128] RGB in [0, 1]; ImageNet normalisation, T and softmax are inside the graph. `probs`: [N, 6] = softmax(logits / T), **T = 2.046**. `embed`: [N, 1,024], the penultimate feature, L2-normalised; used only by the calibration pilot (§3) |
| Deployed file | `app/public/models/farz_beans.onnx` = `models_v2/farz_beans_v2_clean_fp16.onnx` (not in the public repo), **3,082,642 bytes (3.1 MB), fp16**, sha256 `c40c10f915f7…` (`reports_v2/clean_model.json → model`; checked by `shasum`). The app checks the sha256 against its labels file at load |
| Runtime | onnxruntime-web 1.30.0 (MIT), wasm backend, 1 thread, in a Web Worker. Its wasm file is 14,239,897 bytes (EVIDENCE M4) |
| Training data | **7 datasets that state a licence, 92,690 crops**: J4ckDev 582 (CC BY-NC-SA 4.0), vicanadya16 86,123 (CC0), loja_yolo 1,311 (CC BY 4.0), lojano 1,182 (CC BY-NC 4.0), samruddh_grading 1,337 (MIT), afiyah_deteksi 985 and afiyah_bijikopi 1,170 (CC0). Of these, 2,643 crops are sound beans. **No dataset without a licence is in the training pool** (sampler log; `reports_v2/clean_eval.json → final_weights_train_info`) |
| Recipe | Same as the audited v2 recipe (`src/beans_v2/train_common.py`): 3,000 steps × batch 128, AdamW + OneCycle (max lr 2e-3), label smoothing 0.05, seed 0. Sampling is source-balanced (label kind → dataset → crop). Untyped defects use a partial-label loss. GPU augmentation. Shipped weights: `train_folds.py licensed` (`reports_v2/clean_model.md` §1, §9) |
| Per-bean rule | **Sound if P(good) ≥ 0.55; defect if 1 − P(good) ≥ 0.91; otherwise "?"**; touching beans are always "?". T and the defect threshold were fitted on the leave-one-dataset-out predictions (§2.2), with the two afiyah sets as one group (`train_eval.pick_pair`: answered accuracy ≥ 90% and stable, then the largest coverage). That fit gave t_good 0.50, the lowest value on its grid; pooled over the held-out predictions it answered **69.5%** of beans at **90.2%** answered accuracy, in-sample for the threshold choice (`clean_model.json → pair_thresholds`). A pre-registered sweep then raised t_good to **0.55** (§2.2a); coverage and answered accuracy were not re-measured at 0.55 |
| fp16 vs fp32 | All 23,447 CBD crops: 99.64% same top class, 99.53% same three-way call (at 0.50 / 0.91), max probability difference 0.0076, embedding cosine ≥ 0.99988 (`clean_model.json → fp16_vs_fp32`). Kept by the pre-set bar of ≥ 99% |
| Browser vs Python | Model run, 300 CBD crops, onnxruntime-web in Node: max probability difference 4.9 × 10⁻⁴, 300/300 same calls (`clean_model.json → wasm_parity`). App tests, 40 crops: 4.6 × 10⁻⁴, 40/40 same calls, embedding cosine ≥ 0.999996. 3 fixture photos end to end: identical boxes and calls (`app/reports/classifier_parity_report.json`, `pipeline_parity_report.json`) |
| Speed (**MacBook M5 Pro, not a phone**) | Built app in Chromium, Web Worker, tap → result: median 344 ms on the 464 CBD photos (p90 362, max 493), 349–395 ms on the 7 demo trays (`app/reports/browser_measure.json`). Page thread: 332 ms, and 1,310 ms with the CPU slowed 4× (`app/reports/e2e-measurements.json → timing`). Chromium cannot throttle a worker |
| Latency on a phone | not yet measured (iPhone test pending; `docs/RESPONSIBLE_AI.md` §9) |
| Licence | **CC BY-NC-SA 4.0: a non-commercial, ShareAlike research prototype**, because two training datasets are non-commercial (J4ckDev, ShareAlike; lojano). The licence of the ImageNet-pretrained starting weights is UNVERIFIED |

### 2.1 Error direction

On new datasets the model calls many sound beans defective (`reports_v2/clean_model.md` §2–3; lojano and afiyah with thresholds fitted without them, USK at the deployed defect threshold 0.91). A defect call depends only on P(defect) ≥ 0.91, so moving t_good to 0.55 does not change these rates:
- **lojano** (held out): 31.4% of sound beans are confidently called defective.
- **afiyah** (both sets held out): 25.1%.
- **USK** (test only): 14.7%, and 20.8% of its defective beans are confidently called sound.

The failure that costs a farmer money is under-grading her coffee. So the output is a count, never a grade, and the photo rules (§4) send most new samples to a person.

### 2.2 Generalisation: the honest numbers

| Test | What it tells you | Result | Source |
|---|---|---|---|
| **Leave one dataset out**, 6 groups (afiyah twins held out together); T and thresholds fitted on the other 5 | sound vs defect on a dataset never seen | balanced accuracy (no "?"), mean of the **5 groups with sound beans: 0.644** (loja_yolo 0.671, lojano 0.618, afiyah 0.556, J4ckDev 0.575, samruddh 0.801; vicanadya16 has no sound beans: defect recall 1.000). Answered accuracy below 90% on lojano (76.4%), afiyah (76.3%), J4ckDev (79.7%) | `clean_model.json → loso_licensed`; `clean_model.md` §2 |
| **5 datasets with no licence, test only** (shipped model; T and thresholds never saw them) | the same, on data that never touched training | balanced accuracy USK **0.539**, daffa **0.541**, Mindforge 0.625; mfu17 and Notplying (no sound beans): defect recall 0.963 and 0.999. Answered accuracy USK 65.0%, daffa 75.1% | `clean_model.json → unlicensed_test_only` |
| **Defect type, J4ckDev photo never seen** (11 folds, each trained without one photo; the 10 defect photos hold 387 beans; argmax, no "?") | can it name the defect? | **30.0%** bean-weighted. **Always guessing the commonest type ("dark", 148 of 387) scores 38.2%**, so it does not beat the majority baseline. Normales (195 sound beans): 0 reach P(good) ≥ 0.5 and 73.8% are answered, all as "defect" | `clean_model.json → j4ck_lopo_typed_accuracy_same_10_as_v1`; `clean_model.md` §4; class counts from `reports/beans_lopo.json` |
| **CBD, 464 real photos from India** (23,447 beans; size grades, no per-bean labels; never used for training or fitting) | ranking of grades by the share of beans not called good | Spearman **+0.68** with the grade order. AUROC Bits > AAA **1.00**, C > AAA 0.951, PB-I > AAA 0.883, **AA > AAA 0.44** (top grades not separated). Median AAA photo defect share 53% | `clean_model.json → cbd`; `clean_model.md` §6 |
| **The app on those 464 photos** (built app, Chromium, t_good 0.55) | what a farmer would hear | **0 bands**: 421 "not sure (implausible)", 38 "not sure (too many unsure)", 5 blur retakes; AAA "many" **0 of 50**. Node run of the same app code: 0 bands (420 / 39 / 5), AAA "many" 0 of 50, also 0 with every retake gate ignored. At t_good 0.50 it was 392 / 67 / 5, also 0 bands | `app/reports/browser_measure.json`; `app/reports/rules_measure_node.json` |
| 777 software re-shoots of 111 AAA + AA photos (7 perturbations each) | robustness of the refusal | **0 bands** on the auditor's perturbations; 0 on the earlier report's (both at t_good 0.50, so also 0 at 0.55) | `clean_model.json → stress_777_auditor_banded, stress_777_report_banded` |
| Real held-out photos (retake gates ignored; thresholds fitted without that dataset) | new real trays | lojano **0/36** bands; loja_yolo **0/315** (with the deployed thresholds at t_good 0.50, 1 band, which was right: 3 of 7 beans defective → "many"); J4ckDev **0/11** | `clean_model.json → lojano_real, loja_yolo_real, j4ck_lopo_real` |
| **Held-out trays at the shipped thresholds** (each licensed group scored by its leave-one-out fold model, the 5 unlicensed sets by the shipped model; 50- and 100-bean trays at 0, 2, 10, 15, 30, 45 and 100% defects, 200 per cell; plus the auditor's 1,200 J4ckDev trays) | how often it answers; wrong and dangerous answers | **24,800 trays: 16 bands, 1 wrong, 0 dangerous** (samruddh 5 bands, 1 wrong: a clean tray called "some"; J4ckDev 1,200: 11 bands, all right). The app's own `callBean()` + `decide()` give the same counts. **This is one random draw.** 20 further draws of the same design (run afterwards, not pre-registered): 12–29 bands per draw, 0 dangerous in every draw; all wrong bands are samruddh's, **26 of 105 (about 1 in 4)**, each one band off; J4ckDev 1,200: 312 bands, 0 wrong | `reports_v2/threshold_sweep.md`; `app/reports/threshold_app_measure.json`; `reports_v2/threshold_seeds.json` |
| The same trays at t_good 0.50 | why t_good moved | 293 bands, 13 wrong, 0 dangerous; samruddh **13 wrong of 52 bands (25%)**: 12 clean trays called "some", 1 "some" tray called "many". The model run's earlier 50-bean trays showed the same: held-out thresholds 1 band in 7,200 (0 wrong), pooled 0.50 thresholds 42 bands, 11 wrong, 0 dangerous, all samruddh; unlicensed sets 0 in 4,600. Over the 20 further draws: samruddh 217 wrong of 993 bands (22%) | `reports_v2/threshold_sweep.md`; `clean_model.json → trays_heldout_totals`; `reports_v2/threshold_seeds.json` |
| The auditor's 1,200 J4ckDev trays (spatially held-out halves, 80 beans, 0 / 12.5 / 30% defects) | the one set-up where it used to answer often | **11 bands, 0 wrong, 0 dangerous** at t_good 0.55 (all 11 are "many" trays); 241 / 0 / 0 at 0.50; 381 / 0 / 0 with thresholds fitted without J4ckDev. Re-running v1 reproduces the auditor's 793 / 57 / 10 exactly | `reports_v2/threshold_sweep.md`; `clean_model.json → j4ck_1200` |

**Reading it.**
- On a farm it has never seen, the model separates sound from defective beans only somewhat better than a coin flip. It cannot name the defect type better than always guessing the commonest type.
- Every safe outcome above comes from the photo rules (§4), not from per-bean accuracy. Those rules turn weak evidence into "not sure", which is why Farz almost never answers on new data (16 bands in 24,800 held-out trays; 12–29 in each of 20 further draws).
- They cannot catch an unfamiliar sample whose beans are confidently called mostly *sound*.
- Nothing in Farz should be read as a measured accuracy on Yemeni beans or on phone photos.

### 2.2a Why t_good is 0.55

- **The problem.** The pooled fit put t_good at 0.50, the floor of its grid, so a bean just over P(good) = 0.5 counted as sound with no margin, and held-out samruddh trays got 25% wrong bands at that setting (above).
- **The rule, fixed before the run.** Written and hashed at 04:53 BST on 4 Oct, 10 s before the sweep ran (SHA-256 `d8d95c87…`, recorded in `threshold_sweep.json`): the smallest t_good in {0.50, 0.55, 0.60, 0.65, 0.70, 0.75} with (a) 0 dangerous trays in every dataset unit and (b) at most 5% wrong bands in every unit that gets 10 or more bands. Temperature, t_defect and the model are unchanged.
- **Result.** 0.50 fails (b) on samruddh; **0.55** is the smallest that passes; 0.60–0.75 also pass, with 2 or 0 bands.
- **Caveat.** The sweep used the same held-out trays that exposed the samruddh problem, so the 1 wrong band at 0.55 is in-sample for the choice of 0.55.
- **One draw understates the samruddh error.** On 20 further draws of the same design (run afterwards, not pre-registered; `reports_v2/threshold_seeds.json`) the same rule picks 0.55 in 18 and 0.60 in 2, with 0 dangerous trays at every candidate in every draw. On samruddh, 0.55 mainly makes Farz answer less often (105 bands over the 20 draws, against 993 at 0.50); when it does answer, about 1 band in 4 is one band off (26 of 105; 217 of 993 = 22% at 0.50), never clean ↔ many.
- **Cost.** Bands on held-out data fall from 293 to 16 of 24,800 trays; on the 1,200 J4ckDev trays from 241 to 11, and only "many" trays still get one.
- **Shipped** in `app/src/lib/rules.ts V2_THRESHOLDS` and `app/public/models/farz_beans_labels.json`; re-measured on CBD, the demo trays and the browser (`reports_v2/threshold_ship.md`). `models_v2/farz_beans_v2_clean_labels.json` (not in the public repo) still records the pooled fit (0.50).

### 2.3 Is AI actually needed? (the brief's "simpler tool" question)

SMS, a spreadsheet or a search cannot look at a bean. The real question is whether a simple **colour rule** could do the classifier's job. For the shipped model, a colour-only baseline was **not measured** on the same held-out protocol, so we do not claim the CNN beats colour on unseen photos. What we can say:
- insect holes, breaks and parchment are shape and texture cues;
- the model's embedding is what makes the calibration pilot (§3) possible: a colour rule offers no such feature to tune.

Do **not** use bean size as a feature in such a baseline. Within J4ckDev each class comes from one photo with its own scale, so size gives away the photo, and the photo gives away the label.

### 2.4 Intended use, and uses it must not be put to

- **Intended:** a **self-check** that counts likely defects in a sample of about 100 hulled green beans from the farmer's own lot, on a plain light sheet, before the price is set. A person makes every decision; the cooperative grader checks the physical sample.
- **Not intended:**
  - grading or pricing coffee;
  - naming a defect type as fact;
  - certifying "specialty" quality;
  - diagnosing a pest or disease;
  - roasted coffee, cherries or leaves;
  - any decision about a farmer made by someone else from the farmer's slip;
  - any commercial use (licence).

---

## 3. Cooperative calibration (PILOT, off by default)

| | |
|---|---|
| What it is | A **prototype head** on the model's `embed` output. A grader labels 10 sound and 10 defective beans from the cooperative's own trays. The phone stores the normalised mean embedding of each pile (μ_sound, μ_defect) and a margin m_hi. Each bean is then scored by s = cos(e, μ_defect) − cos(e, μ_sound) |
| Per-bean call while a head is active | touching → "?". If the head and the base model's own binary call (P(good) ≥ 0.5, as pre-registered; not the 0.55 sound threshold) agree, that answer. If they disagree but \|s\| ≥ m_hi, the head's answer. Otherwise "?". The photo rules (§4) are unchanged (`reports_v2/localcal_spec.md` §3) |
| m_hi | the larger of (a) the largest \|s\| among calibration beans mis-sorted in leave-one-out and (b) the median \|s\| (`localcal_spec.md` §2) |
| Stored | IndexedDB `farz-localcal`: the two vectors, m_hi, leave-one-out errors, a name and counts. No photos, no embeddings of test beans. A head made with another model file (sha256) is ignored |
| Our extra guard | a head is **refused if more than 10% of its 20 beans are mis-sorted in leave-one-out**. The study lists this as post-hoc and not validated; refusing is the safe direction. In the study it would have kept 13 of 15 loja_yolo draws and rejected all Mindforge and USK draws. It would still have let through one lojano head that called a clean tray "many" (`localcal.md`, exploratory section) |
| Parity | m_hi 0.63483 in onnxruntime-web (Node) vs 0.63481 in Python, 0 leave-one-out errors in both; Chromium 0.6348 (`app/reports/localcal_demo_measure.json`; e2e test 10) |

**Pre-registered study** (`reports_v2/localcal.md`):
- **When the bar was fixed:** 02:14 BST, before any result. The block's SHA-256 is `d97a6694…`; it is recorded in `localcal.json` and was re-checked by us.
- **Set-up:** base model trained without loja_yolo; K ∈ {20, 50, 100} labelled beans; 5 draws; every bean of every calibration photo is removed from the test set.
- **GO rule:** at some K, all three must hold: (1) mean class-balanced answered accuracy ≥ 90% and coverage ≥ 50%; (2) 0 clean trays called "many" in 2,500; (3) 0 CBD AAA photos called "many" in every draw.

| Dataset (role) | K | Answered accuracy, balanced (worst draw) | Coverage, balanced (worst) | Trays: bands / wrong / **clean → "many"** (of 2,500) | Verdict |
|---|---|---|---|---|---|
| **loja_yolo (primary)** | 20 | **96.3%** (95.5%) | **78.3%** (70.8%) | 131 / 17 / **0** | **GO** (all 3 criteria; also at K = 50, 100) |
| loja_yolo, base model alone | — | 97.4% | 65.7% | 0 bands | (comparison) |
| lojano (secondary) | 20 | 84.9% | 72.6% | 152 / 77 / **11** | fails |
| Mindforge (secondary, test-only set) | 20 / 50 | 78.7% / 78.9% | 71.9% / 73.3% | 3 / 2 / **1**; 16 / 9 / **2** | fails |
| USK (secondary, test-only set) | 20 | 63.0% | 62.0% | 0 / 0 / 0 | fails (63.0–67.4% across K = 20–100, against 64.2% for the base model alone: not rescued) |

Source: `reports_v2/localcal.json` (tallies re-summed from the per-draw records).

Limits:
- The GO covers one dataset. The bar could not separate the head from the base model, which alone meets the accuracy bar there.
- With lojano heads, one AA and one A CBD photo were called "many".
- Calibration photos of a single sound tray and a single defective tray probably teach the head "which photo", not "which bean".
- No Yemeni beans, no phone, no re-photographed trays; one base-model seed.

**In the app:** a labelled pilot for the grader (Home → "Calibrate for your cooperative"), inactive until a head is saved, with its own reset. So the pass rate is plain: **it passed the pre-registered bar on 1 of the 4 datasets it was tried on.**
- **The demo is the quality check only** (since 4 Oct 05:18). Labels from the public loja_yolo dataset stand in for a grader's: correct labels give 20/20 (0 leave-one-out errors, m_hi 0.6348) and are accepted; the same 20 beans with 6 deliberately wrong labels give 13/20 (7 errors) and are refused (`app/reports/calib_demo_check.json`, Node = Chromium; e2e test 10). The demo never saves or installs a head, and e2e checks that a grader's saved head is byte-identical after it runs.
- **Why the old before/after demo was dropped.** It used 3 loja_yolo photos, and the shipped model was trained on loja_yolo: on loja_yolo crops it already calls 146/146 sound beans sound and 1,116 of 1,163 defective beans defective, with 47 unsure (`app/reports/loja_insample_check.json`). Re-measured at 0.55 the 3 photos still gave the same outcome before and after calibration (many → many; not sure → not sure twice; `app/reports/localcal_demo_measure.json`).
- **What a real pilot needs:** a grader's own labelled trays plus an acceptance test on trays not used for calibration.

---

## 4. The sample rule and abstention (NOT machine learning)

The rules in `app/src/lib/rules.ts`, `touching.ts` and `wilson.ts` of the final build (Python mirror: `src/beans/pipeline.py`). In order:

1. **Quality gate** (before the model; shipped `gate()` order):
   1. too dark → `c03`;
   2. **fewer than 20 beans, nothing touching and nothing off-colour** → `c05` "use about 100 beans";
   3. blurred → `c02`;
   4. not green coffee (`colourgate.ts`: hue, saturation and dark-bean limits; a bean-coloured big blob while many blobs touch counts as a clump → `c04`) → `c06`;
   5. too many touching → `c04`;
   6. count outside 20–160 → `c05`.

   **Touching** (`touching.ts`): a blob is touching if its area is > 1.8× the median blob, > 1.6× a robust one-bean area, or its shape is one no single bean has (solidity < 0.80 or ellipse fill < 0.85). The spread-out retake fires when touching blobs exceed max(2, 8% of blobs), or when beans hidden in bean-coloured clumps exceed max(2, 8% of beans).

   A fixed clip asks for a retake; the model is not run. The 20-bean minimum is relaxed to 1 only for the grader's own calibration photos (`Calibrate.tsx`, `rules.ts` `minBeans`); a farmer's check always uses 20.

   **Dark lot** (`darklot.ts`, since 4 Oct). When the colour gate would say "not green coffee" because the beans are too dark, the photo gets `c07_unsure` ("not sure, take it to the cooperative") instead of `c06` if ≥ 80% of the blobs look like single beans (not touching, elongation 1.15–2.2), blob areas vary by CV ≤ 0.6, and the dark blobs have median hue 26–62° and median saturation ≤ 0.42. The photo checks, the count range and the touching retake come first. The colour limits rest on one real photo (J4ckDev Negros: hue 31.8°, saturation 0.26) and a synthetic dark roast (21.6°, 0.58); real roasted coffee is untested, and either answer gives no band (`reports_v2/threshold_ship.md` §5).
2. **Per-bean call** (§2): sound if P(good) ≥ 0.55, defect if 1 − P(good) ≥ 0.91, else "?". Touching blobs are always "?".
3. **Nothing answered** → `c07_unsure`.
4. **Implausible sample.** If more than **60%** of the answered beans are called defective (`IMPLAUSIBLE_DEFECT_FRAC = 0.6`), the answer is "not sure, take the sample to the cooperative" (`c07_unsure`). Consequence: Farz never says "many defects" above 60%; those samples go to a person.
5. **Model uncertainty.** If more than **15%** of the beans are "?" (`MAX_UNSURE_FRAC = 0.15`) → `c07_unsure`.
6. **Band** from the point estimate of the defect rate among answered beans (clean < 5%, some 5–20%, many > 20%; our own rule of thumb, not an official grade), always shown with its **95% Wilson interval**.
   - When the interval crosses a band edge, the band is marked **"about"** and `c18_another` asks her to photograph another handful of the same coffee.
   - Up to 3 handfuls are pooled into one count and one interval. If any single handful is "not sure" on its own, the pooled result is "not sure" too.
   - The band clip (`c08`/`c09`/`c10`) is the only result clip. The type clips `c11`–`c14` are not played since 4 Oct.

**What the rule does at 100 answered beans** (formulas in `wilson.ts`; examples in EVIDENCE M5):

| Defects in 100 beans | 95% interval (example) | Output |
|---|---|---|
| 0 | 0.0–3.7% | "clean" |
| 1–4 | 3 → 1.0–8.5% | "about clean" + another handful |
| 5–9 | — | "about some" + another handful |
| 10–12 | 12 → 7.0–19.8% | "some" |
| 13–20 | 18 → 11.7–26.7% | "about some" + another handful |
| 21–27 | — | "about many" + another handful |
| 28–60 | 30 → 21.9–39.6% | "many" |
| 61–100 | — | **not sure** (implausible) → cooperative |

**Demo trays in the built app** (Chromium, t_good 0.55, `app/reports/browser_measure.json → demo`; manifest `app/public/demo/manifest.json`; all 7 asserted by e2e test 4):
- SYNTHETIC trays (in-sample): 0% clean (0/96) · ~3% about clean (3/99) · ~12% some (11/93) · ~30% about many (25/92) · ~40% many (37/94).
- Real CBD photos: `cbd_aaa_2` not sure (implausible: 9 of 14 answered beans defective, 36 of 50 "?"; at 0.50 it was "too many unsure") · `cbd_bits_262` not sure (implausible).

---

## 5. History: the first model (v1), and why it was replaced

| | v1 (J4ckDev only) | v2, all 12 datasets (not shipped) | v2 licence-stated (shipped) |
|---|---|---|---|
| File (the v1 and all-12-datasets files are not in the public repo) | `models/farz_beans_fp32.onnx`, 6,104,656 bytes fp32, 5 classes, abstain if top probability < 0.59 | `models_v2/farz_beans_v2_fp16.onnx`, 3,084,405 bytes | 3,082,642 bytes fp16, 6 classes + embedding |
| Within-photo held-out accuracy (spatial halves, **upper bound**) | 86.0% (folds 81.6% / 90.5%; `reports/beans_model.json`) | not measured | not measured |
| J4ckDev photo never seen, defect type, same 387 beans (majority baseline **38.2%**, 148/387) | **27.4%** (`reports/beans_lopo.json`) | 42.9% (`AUDIT_V2.md` (2)) | **30.0%** |
| Sound vs defect, leave one dataset out (datasets with sound beans) | – | 0.721 over 9; 0.657 with the afiyah twins held out together (`AUDIT_V2.md` (6)) | 0.644 over 5 groups |
| CBD Bits > AAA AUROC | 0.07 (`AUDIT_V2.md` (6)) | 1.00 | 1.00 |
| CBD photos with a confident band | 167 of 464 "many" under the 22:42 rules (27 of 50 AAA), from `reports/beans_ood_cbd_preds.npz`; 0 after the 60% rule | 0 of 464 | 0 of 464 |
| Auditor's 1,200 J4ckDev trays: bands / wrong / dangerous | 793 / 57 / **10** | 229 / 0 / 0 | 11 / 0 / 0 at t_good 0.55 (241 / 0 / 0 at 0.50) |
| Licence of the weights | CC BY-NC-SA 4.0 | non-commercial; includes 5 datasets with no licence | CC BY-NC-SA 4.0; only datasets that state a licence |

Why v1 was replaced:
- **One photo per class.** v1 partly learned which photo a bean came from: 86.0% within a photo, but 27.4% on a photo never seen, below always guessing the commonest type.
- **Real photos.** On real Indian photos it called 54.3% of beans "broken" and 0% of AAA beans "good" (`reports/beans_ood_cbd.json`).
- **Dangerous answers in its own set-up.** On the auditor's crop-level trays from beans it had not seen, it called 10 of 400 clean trays "many" (`AUDIT_V2.md` (3d)).

The audit also corrected two of our earlier claims:
- **"0.78 balanced accuracy on new sources".** Three sources with no sound beans inflate the mean.
- **"v2 is not more accurate than v1 in J4ckDev's set-up".** That figure was v2 *with* local re-calibration; the fair comparison is the 1,200-tray row.

---

## 6. Known failure modes

| Risk | What happens | Mitigation in Farz |
|---|---|---|
| Yemeni natural-process beans look different from every training set | Sound beans may be called defective, so the farmer under-grades her own coffee | Counts not grades; no types as facts; the slip says «فحص ذاتي مش تصنيف» ("self-check, not a grade"); "not sure" sends her to the cooperative grader |
| A new photo set-up (camera, light, cloth, farm) | Per-bean calls near a coin flip (§2.2) | The 15% and 60% rules: 0 bands on every new real set we tested. They do not catch a new sample that mostly looks "sound" |
| A cooperative calibration that does not fit | On two of four datasets, the same method called clean trays "many" | Off by default; the >10% leave-one-out refusal; "Calibrated for" shown on every result; acceptance test on unused trays required for a pilot (not enforced by the app) |
| Roasted beans, hands, rice | Should be refused by the colour gate («هذا مش بن أخضر», "this isn't green coffee"). Measured on real photos: all 464 CBD photos and 9 of 11 J4ckDev photos pass the gate (`app/reports/colourgate_all_photos.json`). Hands, roasted beans and leaves were tested only as synthetic images: dark and medium roast and hands over a tray get "not green coffee". At t_good 0.50 a reviewer got a confident "clean" on 3 of 36 synthetic light-roast trays made from J4ckDev beans (`reports/REGRESSION_B_app.md` F2); at 0.55 all 38 light-roast trays get no band (`reports_v2/threshold_ship.md` §5) | Test with real roasted-bean and hand photos on the phone (pending) |
| A very bad lot, mostly black beans | Until 4 Oct the colour gate refused the J4ckDev all-black photo (Negros, 100 black beans) as "not green coffee", the wrong message for the worst kind of lot. The dark-lot rule (§4) now answers "not sure, take it to the cooperative"; the other J4ckDev photo that fails the gate (CerezaSeca, 9 dried cherries) still gets "not green coffee" | Either way no band is given; the colour limits rest on one real photo |
| Bean cut by the photo edge | Kept and classified (the edge check never fires, §1) | Tell users to keep beans away from the edges |
| **Touching beans: all in tight pairs (R1)** | If fewer than 5 single beans lie apart, the one-bean reference is itself a pair. On synthetic trays of 50 side-by-side pairs, 0 pairs were flagged on the 10 trays with ≤ 4 single beans (bean-finder level, model-independent). With the v1 model, some all-sound trays then got "about many" (`reports/REGRESSION_A2.md` check 3c). Re-run with the shipped model at t_good 0.50: all 36 aligned-pair trays end in a spread retake (26) or "not sure" (10), and a "not sure" cannot become a band at 0.55 (`reports/REGRESSION_B_app.md` F5) | Spread the beans out (`c01_welcome`, `c04_spread`); needs real photos |
| **Touching beans: a broken piece pressed against a whole bean (R2)** | The merged blob looks like one bean; 0–1 of 10 such pieces flagged. With the v1 model, one synthetic tray with 10% defects got a confident "clean" in the browser (REGRESSION_A2 check 3c). Re-run with the shipped model at t_good 0.50: 7 of 9 such trays "about clean" where the truth is 10% (`reports/REGRESSION_B_app.md` F5); not re-run at 0.55 | The error runs in the farmer's favour; spreading the beans is the defence |
| **Widely mixed bean sizes (R3)** | A large single bean can be flagged as touching; 4 of 14 synthetic trays at ±25–30% size jitter got a false "spread them out" retake (REGRESSION_A2 check 3c; the gate runs before the model, so this does not depend on it) | Fails safe (a retake), but such a lot may never get a result; a shape guard was proposed, not built |
| Rotated photo | Per-bean calls can change with rotation (measured for v1: a borderline tray moved "some" → "about some"; `reports/REGRESSION_A.md` check 2). **Not re-measured for the shipped model** | Treat borderline results as borderline |
| A bean with two defects | Only the sound/defect call counts | The count is per bean, not per defect |

## 7. Second "country pack": the leaf pipeline (smoke test only)

`src/prep.py` and `src/train.py` build a 5-class coffee-leaf classifier (healthy, rust, miner, phoma, cercospora). The data comes from BRACOL (Brazil), JMuBEN (Kenya) and Uganda, de-duplicated to 3,935 images. Uganda is held out as an unseen country.

The 2-epoch smoke test proves the pipeline only:
- MobileNetV3-Small ONNX is 6,102,733 bytes at fp32 and 1,876,904 bytes at INT8 (`models/coffee_mnv3s_smoke_*.onnx`, not in the public repo), at about 1 ms per image on CPU.
- **Its accuracy is meaningless and is not reported.**

It is not part of the Yemen build, because UNDP 2022 says rust is not a major issue in Yemen (EVIDENCE E9).
