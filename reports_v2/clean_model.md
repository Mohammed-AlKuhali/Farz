# Farz v2 CLEAN model: retrained only on sources that state a licence, re-tested, exported with an embedding output

STATUS: COMPLETE, 03:15 BST Sun 4 Oct 2026. Fold training ran 02:11–03:10 BST. Every number comes from a script in `src/beans_v2/clean_*.py`, and its raw value is in `clean_eval.json`, `clean_export.json` or `clean_deploy.json`; `clean_model.json` collects the headline figures. Every decision taken during the run is recorded in §9. Nothing in `app/`, `models/` or `docs/` was changed. In `src/beans/` the only change is the requested `logs/` fix in `train_beans.py`.

## Headline table (shipped file `models_v2/farz_beans_v2_clean_fp16.onnx`)

| What | Number | Protocol |
|---|---|---|
| Training data | 7 licence-stated sources, 92,690 crops | afiyah_deteksi, afiyah_bijikopi, vicanadya16 (CC0); loja_yolo (CC BY 4.0); samruddh_grading (MIT); lojano (CC BY-NC 4.0); J4ckDev (CC BY-NC-SA 4.0). Same recipe as v2 (§1) |
| Licence of the weights | **non-commercial, ShareAlike research prototype** (CC BY-NC-SA 4.0 via J4ckDev) | – |
| File | 3,082,642 B fp16 (fp32: 6,107,254 B). Outputs `probs` [N,6] and `embed` [N,1024] | §5 |
| fp16 vs fp32 | **99.64%** same top class, 99.53% same good/defect/unsure call, max prob diff 0.0076, embed cosine >= 0.99988 | all 23,447 CBD crops |
| onnxruntime-web wasm vs Python (fp16) | max prob diff **4.9e-4**, max embed diff 1.5e-3; **300/300** same top class and same call | 300 CBD crops, Node, 1 thread |
| Temperature, thresholds | T = 2.046; **good if P(good) >= 0.50, defect if P(defect) >= 0.91** | fitted on clean LOSO held-out predictions; afiyah twins held out together |
| Pooled held-out coverage / answered accuracy | 69.5% / 90.2% | in-sample for the threshold choice |
| LOSO, groups with good beans: mean good-vs-defect balanced accuracy | **0.644** (5 groups: 0.556–0.801) | no abstention |
| Unlicensed sources (test only, never trained on): balanced accuracy | USK 0.539, daffa 0.541, Mindforge 0.625; mfu17 / Notplying (no good beans) defect recall 0.963 / 0.999 | shipped model |
| CBD 464 real photos | **0 bands, 0/50 AAA "many"** (with every gate ignored) | shipped fp16 |
| CBD ranking | Spearman **+0.68** vs grade-folder order; AUROC Bits > AAA **1.00**; AA > AAA 0.44 | per-photo argmax defect share |
| 777 CBD re-shoots | **0 bands** with the auditor's perturbations; 0 with the v2 report's | gates ignored |
| Real held-out photos | lojano 0/36 bands; loja_yolo 0/315 bands (thresholds fitted without loja); J4ckDev LOPO 0/11 bands | gates ignored |
| Auditor's 1,200-tray J4ckDev protocol | **241 bands, 0 wrong, 0 dangerous** (v1 re-run: 793 bands, 57 wrong, 10 dangerous, the auditor's exact numbers) | clean_half models |
| 50-bean trays from held-out crops | licensed groups: 1 band in 7,200 trays (held-out thresholds; 0 wrong); unlicensed sources: 0 in 4,600 | auditor design |

## Bottom line
1. **The licence-stated model is as safe as the all-sources v2, and about as weak per bean.**
   - Every safety test that the auditor ran on v2 gives the same answer here: 0 confident bands on CBD, 0 on 777 re-shoots, 0 wrong and 0 dangerous on the 1,200 J4ckDev trays.
   - On held-out sources, per-bean good-vs-defect balanced accuracy is 0.54–0.80.
2. **No unlicensed source is needed for the shipped behaviour.** The 5 sources with no licence are used here only as test sets, and the model behaves on them as v2 did with them held out:
   - USK 0.539 vs 0.532;
   - daffa 0.541 vs 0.570;
   - Mindforge 0.625 vs 0.733 (worse: v2's Mindforge fold had USK and daffa good beans in training).
3. **The app would still answer "not sure" on almost every new photo.**
   - With thresholds fitted without the scored source, the model gives:
     - 1 band in 7,200 trays from held-out licensed groups;
     - 0 bands in 4,600 trays from the unlicensed sources;
     - 0 bands on the real lojano and loja_yolo photos.
   - On the app's own 5 synthetic demo trays the bands are right: clean, clean, some, many, many. That is **in-sample**: the trays are pasted from J4ckDev crops, and J4ckDev is in training.
4. **The embedding output works for on-device calibration.** The pre-registered local-calibration study (`localcal.md`) reached GO on loja_yolo. The same study shows the method is not safe on two other sources. Read it before using the feature.

## 1. Recipe (unchanged from v2; `train_common.train`)
- **Classes and loss.** 6 classes. Partial-label loss for untyped defects.
- **Sampling.** Label kind → source uniform → crop.
- **Optimiser.** 3,000 steps × batch 128, AdamW, OneCycle with max lr 2e-3, label smoothing 0.05, seed 0. Same GPU augmentation.
- **Shipped weights.** The shipped weights are `licensed.pt`, trained at 01:09–01:15 by `train_folds.py licensed`.
  - Its sampler log lists exactly the 7 licensed sources.
  - Re-scored now, its logits on the 25,453 unlicensed crops match the saved ones bit for bit (max abs diff 0.0).
  - Fold models: `src/beans_v2/clean_folds.py`, 19 folds, same recipe and seed.
- **Training pools.** Every fold's training pool was checked: **no unlicensed source in any of them** (`clean_*.json → train_sources`).
- **No vicanadya16 in training changes a lot.** The vicanadya16-held-out fold trains on only 6,567 crops, so typed-defect evidence comes almost entirely from vicanadya16.

## 2. Leave-one-source-out over the licensed sources (6 groups; afiyah twins held out TOGETHER)
Protocol for each held-out group:
- The fold never saw that group.
- T and the two thresholds are fitted on the other 5 groups' held-out predictions.
- Weights: each group counts 1 in total, split equally between its good and defect beans.
- "Balanced acc." is good-vs-defect with no abstention (defect if P(good) < 0.5).

| Held-out group | n | T | thresholds (good / defect) | balanced acc. | good recall | defect recall | coverage at thresholds | answered accuracy | good beans called good / defect | defect beans called defect / good |
|---|---|---|---|---|---|---|---|---|---|---|
| loja_yolo | 1,311 | 2.29 | 0.58 / 0.91 | 0.671 | 0.349 | 0.992 | 68.1% | 99.4% | 15.8% / 2.1% | 74.2% / 0.2% |
| lojano | 1,182 | 2.00 | 0.50 / 0.88 | 0.618 | 0.251 | 0.985 | 70.7% | **76.4%** | 25.1% / **31.4%** | 83.7% / 1.5% |
| afiyah (both) | 2,155 | 2.04 | 0.50 / 0.89 | 0.556 | 0.113 | 0.999 | 60.9% | **76.3%** | 11.3% / **25.1%** | 93.8% / 0.1% |
| j4ckdev | 582 | 2.14 | 0.50 / 0.89 | 0.575 | 0.303 | 0.848 | 66.0% | **79.7%** | 30.3% / 9.7% | 63.8% / **15.2%** |
| samruddh_grading | 1,337 | 2.23 | 0.54 / 0.91 | 0.801 | 0.666 | 0.936 | 65.0% | 93.1% | 58.9% / 3.4% | 61.4% / 5.1% |
| vicanadya16 (no good beans) | 86,123 | 1.61 | 0.70 / 0.96 | 1.000* | – | 1.000 | 97.0% | 100.0% | – | 97.0% / 0.0% |
| **Mean, 5 groups with good beans** | | | | **0.644** | | | | | | |

\* Defect recall only.

- **The deployed values are pooled over all 6 groups:** T = 2.046, good >= 0.50, defect >= 0.91. Pooled coverage is 69.5% at 90.2% answered accuracy, but that is in-sample for the threshold choice. 0.50 is the lowest value on the threshold grid.
- **3 of 6 groups fall below 90% answered accuracy** with thresholds fitted without them: lojano, afiyah and J4ckDev.
- **Calibration does not transfer evenly.** Pooled weighted ECE is 0.115 (0.127 at T = 1).
- **Comparison with the all-sources v2** (AUDIT_V2 (2)): its mean over sources with good beans was 0.721, or 0.657 once the afiyah twin leak is removed. The clean model's 0.644 is comparable.

## 3. The 5 unlicensed sources: TEST ONLY (shipped clean model; T and thresholds never saw them)

| Source | n | balanced acc. (no abstention) | coverage at 0.50 / 0.91 | answered accuracy | good beans called good / defect | defect beans called defect / good | v2 all-sources, this source held out |
|---|---|---|---|---|---|---|---|
| usk_coffee | 7,969 | **0.539** | 46.3% | **65.0%** | 28.6% / 14.7% | 34.9% / 20.8% | 0.532 |
| daffa_defect | 1,000 | **0.541** | 53.8% | **75.1%** | 11.4% / 23.6% | 69.4% / 3.2% | 0.570 |
| mindforge_doubleside | 7,372 | 0.625 | 58.7% | 91.2% | 29.2% / 8.9% | 59.5% / 4.2% | 0.733 |
| mfu17 (no good beans) | 962 | 0.963* | 70.5% | 94.7% | – | 66.7% / 3.7% | 0.883* |
| notplying_defects (no good beans) | 8,150 | 0.999* | 85.1% | 99.9% | – | 85.0% / 0.1% | 0.996* |

\* Defect recall only.

- **The model is confidently wrong on USK and daffa,** as v2 was. USK: 20.8% of defect beans are called good, and 14.7% of good beans are called defect. Those are the beans most likely to mislead a band; the photo rules are what catch them (§7).
- **Earlier numbers for this same model are not comparable.** `model_v2.md` §2.4 used the same weights with a different temperature (fitted on the old 7-source LOSO, which never existed), so its numbers (USK 0.568, Mindforge 0.693) differ from these. The logits are identical; T = 2.05 moves P(good) below 0.5 more often.

## 4. J4ckDev leave-one-photo-out (11 clean LOPO folds)
Protocol for each photo:
- The fold trains on the 7 licensed sources minus that one photo, so the other 10 J4ckDev photos are in training.
- T and thresholds are the J4ckDev held-out-group values (T = 2.141, 0.50 / 0.89).

| Photo | class | n | typed recall | good-vs-defect correct (no abstention) | coverage at thresholds | answered correct |
|---|---|---|---|---|---|---|
| BrocadoLeve | insect | 30 | 0.267 | 0.967 | 56.7% | 94.1% |
| BrocadoSevero | insect | 25 | 0.320 | 1.000 | 100.0% | 100.0% |
| CerezaSeca | unhulled | 9 | 0.000 | 1.000 | 100.0% | 100.0% |
| Concha | broken | 33 | 0.727 | 0.970 | 81.8% | 96.3% |
| DXHongo | dark | 12 | 0.250 | 1.000 | 91.7% | 100.0% |
| Inmaduro | broken | 16 | 0.188 | 0.938 | 62.5% | 90.0% |
| MarronAVinagre | dark | 36 | 0.444 | 0.972 | 83.3% | 96.7% |
| Negros | dark | 100 | 0.190 | 1.000 | 98.0% | 100.0% |
| Normales | good | 195 | 0.015 | 0.000 | 73.8% | 0.0% |
| PMordidoCortado | broken | 36 | 0.917 | 1.000 | 83.3% | 100.0% |
| Pergamino | unhulled | 90 | 0.022 | 0.800 | 53.3% | 62.5% |

- **The same 10 photos as v1's LOPO:** bean-weighted typed accuracy is **30.0%**, against v1 27.4% and the all-sources v2 42.9%. The clean model has far fewer typed defect examples, because mfu17, Notplying and Mindforge are gone.
- **Normales (195 good beans) is a confident wrong answer, as in v2.** 0 good beans reach P(good) >= 0.5, and 73.8% are answered, every one as "defect". In this fold the other 10 J4ckDev photos in training are all defect photos, so the model learns "J4ckDev look = defect".
- **On the real photos all 11 end as "not sure (implausible)"** with gates ignored (§7). The 60% rule catches Normales.


## 5. Export (`src/beans_v2/clean_export.py` → `clean_export.json`)
- **Graph.** Input `image` float32 [N,3,128,128] RGB in [0,1], the existing crop contract. ImageNet normalisation, T = 2.0459 and softmax are inside the graph.
  - `probs`: [N,6] (good, dark, insect, broken, unhulled, other_defect).
  - `embed`: [N,1024], the penultimate feature, L2-normalised. Computed as u = h / max|h|, then u / ||u||: the same maths as h / ||h||, but the fp16 graph cannot overflow when squaring.
- **fp32 vs PyTorch.** Max prob diff 2.6e-6; max embed diff 1.7e-6. The embed is identical to the PyTorch helper the calibration study used.
- **fp16.** `convert_float_to_float16(keep_io_types=True)`, compared on 23,447 CBD crops:
  - top class 99.64%;
  - three-way call at 0.50 / 0.91: 99.53%;
  - max prob diff 0.0076;
  - embed cosine min 0.99988, mean 0.99998;
  - no NaN or Inf.
  - **Kept** (bar: >= 99%).
  - v1's own browser rule (max prob diff <= 0.01) also passes, unlike the all-sources v2 fp16 (0.0175, AUDIT_V2 (4)).
- **Browser runtime.** onnxruntime-web 1.30.0 (the app's package), wasm, 1 thread, Node 26, 300 CBD crops (`clean_wasm_parity.mjs`):

  | File | max prob diff | max embed diff | top class | call |
  |---|---|---|---|---|
  | fp16 | 4.9e-4 | 1.5e-3 | 300/300 | 300/300 |
  | fp32 | 8.9e-7 | 8.3e-7 | 300/300 | 300/300 |

- **Laptop CPU** (1 thread, batch 50; not a phone): 75.8 ms for fp16, 75.1 ms for fp32.
- **Files.** `models_v2/farz_beans_v2_clean_fp16.onnx` (sha256 c40c10f9…bc87a), `…_fp32.onnx`, `farz_beans_v2_clean_labels.json`.

## 6. CBD (464 real photos, India light box; never used for training or fitting)

| Grade | photos | good | defect | unsure | defect share of answered | argmax good (no abstention) |
|---|---|---|---|---|---|---|
| AAA | 50 | 20.9% | 32.5% | 46.6% | 60.8% | 47.6% |
| AA | 61 | 20.9% | 31.9% | 47.2% | 60.4% | 48.7% |
| A | 50 | 20.5% | 34.3% | 45.2% | 62.6% | 47.9% |
| AB | 50 | 15.0% | 40.8% | 44.2% | 73.1% | 40.6% |
| PB-I | 51 | 10.9% | 47.4% | 41.8% | 81.3% | 32.4% |
| PB-II | 50 | 7.3% | 55.1% | 37.6% | 88.3% | 28.0% |
| C | 51 | 8.2% | 54.2% | 37.5% | 86.9% | 29.1% |
| Bulk | 50 | 8.8% | 41.8% | 49.4% | 82.7% | 37.2% |
| Bits | 51 | 3.8% | 71.3% | 24.9% | 94.9% | 14.1% |

**Ranking.** Per-photo share of beans whose argmax is not "good":
- Spearman vs grade-folder order: **+0.680** (p = 3e-64); the all-sources v2 had +0.642.
- AUROC Bits > AAA **1.00**; C > AAA 0.951; PB-I > AAA 0.883; AA > AAA 0.439 (the top grades are not separated).
- Median AAA photo defect share: **53%** (all-sources v2: 36%). The absolute level is worse, so the ranking is usable, but not an absolute band.

**Photo outcomes.** The 23:05 app rules decide:
- gates ignored: 381 not sure (implausible), 83 not sure (too many unsure), **0 bands, 0/50 AAA "many"**;
- with the gate port: the same, except that 5 photos become blur retakes.

## 7. End-to-end deploy simulation (`clean_simulate.py` → `clean_deploy.json`)
How each photo is decided:
1. Per bean: touching → unsure; P(good) >= t_good → good; P(defect) >= t_defect → defect; otherwise unsure.
2. `src/beans/pipeline.decide`, the Python mirror of `app/src/lib/rules.ts decide()`, run unchanged.
3. Gates, reported two ways:
   - **"Gates ignored"** is the safety number (stricter than the app).
   - **"With gates"** uses a port of the app's `gate()` order on the Python finder's checks. The newer `touching.ts` rules are not ported.
4. Thresholds, also two ways:
   - **"held-out"**: T and thresholds fitted without the scored group;
   - **"deployed"**: 2.046, 0.50 / 0.91.

| Set (model) | n | bands, gates ignored | wrong | dangerous | with gates | notes |
|---|---|---|---|---|---|---|
| CBD 464 (shipped fp16) | 464 | **0** | 0 | 0 | 0 bands (5 blur) | AAA "many" 0/50 |
| 777 re-shoots, auditor's perturbations and crops (shipped) | 777 | **0** | 0 | 0 | 0 (2 blur) | 452 implausible, 325 too many unsure |
| 777 re-shoots, v2 report's perturbations (shipped) | 777 | **0** | 0 | 0 | 0 (9 blur) | |
| lojano 36 real photos (clean_loso_lojano) | 36 | 0 held-out / 0 deployed | – | – | 0 (8 too dark) | |
| loja_yolo 315 real photos (clean_loso_loja_yolo) | 315 | 0 held-out / 1 deployed | 0 | 0 | 0 (173 count, 124 not green, 18 blur) | the 1 band: IMG_4907, 7 beans, truth 3 of 7 defects (43%) → "many" is right |
| J4ckDev 11 real photos (clean LOPO) | 11 | **0** held-out / 0 deployed | 0 | 0 | 0 bands (6 implausible, 3 count, 2 not green; held-out thresholds) | |
| **Auditor's 1,200 J4ckDev trays** (clean_half, deployed thresholds) | 1,200 | **241** | **0** | **0** | n/a (crop-level) | J4ckDev-held-out thresholds 0.50 / 0.89: 381 bands, 0 wrong. v1 re-run: 793 / 57 / 10, the auditor's exact numbers, so the protocol is reproduced |
| Held-out licensed groups, 50-bean crop trays (held-out thresholds) | 7,200 | 1 | 0 | 0 | n/a | the 1 band is samruddh |
| same, deployed thresholds | 7,200 | 42 | **11** | 0 | n/a | all samruddh; the thresholds are in-sample for the pooled fit |
| Unlicensed sources, 50-bean crop trays (shipped) | 4,600 | 0 | 0 | 0 | n/a | |
| App's 7 demo photos (shipped; **in-sample**) | 7 | 5 | 0 | 0 | same | synthetic 00 / 03 / 12 / 30 / 40 → clean, clean, some, many, many; the 2 CBD photos → not sure |

- **Responsible AI:** 0 dangerous answers in every held-out set.
- **Wrong bands:** 11 non-dangerous wrong bands appear only with the pooled (in-sample) thresholds, all on samruddh.
- **Where the safety comes from.** As for v2, the safety comes from the app's 15% unsure and 60% implausible photo rules, not from per-bean accuracy.

## 8. Comparison with the all-sources v2 (AUDIT_V2-corrected numbers)

| | all-sources v2 (`farz_beans_v2_fp16.onnx`) | clean v2 (`farz_beans_v2_clean_fp16.onnx`) |
|---|---|---|
| Training sources | 12, incl. 5 with no licence and 2 NC | **7, all state a licence** (2 NC) |
| Outputs | probs | probs + **embed** |
| T, thresholds | 1.558, 0.66 / 0.89 | 2.046, 0.50 / 0.91 |
| LOSO balanced acc., sources with good beans | 0.657 (afiyah fixed, auditor) | 0.644 (afiyah held out together) |
| USK / daffa / Mindforge | 0.532 / 0.570 / 0.733 (each held out) | 0.539 / 0.541 / 0.625 (test only) |
| CBD bands, AAA "many" | 0/464, 0 | 0/464, 0 |
| CBD Spearman, AUROC Bits > AAA, median AAA defect share | +0.64, 1.00, 36% | +0.68, 1.00, 53% |
| 777 re-shoots | 0 bands | 0 bands (both perturbation sets) |
| 1,200 J4ckDev trays: bands / wrong / dangerous | 229 / 0 / 0 | 241 / 0 / 0 |
| fp16 bytes, agreement with fp32 | 3,084,405 B, 99.31% | 3,082,642 B, 99.64% |
| wasm fp16 max prob diff (300 crops) | 7.4e-4 | 4.9e-4 |

## 9. Decisions taken during the run (within the planner's rules)
1. **The shipped weights are the existing `licensed.pt`, not a new training run.**
   - It is the same recipe, seed and licensed pool, verified from its sampler log, and its logits re-scored bit-identical.
   - Retraining would have produced the same file, so the 6 minutes went to the fold models instead.
2. **The afiyah twins are one group everywhere:** in LOSO (held out together) and in the threshold weights (they count once).
3. **Embedding = the 1,024-d penultimate feature,** not the 576-d pooled backbone feature: the task asks for the penultimate vector. The normalisation was rewritten to be safe in fp16 (§5).
4. **Thresholds use `train_eval.pick_pair` unchanged** (answered accuracy >= 90% and stable, then largest coverage). Nothing was tuned on the unlicensed sources or on CBD.
5. **Safety numbers are reported with gates ignored.** The Python gate port lacks `touching.ts`, so the "with gates" columns are indicative only.
6. **The local-calibration study was pre-registered** at 02:14, before results, and its verdict follows the pre-registered rule, not my judgement.

## 10. For the app owner (nothing in `app/` was changed)
- **Load** `models_v2/farz_beans_v2_clean_fp16.onnx` and `models_v2/farz_beans_v2_clean_labels.json`. The input contract is unchanged; output `probs` has 6 classes; output `embed` is optional.
- **`rules.ts` changes:**
  - add `other_defect` as a defect;
  - `callBean` becomes the two-threshold rule (good if probs[0] >= 0.50, else defect if 1 - probs[0] >= 0.91, else unsure);
  - leave `decide()` unchanged.
- **Licence text** for README and model card: "non-commercial research prototype; weights derived from CC BY-NC-SA 4.0 data (J4ckDev) and CC BY-NC 4.0 data (lojano), and so shared under CC BY-NC-SA 4.0".
  - Attribution is required for loja_yolo (CC BY 4.0), lojano, J4ckDev and samruddh (MIT notice).
  - CC0 sets from anonymous uploaders (vicanadya16, afiyah) have unverified provenance (AUDIT_V2 (5)).
  - The 5 unlicensed sources were used only to test, and are not redistributed.
- **Still to test:** the iPhone, and the browser path with the 6-class `rules.ts`.

## 11. Limits
- **No Yemeni beans and no farmer phone photos.** Trays are built from crops, and nothing was re-photographed except the 777 software re-shoots.
- **One seed per fold.** No seed spread was measured.
- **The pooled thresholds are in-sample** for the threshold choice.
- **CBD has no per-bean labels,** and its grade order is the folder order.
- **The app demo is in-sample** (J4ckDev crops are in training).
- **Timings are from a laptop CPU.**

## 12. Reproduce (FARZ_SCRATCH = the scratchpad that holds `v2train/`; about 60 min of MPS with 3 workers)
`cd src/beans_v2`, then:
1. `python clean_folds.py worker`, run as 3 parallel workers.
2. `python clean_eval.py`
3. `python clean_export.py`
4. `python clean_simulate.py`
5. `python localcal.py`
6. `python localcal_export.py`
7. `python clean_report.py`
8. `node localcal_spec_check.mjs`

The shipped weights are `v2train/folds/licensed.pt`, from `python train_folds.py licensed`.

## Files
- **Models:** `models_v2/farz_beans_v2_clean_fp16.onnx` (ship), `models_v2/farz_beans_v2_clean_fp32.onnx`, `models_v2/farz_beans_v2_clean_labels.json`, `models_v2/localcal_demo/`.
- **Reports:** `reports_v2/clean_model.md` (this file), `clean_model.json`, `clean_eval.json`, `clean_export.json`, `clean_deploy.json`, `localcal.md`, `localcal.json`, `localcal_spec.md`, `localcal_sensitivity.txt`.
- **Code:** `src/beans_v2/clean_folds.py`, `clean_common.py`, `clean_eval.py`, `clean_export.py`, `clean_wasm_parity.mjs`, `clean_simulate.py`, `clean_report.py`, `localcal.py`, `localcal_export.py`, `localcal_spec_check.mjs`.
- **Fix:** `src/beans/train_beans.py` now creates `logs/` before opening its log.
