> **This report describes the all-sources research model, NOT the model shipped in this repository (see `reports_v2/clean_model.md`).** The model described here was trained on all 12 bean datasets, including 5 that state no licence; where it says "deployed" or "shipped", it means that model as it stood at 01:50 BST on 4 Oct. The shipped model (`app/public/models/farz_beans.onnx`) was trained only on sources that state a licence and was re-tested in `clean_model.md`; its sound threshold is now 0.55 (`threshold_sweep.md`). Several claims below were corrected by the independent audit (`AUDIT_V2.md`), which wins where the two disagree.

# Farz v2 bean classifier: retrained on 12 sources, tested honestly, and a deploy rule

STATUS: COMPLETE, 01:50 BST Sun 4 Oct 2026. Every number below was measured by a script in `src/beans_v2/train_*.py`, and each one names its protocol. Raw numbers are in `model_v2_eval.json`, `model_v2_export.json`, `model_v2_deploy.json` and `model_v2_cbd_rank.json`; `model_v2.json` collects the headline figures. No file in `app/`, `models/`, `reports/`, `src/beans/` or `docs/` was changed. `src/beans/beanfinder.py` is imported read-only.

## Bottom line

1. **On a source it has never seen, v2 is better than v1 but still not good enough to give a band.**
   - Leave-one-source-out (LOSO), 12 held-out sources, argmax with no abstention: mean good-vs-defect balanced accuracy is **0.78** (range 0.53 to 1.00).
   - On the same 10 J4ckDev photos as v1's leave-one-photo-out (LOPO) test, typed accuracy rises from **27.4% to 42.9%**.
2. **On CBD (India light box, never used for training or fitting) v2 ranks the grades the right way round. v1 ranked them backwards.**
   - Per-photo defect share vs grade folder: Spearman **+0.64** for v2, **−0.48** for v1.
   - AAA vs Bits photos: AUROC **1.00** for v2, **0.07** for v1.
   - With no abstention v2 calls **62.5%** of AAA beans good; v1 called **0.08%**.
   - The absolute level is still wrong: the median AAA photo is **36%** defects.
3. **The abstain threshold was chosen on held-out sources.** Rule: answered accuracy ≥ 90%.
   - **Single threshold: t = 0.89.** Pooled held-out coverage is 53.8%, at 90.5% answered accuracy.
   - **Two thresholds: good ≥ 0.66, defect ≥ 0.89.** Pooled coverage is 68.9%, at 90.1% answered accuracy.
   - The guarantee does **not** hold for every source. Scored with thresholds fitted on the other 11 sources, 5 of the 12 held-out sources (single threshold) or 6 (two thresholds) fall below 90% answered accuracy. USK is the worst at **36% / 49%**: its good beans are confidently called broken or insect.
4. **Size.**
   - **fp16 is kept:** 3,084,405 B against 6,108,756 B for fp32. On 23,447 held-out CBD crops it agrees with fp32 on **99.31%** of argmax calls and **99.20%** of good / defect / unsure calls.
   - **INT8 static failed.** Six attempts, all with HardSwish, HardSigmoid, Mul and SE convs excluded, reached only 2.0–55.0% agreement. No INT8 file was kept.
5. **Deploy rule P_pair** (v2 + two thresholds + the app's 23:05 rules) was simulated on 3,427 held-out photos and trays:
   - **0 wrong bands** and **0 of 50 CBD AAA photos called "many"**.
   - It gives a band on only **40 of 3,427 (1.2%)**: "not sure" 82.9%, retake 15.9%.
   - All 40 bands are correct. 39 come from afiyah_bijikopi trays and 1 from afiyah_deteksi.
   - It gives **no band on any of the 1,030 real held-out photos**, on the 60 J4ckDev-style held-out trays, or on the app's own 7 demo photos.
6. **Freeze recommendation: do not swap v1 for v2 in the app before the 08:30 freeze.** v2 makes the app safe only by making it almost silent, and it is not more accurate than v1 in the J4ckDev set-up:
   - **v2 with local calibration:** the 60 J4ckDev-style trays give 22 correct bands, 38 wrong bands and 2 dangerous ones.
   - **v1 under the 23:05 rules** (`robust_summary.md`): 30/30 trays correct, with 0/464 CBD photos given a band.
   - Use the v2 numbers as the evidence in the README and video. Switch to v2 only if the team accepts that the app will answer "not sure" to nearly every photo. The exact app changes are in §6.

## 1. What changed from v1

| | v1 | v2 (this report) |
|---|---|---|
| Training data | 582 crops, 1 source (J4ckDev, 11 photos, one photo per class) | **118,143 crops, 12 sources** (582 J4ckDev + 117,561 v2 crops, `keep == 1`) |
| Output classes | 5: good, dark, insect, broken, unhulled | **6**: + other_defect, which has 4,572 typed crops (floater, withered, fade, foreign matter, Mindforge fissure and over-dried). The 4,871 "defective, type not given" crops train only "is a defect", via a partial-label loss −log P(any defect class) |
| Sampling | class-balanced | label kind (good 0.40; dark, insect, broken, unhulled 0.11 each; other 0.06; untyped defect 0.10), **then source uniformly among the sources that have that kind**, then crop. No source can dominate an epoch: vicanadya16 has 86k crops but gets the same share as J4ckDev |
| Recipe | 40 × 512 samples | 3,000 steps × batch 128, AdamW (weight decay 1e-4), OneCycle with max lr 2e-3, label smoothing 0.05, seed 0. v1's augmentations, batched on the GPU, with a wider resolution-degrade range (24–96 px vs 36–80 px), a per-channel gain of 0.9–1.1 to simulate white-balance error, and p 0.15 blur. Fixed recipe: no early stopping, nothing tuned on test data |
| Held-out test | spatial halves inside each photo (an upper bound) | **LOSO** (12 folds), **J4ckDev LOPO** (11 folds; Normales can now be held out), J4ckDev spatial halves (2 folds) |
| Temperature and threshold | fitted on in-photo validation | **fitted only on held-out-source predictions.** For source s, T and t come from the other 11 LOSO folds. The deployed T and t come from all 12. Every source has weight 1, split equally between its good rows and its defect rows |
| Deployed file | `models/farz_beans_fp32.onnx`, 6.1 MB | `models_v2/farz_beans_v2_fp16.onnx`, 3.08 MB. Input `image` [N,3,128,128] RGB in [0,1]; output `probs` [N,6]. ImageNet normalisation, temperature (T = 1.558) and softmax are inside the graph |

### Data (counts from the training cache; licences as recorded in `docs/DATA_SOURCES_V2.md`)
| Source | Licence | good | dark | insect | broken | unhulled | other (typed) | defect, untyped | Total |
|---|---|---|---|---|---|---|---|---|---|
| j4ckdev | CC BY-NC-SA 4.0 | 195 | 148 | 55 | 85 | 99 | 0 | 0 | 582 |
| mfu17 | "Unknown" | 0 | 303 | 111 | 250 | 161 | 137 | 0 | 962 |
| loja_yolo | CC BY 4.0 (Kaggle page: Apache 2.0) | 146 | 472 | 135 | 367 | 191 | 0 | 0 | 1,311 |
| lojano | CC BY-NC 4.0 | 598 | 0 | 0 | 0 | 0 | 0 | 584 | 1,182 |
| usk_coffee | "Unknown"; authors state none | 5,970 | 0 | 0 | 0 | 0 | 0 | 1,999 | 7,969 |
| samruddh_grading | MIT | 467 | 0 | 0 | 0 | 0 | 0 | 870 | 1,337 |
| afiyah_deteksi | CC0 | 658 | 0 | 0 | 0 | 0 | 0 | 327 | 985 |
| afiyah_bijikopi | CC0 | 579 | 0 | 0 | 0 | 0 | 0 | 591 | 1,170 |
| daffa_defect | none stated | 500 | 0 | 0 | 0 | 0 | 0 | 500 | 1,000 |
| mindforge_doubleside | none stated | 1,447 | 1,026 | 1,039 | 2,357 | 0 | 1,503 | 0 | 7,372 |
| notplying_defects | none stated | 0 | 1,363 | 1,362 | 5,142 | 257 | 26 | 0 | 8,150 |
| vicanadya16 | CC0 | 0 | 22,316 | 16,138 | 24,749 | 20,014 | 2,906 | 0 | 86,123 |
| **Total** | | **10,560** | **25,628** | **18,840** | **32,950** | **20,722** | **4,572** | **4,871** | **118,143** |

**Dedupe.** The data hunter removed duplicates before this step:
- pixel MD5 of source images and of crops;
- near-duplicates at r ≥ 0.998 within each source.

It found 0 cross-dataset duplicates and 0 copies of any J4ckDev photo. Splits here are by source or by J4ckDev photo, so a within-source near-duplicate can never cross a train/test boundary. CBD is used only as a test set.

**Licence consequence.**
- The deployed weights were trained on 5 sources that state no licence (by default "all rights reserved") and on 2 non-commercial ones (J4ckDev, lojano).
- Treat the model as a non-commercial prototype, and say so in the README.
- A comparison with a model trained only on sources that state a licence is in §2.4.

## 2. Honest evaluation

### 2.1 Leave-one-source-out (12 folds)
Protocol for each held-out source:
- Train on the other 11 sources (3,000 steps, same seed) and score every crop of the held-out source.
- T_s and t_s are fitted on the other 11 sources' held-out predictions.
- "Good-vs-defect" means: truth = defect if the label is any defect (typed or untyped); prediction = defect if P(good) < 0.5.
- Macro accuracy = mean recall over the label kinds the source has. Untyped defects count as correct when called any defect class.
- ECE uses 10 bins on the good-vs-defect confidence.

| Held-out source | n | T_s | macro acc (typed) | good-vs-defect balanced acc | good recall | defect recall | ECE (T_s) | ECE (T = 1) |
|---|---|---|---|---|---|---|---|---|
| j4ckdev | 582 | 1.55 | 0.508 | 0.606 | 0.221 | 0.992 | 0.163 | 0.166 |
| mfu17 (no good beans) | 962 | 1.51 | 0.455 | 0.883* | – | 0.883 | 0.067 | 0.074 |
| loja_yolo | 1,311 | 1.58 | 0.574 | 0.710 | 0.425 | 0.996 | 0.032 | 0.022 |
| lojano | 1,182 | 1.59 | 0.830 | 0.753 | 0.545 | 0.961 | 0.074 | 0.084 |
| usk_coffee | 7,969 | 1.50 | 0.586 | **0.532** | 0.249 | 0.816 | **0.419** | 0.418 |
| samruddh_grading | 1,337 | 1.57 | 0.786 | 0.719 | 0.505 | 0.932 | 0.067 | 0.105 |
| afiyah_deteksi | 985 | 1.61 | 0.919 | 0.876 | 0.752 | 1.000 | 0.098 | 0.039 |
| afiyah_bijikopi | 1,170 | 1.63 | 0.997 | 0.997 | 0.995 | 0.998 | 0.125 | 0.029 |
| daffa_defect | 1,000 | 1.54 | 0.610 | 0.570 | 0.324 | 0.816 | 0.185 | 0.208 |
| mindforge_doubleside | 7,372 | 1.55 | 0.462 | 0.733 | 0.572 | 0.894 | 0.018 | 0.078 |
| notplying_defects (no good beans) | 8,150 | 1.57 | 0.273 | 0.996* | – | 0.996 | 0.096 | 0.063 |
| vicanadya16 (no good beans) | 86,123 | 1.50 | 0.287 | 1.000* | – | 1.000 | 0.030 | 0.011 |
| **Mean over the 12 sources** | | | **0.607** | **0.781** | | | | |

\* For a source with no good beans this is defect recall only. Calling everything "defect" would score 1.0, so these rows do not show that the model can tell good from defective.

**Per-kind recall** (argmax, no abstention; `model_v2_coverage.png`, right):

| Held-out source | good | dark | insect | broken | unhulled | other_defect | untyped defect (called any defect) |
|---|---|---|---|---|---|---|---|
| j4ckdev | 0.456 | 0.818 | 0.364 | 0.671 | 0.232 | – | – |
| mfu17 | – | 0.558 | 0.613 | 0.628 | 0.478 | 0.000 | – |
| loja_yolo | 0.664 | 0.392 | 0.200 | 0.665 | 0.948 | – | – |
| lojano | 0.747 | – | – | – | – | – | 0.913 |
| usk_coffee | 0.434 | – | – | – | – | – | 0.739 |
| samruddh_grading | 0.692 | – | – | – | – | – | 0.880 |
| afiyah_deteksi | 0.839 | – | – | – | – | – | 1.000 |
| afiyah_bijikopi | 1.000 | – | – | – | – | – | 0.995 |
| daffa_defect | 0.588 | – | – | – | – | – | 0.632 |
| mindforge_doubleside | 0.733 | 0.490 | 0.451 | 0.622 | – | 0.012 | – |
| notplying_defects | – | 0.282 | 0.257 | 0.593 | 0.004 | 0.231 | – |
| vicanadya16 | – | 0.248 | 0.054 | 0.907 | 0.220 | 0.005 | – |

Full 7 × 6 confusion matrices (rows = label kind, columns = predicted class) for every source are in `model_v2_eval.json → per_source.<source>.no_abstain.confusion_rows_kinds_cols_classes`.

What these tables show:
- **The source-style confound is real but not total.** loja_yolo and Mindforge are the only sources with good beans and typed defects in the same photo style. Held out, their good-vs-defect balanced accuracy is 0.71 and 0.73. The model learned some real good-vs-defect signal, but far from 90%.
- **Defect types barely transfer.** For example, vicanadya16 insect recall is 0.05 and notplying unhulled recall is 0.004. Do not show defect types as facts.
- **"good" is the weak class on a new source.** Good recall (P(good) ≥ 0.5) is 0.22–0.57 on 7 of 9 sources. The exceptions are afiyah_deteksi (0.75) and afiyah_bijikopi (0.995).
- **USK good beans are confidently called defects.** Argmax calls them broken 32% and insect 21% of the time. This holds even for USK "premium" beans: 27% have P(defect) ≥ 0.91 (`train_eval` spot check in this session).

### 2.2 Abstain threshold, chosen on held-out-source data
**Single-threshold rule (A).**
- Confidence = max(P(good), 1 − P(good)).
- t = the lowest value in 0.50–0.99 whose source- and class-balanced answered good-vs-defect accuracy is ≥ 90% and stays ≥ 90% for every higher t.
- On all 12 LOSO sets pooled: **t = 0.89, coverage 53.8%, answered accuracy 90.5%.**

**Two-threshold rule (P).** Needed because P(good) has a ceiling:
- Label smoothing over 6 classes and T = 1.56 > 1 push P(good) down.
- P(defect) sums 5 classes, so it does not have that ceiling.
- Under rule A, between 0.0% and 0.6% of held-out good beans are ever called good.
- Even in-sample, the final model's median P(good) on J4ckDev good crops is 0.747.

Rule P:
- Good if P(good) ≥ t_good; defect if 1 − P(good) ≥ t_defect; otherwise unsure.
- For each t_good, take the lowest stable t_defect that keeps answered accuracy ≥ 90%. Keep the pair with the largest coverage.
- On all 12 LOSO sets pooled: **t_good = 0.66, t_defect = 0.89, coverage 68.9%, answered accuracy 90.1%.**

Each row below uses thresholds fitted on the other 11 sources only (curves in `model_v2_coverage.png`, left):

| Held-out source | rule A t | A coverage | A answered acc | rule P (t_good / t_defect) | P coverage | P answered acc | good beans: called good / defect / unsure | defect beans: called defect / good / unsure |
|---|---|---|---|---|---|---|---|---|
| j4ckdev | 0.87 | 72.3% | 86.2% | 0.66 / 0.87 | 75.3% | 86.5% | 8.2% / 29.7% / 62.1% | 93.8% / 0.3% / 5.9% |
| mfu17 | 0.91 | 68.4% | 100.0% | 0.52 / 0.91 | 79.2% | 86.4% | – | 68.4% / 10.8% / 20.8% |
| loja_yolo | 0.88 | 85.9% | 98.0% | 0.68 / 0.88 | 88.3% | 97.8% | 19.9% / 15.1% / 65.1% | 94.8% / 0.3% / 5.0% |
| lojano | 0.88 | 42.3% | 88.8% | 0.66 / 0.89 | 60.3% | 91.6% | 36.5% / 8.7% / 54.8% | 74.5% / 1.4% / 24.1% |
| usk_coffee | 0.85 | 47.4% | **35.7%** | 0.56 / 0.87 | 63.5% | **49.4%** | 20.6% / 37.5% / 41.9% | 63.6% / 16.1% / 20.3% |
| samruddh_grading | 0.88 | 55.0% | 89.4% | 0.64 / 0.88 | 68.7% | 88.2% | 33.2% / 16.7% / 50.1% | 75.3% / 3.4% / 21.3% |
| afiyah_deteksi | 0.89 | 35.7% | 92.0% | 0.70 / 0.90 | 71.2% | 96.3% | 53.3% / 4.0% / 42.7% | 99.1% / 0.0% / 0.9% |
| afiyah_bijikopi | 0.90 | 48.5% | 100.0% | 0.76 / 0.90 | 88.4% | 100.0% | 80.7% / 0.0% / 19.3% | 95.9% / 0.0% / 4.1% |
| daffa_defect | 0.88 | 27.9% | **74.9%** | 0.60 / 0.88 | 42.4% | **69.6%** | 17.2% / 14.0% / 68.8% | 41.8% / 11.8% / 46.4% |
| mindforge_doubleside | 0.88 | 54.2% | 95.9% | 0.62 / 0.89 | 66.0% | 89.0% | 43.0% / 10.2% / 46.9% | 62.6% / 6.5% / 30.8% |
| notplying_defects | 0.90 | 62.3% | 100.0% | 0.64 / 0.91 | 55.3% | 99.8% | – | 55.2% / 0.1% / 44.7% |
| vicanadya16 | 0.91 | 98.6% | 100.0% | 0.64 / 0.92 | 97.9% | 100.0% | – | 97.9% / 0.0% / 2.1% |

- **Calibration.** The deployed T = 1.558 lowers pooled weighted good-vs-defect ECE from 0.103 (T = 1) to **0.069**.
- **Calibration does not transfer evenly.** ECE on a single held-out source ranges from 0.018 (Mindforge) to 0.419 (USK).
- **Below 90% answered accuracy on its own held-out threshold:** rule A fails on 5 of 12 sources (j4ckdev, lojano, usk, samruddh, daffa); rule P fails on 6 (j4ckdev, mfu17, usk, samruddh, daffa, mindforge).
- **The pooled 90% is not a promise for a new farm.** The photo-level rules (§5) are the real safety net.

### 2.3 J4ckDev leave-one-photo-out (11 folds)
Protocol for each held-out photo:
- Train on all 11 other sources plus the other 10 J4ckDev photos, and score every bean of the held-out photo.
- T and thresholds are the J4ckDev LOSO values (other sources only).
- The v1 column comes from `reports/beans_lopo.json` (argmax, same protocol).

| Photo | class | n | v1 recall | **v2 recall (typed)** | v2 good-vs-defect correct | coverage at t = 0.87 | v2 argmax shares |
|---|---|---|---|---|---|---|---|
| BrocadoLeve | insect | 30 | 0.267 | 0.467 | 0.833 | 66.7% | good 0.20, dark 0.17, insect 0.47, broken 0.17 |
| BrocadoSevero | insect | 25 | 0.400 | 0.280 | 1.000 | 100% | dark 0.56, insect 0.28, broken 0.16 |
| CerezaSeca | unhulled | 9 | 0.000 | 0.111 | 1.000 | 100% | dark 0.89, unhulled 0.11 |
| Concha | broken | 33 | 0.758 | 0.727 | 1.000 | 81.8% | broken 0.73, insect 0.12, dark 0.09, good 0.06 |
| DXHongo | dark | 12 | 0.000 | 0.500 | 1.000 | 91.7% | dark 0.50, broken 0.42, insect 0.08 |
| Inmaduro | broken | 16 | 0.250 | 0.125 | 1.000 | 87.5% | dark 0.62, insect 0.25, broken 0.12 |
| MarronAVinagre | dark | 36 | 0.333 | 0.500 | 1.000 | 94.4% | dark 0.50, insect 0.17, broken 0.14, unhulled 0.14, good 0.06 |
| Negros | dark | 100 | 0.190 | 0.500 | 1.000 | 100% | dark 0.50, insect 0.36, unhulled 0.14 |
| Normales | good | 195 | n/a (v1 could not hold it out) | **0.041** | **0.000** | 84.1% (all wrong) | insect 0.74, dark 0.18 |
| PMordidoCortado | broken | 36 | 0.778 | 0.861 | 1.000 | 100% | broken 0.86, insect 0.08 |
| Pergamino | unhulled | 90 | 0.000 | 0.144 | 1.000 | 83.3% | dark 0.56, insect 0.17, unhulled 0.14 |
| **Same 10 photos as v1** | | 387 | **0.274** (bean-weighted) | **0.429** (bean-weighted), 0.422 photo-mean | | | |

**Normales is a confident wrong answer.** The fold that holds it out still trains on the other 10 J4ckDev photos, and all 10 are defect photos. So the model learns "J4ckDev camera = defect", which is v1's one-photo-per-class confound again at photo level. Of 195 good beans, it answers 84% at t = 0.87, and every one of those answers is "defect".

The safety net catches it in the full pipeline: the real Normales photo goes to retake (count), and the defect photos go to "not sure" (implausible). See §5.

### 2.4 Comparison with a model trained only on sources that state a licence
Protocol:
- Train only on the 7 sources that state a licence (j4ckdev, vicanadya16, afiyah ×2, loja_yolo, lojano, samruddh).
- Test on every crop of the 5 that state none.
- T and t are fitted on the LOSO predictions of the 7 licensed sources.

| Unseen source | licensed-only model: good-vs-defect balanced acc | all-sources model, same source held out (§2.1) |
|---|---|---|
| daffa_defect | 0.590 | 0.570 |
| mfu17 | 0.936 | 0.883 |
| mindforge_doubleside | 0.693 | 0.733 |
| notplying_defects | 0.998 | 0.996 |
| usk_coffee | 0.568 | 0.532 |

The licensed-only model is about as good on these 5 unseen sources. That is evidence (not proof: its own LOSO was not run) that the unlicensed sources are not needed for the result. Shipping a model trained only on sources that state a licence would need the full evaluation re-run on it, which takes about 80 minutes. It would still be non-commercial because of J4ckDev and lojano.

## 3. CBD (464 real photos, India light box; never used for training, temperature or thresholds)
Protocol:
- The frozen bean finder cuts 23,447 crops (the robust study's cache; the same count as v1).
- Score them with the deployed fp16 model, using the rule P thresholds.
- The v1 columns come from `reports/beans_ood_cbd.json`.
- The CBD page does not define its grades, so grade names are folder labels only.

| Grade | photos | beans | v2 good | v2 defect | v2 unsure | v2 defect share of answered | v2 argmax good (no abstention) | v1 good (with abstention) | v1 broken (with abstention) |
|---|---|---|---|---|---|---|---|---|---|
| AAA | 50 | 2,498 | 14.7% | 23.2% | 62.1% | 61.2% | **62.5%** | **0.0%** | 61.3% |
| AA | 61 | 3,059 | 16.1% | 24.2% | 59.8% | 60.0% | 61.4% | 1.0% | 61.8% |
| A | 50 | 2,572 | 18.6% | 24.8% | 56.6% | 57.2% | 62.9% | 0.1% | 63.6% |
| AB | 50 | 2,510 | 14.7% | 26.9% | 58.4% | 64.8% | 59.1% | 0.1% | 60.8% |
| PB-I | 51 | 2,546 | 7.3% | 47.1% | 45.6% | 86.6% | 37.1% | 1.2% | 38.6% |
| PB-II | 50 | 2,536 | 6.9% | 44.6% | 48.5% | 86.7% | 40.0% | 0.9% | 47.8% |
| C | 51 | 2,594 | 6.6% | 46.9% | 46.5% | 87.7% | 39.0% | 2.4% | 56.2% |
| Bulk | 50 | 2,525 | 8.4% | 30.3% | 61.3% | 78.2% | 54.4% | 0.3% | 56.0% |
| Bits | 51 | 2,607 | 2.9% | 63.1% | 34.0% | 95.6% | 22.6% | 6.1% | 41.4% |

**Ranking** (`train_cbd_rank.py` → `model_v2_cbd_rank.json`). Per photo, take the share of beans whose argmax is not "good":

| | v2 | v1 |
|---|---|---|
| Spearman vs folder order (AAA … Bits), 464 photos | **+0.64** (p = 3e-55) | **−0.48** (p = 5e-28) |
| AUROC, Bits above AAA | **1.00** | 0.07 |
| AUROC, C above AAA | **0.98** | 0.19 |
| AUROC, PB-I above AAA | **0.99** | 0.22 |
| Median photo defect share, AAA / C / Bits | 36% / 60% / 78% | 100% / 98% / 96% |

v2 puts CBD photos in the right order, which v1 did not. But it calls about a third of AAA beans defective.
- **Possible future use:** comparing lots photographed in one set-up (relative ranking within one cooperative).
- **Not yet possible:** an absolute band in a new set-up.

## 4. Size: fp16 kept, INT8 static rejected
Agreement set: all 23,447 CBD crops. They were never used for training, temperature, thresholds or INT8 calibration. INT8 calibration used 1,200 training crops (100 per source, seed 0, MinMax unless stated). Latency was measured with batch 50, 1 thread, on the MacBook CPU with onnxruntime 1.30, **not a phone**.

| File | Bytes | argmax agreement with fp32 | good / defect / unsure agreement (rule P) | Kept (needs ≥ 99%) | CPU ms per 50 beans |
|---|---|---|---|---|---|
| fp32 (max diff vs PyTorch 3.7e-6) | 6,108,756 | – | – | reference | 89.7 |
| **fp16** (`convert_float_to_float16`, keep_io_types=True) | **3,084,405** | **99.31%** | **99.20%** | **yes** | 108.7 |
| INT8 QDQ, HardSwish + HardSigmoid + Mul + 18 SE convs excluded | 3,171,531 | 2.0% | 47.1% | no | 41.9 |
| same + stem conv and classifier kept fp32 | 4,941,392 | 48.2% | 32.3% | no | 41.9 |
| same + first 4 convs and classifier kept fp32 | 4,940,308 | 55.0% | 59.0% | no | 44.7 |
| same exclusions, Percentile calibration | 3,171,547 | 21.1% | 46.8% | no | – |
| same exclusions, Entropy calibration | 3,171,531 | 2.0% | 47.1% | no | – |
| only the 22 pointwise (1×1) convs quantised | 5,038,639 | 54.0% | 56.2% | no | – |

The rejected INT8 files were deleted (`train_export.py`, `train_int8_extra.py`). **Deployed file: `models_v2/farz_beans_v2_fp16.onnx`.** Its labels file is `models_v2/farz_beans_v2_labels.json`.

## 5. Deploy rule: what was combined and how it was simulated
Every candidate shares the same pipeline:
1. The app's retake gates (23:05): too dark, blurred, colour "not green", 20–160 beans, too many touching.
2. Optional photo-level OOD gate.
3. Per-bean call: a touching bean is unsure; otherwise apply rule A or rule P.
4. Optional colour override.
5. The app's 23:05 photo rules: nothing answered, or more than 60% of answered beans are defects ("implausible"), or more than 15% unsure → **not sure**. Otherwise the band of the point estimate: clean < 5% ≤ some ≤ 20% < many.

The Python port of the rules is `robust_common.decide`, unchanged, and `train_simulate.py` mirrors it.

Rule variants simulated:
- **A_sym_t:** v2 + one threshold t = 0.89.
- **P_pair:** v2 + t_good 0.66 / t_defect 0.89.
- **P_pair+ood_gate:** adds the hand-Mahalanobis photo gate from `robust_gate_params.json` (unchanged, threshold 4.9106).
- **P_pair+dark_override:** adds the fixed colour rule R0 (bean mean-colour HSV value < 0.30 → defect).
- **P_pair+ood+dark:** both.

Every test bean is scored by a model that never saw its source or photo, with thresholds fitted without it:

| Test set (model) | n | **P_pair: given a band** | P_pair: not sure | P_pair: retake | Wrong bands | Notes |
|---|---|---|---|---|---|---|
| CBD 464 real photos (final model; CBD never used) | 464 | **0** | 459 | 5 | 0 | **AAA "many" = 0/50** in every variant |
| CBD stress: 111 AAA+AA photos × 7 software re-shoots (exposure, cast, ⅓ resolution, blur, JPEG; `robust_ood.PERTURB`) | 777 | **0** | 768 | 9 | 0 | v1 under the 23:05 rules let 1/777 through (`robust_summary.md`) |
| J4ckDev 11 real photos (LOPO model) | 11 | 0 | 6 (implausible) | 5 | 0 | Normales: retake (count); the 10 defect photos: implausible or retake |
| lojano 36 real tray photos (LOSO model; 18 good, 18 defective) | 36 | 0 | 28 | 8 (brown board too dark) | 0 | good photos: "too many unsure"; defect photos: implausible |
| notplying 204 real photos (LOSO) | 204 | 0 | 0 | 204 (colour "not green": dense piles) | 0 | |
| loja_yolo 315 real photos (LOSO) | 315 | 0 | 0 | 315 (count 144, not green 124, blur 47) | 0 | |
| J4ckDev-style trays, unseen beans: 100 beans pasted with the demo recipe, then the full pipeline (spatial-half models) | 60 | **0** | 60 | 0 | 0 | 20 clean, 20 some, 20 many by construction |
| 50-bean trays from each held-out source's own crops (LOSO logits; 0 / 10 / 40 / 100% defects; no re-photographing) | 1,560 | **40** | 1,520 | 0 | **0** | all correct: 39 afiyah_bijikopi (5 clean, 11 some, 23 many) + 1 afiyah_deteksi (many) |
| **All held-out photos and trays** | **3,427** | **40 (1.2%)** | **2,841 (82.9%)** | **546 (15.9%)** | **0** | |
| App demo photos (`app/public/demo`, read-only; final model, **in-sample**: built from J4ckDev training crops) | 7 | 0 | 7 | 0 | – | the demo trays would all show "not sure" |

Effect of each add-on, measured on the same photos and trays as P_pair:
- **OOD gate (hand-Mahalanobis).**
  - It changes no wrong band into a refusal, because there were none.
  - It takes away 39 of the 40 correct tray bands, all of them afiyah_bijikopi (afiyah beans do not look like J4ckDev beans).
  - On CBD it only relabels refusals: 351 photos become "ood_gate" instead of implausible or unsure.
  - **Not added.**
- **Colour override R0.**
  - On held-out sources it catches **1 of 927** defects that v2 called good (at t_good), and flips 0 good beans.
  - It changes no verdict in any set.
  - **Not added.**
- **Colour rule for the dark type, on held-out sources.** Balanced accuracy for dark vs not:

  | Held-out source | v2 argmax | R1: median luminance < 97.152 (fitted on J4ckDev) | R0 |
  |---|---|---|---|
  | mfu17 | **0.704** | 0.490 | 0.495 |
  | loja_yolo | **0.642** | 0.531 | 0.499 |
  | Mindforge | **0.691** | 0.687 | 0.583 |
  | notplying | **0.608** | 0.511 | 0.500 |
  | vicanadya16 | 0.590 | **0.768** | 0.525 |

  The colour rule beats the model on 1 of 5 fair sources. J4ckDev is excluded because R1 was fitted on it. Types are not used for the band anyway.
- **Unusual-bean detector.** Not used for the verdict. `robust_summary.md` measured that it cannot count or band (ρ = −0.12 vs grade; blind to a uniform tray). At most it is an optional "look closer" highlight.

**Sensitivity only** (rule A, one threshold swept; nothing was chosen from this; `model_v2_deploy.png`, right):

| t | Held-out trays: correct / wrong / dangerous (of 1,560) | CBD photos given a band (AAA "many") | J4ckDev-style trays given a band (of 60) |
|---|---|---|---|
| 0.60 | 164 / 122 / **56** | 3 (0) | 0 |
| 0.70 | 127 / 0 / 0 | 0 (0) | 0 |
| ≥ 0.80 | 0 / 0 / 0 | 0 (0) | 0 |

Lowering the threshold enough to give bands brings dangerous errors (a clean tray called "many", or the reverse) before it gives useful coverage.

**Local calibration, sensitivity only.** Could a cooperative fix this by calibrating v2 on its own set-up? Protocol:
- Re-fit T and the two thresholds on the J4ckDev beans of the *other* spatial half, scored by that half's model (cross-fitted, so no tray bean is used).
- Fitted values: T = 0.85 / 0.83, thresholds 0.50 / 0.50–0.52.
- Then score the same 60 J4ckDev-style trays.

Results:
- Clean trays: 0/20 correct, 20 wrong, of which **2 called "many"**.
- Some trays: 2/20 correct.
- Many trays: 20/20 correct.

v2 counts 11–24% of the good beans on a clean J4ckDev tray as defects (estimated defect share, min to max). On comparable trays, v1 under the 23:05 rules got 30/30 correct (`robust_summary.md`). **Adding 11 other sources made v2 worse than v1 in J4ckDev's own set-up.**

## 6. Recommended deploy rule (if v2 ships), and the freeze decision
**Rule P_pair, in order:**
1. Retake gates, unchanged (dark, blur, colour not green, 20–160 beans, touching).
2. Per bean, with `models_v2/farz_beans_v2_fp16.onnx`:
   - touching → unsure;
   - otherwise pGood = probs[0];
   - **pGood ≥ 0.66 → good**;
   - **else 1 − pGood ≥ 0.89 → defect**, with type = argmax of probs[1..5], shown only as "possible";
   - else **unsure**.
3. The 23:05 photo rules, unchanged: implausible > 60% → not sure; unsure > 15% → not sure; otherwise band of the point estimate with "about".
4. No OOD gate, no colour override, no unusual-bean detector in the verdict. Each was measured above to add nothing.

**What it does (measured):**
- **Responsible AI:** on every held-out set, 0 wrong bands and 0/50 CBD AAA photos called "many".
- **Usefulness:** a band on 40/3,427 held-out photos and trays (1.2%), and on 0 of 1,030 real held-out photos.
- **Demo:** the app's own demo trays would all say "not sure".
- Describe it honestly: "Farz v2 answers only when the data supports it; with no labelled photos from your own set-up, it almost always sends the sample to the cooperative."

**Freeze recommendation: keep v1 + 23:05 rules in the app** (no change needed).

| Option | CBD photos with a band (AAA "many") | Stress set with a band | J4ckDev-style held-out trays | Other-source held-out trays | App demo trays |
|---|---|---|---|---|---|
| v1 + 23:05 rules (`robust_summary.md`) | 0/464 (0) | 1/777 | **30/30 correct**, 0 wrong | not measured | bands (in-sample) |
| v2 + P_pair (this report) | 0/464 (0) | **0/777** | 0/60 (all "not sure") | 40/1,560, 0 wrong | all "not sure" |

Reasons:
- v2's only safety gain over v1 under the 23:05 rules is the one stress photo (1 → 0 of 777).
- v2 costs every band in the J4ckDev set-up.
- Even with local calibration, v2 is less accurate than v1 there.

Use the v2 results as the **evidence** in the README, V3 and V4:
- the honest cross-source test;
- the threshold picked on held-out sources;
- the CBD ranking (+0.64 vs −0.48);
- USK as the example of a confident wrong answer on a new farm;
- the conclusion that local labelled photos are what a pilot needs.

**If the team prefers v2 anyway,** these are the app changes. Nothing has been copied into `app/`:
- **Load** `models_v2/farz_beans_v2_fp16.onnx` (3.08 MB) and `models_v2/farz_beans_v2_labels.json`.
- **`rules.ts`:**
  - `CLASSES` gains `"other_defect"` (6 outputs), and `DEFECTS` gains `"other_defect"`, with no type clip for it.
  - `callBean` becomes the two-threshold rule above. The current version takes argmax with one `abstainBelow`; applied to v2 it would never call a bean good above 0.89.
- **`classify.ts`:** unchanged (same input contract).
- **Re-test** the browser path: wasm parity of fp16, and the iPhone.

## 7. Limits (read before quoting any number)
- **No Yemeni beans, no phone photos of a farmer's sheet.** Every "new farm" number is a proxy: another dataset's camera and beans.
- **Weak "good" evidence in some folds.** Three sources have no good beans (mfu17, notplying, vicanadya16), so their good-vs-defect numbers are defect recall only.
- **Partly synthetic tests.** The held-out source trays are built from crops (no re-photographing). The J4ckDev-style trays are pastes of real crops.
- **CBD has no per-bean labels.** "AAA many is wrong" is presumed from the grade name. The grade order is the folder order, not an official scale.
- **The final model has no held-out data of its own.** Its T and thresholds are the pooled LOSO values, the same approach as v1.
- **One seed per fold.** No seed spread was measured for v2 (v1's within-photo seed spread is in `reports/beans_model.json`).
- **Licences.** The deployed weights include 5 unlicensed and 2 non-commercial sources.
- **Laptop timings.** Speeds are on a MacBook CPU, not a phone.

## 8. Reproduce (about 80 min of MPS training with two jobs in parallel, then about 10 min)
Set `FARZ_SCRATCH=<dir>` (cache and fold weights live there, outside the repo). Then, from `src/beans_v2/`:
1. `python -c "import train_common as tc; tc.build_cache()"`
2. `python train_hand.py`
3. `python train_folds.py loso`
4. `python train_folds.py lopo`
5. `python train_folds.py halves`
6. `python train_folds.py final`
7. `python train_folds.py licensed`
8. `python train_eval.py`
9. `python train_export.py`
10. `python train_int8_extra.py`
11. `python train_simulate.py precompute`
12. `python train_simulate.py`
13. `python train_cbd_rank.py`
14. `python train_plot.py`
15. `python train_report.py`

`train_simulate.py` also needs the robust study's cache in the same `FARZ_SCRATCH` (`robust_cache.py`).

## Files
- **Models:**
  - `models_v2/farz_beans_v2_fp16.onnx` (deployed candidate)
  - `models_v2/farz_beans_v2_fp32.onnx`
  - `models_v2/farz_beans_v2_labels.json`
- **Reports:**
  - `reports_v2/model_v2.md` (this file)
  - `model_v2.json` (headline numbers)
  - `model_v2_eval.json` (LOSO, LOPO, calibration, thresholds, confusions, curves, licensed-only)
  - `model_v2_export.json` (sizes, agreement)
  - `model_v2_deploy.json` (every simulated photo and tray, all variants, sweeps)
  - `model_v2_cbd_rank.json`
  - `model_v2_coverage.png` (coverage vs accuracy per held-out source; recall heatmap)
  - `model_v2_deploy.png` (rule P_pair outcomes; threshold sweep)
- **Code:** `src/beans_v2/train_common.py`, `train_hand.py`, `train_folds.py`, `train_eval.py`, `train_export.py`, `train_int8_extra.py`, `train_simulate.py`, `train_cbd_rank.py`, `train_plot.py`, `train_report.py`
