# Farz v2 local-calibration study (licence-stated base model)

STATUS: PRE-REGISTRATION ONLY. Written 02:14 BST Sun 4 Oct 2026, **before any local-calibration number was computed**. At this moment the base fold models are still training (`src/beans_v2/clean_folds.py`, started 02:11) and no embedding of any test bean exists. The results section will be appended below this block by `src/beans_v2/localcal.py`; nothing in this block may be edited after results exist (the file's pre-registration block is also hashed into `localcal.json`).

## Pre-registration (fixed before any result)

**Question.** Can a cooperative make the licence-stated v2 model useful in its own set-up with **K <= 100 beans labelled by a grader** (good / defect), taken from a few calibration photos, without making it unsafe?

**Base models (frozen; never see the evaluated source).**
- loja_yolo (primary): `clean_loso_loja_yolo` = the clean recipe trained on the 6 other licensed sources (afiyah x2, j4ckdev, lojano, samruddh_grading, vicanadya16); loja_yolo never in training.
- lojano (secondary): `clean_loso_lojano`.
- mindforge_doubleside, usk_coffee (secondary, test-only, no licence): the shipped clean model (`licensed.pt`), which never saw them.
- CBD safety check: the same base as the loja study (`clean_loso_loja_yolo`) with the head calibrated on loja_yolo.

**Embedding.** The penultimate feature of MobileNetV3-Small: the 1,024-d Hardswish output that feeds the last Linear layer (eval mode, dropout off), **L2-normalised**. This is exactly the ONNX `embed` output of the shipped file.
**Base probabilities.** softmax(logits / T), T = the clean pooled LOSO temperature. **Base binary call:** good if P(good) >= 0.5, else defect.

**Calibration draw** (per source, K in {20, 50, 100}, 5 draws d = 0..4, numpy seed = 10,000·source_index + 100·K + d).
- Only non-touching crops are used (touching blobs are always "unsure" in the app).
- Photos (the `group` column: tray photo for loja_yolo / lojano, bean for Mindforge, file for USK) are shuffled.
- Phase 1: add whole photos that contain >= 1 good bean, in shuffled order, until the pool holds >= K/2 good beans.
- Phase 2: if the pool holds < K/2 defect beans, add further photos (any remaining, shuffled order) until it holds >= K/2.
- Then K/2 good and K/2 defect beans are drawn uniformly from the pool. **Every bean of every calibration photo is removed from the test set**, used or not. The number of calibration photos is reported.
- A source/K pair that cannot supply K/2 of each class is skipped and reported as such.

**Local head: PROTOTYPES (chosen in advance).**
- mu_good = L2-normalise(mean embedding of the K/2 good calibration beans); mu_defect likewise.
- Score s(e) = cos(e, mu_defect) - cos(e, mu_good). Local call = defect if s >= 0, else good.
- Leave-one-out on the calibration beans: s_i^LOO with both prototypes recomputed without bean i.
- **m_hi = max( largest |s_i^LOO| among LOO-misclassified calibration beans (0 if none), median of |s_i^LOO| over all calibration beans ).**
- **Per-bean answer:** touching -> unsure; else if local call == base binary call -> answer the local call ("agree"); else if |s| >= m_hi -> answer the local call ("high margin"); else unsure.

**Reported alternative (not the primary): logistic regression** on the same L2-normalised embeddings (scikit-learn `LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)`), local call = defect if P(defect) >= 0.5, margin = |P(defect) - 0.5|, m_hi by the same LOO formula (K refits), same answer rule. **Baseline:** base model alone with the clean deployed thresholds (good if P(good) >= t_good, defect if P(defect) >= t_defect).

**Bean metrics (test beans = all non-touching beans of the non-calibration photos).**
- coverage = share of test beans answered, **class-balanced** (good and defect test beans weighted 50/50);
- answered accuracy = good-vs-defect accuracy of answered beans, **class-balanced** (each true class carries half the weight). Plain (unweighted) values are reported too.

**Trays.** 500 per (source, K, draw): 50 beans each, drawn without replacement from that draw's test beans; 167 trays at 0% defects (truth "clean"), 167 at 12% (6 of 50, truth "some"; the 12.5% mix rounded to whole beans), 166 at 30% (15 of 50, truth "many"); numpy seed fixed per draw. The band comes from `src/beans/pipeline.decide` (the Python mirror of `app/src/lib/rules.ts decide()`: nothing answered / > 60% of answered defective / > 15% unsure -> "not sure"; else band of the point estimate, clean < 5% <= some <= 20% < many). **Dangerous = a 0% tray called "many"** (a 30% tray called "clean" is counted and reported too).

**CBD check.** All 464 CBD photos (bean-finder crops and touching flags from the robust cache), base `clean_loso_loja_yolo` + head calibrated on loja_yolo, every K and draw. "Confident many" = any "many" band (with or without "about"), counted on the 50 AAA photos, **with every retake gate ignored** (stricter than the app).

**GO / NO-GO (primary: prototypes, loja_yolo).** GO if there is a K in {20, 50, 100} at which ALL hold:
1. mean over the 5 draws of class-balanced answered accuracy >= 90% AND mean class-balanced coverage >= 50% (the worst draw is reported next to it);
2. 0 dangerous trays in the 5 x 500 trays at that K;
3. 0 CBD AAA photos called "many" in every one of the 5 draws at that K.
The spec uses the smallest such K. Otherwise **NO-GO**.
**Pre-registered exception for logistic regression** (it is a second look, so a stricter bar): if prototypes fail and logistic meets 2 and 3 with mean answered accuracy >= 92% and mean coverage >= 55% at the same K, the verdict is "GO (logistic)". lojano, Mindforge and USK are secondary evidence and do not change the verdict.


## Results (appended 02:39 BST; the pre-registration block above is unchanged)

Integrity of the pre-registration: everything in this file before the blank line that precedes `## Results` has SHA-256 `d97a6694d2f253e877b831b2758602d42135235c1e552f730f5795794b34837c`. That is identical to the whole file at 02:14 BST, before any result existed, and to the hash `localcal.py` recorded when the run started (`localcal.json` → `prereg_block_sha256_now`). Check it with `python3 -c "import hashlib;print(hashlib.sha256(open('reports_v2/localcal.md').read().split(chr(10)*2+'## Results')[0].encode()).hexdigest())"`. The status line at the top is part of that frozen block, which is why it still reads "pre-registration only". The run took 42 s (`src/beans_v2/localcal.py`, 02:30:52–02:31:24). Every number below is in `reports_v2/localcal.json`.

### Verdict against the pre-registered bar: **GO (prototypes, K = 20)**

| Pre-registered criterion (loja_yolo, prototypes) | K = 20 | K = 50 | K = 100 |
|---|---|---|---|
| 1. Mean balanced answered accuracy >= 90% | **96.3%** (worst draw 95.5%) | 96.6% (95.9%) | 96.5% (95.4%) |
| 1. Mean balanced coverage >= 50% | **78.3%** (worst draw 70.8%) | 80.8% (78.8%) | 77.2% (72.9%) |
| 2. Dangerous trays (a 0% tray called "many"), 5 x 500 trays | **0** | 0 | 0 |
| 3. CBD AAA photos called "many", per draw (gates ignored) | **0, 0, 0, 0, 0** | 0 in every draw | 0 in every draw |
| CBD photos given any band, per draw (gates ignored) | 0, 0, 0, 0, 0 | 0 in every draw | 0 in every draw |
| Calibration photos per draw | 4, 3, 2, 4, 6 | 11, 12, 8, 12, 9 | 26, 19, 18, 22, 19 |

All three criteria hold at every K, so by the pre-registered rule the verdict is GO at the smallest K, **K = 20** (10 good + 10 defect beans). The logistic head also meets its stricter exception bar at every K, but it is not needed. The spec (`localcal_spec.md`) uses prototypes.

### How to read this GO (it is narrower than it sounds)

1. **The bar could not tell the head apart from the base model on loja_yolo.** The base model, trained without loja_yolo, already meets criterion 1 on its own: on the same test beans at K = 20 it reaches 97.4% balanced answered accuracy at 65.7% balanced coverage, using the clean thresholds 0.50 / 0.91. The local head adds about 13 points of coverage (65.7% → 78.3%) for about 1 point of accuracy (97.4% → 96.3%). Good beans called defect go from 2.2% (base alone) to 3.0–5.9% per draw (head). Good beans called good go from 49–53% to 51–74%.
2. **Most trays are still "not sure".**
   - At K = 20, **131 of 2,500 trays (5.2%)** got a band, against 0 for the base model alone.
   - 17 of those 131 bands were wrong, all one band off: 12 trays at 12% called "many", and 5 clean trays called "some".
   - The 15% unsure rule still fires on most trays, because about 22% of beans (class-balanced) stay unsure.
   - The test pool held about 135 good beans per draw, so the 167 clean trays per draw share beans and are not independent trials.
3. **The same method failed on every secondary source, and it was dangerous on two of them:**
   - **lojano**, base model held out from lojano. Answered accuracy is 84.9–85.9%, below the bar. At K = 20, **11 clean trays were called "many"**, 10 of them in one draw: draw 4, with 6 of 20 calibration beans wrong in leave-one-out and m_hi = 0.109. With lojano heads, **2 top-grade CBD photos were called "many"**: an AA photo in draw 0 and an A photo in draw 4. No AAA photo was called "many" with the prototype head. The logistic alternative called **2 CBD AAA photos "many"**: one with a lojano head (K = 20, draw 0) and one with a Mindforge head (K = 100, draw 4). Calibrating on 1 good tray photo and 1 defect tray photo most likely teaches the head the difference between two photos, not between good and defective beans.
   - **Mindforge**, test-only. Accuracy is 78.7–80.3%, and **3 clean trays were called "many"**: 1 at K = 20 and 2 at K = 50.
   - **USK**, test-only. Accuracy is 63.0–67.4%; calibration does not rescue it.
4. **Crops, not new photos.** Test beans are the bean finder's crops of real photos. Nothing was re-photographed, nothing ran on a phone, and there are no Yemeni beans. loja_yolo photos hold only about 4 usable beans each (1,309 non-touching beans in 315 photos), so "a few photos" meant 2–6 photos at K = 20.
5. **One base-model seed per fold** (`clean_folds.py`, seed 0). No seed spread was measured.

### Full tables (prototypes = pre-registered head; logistic = reported alternative; baseline = base model alone, thresholds 0.50 / 0.91)

### loja_yolo (base `clean_loso_loja_yolo`; 1309 non-touching beans in 315 photos/groups: 146 good, 1163 defect)

Base model alone (thresholds 0.50/0.91), all beans of the source: balanced coverage 65.7%, balanced answered accuracy 97.5%.

| K | head | calib. photos per draw | mean balanced coverage (min) | mean balanced answered acc (min) | plain coverage / acc | trays: bands / wrong / dangerous (clean->many) / many->clean (of 2,500) | CBD AAA 'many' per draw (gates ignored) | CBD photos banded per draw |
|---|---|---|---|---|---|---|---|---|
| 20 | proto | [4, 3, 2, 4, 6] | 78.3% (70.8%) | 96.3% (95.5%) | 85.0% / 98.2% | 131 / 17 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 20 | logistic | [4, 3, 2, 4, 6] | 80.2% (75.1%) | 96.2% (95.1%) | 88.0% / 98.2% | 149 / 25 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 20 | baseline | [4, 3, 2, 4, 6] | 65.7% (64.8%) | 97.4% (97.3%) | 75.9% / 98.3% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |
| 50 | proto | [11, 12, 8, 12, 9] | 80.8% (78.8%) | 96.6% (95.9%) | 87.0% / 98.3% | 133 / 8 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 50 | logistic | [11, 12, 8, 12, 9] | 80.9% (75.8%) | 95.9% (93.0%) | 89.5% / 98.2% | 103 / 17 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 50 | baseline | [11, 12, 8, 12, 9] | 65.0% (63.7%) | 97.7% (97.1%) | 76.0% / 98.4% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |
| 100 | proto | [26, 19, 18, 22, 19] | 77.2% (72.9%) | 96.5% (95.4%) | 87.0% / 98.3% | 93 / 20 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 100 | logistic | [26, 19, 18, 22, 19] | 76.8% (73.8%) | 95.5% (94.4%) | 91.7% / 98.3% | 40 / 4 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 100 | baseline | [26, 19, 18, 22, 19] | 63.8% (62.4%) | 97.6% (96.5%) | 76.2% / 98.3% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |

### lojano (base `clean_loso_lojano`; 1176 non-touching beans in 28 photos/groups: 598 good, 578 defect)

Base model alone (thresholds 0.50/0.91), all beans of the source: balanced coverage 60.3%, balanced answered accuracy 79.7%.

| K | head | calib. photos per draw | mean balanced coverage (min) | mean balanced answered acc (min) | plain coverage / acc | trays: bands / wrong / dangerous (clean->many) / many->clean (of 2,500) | CBD AAA 'many' per draw (gates ignored) | CBD photos banded per draw |
|---|---|---|---|---|---|---|---|---|
| 20 | proto | [3, 2, 2, 2, 2] | 72.6% (61.8%) | 84.9% (80.1%) | 72.6% / 85.0% | 152 / 77 / 11 / 0 | [0, 0, 0, 0, 0] | [1, 0, 0, 0, 1] |
| 20 | logistic | [3, 2, 2, 2, 2] | 72.1% (61.2%) | 84.7% (80.6%) | 72.2% / 84.8% | 223 / 114 / 13 / 0 | [1, 0, 0, 0, 0] | [2, 0, 0, 0, 1] |
| 20 | baseline | [3, 2, 2, 2, 2] | 60.6% (59.9%) | 79.3% (78.8%) | 60.8% / 79.6% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |
| 50 | proto | [2, 2, 2, 2, 2] | 70.8% (62.9%) | 85.9% (79.8%) | 70.5% / 85.6% | 3 / 1 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 50 | logistic | [2, 2, 2, 2, 2] | 68.6% (56.7%) | 87.9% (80.6%) | 68.3% / 87.6% | 3 / 1 / 1 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 50 | baseline | [2, 2, 2, 2, 2] | 60.3% (60.1%) | 79.5% (78.7%) | 60.0% / 79.1% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |
| 100 | proto | [5, 3, 4, 3, 5] | 65.8% (62.0%) | 85.2% (82.7%) | 65.6% / 84.9% | 0 / 0 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 100 | logistic | [5, 3, 4, 3, 5] | 64.8% (60.5%) | 87.3% (85.5%) | 64.5% / 87.0% | 0 / 0 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 100 | baseline | [5, 3, 4, 3, 5] | 60.6% (59.6%) | 79.5% (77.8%) | 60.4% / 79.1% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |

### mindforge_doubleside (base `FINAL`; 7372 non-touching beans in 5066 photos/groups: 1447 good, 5925 defect)

Base model alone (thresholds 0.50/0.91), all beans of the source: balanced coverage 50.9%, balanced answered accuracy 87.1%.

| K | head | calib. photos per draw | mean balanced coverage (min) | mean balanced answered acc (min) | plain coverage / acc | trays: bands / wrong / dangerous (clean->many) / many->clean (of 2,500) | CBD AAA 'many' per draw (gates ignored) | CBD photos banded per draw |
|---|---|---|---|---|---|---|---|---|
| 20 | proto | [14, 15, 16, 14, 14] | 71.9% (70.4%) | 78.7% (76.2%) | 77.9% / 87.9% | 3 / 2 / 1 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 20 | logistic | [14, 15, 16, 14, 14] | 73.9% (72.4%) | 78.7% (75.5%) | 78.8% / 86.7% | 13 / 9 / 1 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 20 | baseline | [14, 15, 16, 14, 14] | 50.9% (50.8%) | 87.1% (87.0%) | 58.7% / 91.2% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |
| 50 | proto | [32, 35, 39, 33, 36] | 73.3% (70.5%) | 78.9% (76.1%) | 79.6% / 88.2% | 16 / 9 / 2 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 50 | logistic | [32, 35, 39, 33, 36] | 75.1% (71.5%) | 78.8% (76.5%) | 81.1% / 87.9% | 42 / 21 / 6 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 50 | baseline | [32, 35, 39, 33, 36] | 51.0% (50.8%) | 87.0% (86.9%) | 58.7% / 91.2% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |
| 100 | proto | [77, 67, 67, 71, 67] | 66.1% (64.6%) | 80.3% (79.2%) | 75.2% / 89.9% | 0 / 0 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 100 | logistic | [77, 67, 67, 71, 67] | 65.3% (64.6%) | 82.1% (81.1%) | 75.1% / 90.5% | 0 / 0 / 0 / 0 | [0, 0, 0, 0, 1] | [0, 0, 0, 0, 1] |
| 100 | baseline | [77, 67, 67, 71, 67] | 50.9% (50.9%) | 87.1% (87.0%) | 58.7% / 91.3% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |

### usk_coffee (base `FINAL`; 7969 non-touching beans in 7969 photos/groups: 5970 good, 1999 defect)

Base model alone (thresholds 0.50/0.91), all beans of the source: balanced coverage 49.4%, balanced answered accuracy 64.1%.

| K | head | calib. photos per draw | mean balanced coverage (min) | mean balanced answered acc (min) | plain coverage / acc | trays: bands / wrong / dangerous (clean->many) / many->clean (of 2,500) | CBD AAA 'many' per draw (gates ignored) | CBD photos banded per draw |
|---|---|---|---|---|---|---|---|---|
| 20 | proto | [37, 27, 46, 21, 66] | 62.0% (55.8%) | 63.0% (56.3%) | 59.0% / 51.3% | 0 / 0 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 20 | logistic | [37, 27, 46, 21, 66] | 63.9% (55.1%) | 64.2% (57.4%) | 59.9% / 51.5% | 0 / 0 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 20 | baseline | [37, 27, 46, 21, 66] | 49.5% (49.4%) | 64.2% (64.1%) | 46.4% / 65.1% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |
| 50 | proto | [84, 83, 107, 58, 105] | 71.9% (69.1%) | 67.3% (64.2%) | 66.4% / 61.1% | 4 / 2 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 50 | logistic | [84, 83, 107, 58, 105] | 67.9% (64.3%) | 73.1% (70.0%) | 62.3% / 67.3% | 0 / 0 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 50 | baseline | [84, 83, 107, 58, 105] | 49.4% (49.4%) | 64.2% (64.1%) | 46.3% / 65.1% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |
| 100 | proto | [193, 190, 201, 202, 159] | 69.4% (64.0%) | 67.4% (65.7%) | 62.9% / 61.8% | 1 / 1 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 100 | logistic | [193, 190, 201, 202, 159] | 64.8% (56.6%) | 78.2% (76.3%) | 59.5% / 73.6% | 4 / 1 / 0 / 0 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| 100 | baseline | [193, 190, 201, 202, 159] | 49.5% (49.3%) | 64.2% (63.9%) | 46.4% / 65.0% | 0 / 0 / 0 / 0 | ['-'] | ['-'] |

### Exploratory, post-hoc (NOT part of the verdict, NOT validated)
Consider a guard that refuses a head whose leave-one-out error rate on its own calibration beans is above 10%. Here is what it would have done:
- It keeps 13 of 15 loja_yolo draws, with 0 dangerous trays.
- It rejects all 15 Mindforge and all 15 USK draws.
- It keeps 4 of 15 lojano draws. One of those (K = 20, draw 3: 2 of 20 LOO errors) still called 1 clean tray "many".

So a calibration-only check would catch most failures but not all. A real deployment needs an acceptance test on labelled photos that were not used for calibration.

### Sensitivity check, post-hoc: could the same bean sit in a calibration photo and a test photo?
loja_yolo photos are numbered IMG_4743 to IMG_5791. If neighbouring shots re-use beans, a calibration bean could reappear in a test photo. To test this, the 5 K = 20 draws were re-scored after dropping every test photo whose IMG number is within ±20, or within ±50, of any calibration photo. Output: `reports_v2/localcal_sensitivity.txt` (inline script; the K = 100 row at ±50 failed because no good test beans were left).

| Test photos kept | Mean balanced answered accuracy (min) | Mean balanced coverage |
|---|---|---|
| All test photos (as reported) | 96.3% (95.5%) | 78.3% |
| > 20 away | 95.8% (94.4%) | 77.5% |
| > 50 away | 96.1% (93.6%) | 75.5% |

- The mean nearest-neighbour cosine from a test bean to the calibration beans was 0.585 at K = 20. A re-shot bean would sit near 1.
- For K = 50 and 100 this check runs out of good test beans (0–42 left per draw), so it is not informative there.
- So the K = 20 result does not depend on neighbouring shots.

### Files
- `reports_v2/localcal.json`: every draw. It holds the calibration photo IDs and rows, the head parameters (m_hi, LOO errors), bean metrics, the tray tallies by truth and outcome, and the CBD outcomes with gates ignored and with the gate port.
- `reports_v2/localcal_spec.md`: the JS-portable algorithm.
- `models_v2/localcal_demo/`: the demo set (CC BY 4.0, `CREDITS.md`).
- Code: `src/beans_v2/localcal.py` (study), `src/beans_v2/localcal_export.py` (demo set), `src/beans_v2/clean_common.py` (embedding, rules mirror).
