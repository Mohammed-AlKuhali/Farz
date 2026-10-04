# Robust signals (v2): colour baseline, unusual-bean detector, OOD gate

STATUS: COMPLETE, 23:30 BST Sat 3 Oct 2026. Every number below was measured by the script named in its section heading; the matching JSON holds the full detail. No v1 file was modified.

**Rule-version warning.** `app/src/lib/rules.ts` and `wilson.ts` changed at 23:05 while this study was running. The new version has two changes: the band now comes from the point estimate (with an "about" flag), and there is a new "implausible" fail-safe (more than 60% of answered beans called defective → "not sure"). Section (3) uses a Python port of the **23:05** rules and also re-runs the **22:40** rules for comparison. If the rules change again, re-run `robust_ood.py`.

**Bottom line**

1. **Simpler tool.** Across photos, one luminance threshold beats the v1 CNN at "dark vs not": balanced accuracy 0.71 vs 0.42. Neither the rule nor the CNN can find unhulled beans in a photo it has not seen (≤ 0.49).
2. **Unusual-bean detector.** A per-photo score flags whatever differs from the rest of that tray.
   - Inside one capture set-up (CBD) it carries real signal: AUROC 0.88.
   - Recall at the cut-off is only 30%.
   - On real photos, lower grades are **not** flagged more often (ρ = −0.12).
   - It cannot count defects.
3. **OOD gate.**
   - With the 22:40 rules, the v1 app gave a confident band to **163/464** CBD photos, including "many defects" on **27/50** AAA photos.
   - With the 23:05 rules it gives **0/464**.
   - Held-out J4ckDev-style trays still get **30/30 correct** bands.
   - This works only because v1 calls ≥ 89% of answered CBD beans defective, which trips the 60% "implausible" rule. That rule is a content fail-safe tuned on these photos. It does not detect a change of distribution.
   - A held-out stress set (777 AAA/AA photos re-shot in software: exposure, colour cast, resolution, blur, JPEG) lets **1/777** through as a confident "many".
   - A cheap, model-free colour/shape Mahalanobis gate closes that leak (**0/777**). Its cost: 4/30 J4ckDev-style trays become "not sure".

## Data, licences, dedupe
| Data | Use | Licence (checked on the source page, 23:13 BST) |
|---|---|---|
| J4ckDev Green Coffee Beans: 11 photos, 582 bean crops (`data/raw/green`, `data/crops`) | Labels at photo level; training bank | CC BY-NC-SA 4.0 (github.com/J4ckDev/GreenCoffeeBeansDataset) |
| CBD Coffee Bean Dataset: 464 photos, 9 grade folders (`data/raw/cbd`) | Unlabelled out-of-distribution behaviour | CC BY 4.0 (data.mendeley.com/datasets/52877z55vr/1). The page does **not** define the grades, so grade names are used as folder labels only |
| torchvision MobileNetV3-Small IMAGENET1K_V1 | Frozen feature extractor | The same weights v1 is fine-tuned from, so this adds no new dependency. Licence not re-checked tonight |

No new data was downloaded.

Dedupe used the MD5 of decoded pixels (`src/beans_v2/robust_cache.py`):
- **475/475 photos unique** and **24,029/24,029 crops unique**.
- The frozen finder re-cut the J4ckDev crops **pixel-identical** to `data/crops` (582/582).
- The CBD bean count is 23,447, the same as v1.

The Python port of the app's colour gate reproduces the app's own comment: all 464 CBD photos pass; 9/11 J4ckDev photos pass (CerezaSeca and Negros are refused as `too_dark_beans`).

## (1) Colour-only baseline vs v1 (`robust_colour.py` → `robust_colour.json`, `.png`)
**Protocol.** Leave-one-photo-out (LOPO) on J4ckDev: each rule is fitted only on the other photos. The v1 numbers come from `reports/beans_lopo.json`, which uses the same protocol (argmax, no abstention). v1 could not leave out Normales, so the head-to-head uses the same 10 photos. Results are bean-weighted.

| Task | Method | Sensitivity | False-positive rate | Balanced accuracy | Photo-balanced accuracy |
|---|---|---|---|---|---|
| dark vs not | **v1 CNN (LOPO)** | 0.210 | 0.368 | **0.421** | 0.361 |
| | R0, fixed rule: mean-colour HSV V < 0.30 (nothing fitted) | 0.142 | 0.000 | 0.571 | 0.535 |
| | **R1, one threshold** (fitted per fold; 8/11 folds pick median luminance < 94–98) | 0.588 | 0.163 | **0.712** | 0.539 |
| | R2 depth-2 tree / R3 logistic regression on 16 stats | – | – | 0.491 / 0.619 | 0.474 / 0.559 |
| unhulled vs not | v1 CNN (LOPO) | 0.000 | 0.118 | 0.441 | 0.471 |
| | R1, one threshold | 0.000 | 0.017 | 0.491 | 0.488 |
| | R2 / R3 | – | – | 0.375 / 0.479 | – |

- **R1, share of each photo called dark:**
  - True dark photos: Negros 0.73, MarronAVinagre 0.36, DXHongo 0.08.
  - False positives: BrocadoSevero 1.00 and CerezaSeca 1.00. These beans look black but are labelled insect and unhulled.
  - Normales (good, held out): 0.005.
- **Why unhulled fails for every method.** Its two photos look opposite: pale parchment and black dried cherry. When one is held out, the other cannot teach it.
- **Within-photo halves** (v1's own upper-bound protocol): dark R1 0.874 vs v1 0.894; unhulled R1 0.908 vs v1 0.974.
- **CBD behaviour** (no labels). R1 fitted on all of J4ckDev calls **10.6–22.0%** of CBD beans dark (AAA 13.1%), against v1's 5.5–31.8% (AAA 5.8%). R0 calls ≤ 0.44%. The unhulled rule calls 0.2–1.8% of CBD beans unhulled, against v1's 11.4–21.8%.
- **Answer to "would a simpler tool do the job?"**
  - For dark beans, yes: across photos a one-line luminance rule is at least as good as the CNN. On the same photos the CNN is slightly better.
  - Neither tool generalises to unhulled beans or to a new camera, because the absolute threshold does not transfer to CBD's light box.

## (2) Unusual-bean detector (`robust_unusual.py`, `robust_spread.py`, `robust_export.py`)
Outputs: `robust_unusual.json/.png`, `robust_unusual_examples.png`, `robust_spread.json/.png`, `robust_export.json`.

Each bean is scored **only against the other beans in its own photo**. Four scores:
- `hand`: robust z (median/MAD) of 16 colour/shape stats, combined as an RMS.
- `emb`: leave-one-out 5-nearest-neighbour cosine distance on frozen ImageNet MobileNetV3-Small 576-d features.
- `both`: the mean of the two.
- `v1f`: the same 5-NN distance on the v1 model's own features.

A bean is flagged when its per-photo robust z is above 3.5. That cut-off was fixed before any result was seen.

**(a) Synthetic trays.** 300 trays per condition, each 50 base beans + 3 inserted beans, `both` score. Each cell shows raw crops / exposure-normalised crops; normalisation = per-crop background white balance + a common 40 px resolution.

| Condition | AUROC | Recall of inserted beans | False-flag rate on base beans |
|---|---|---|---|
| A: J4ck good + J4ck **defects** (from other photos) | 0.965 / 0.960 | 0.69 / 0.68 | 0.009 / 0.028 |
| B: CONTROL, J4ck good + CBD AAA **good** beans | 0.987 / 0.976 | 0.55 / 0.64 | 0.009 / 0.027 |
| B2: CONTROL, the same good beans with a synthetic exposure shift | 1.000 / 0.707 | 0.98 / 0.10 | 0.014 / 0.032 |
| C: real CBD AAA photo + J4ck **defects** | 0.994 / 0.977 | 0.91 / 0.66 | 0.019 / 0.019 |
| D: CONTROL, real CBD AAA photo + J4ck **good** | 0.989 / 0.974 | 0.78 / 0.49 | 0.020 / 0.019 |
| E: real CBD AAA photo + CBD **Bits** beans (same light box) | 0.880 / 0.908 | 0.30 / 0.30 | 0.021 / 0.022 |
| F: CONTROL, real CBD AAA photo + AAA beans from **other AAA photos** | 0.527 / 0.516 | 0.03 / 0.03 | 0.030 / 0.027 |

What the table shows:
- **Trays that mix photos leak photo identity.** Good beans from another camera (B, D) are flagged almost as often as defects (A, C).
  - Per-crop normalisation removes a pure exposure shift (B2 recall: 0.98 → 0.10).
  - It does not remove a camera or dataset difference (B recall: 0.55 → 0.64).
  - The `v1f` score leaks completely: B recall 0.99.
- **Inside one capture set-up there is signal and no leakage.** E vs F: AUROC 0.88 vs 0.53. At z > 3.5, though, only 30% of the inserted beans are flagged.
- **Merged blobs get flagged too.** The strongest flag in AAA/1.jpg is a merged two-bean blob, and 30 of the 40 touching blobs in the data are flagged. v1 already treats touching blobs as "unsure".

**(b) Real CBD photos.** Share of beans flagged (`both`, raw crops):

| AAA | AA | A | AB | PB-I | PB-II | C | Bulk | Bits |
|---|---|---|---|---|---|---|---|---|
| 2.6% | 2.7% | 2.0% | 3.5% | 0.9% | 1.7% | 2.1% | 2.4% | 1.5% |

- That is about 0.4–1.8 flagged beans per 50-bean photo.
- **The expected trend is absent.** Spearman ρ against grade order is −0.12 (p = 0.007); the `hand` score gives −0.29.
- **A relative score cannot see a uniform sample.** On the single-class J4ckDev photos it flags 0% (Normales 1.0%), so it is blind to a sample that is uniformly bad or uniformly mixed.
- **Photo-level spread is weak** (`robust_spread`). Within CBD, spread correlates with grade: embedding spread ρ +0.29, AUROC AAA vs Bits 0.82. The distributions overlap heavily, and spread depends on the camera.

**(c) Browser cost** (laptop CPU, not a phone; extractor has 927,008 parameters):

| Format | Size | Speed, batch 50 | Fidelity |
|---|---|---|---|
| fp32 | 3,711,429 B | 76 ms (Python ONNX Runtime, 1 thread); **116 ms** (onnxruntime-web 1.30 wasm, 1 thread) | max diff 3.3e-5 vs PyTorch |
| fp16 | 1,878,403 B | 114 ms (wasm) | min cosine 0.9992; flag agreement on 2,037 CBD beans: `both` 100%, `emb` 99.85% |

The `hand` score needs no model at all.

## (3) Photo-level OOD gate (`robust_lopo_v1.py`, `robust_ood.py`, `robust_export_gate.py`)
Outputs: `robust_ood.json`, `robust_ood.png`, `robust_ood_gates.png`, `robust_gate_params.json`.

**Pipeline ported from the app:** retake gates → per-bean call (max-prob ≥ 0.59 and not touching) → implausible (> 60% defects) → more than 15% unsure → band.

**Test sets**
- 22 real J4ckDev half-photos, each scored by the fold model that never saw it.
- **30 held-out synthetic trays**: 100 beans each (10 clean, 10 "some", 10 "many"), built with `make_demo`'s recipe from fold-test beans only.
- 11 J4ckDev photos scored LOPO with v1-recipe models retrained without that photo. The retrain reproduces `beans_lopo.json` exactly, on all 10 photos.
- 464 CBD photos.
- **STRESS set (never used for calibration): 777 photos** = 111 AAA/AA photos × 7 software re-shoots (×0.6 darker, ×1.3 brighter, warm cast, cool cast, ⅓ resolution, blur r3, JPEG q15).
- 77 perturbed J4ckDev photos and the 7 app demo photos.

**Calibration ("@zeroAAA").** The largest threshold that still refuses all 27 AAA photos v1 calls "many" under the 22:40 rules.

"Confident" means a band of clean, some or many.

| Verdict after the gate | CBD given a band | AAA "many" | Held-out CBD grades given a band | **STRESS given a band** | J4ck trays correct (wrong) | J4ck real halves refused by score¹ |
|---|---|---|---|---|---|---|
| no gate, app rules **22:40** (reference) | 163/464 | 27/50 | 136/414 | 419/777 | 16/30 (0) | – |
| **no gate, app rules 23:05 (today)** | **0/464** | **0/50** | **0/414** | **1/777** | **30/30 (0)** | – |
| + hand-stat Mahalanobis @zeroAAA (no model) | 0/464 | 0/50 | 0/414 | **0/777** | 26/30 (0) | 2/22 (both CerezaSeca) |
| + v1 energy @zeroAAA (needs logits output) | 0/464 | 0/50 | 0/414 | 0/777 | 30/30 (0) | **12/22** |
| + ImageNet 5-NN to J4ck bank @zeroAAA (+1.9–3.7 MB) | 0/464 | 0/50 | 0/414 | 1/777 | 30/30 (0) | 4/22 |
| + S7: bean-familiarity → "unsure" (95th percentile) | 0/464 | 0/50 | 0/414 | 0/777 | 9/30 (**2 wrong**) | 0 lost |
| + S7 (99th percentile) | 0/464 | 0/50 | 0/414 | 1/777 | 20/30 (0) | 0 lost |

¹ "Refused by score" means the photo score is above the threshold, whatever its band. Most half-photos have fewer than 20 beans, so they are already a "retake". This column is the best available proxy for refusing a *real* J4ckDev-style photo.

- **Margin of the implausible rule on CBD.** Among the 163 CBD photos that the 15% rule would not stop, v1 calls **at least 89.4%** of answered beans defective (median 100%). That is far above the 0.6 cut-off.
- **The one stress leak** is `AA/82.jpg` blurred with r3: 39.6% defects and 12.7% unsure → confident "many". A moderate shift that makes v1 *moderately* wrong is exactly what the implausible rule cannot see.
- **False refusal / false confidence of the app as of 23:05:**
  - False confidence: 0/464 on CBD, 1/777 under stress (0.13%), 0 wrong on the 30 held-out trays.
  - False refusal on held-out J4ckDev-style trays: 0/30.
  - Real J4ckDev photos: every LOPO photo is "not sure" or a retake. They are single-class defect photos with more than 60% defects, so they are refused by design.
  - With the hand-Mahalanobis gate added: stress 0/777; false refusal on trays 4/30.
- **S7 is no longer recommended.** Under point-estimate bands, making unfamiliar beans "unsure" changes the defect count, because unfamiliar beans are disproportionately defects. That produced 2 wrong confident tray verdicts at the pre-chosen 95th percentile. Its hand-off files (v1 graph + `feat` output, probs bit-identical; 670,464 B fp16 bank) remain in `robust_gate_params.json` / scratchpad for reference.
- **The hand-Mahalanobis gate:**
  - It is 288 constants in `robust_gate_params.json`, with threshold 4.9106.
  - Its margin is thin: J4ckDev halves reach 4.53 (excluding CerezaSeca) and the CBD AAA minimum is 4.91.
  - It *passes* CBD C and Bits photos, whose dark beans resemble J4ckDev defect beans.
  - It would need a JS port of the 16 per-bean stats (mask, Lab, convex hull, perimeter, erosion).

## What the app should show
1. **Keep the 23:05 rules.** On these data they already meet the target: zero confident verdicts on all 464 CBD photos, and 30/30 correct on held-out J4ckDev-style trays. Describe the 60% rule honestly in the README and videos: it is a **fail-safe tuned on out-of-distribution photos**. It routes very bad samples and unfamiliar photos alike to "not sure, take the sample to the cooperative". It is **not** a detector of new cameras. Evidence to quote: 163 → 0 CBD photos given a band; 1 of 777 re-shot top-grade photos still given a band.
2. **Optional defence in depth before the freeze:** add the hand-Mahalanobis photo gate. It needs no model download and closes the stress leak (1 → 0 of 777). The cost is that 4/30 J4ckDev-style trays get "not sure". Only ship it with a JS↔Python parity test of the 16 stats. If that cannot be done in time, skip it.
3. **Do not ship per-type counts as facts** ("dark", "unhulled") from either the CNN or the colour rule. If types are shown, label them "possible". For the "simpler tool" part of the video: *"a one-line colour threshold matched or beat our CNN on dark beans across photos; neither works for parchment or a new camera."*
4. **Do not use the unusual-bean detector for a count or a band.** At most it is an optional "look closer at these beans" highlight (about 1 bean per photo). It cannot say a whole sample is bad.

## Limits (read before quoting)
- CBD has no per-bean labels. Calling every confident CBD band a false confidence assumes the model should not answer about a different set-up. That AAA "many" is wrong is presumed from the grade name, not checked bean by bean.
- The 60% rule was tuned on CBD-like photos (`rules.ts` comment), so the 0/464 is partly in-sample. The STRESS set is the held-out check.
- The held-out J4ckDev trays are synthetic pastes of real crops.
- Timings were measured on a MacBook M5 Pro CPU, not a phone.
- Nothing here was measured on Yemeni beans or on phone photos.

## Reproduce (about 9 min, plus 11 min of LOPO retraining on MPS)
Set `FARZ_SCRATCH=<dir>`, then run these from `src/beans_v2/`, in order:
1. `robust_cache.py`
2. `robust_colour.py`
3. `robust_unusual.py`
4. `robust_unusual_plots.py`
5. `robust_spread.py`
6. `robust_export.py`
7. `robust_lopo_v1.py`
8. `robust_ood.py`
9. `robust_export_gate.py`

The cache (about 2.5 GB) lives outside the repo.
