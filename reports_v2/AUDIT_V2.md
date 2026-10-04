> **This audit describes the all-sources research model, NOT the model shipped in this repository (see `reports_v2/clean_model.md`).** It finished at about 02:10 BST on 4 Oct, before the shipped model existed. Where it says "the shipped file", "the deployed weights" or that unlicensed sets "are inside the shipped weights", it means the all-sources model as it stood then. The shipped model was trained only on sources that state a licence and was re-tested with this audit's protocols in `clean_model.md`; the auditor did not test it.

# AUDIT_V2 — adversarial audit of the v2 bean classifier and robustness work

Status: COMPLETE, 2026-10-04 ~02:10 BST (machine clock). Auditor scripts and outputs: /tmp/audit_v2/ (outside the repo). Only this file was written in the repo; `app/` and `node_modules` were read, never written.
Legend: CONFIRMED / WRONG (with auditor number) / UNVERIFIABLE.

## (1) Cross-source leakage (auditor scripts /tmp/audit_v2/leak_crops.py, leak_analyse.py, leak_sheet.py)

Population: all 118,143 training-cache crops (12 sources) + the robust cache's 24,029 crops (CBD 23,447 + the 582 J4ckDev crops again, labelled `j4ck_robust`) = 142,172 crops.
Methods: (a) MD5 of the 128x128x3 pixel array; (b) 63-bit DCT pHash on 32x32 luminance, each query hashed under all 8 rotations/flips, candidates from 4-band LSH, Hamming <= 8 kept; (c) ImageNet MobileNetV3-Small 576-d embedding, cosine of every crop's nearest neighbour in a *different* source (exhaustive, GPU). Then the 28 cross-source pHash-0 pairs and the 24 highest-cosine pairs were inspected by eye (/tmp/audit_v2/leak_pairs.jpg).

| Check | Result | Verdict |
|---|---|---|
| Exact pixel duplicates across sources (excluding j4ckdev vs its own robust-cache copy, 582/582 identical as expected) | **0** | CONFIRMED ("0 cross-dataset duplicates") |
| Exact duplicates within a source that cross a group (photo) boundary | **0** | CONFIRMED |
| Cross-source pHash Hamming 0 (dihedral) | 28 pairs, every one a **different bean** by eye (pHash of a smooth oval on grey is low-entropy; bands at h<=8 match >1M pairs, i.e. pHash cannot separate beans) | no duplicate found |
| CBD vs any training crop: embedding NN cosine | max < 0.98 for all 23,447 CBD crops; p99 0.943; 70 crops >= 0.95, none a duplicate by eye | CONFIRMED: no CBD crop leaked into training |
| Any cross-source pair with embedding cosine >= 0.99 | **0** | no near-duplicate found |
| afiyah_bijikopi <-> afiyah_deteksi | Each other's nearest neighbour: 154 of 1,170 bijikopi and 182 of 985 deteksi crops have a cross-source NN cosine >= 0.95, **all in the sibling set**; top pairs (0.97-0.98) are different beans on the same white paper, same lighting. EXIF: both sets **samsung Galaxy A05s** (40/40 sampled each), same Kaggle uploader (afiyahmusrah) | **SOURCE-LEVEL LEAK (not a pixel duplicate)** |

**Finding L1 (material).** afiyah_deteksi and afiyah_bijikopi are one capture set-up (same uploader, same phone model, same backdrop, shot weeks apart: bijikopi EXIF 2024-07-06, deteksi uploaded 2024-06-18). In LOSO, holding out one leaves its twin in training, so those two folds are **not "a source it has never seen"**. They are the two best LOSO rows (balanced acc 0.997 and 0.876) and they are the source of **all 40 of the 40 bands** the deploy simulation hands out (39 bijikopi + 1 deteksi). The report never mentions that the two afiyah sets are siblings. With both afiyah sets treated as one source, the deploy rule's real "new-source" usefulness evidence is **0 bands** (see (3)).

Limits: pHash/embedding cannot prove the absence of a re-shot bean at arbitrary rotation and different lighting; arbitrary-angle rotations are only covered by the embedding.

## (2) Leave-one-source-out accuracy (auditor scripts /tmp/audit_v2/loso_rescore.py, loso_metrics.py)

Method: loaded every `loso_<source>.pt` (and `lopo_<photo>.pt`) from the scratchpad into an independently written MobileNetV3-Small(6) + ImageNet-normalisation module, re-ran it on its held-out crops, and recomputed everything from **labels read from the CSVs** (`data/crops/crops.csv`, `data/crops_v2/crops_v2.csv`, keep==1), not from the trainer's meta.json. Temperature re-fitted by my own grid search (400 log-spaced T in 0.3-5) on the other 11 sources' predictions, source- and class-balanced weights.

Integrity checks (all pass): held-out index set == every row of that source (12/12); the held-out source is absent from that fold's training pool (12/12, from the fold's sampler log); re-run logits == saved logits (max abs diff **0.0**, 12/12 LOSO and 11/11 LOPO); label kinds from CSV == trainer kinds (0 disagreements; 10,560 good / 107,583 defect rows).

| Claim | Auditor number | Verdict |
|---|---|---|
| Good-vs-defect balanced accuracy averages 0.78 across 12 held-out sources (0.53-1.00) | 0.781, range 0.532-1.000 | CONFIRMED as arithmetic, **MISLEADING as a headline**: 3 of the 12 (mfu17 0.883, notplying 0.996, vicanadya16 1.000) have no good beans, so they score defect recall only. **Mean over the 9 sources that have good beans = 0.721**; without afiyah_bijikopi (sibling leak, see (1)) = **0.687**. USK 0.532 and daffa 0.570 are near chance (0.5) |
| Typed accuracy averages 0.61 | 0.607 | CONFIRMED (note: untyped-defect sources count "any defect" as correct, which inflates it) |
| Per-source table §2.1 (T_s, macro, balanced, good/defect recall) | every row within 0.006 of the report (e.g. j4ckdev good recall 0.215 vs 0.221, from the T fit) | CONFIRMED |
| Good recall 0.22-0.57 on 7 of 9 sources with good beans | median good recall over the 9 = **0.505** | CONFIRMED: on a new source v2 calls about half of good beans defective |
| Pooled T = 1.558 | 1.562 (grid) | CONFIRMED |
| Rule A pooled t = 0.89, coverage 53.8%, answered acc 90.5% | 0.89 / 53.8% / 90.46% | CONFIRMED, but this is **in-sample for the threshold choice** (threshold picked and scored on the same pooled data) |
| Rule P pooled 0.66 / 0.89, coverage 68.9%, acc 90.1% | 0.66 / 0.89 / 68.85% / 90.10% | CONFIRMED (same in-sample caveat) |
| Per-source below 90% with thresholds from the other 11: A fails 5 (j4ckdev, lojano, usk, samruddh, daffa), P fails 6 (j4ckdev, mfu17, usk, samruddh, daffa, mindforge); USK 36% / 49% | A: same 5; P: same 6; USK 35.7% / 49.4% | CONFIRMED |
| Under rule A, 0.0-0.6% of held-out good beans called good | 0.0-0.6% | CONFIRMED |
| J4ckDev LOPO same 10 photos 27.4% -> 42.9% (photo-mean 42.2%) | 42.89% bean-weighted, 42.16% photo-mean | CONFIRMED. Caveat: on those 10 defect photos good-vs-defect is 100% because the model calls J4ckDev-looking beans "defect"; on Normales (good) it is **0%** |
| Normales: 74% called insect; 84% answered at t=0.87, all wrong | insect 73.8%, answered 84.1%, answered accuracy 0.0% | CONFIRMED (a confident wrong answer) |
| Model trained only on sources that state a licence "about as good" (0.59-1.00 vs 0.53-1.00) | not re-run (only its saved logits exist; LOSO of it never run, as the report says) | UNVERIFIABLE beyond the 5 numbers |

**What LOSO means for the shipped file.** `models_v2/farz_beans_v2_fp16.onnx` was trained on all 12 sources, so it has no LOSO of its own; the fold models are the same recipe with one source removed. The numbers above are the honest proxy and they say: on a new camera/farm, v2 is roughly a coin-flip-plus (0.69-0.72 balanced accuracy where good beans exist) and miscalibrated per source (USK ECE 0.42).

## (3) Deploy-rule simulation (auditor scripts /tmp/audit_v2/photos_find.py, sim_photos.py, sim_v1.py, sim_trays.py, j4ck_trays*.py, retrain_afiyah.py)

Independent pieces: my own run of the frozen bean finder (imported read-only) on every photo; my own port of `app/src/lib/rules.ts` `decide()` as it is on disk now (mtime 23:52: nothing answered / >60% of answered defective / >15% unsure -> "not sure", else band of the point estimate, clean <5% <= some <= 20% < many); per bean: touching -> unsure, P(good) >= t_good -> good, else 1-P(good) >= t_defect -> defect, else unsure; `other_defect` counted as a defect. Retake gates: the bean finder's own checks (dark, blur, 20-160 count, touching); the app's colour gate was **not** ported, and every result is also reported **as if all gates passed** (stricter than the app, because a retake is a safe outcome). My CBD crops are pixel-identical to the trainer's robust cache (464/464 photos, 23,447 crops).

### (3a) CBD and stress (shipped `farz_beans_v2_fp16.onnx`, T baked, 0.66 / 0.89)
| Set | Bands given | AAA "many" | Verdict |
|---|---|---|---|
| CBD 464 real photos | **0/464** even with all gates ignored (AAA: 23 too-many-unsure, 27 implausible) | **0/50** | CONFIRMED |
| Stress: 111 AAA+AA photos x 7 **auditor-chosen** re-shoots (x0.6 and x1.35 exposure, warm and cool cast, 1/3 resolution, Gaussian blur r=2.5, JPEG q=15) = 777 | **0/777** even with gates ignored | 0 | CONFIRMED (different perturbations from the report's, same result) |
| Same 777 + 464 with **v1** (`models/farz_beans_fp32.onnx`, argmax, abstain 0.59, same rules) | 0/464, 0/777 | 0 | CONFIRMED for CBD (report 0/464). My stress set did not reproduce the report's single v1 leak (`AA/82.jpg` blurred); not a contradiction, different perturbations |

Why it is safe: on CBD v2 calls 14.7% of AAA beans good and 23.2% defect at 0.66/0.89 (auditor: 14.7% / 23.2%), so >15% unsure or >60% implausible always fires. **The safety comes from the app's 15% / 60% photo rules, not from v2's accuracy** — v1 is equally silent on CBD (all 464 "implausible").

### (3b) Every held-out source: 50-bean trays from LOSO logits (auditor design: 0, 2, 10, 15, 30, 45, 100% defects, 200 trays each, seed 12345)
| Thresholds | Trays | Bands | Wrong | Dangerous (clean<->many) | Sources that ever get a band |
|---|---|---|---|---|---|
| T and (t_good, t_defect) fitted on the other 11 sources | 13,200 | 450 | **0** | 0 | afiyah_deteksi 8, afiyah_bijikopi 442; **0 on the other 10 sources** |
| Deployed T 1.558, 0.66 / 0.89 (pooled fit includes the scored source) | 13,200 | 1,243 | **2** (afiyah_deteksi: 2%->some, 15%->many) | 0 | afiyah only |

Report claim "40 bands, all correct, all afiyah, 0 wrong" (their 1,560 trays at 0/10/40/100%): CONFIRMED in direction (only afiyah ever gets a band; 0 wrong with held-out thresholds). With the deployed thresholds my denser design finds 2 wrong (non-dangerous) bands.

**Sibling test (new).** Retrained the same recipe (tc.train, 3,000 steps, seed 0) with **both** afiyah sets held out (training pool verified: 10 sources, no afiyah):
| | afiyah_deteksi bal. acc | afiyah_bijikopi bal. acc | bijikopi good beans called good (rule P) | Trays with a band |
|---|---|---|---|---|
| Report LOSO (twin in training) | 0.876 | 0.997 | 80.7% | bijikopi 442 / deteksi 8 (of 1,400 each) |
| Both held out (auditor) | **0.588** | **0.703** | **1.0%** | **0 of 1,400** |
So every band v2 has ever given on a held-out source comes from the afiyah twin leak. **On a genuinely new set-up v2 gives no band at all.** (One seed; the drop is large enough not to hinge on it.)

### (3c) Real held-out photos
| Set (model) | Auditor result | Report | Verdict |
|---|---|---|---|
| J4ckDev 11 photos (each by its LOPO model, deployed and J4ckDev-fitted thresholds) | 11/11 "not sure" (implausible); 4 also gated (Normales count 195 > 160) | 0 bands | CONFIRMED |
| lojano 36 tray photos (LOSO-lojano model) | 0/36 bands; all 14 good photos "too many unsure" (gates ignored), all defect photos implausible | 0 bands | CONFIRMED |
| loja_yolo 491 photos (LOSO model) | 0 bands after gates (all fail count/blur); 1 "many" if gates ignored | 0 bands | CONFIRMED |
| notplying 204 | not re-run | 0 bands (all retake) | UNVERIFIABLE (not re-run) |
| App demo photos (7, final model, in-sample for the 5 synthetic trays) | 7/7 "not sure" (v1: 5 trays correct bands in-sample, 2 CBD "not sure") | 7/7 not sure | CONFIRMED |

### (3d) J4ckDev set-up, unseen beans (the freeze argument): crop-level trays from the spatially held-out half (v1 fold ONNX models vs v2 `half_k` models), 80 beans, 1,200 trays at the report's own band-centre compositions (0 / 12.5 / 30% defects)
| Model + rule | Bands | Correct | Wrong | **Dangerous: a 0-defect tray called "many"** |
|---|---|---|---|---|
| v1 fold models, abstain 0.59, 23:05 rules | 793 / 1,200 | 736 | 57 | **10 / 400 clean trays** (and 46/400 "some" trays called "many") |
| v2 half models, deployed 0.66/0.89 | 229 / 1,200 | 229 | **0** | 0 |
| v2 half models, J4ckDev-LOSO thresholds 0.66/0.87 | 322 / 1,200 | 322 | 0 | 0 |
| v2 "local calibration" (approx. T 0.84, 0.50/0.51) | 1,200 | 726 | 474 | 118 |
A wider design (0-3 / 5-16 / 17-40 defects of 80): v1 116 wrong of 751 bands, **6 dangerous**; v2 deployed 23 wrong of 255, **0 dangerous**.
Mechanism (measured): on fold 1's held-out half of Normales (95 good beans) v1 abstains on 18.9% and calls **22.1% of the answered good beans defective**; fold 2: 4% / 1%. So a zero-defect fold-1 tray is "not sure" when unsure > 15% and **"many"** when it dips below. The trays share beans (80 drawn from 95), so this is one half-photo's behaviour, not 400 independent trials: read it as "v1's safety in its own set-up is not established", not as a precise rate.
The report's "v1 + 23:05 rules: 30/30 held-out trays correct" (`robust_summary.md`) is a 30-tray sample (10 clean); with 5 clean trays per fold it can easily miss this. **"v2 is not more accurate than v1 in that set-up" is WRONG for safety**: with unseen J4ckDev beans, v1 gives confident dangerous answers and v2 (deployed rule) gave none in 1,200 trays; v2 is only less *useful* (19% vs 66% bands). My trays are crop-level (no re-photographing, no touching beans), unlike the report's pasted-and-re-found 60 trays where v2 gave 0 bands; both designs agree v2 rarely answers. The report's "local calibration is dangerous" (2 dangerous in 60): CONFIRMED in direction (118/400).

## (4) ONNX vs PyTorch parity of the shipped file (auditor scripts /tmp/audit_v2/parity.py, wasm_parity.mjs)

Reference: `final.pt` (scratchpad `v2train/folds/final.pt`, the all-12-sources weights) loaded into my own MobileNetV3-Small(6)+normalisation module, softmax(logits / 1.5583733) with T read from `farz_beans_v2_labels.json`; PyTorch on MPS (MPS vs CPU max prob diff 2.4e-6 on 512 crops). Test crops: all 23,447 CBD crops + 3,000 random training crops (seed 1). ONNX run with onnxruntime 1.30.0 CPU. `onnx.checker` passes on both files; opset 17; input `image` float32 [n,3,128,128]; output `probs` float32 [n,6]; rows sum to 1 (max dev 2.4e-7).

| File | Bytes (sha256 prefix) | max abs prob diff vs PyTorch (CBD / train) | argmax agreement (CBD) | good/defect/unsure call agreement at 0.66/0.89 (CBD) | Verdict |
|---|---|---|---|---|---|
| `farz_beans_v2_fp32.onnx` | 6,108,756 (d7d2deef) | 4.8e-6 / 3.3e-6 | 100% | 100% | CONFIRMED (report: 3.7e-6) |
| `farz_beans_v2_fp16.onnx` (shipped candidate) | 3,084,405 (b486537c) | **0.0175** / 0.0120 | **99.31%** | **99.20%** | CONFIRMED (report: 99.31% / 99.20%) |
| fp16 vs fp32 ONNX directly, CBD | | 0.0175 | 99.31% | 99.20% | CONFIRMED |

This also confirms the ONNX files are the `final` weights with T = 1.558 baked in (not a fold model). 
**Browser path (new, not in the v2 report):** onnxruntime-web 1.30.0 (`app/node_modules`, read only), wasm, 1 thread, Node, 300 random CBD crops: fp16 wasm vs Python ORT fp16 max abs prob diff **7.4e-4**, 300/300 identical argmax and 3-way calls; fp32 1.5e-6. So the fp16 file runs in the app's runtime. Not measured: iPhone Safari, Android Chrome.
Caveat: the fp16 file moves 0.8% of CBD bean calls across a threshold relative to fp32; v1's own browser rule (`reports/beans_model.json`, max abs prob diff <= 0.01) would **reject** fp16 against fp32 (0.0175 > 0.01) if applied the same way. Size claim 3,084,405 vs 6,108,756 B: CONFIRMED.
INT8 claims (6 variants, 2.0-55.0% argmax agreement): UNVERIFIABLE — the INT8 files were deleted; only the JSON remains.

## (5) Licences (re-checked by the auditor 01:50-02:05 BST from the live source pages / public APIs, no login)

Method: Kaggle `GET https://www.kaggle.com/api/v1/datasets/view/<owner>/<slug>` (public, no auth; field `licenseName`), GitHub `GET api.github.com/repos/<repo>` (`license`) + root listing, Hugging Face `GET huggingface.co/api/datasets/<id>` (`cardData.license`) + raw README, Mendeley `GET data.mendeley.com/public-api/datasets/52877z55vr` (`data_licence`). Raw responses saved in /tmp/audit_v2/kg_*.json.

| Source | Licence found by auditor | Matches report? | NC? | ND? | Note |
|---|---|---|---|---|---|
| j4ckdev | GitHub repo file `LICENSE-CC-BY-NC-SA` = "Attribution-NonCommercial-ShareAlike 4.0 International" (GitHub API spdx NOASSERTION); Kaggle j4ckdev/green-coffee-beans-dataset `CC BY-NC-SA 4.0` | CONFIRMED | **yes** | no | **ShareAlike**: a model trained on it is arguably an adapted work -> must be released under BY-NC-SA if distributed |
| mfu17 (sujitraarw/coffee-green-bean-with-17-defects-original) | `Unknown`; description asks to cite doi 10.1016/j.atech.2024.100680; no licence text | CONFIRMED | unknown | unknown | default all-rights-reserved |
| usk_coffee (mfaisalriftiarrasyid/duardata) | Kaggle mirror `Unknown`, empty description. Authors' page coffee.comvislab-usk.org: no licence, "please cite", download via Google Form (forms.gle/...) and README (notepad.pw) that also states no licence | CONFIRMED | unknown | unknown | **Third-party mirror of a form-gated dataset: redistribution right of the mirror is unverifiable** |
| vicanadya16 (vicanadya/coffee-defect-16-classes) | `CC0: Public Domain`, no description/provenance | CONFIRMED | no | no | provenance unknown: CC0 from an anonymous uploader cannot be verified as the rights holder's grant |
| loja_yolo (cristianyagg/...loja) | Kaggle `Apache 2.0` (download README says CC BY 4.0) | CONFIRMED (both permissive) | no | no | |
| lojano (patopucho/lojano-arabica-coffee) | `Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)` | CONFIRMED | **yes** | no | |
| afiyah_deteksi / afiyah_bijikopi | `CC0: Public Domain` both, empty description | CONFIRMED | no | no | same uploader, same phone (EXIF samsung Galaxy A05s in 40/40 sampled files of each) -> see leakage (1) |
| samruddh_grading (HF SamruddhK/coffee-bean-grading-dataset) | `license: mit` (card + API tag), not gated | CONFIRMED | no | no | card says Coorg, Karnataka, India; card contradicts itself Arabica vs Robusta (as reported) |
| daffa_defect (GitHub daffakurnia11/...) | API `license: null`, no LICENSE file in root | CONFIRMED | unknown | unknown | default all-rights-reserved |
| mindforge_doubleside (GitHub Mindforge-inc/...) | API `license: null`; root = 1st, 2nd, README.md; README (Korean) states no licence | CONFIRMED | unknown | unknown | default all-rights-reserved |
| notplying_defects (GitHub Notplying/...) | API `license: null`; no LICENSE file | CONFIRMED | unknown | unknown | default all-rights-reserved |
| CBD (Mendeley 52877z55vr v1) | `CC BY 4.0` (API data_licence), "collected from our own setup" | CONFIRMED | no | no | test only |

**No dataset is No-Derivatives (ND)**, so no source is forbidden for training on ND grounds. CONFIRMED.
Flags: 2 NC sources (j4ckdev BY-NC-SA, lojano BY-NC) and 5 with no licence (mfu17, usk_coffee, daffa, mindforge, notplying) are inside the shipped weights -> the v2 weights may only be described as a **non-commercial research prototype**; the BY-NC-SA ShareAlike term was not mentioned in reports_v2 (MISSING). USK is a third-party re-upload of a form-gated set (MISSING from reports_v2).

## (6) Claims in reports_v2/*.md (and the hand-over text) checked against auditor numbers

Extra auditor measurements used here: CBD ranking (/tmp/audit_v2/cbd_rank.py, my own bean-finder crops), ECE and licensed-only re-fit (inline), USK fold retrained with a **second seed** (seed 1, retrain_afiyah.py usk_repeat), v1 under the 22:40 rules (inline), CSV dedupe counts, decoded-pixel MD5 of the 475 photos.

| # | Claim (where) | Auditor number | Verdict |
|---|---|---|---|
| 1 | 118,143 crops from 12 sources; v2 data 117,561 kept (model_v2.md §1, DATA_SOURCES_V2) | 118,143 rows = 582 + 117,561 keep==1 | CONFIRMED |
| 2 | other_defect 4,572 typed; 4,871 untyped | 4,572 / 4,871 | CONFIRMED |
| 3 | CBD never used for training or fitting | no CBD source in any training pool; no CBD crop has a training near-duplicate (max cos < 0.98) | CONFIRMED |
| 4 | Source-balanced sampling | `Sampler`: kind -> source uniform -> crop | CONFIRMED (code read) |
| 5 | T and thresholds only fitted on other sources | true for every per-source row; the **pooled** 53.8%@90.5% and 68.9%@90.1% are in-sample for the threshold choice | CONFIRMED with caveat |
| 6 | "good-vs-defect balanced accuracy averages 0.78 across the 12 held-out sources" (headline) | 0.781 only because 3 sources with no good beans score ~1.0 by calling everything defect; 0.721 over the 9 with good beans; **0.657** after replacing the two afiyah rows with the both-held-out retrain (0.588, 0.703) | **WRONG as a headline** (arithmetic right, meaning wrong). Say "about 0.66-0.72 on sources that have good beans; near chance (0.53-0.57) on USK and daffa" |
| 7 | v2 "less wrong than v1 on new data" | LOPO 27.4% -> 42.9% (confirmed); CBD ranking flips sign (confirmed). But on new sources good recall median 0.505 and USK good beans called defect 37.5% (seed 0) / 27.2% (seed 1) | CONFIRMED with caveat |
| 8 | J4ckDev LOPO 42.9%; Normales 74% insect, 84% answered, all wrong | 42.89%; 73.8%; 84.1%, 0% correct | CONFIRMED |
| 9 | CBD Spearman +0.64 vs v1 -0.48; AUROC AAA vs Bits 1.00 vs 0.07; C 0.98; PB-I 0.99 | +0.642 / -0.479; 1.000 / 0.065; 0.980; 0.985 | CONFIRMED. Caveat: PB-I/PB-II are **peaberry** (whole beans, a shape grade) and v2 calls them defective as often as grade C (median photo defect share 64% / 62% vs C 60%); AAA/AA/A are indistinguishable (36% / 36% / 35%). The +0.64 rests on Bits/C/PB, not on separating top grades |
| 10 | v2 calls 62.5% of AAA beans good (no abstention) vs v1 0.08%; median AAA photo 36% defects | 62.53% vs 0.08% (both argmax, like for like); 36.4% | CONFIRMED |
| 11 | Thresholds A 0.89 / P 0.66+0.89; per-source failures 5 / 6; USK 36% / 49% | identical | CONFIRMED. Seed sensitivity (new): USK fold with seed 1 -> answered acc 58.7% (seed 0: 49.4%), balanced 0.564 (0.532); per-bean argmax agreement between the two seeds only **74.4%** |
| 12 | ECE pooled 0.069 after T, 0.103 before; per source 0.018-0.419 | 0.0688 / 0.1031; 0.018-0.420 | CONFIRMED. Note: T = 1.56 makes ECE **worse** on 5 of 12 sources (loja_yolo, both afiyah, notplying, vicanadya16) |
| 13 | fp16 3,084,405 B vs fp32 6,108,756 B; 99.31% argmax / 99.20% 3-way agreement | identical (see (4)) | CONFIRMED |
| 14 | INT8: six attempts, 2.0-55.0% agreement | files deleted | UNVERIFIABLE |
| 15 | Model trained only on sources that state a licence: 0.59-1.00 vs 0.53-1.00 on the 5 unlicensed sources | 0.590, 0.936, 0.693, 0.998, 0.568 | CONFIRMED numbers; **"licence-clean" is a misnomer**: its 7 sources include 2 non-commercial (J4ckDev BY-NC-SA, lojano BY-NC) and 3 CC0 uploads with no provenance. Call it "licence-stated" |
| 16 | Deploy rule: 0 wrong bands on 3,427 held-out photos/trays | held-out thresholds: 0 wrong in 13,200 auditor trays; deployed thresholds: 2 wrong (non-dangerous) in 13,200 | CONFIRMED with caveat |
| 17 | 0/50 CBD AAA "many"; 0/464 CBD band | 0/50; 0/464 (even ignoring all retake gates) | CONFIRMED |
| 18 | 0/777 stress (v1 let 1 through) | v2 0/777 on auditor perturbations; v1 also 0/777 on mine | v2 CONFIRMED; v1 "1/777" UNVERIFIABLE (perturbation-specific) |
| 19 | 40 bands (1.2%), all correct, all afiyah | only afiyah ever banded in my trays too; **both afiyah held out -> 0 bands** | CONFIRMED as stated, **MISLEADING**: the 40 bands are evidence of the afiyah twin leak, not of usefulness on a new source. The report never says the two afiyah sets are one uploader/phone/backdrop |
| 20 | 82.9% "not sure", 15.9% retakes | depends on their tray set; not re-run exactly | UNVERIFIABLE |
| 21 | No band on any of the 1,030 real held-out photos | CBD 464, J4ckDev 11, lojano 36, loja_yolo 491 photos: 0 bands (after gates) | CONFIRMED for 1,002 photos; notplying 204 not re-run |
| 22 | App's 7 demo photos all "not sure" under v2 | 7/7 | CONFIRMED |
| 23 | v1 22:40 rules: 163/464 CBD bands, 27/50 AAA "many" (robust_summary) | 163/464 all "many"; AAA 27/50 | CONFIRMED |
| 24 | v1 23:05 rules: 0/464 CBD band (robust_summary) | 0/464 (all implausible) | CONFIRMED |
| 25 | v1 23:05 rules: 30/30 held-out J4ckDev-style trays correct (robust_summary; used for the freeze) | not re-run with their paste recipe; crop-level trays from the same held-out halves: v1 **10 of 400 zero-defect trays called "many"**, 57 wrong of 793 bands | **WRONG as evidence that v1 is safe/accurate in its own set-up** (30 trays, 10 clean, under-powered) |
| 26 | "v2 is not more accurate than v1 in that set-up: even recalibrated it got 22/60 right, 38 wrong, 2 dangerous" (freeze reasoning) | that figure is **v2 with local re-calibration**, not the proposed v2 rule. Deployed v2 rule on the same held-out halves: 229 bands, **0 wrong, 0 dangerous** vs v1 57 wrong, 10 dangerous | **WRONG** comparison (apples to oranges). True statement: v2 answers far less often (19% vs 66% of trays) but was never confidently wrong there |
| 27 | Dedupe: 475/475 photos, 24,029/24,029 crops unique; J4ckDev crops re-cut pixel-identical | 475/475 decoded-pixel MD5 unique; 0 duplicate CBD crops; 582/582 identical | CONFIRMED |
| 28 | Within-source drops "79 exact + 54 near" | CSV rows with `dup_of` and keep==0: **135** (= the per-source table's own sum 17+14+31+33+1+30+1+2+6) | minor inconsistency (133 vs 135) |
| 29 | 5 exact duplicates with conflicting labels (1 mfu17, 4 vicanadya16) | 1 / 4 | CONFIRMED |
| 30 | loja_yolo "CC BY 4.0 in README.roboflow.txt" | the line `License: CC BY 4.0` is in `README.dataset.txt` | trivial location error |
| 31 | Robust study colour baseline (0.712 vs 0.421), unusual-bean detector AUROCs, OOD-gate table rows other than 23/24 | not re-run | UNVERIFIABLE (not shipped, so not on the safety path) |
| 32 | model_v2.md "STATUS: COMPLETE, 01:50 BST" | file mtime 01:43:15 BST | timestamp written ahead of time (cosmetic) |
| 33 | "Re-test the browser path: wasm parity of fp16" listed as a to-do | auditor: wasm fp16 vs Python max diff 7.4e-4, 300/300 identical calls | now MEASURED (Node, not a phone) |

## WRONG items (summary)
1. **Headline "0.78 balanced accuracy on new sources"** — 3 of 12 sources have no good beans and score ~1.0 by calling everything defect. 0.72 over the 9 sources with good beans; **0.66** once the afiyah twin leak is removed (both held out: 0.59 and 0.70 instead of 0.88 and 1.00).
2. **"40 bands, all correct" used as usefulness evidence** — all 40 come from afiyah_deteksi/afiyah_bijikopi. These are one uploader, one phone model (Galaxy A05s) and one backdrop, so each was "held out" with its twin still in training. With both held out: **0 bands in 1,400 trays**, and bijikopi good beans called good drop from 80.7% to 1.0%. The report never says the two sets are related.
3. **Freeze reasoning "v2 is not more accurate than v1 in the J4ckDev set-up (22/60 right, 38 wrong, 2 dangerous)"** — those figures are for v2 *with local re-calibration*, not for the proposed v2 rule. Same held-out halves, 1,200 crop-level trays at the report's own compositions: the v2 deploy rule gave **229 bands, 0 wrong, 0 dangerous**; v1 gave **793 bands, 57 wrong and 10 dangerous** (a zero-defect tray called "many").
4. **"v1 + 23:05 rules: 30/30 held-out trays correct" cited as evidence that v1 is accurate or safe** — only 30 trays, 10 of them clean. On one held-out half of Normales, v1 calls 22% of the good beans it answers defective, so clean trays flip to "many" whenever the unsure rule does not fire.
Minor: the model trained only on sources that state a licence still includes two non-commercial sources (call it "licence-stated"); dedupe count 133 vs 135; loja_yolo licence sits in a different README file; model_v2.md status timestamp is ahead of its file time; the J4ckDev ShareAlike obligation and the USK third-party mirror are not mentioned.

## Verdict
**v2 is safe to ship under the Responsible-AI pass/fail rule, but only because it refuses, not because it is accurate.** I tested the proposed rule: fp16 file, P(good) ≥ 0.66 → good, P(defect) ≥ 0.89 → defect, otherwise unsure, then the app's current 60% / 15% photo rules. On genuinely new data it gave **0 confident bands** and so **0 confident wrong answers**: 464 real CBD photos (0/50 AAA "many", even with every retake gate ignored), 777 software re-shoots, 36 lojano, 11 J4ckDev LOPO and 491 loja_yolo photos, 13,200 trays from the 10 non-afiyah held-out sources, and the afiyah-both retrain. In J4ckDev's own set-up with unseen beans it answered 19% of trays and was never wrong, while v1 was confidently wrong on clean trays. Per bean, v2 is often confidently wrong on a new farm: USK good beans called defective at ≥ 0.87 confidence 37% of the time (27% with another seed), J4ckDev Normales 84%, and whole peaberries called defective as often as grade-C beans. The 15% and 60% photo rules are what turn that into a signpost to the cooperative. v1 is also silent on CBD under those rules (0/464), so neither model is "the" safety mechanism. The fp16 file matches PyTorch (99.2% identical calls) and runs in onnxruntime-web wasm (300/300 identical calls). The app changes it needs (6 classes, two-threshold `callBean`) have not been run in the app or on an iPhone, and every demo photo would say "not sure". The freeze decision is the team's, but it must not rest on WRONG items 3 and 4.

**The README and videos may claim:**
- trained on 118,143 bean crops from 12 public datasets, tested leave-one-source-out;
- on sources that contain good beans, good-vs-defect balanced accuracy is **about 0.66–0.72**, near chance (0.53–0.57) on two of them, so per-bean calls on a new farm are not reliable;
- on real India photos, v2 ranks broken bits above AAA (AUROC 1.00; v1 0.07) but cannot tell AAA from AA from A;
- with Farz's fixed photo rules, **0 of 464 real CBD photos, 0 of 777 re-shoots and 0 of about 1,000 real held-out photos** got a confident band, and none of the 50 AAA photos was called "many", so Farz says "not sure — take it to the cooperative" instead of guessing;
- what a pilot needs is local labelled photos;
- the model is a **non-commercial research prototype**: it includes 2 non-commercial sources (one ShareAlike) and 5 that state no licence.

**They must not claim:**
- "0.78 / 78% accurate on new farms", "works on a new farm", any accuracy on Yemeni beans or phone photos;
- the 40 correct bands as proof of usefulness;
- "v1 30/30 correct" as proof that v1 is safe;
- "v2 is less accurate than v1";
- defect types or counts as facts;
- "licence-clean".
