# Farz: data card

The brief (§7.2) asks for two kinds of data:
1. data that shows the problem;
2. data we build with, each named with its source, licence and size, plus "what your data does not cover" (scored).

The problem evidence is in `docs/EVIDENCE.md`. This file covers the build data. The full report on the 12 bean datasets covers licences as stated, class mapping, crop making and de-duplication. It is `docs/DATA_SOURCES_V2.md`, and the licences were re-checked by an independent audit (`reports_v2/AUDIT_V2.md` (5)).

## 1. Build data actually used

### 1.1 Training data of the shipped classifier: only datasets that state a licence

Crop counts are the shipped model's training pool, read from its sampler log (`reports_v2/clean_eval.json → final_weights_train_info`). "Sound" = crops labelled good.

| Dataset | Licence (as stated by the source) | Kept crops (sound) | What it is | URL |
|---|---|---|---|---|
| **J4ckDev Green Coffee Beans** (2022) | **CC BY-NC-SA 4.0** (licence file in the download) | 582 (195) | 11 tray photos, one per defect type, light-grey sheet; README states 642 beans, the bean finder extracts 582 complete beans (§3). Spanish-language thesis; country UNVERIFIED | https://github.com/J4ckDev/GreenCoffeeBeansDataset |
| **vicanadya16** | CC0 (Kaggle). No description, no provenance: a CC0 grant from an anonymous uploader cannot be verified | 86,123 (0) | 16 SCA defect classes, tight crops cut from crowded trays; **no sound beans** | https://www.kaggle.com/datasets/vicanadya/coffee-defect-16-classes |
| **loja_yolo** | CC BY 4.0 in the download's `README.dataset.txt` (Kaggle page: Apache 2.0) | 1,311 (146) | Loja province, Ecuador; tray photos with a polygon per bean | https://www.kaggle.com/datasets/cristianyagg/defectos-grano-de-cafe-de-la-provincia-de-loja |
| **lojano** | **CC BY-NC 4.0** | 1,182 (598) | Loja, Ecuador; good vs defective tray photos | https://www.kaggle.com/datasets/patopucho/lojano-arabica-coffee |
| **samruddh_grading** | MIT (Hugging Face card) | 1,337 (467) | Coorg, India; single beans; grade A used as sound, grade D as defective | https://huggingface.co/datasets/SamruddhK/coffee-bean-grading-dataset |
| **afiyah_deteksi** + **afiyah_bijikopi** | CC0 (Kaggle), no description | 985 (658) + 1,170 (579) | Single beans on white paper. **One uploader, one phone model (Galaxy A05s), one backdrop**, so they are held out together in every test | https://www.kaggle.com/datasets/afiyahmusrah/deteksi-biji-kopi · https://www.kaggle.com/datasets/afiyahmusrah/bijikopideteksi |
| **Total** | | **92,690 (2,643)** | | |

The two non-commercial licences (J4ckDev, ShareAlike; lojano) make the shipped weights a **non-commercial, ShareAlike research prototype**.

### 1.2 Test-only data (never trained on)

| Dataset | Licence | Use | Redistributed? | URL |
|---|---|---|---|---|
| mfu17 (Thailand, 17 defect classes, no sound beans) | **none stated** (Kaggle "Unknown") | held-out test of the shipped model | no | https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original |
| usk_coffee (Kaggle mirror of USK-Coffee, Banda Aceh, Indonesia) | **none stated**; the mirror is a third-party re-upload of a form-gated set | held-out test; calibration study (secondary) | no | https://www.kaggle.com/datasets/mfaisalriftiarrasyid/duardata |
| daffa_defect | **none stated** (no LICENSE file) | held-out test | no | https://github.com/daffakurnia11/Coffee-Beans-Defect-Classification |
| mindforge_doubleside | **none stated** | held-out test; calibration study (secondary) | no | https://github.com/Mindforge-inc/GreenCoffeeBeanDoubleSide |
| notplying_defects (no sound beans) | **none stated** | held-out test | no | https://github.com/Notplying/DatasetCoffeeBeanDefects |
| **CBD Coffee Bean Dataset** (Mendeley 52877z55vr, 2024; *Data in Brief* 2025): 464 light-box photos, 50 beans each, 9 size grades, Wayanad, Kerala, **India** | CC BY 4.0 | bean-finder count test; photo-rule safety test; grade-ranking check. No per-bean labels, so never used to train, fit thresholds or measure per-bean accuracy | 2 demo photos (below) | https://data.mendeley.com/datasets/52877z55vr/1 (329,131,973-byte zip, measured) |

With no licence stated, the default is "all rights reserved". So the five unlicensed sets are used only to measure the model, they appear in **no training pool** of the shipped model (every fold's training log checked, `reports_v2/clean_model.md` §1), and none of their images is in this repository.

### 1.3 In the app

| Asset | Use | Licence | Size |
|---|---|---|---|
| **SYNTHETIC: 5 demo trays** `app/public/demo/tray_synthetic_00.jpg`, `_03`, `_12`, `_30`, `_40` (0, ~3, ~12, ~30 and ~40% defects) | demo button only: shows the flow, **not** accuracy. About 100 J4ckDev bean crops per tray pasted on a grey sheet (`src/beans/make_demo.py`, `make_demo_extra.py`). The shipped model was trained on J4ckDev, so these are **in-sample**. Outcomes in the built app: clean, about clean, some, about many, many (`app/reports/browser_measure.json → demo`). Labelled "synthetic" on screen | CC BY-NC-SA 4.0 (from J4ckDev) | 145,283 / 147,248 / 148,115 / 149,938 / 151,966 bytes |
| CBD demo photos `app/public/demo/cbd_aaa_2.jpg`, `cbd_bits_262.jpg` | demo button: real photos from a dataset Farz never trained on; both give "not sure" | CC BY 4.0 (credit shown in the app) | 822,703 / 557,748 bytes |
| Calibration demo `app/public/calib-demo/` | the quality-check demo: 20 labelled loja_yolo crops (10 sound, 10 defect); the dataset's labels stand in for a grader's, and the screen says so. Also 3 loja_yolo photos in `test/`, resized from 6000×4000 to 2400×1600 (the change is stated in `CREDITS.md`). **In-sample** for the shipped model | CC BY 4.0, credit in `calib-demo/CREDITS.md` | photos 356,563 / 336,104 / 332,959 bytes |
| **Voice clips** (`app/public/audio/`, lines in `docs/CLIPS.md`) | **none in the default build**: the app shows every message on screen and says "Voice coming soon". Recordings in Yemeni Arabic, added under the 19 file names listed in `app/public/audio/README.md`, switch the voice on. Text-to-speech placeholders (macOS voice "Majed", Modern Standard Arabic) exist for local testing only and are not in the public repo | — | 0 audio files in the default build |
| ElevenLabs | **not used in this build**. If a final clip is generated with it at build time, a native speaker checks each clip; the app never calls ElevenLabs | ElevenLabs terms for generated audio: UNVERIFIED | — |
| Own photos for device tests (hand, roasted or green beans) | refusal checks on the phone; video footage | own | not in the repo |

## 2. Data for the second "country pack" (leaf pipeline: smoke test only, not in the Yemen app)

| Dataset | Country / conditions | Licence | Size (measured download) | Use | URL |
|---|---|---|---|---|---|
| BRACOL (Mendeley yy2k5y8mxg, 2019) | Brazil; detached leaves on a white background | CC BY 4.0 | 164,516,964 bytes. **The archive is broken at the source**: 1,402 of 1,747 leaf images recovered | 1,342 images in the leaf manifest | https://data.mendeley.com/public-files/datasets/yy2k5y8mxg/files/c16b08ee-3ca6-4bf0-8f4e-4285a53a4a24/file_downloaded |
| JMuBEN (Mendeley t2r6rszp5c, 2021) | Kenya; field, 128 px augmented crops | CC BY 4.0 | 245,777,719 bytes (+ 69,450,673-byte rust zip) | 784 images after de-duplication (see §4) | Mendeley t2r6rszp5c; paper https://pmc.ncbi.nlm.nih.gov/articles/PMC8165403/ ; rust file https://data.mendeley.com/public-files/datasets/t2r6rszp5c/files/8c7c2915-f979-43f6-b3fd-b3bc7407da87/file_downloaded |
| Uganda coffee leaf (Mendeley k36wnd6knb, 2025, Soroti University) | Uganda; field close-ups | CC BY 4.0 | 26,024,910 bytes | 1,809 images; held out as an unseen country | https://data.mendeley.com/public-api/zip/k36wnd6knb/download/1 |

Downloaded but **not used in any reported result**: JMuBEN2 (475,463,203 bytes), RoCoLe (2,245,588,288 bytes), DECAFIA / CoffeeLeaf-CO (208,325,327 bytes).

## 3. How the training data is prepared

- **One crop contract for everything.** Every source ends as 128 px crops on a 235-grey sheet, padded by 18% of the bean's longer side, the same crop the app makes.
  - Tray photos (J4ckDev, loja_yolo, lojano) go through the frozen bean finder (`src/beans/beanfinder.py`) exactly as the app runs it.
  - Other sources use the modes in `docs/DATA_SOURCES_V2.md` §4. Each crop records its mode.
- **De-duplication before any split.** Pixel MD5 of source images and crops, then near-duplicates (dot product ≥ 0.998 of 32×32 grey thumbnails), within and across sources (`DATA_SOURCES_V2.md` §5).
  - Cross-dataset exact duplicates: **0**.
  - The independent audit found no cross-source near-duplicate either (embedding cosine ≥ 0.99: 0 pairs; `AUDIT_V2.md` (1)).
- **Splits are by dataset, never by row.** Every accuracy number for the shipped model holds out a whole dataset. The two afiyah sets are held out **together**, because the audit showed they are one capture set-up.
- **Class mapping.** Source labels → good / dark / insect / broken / unhulled / other defect (`DATA_SOURCES_V2.md` §3; `src/beans_v2/data_sources.py`). Untyped "defect" labels go to other defect, and the trainer treats them with a partial-label loss.
- **J4ckDev photo → class** (`src/beans/common.py`):

| Farz class | J4ckDev photos (README count) | Crops |
|---|---|---|
| good | Normales (README 255; **finder 195**) | 195 |
| dark | Negros (100), MarronAVinagre: full/partial sour (36), DXHongo: fungus (12) | 148 |
| insect | BrocadoLeve: slight insect damage (30), BrocadoSevero: severe (25) | 55 |
| broken | PMordidoCortado: broken/chipped/cut (36), Inmaduro: immature (16), Concha: shell (33) | 85 |
| unhulled | Pergamino: parchment (90), CerezaSeca: dried cherry (9) | 99 |
| **Total** | 642 stated | **582** |

- **The Normales gap.** The bean finder counts 195 beans where the README says 255. No detected bean touches the frame edge, there are no specks, and a visual check (by two reviewers) shows about 195 beans, all boxed. The gap is in the README; this is not a hand count. All other 10 photos match the README exactly.
- **The majority baseline.** Of the 387 beans in the 10 J4ckDev defect photos, 148 are "dark" (38.2%). Always guessing "dark" therefore scores 38.2% on the leave-one-photo-out test. Both v1 (27.4%) and the shipped model (30.0%) score below it (`reports/beans_lopo.json`; `reports_v2/clean_model.json`).

## 4. Data-integrity findings

**JMuBEN is mostly duplicates.** We hashed every file in the public JMuBEN coffee-leaf dataset (measured, EVIDENCE M3):

| Folder | Files | Byte-unique | Pixel-unique |
|---|---|---|---|
| Cercospora | 7,681 | 322 | **82** |
| Leaf rust | 8,336 | 1,042 | — |
| Phoma | 6,571 | 691 | — |

So the Cercospora folder of one public coffee-leaf dataset holds 7,681 images but only 82 distinct pictures. A random train/test split on it would put copies of the same image on both sides and inflate accuracy. We de-duplicate by hash before any split (`src/prep.py`). Uganda "healthy" also shrinks, from 1,179 to 740 byte-unique files. 102 Uganda files cannot be decoded and are dropped.

**Two bean datasets are one set-up.** afiyah_deteksi and afiyah_bijikopi share an uploader, a phone model and a backdrop. Held out one at a time, each kept its twin in training, and that "leak" produced every band the earlier v2 report counted as usefulness. With both held out, the earlier model gave 0 bands in 1,400 trays (`AUDIT_V2.md` (1), (3b)).

**Exact duplicates with conflicting labels.** 5 cases: 1 in mfu17, 4 in vicanadya16. The first copy is kept (`DATA_SOURCES_V2.md` §5).

## 5. What our data does not cover (scored by the brief)

| Gap | Why it matters | What we do about it |
|---|---|---|
| **No Yemeni beans.** None of the four main varieties named by UNDP (Udaini, Dawairi, Tufahi, Bura'ai), and no natural-process (sun-dried-in-the-cherry) beans | Yemeni naturals may look different from every training set and be called defective, **under-grading the farmer's own coffee** | Counts not grades; no types as facts; colour gate; "?" per bean; the 15% and 60% rules; "not sure" → the cooperative grader; calibration pilot for the grader |
| **Per-bean calls do not transfer between set-ups.** Sound vs defect on a held-out dataset: 0.644 balanced accuracy on average (0.556–0.801), 0.539 and 0.541 on two unlicensed test sets | A new farm's photos will be misread bean by bean | The photo rules send such samples to "not sure" (0 bands on 464 CBD photos and every real held-out set); local labels are the fix (calibration pilot, then a labelled pilot set) |
| **Sound beans are scarce**: 2,643 of 92,690 training crops, from 6 datasets; the biggest dataset (vicanadya16) has none | The model learns "defect" from far more examples than "sound", and calls many sound beans defective on new data | Source-balanced sampling; the defect threshold (0.91) is fitted on held-out data; the sound threshold, fitted at 0.50, was raised to 0.55 by a pre-registered sweep so that a bean needs a margin to count as sound (`reports_v2/threshold_sweep.md`) |
| **Licences limit what we can train on**: 5 of the 12 bean datasets we found state no licence; 2 of the 7 we use are non-commercial | The shipped model is a non-commercial prototype; the unlicensed sets can only be used to test | Train only on datasets that state a licence; a consented cooperative pilot could produce openly licensed Yemeni labels |
| **Studio-like photos only**: light boxes, single-bean close-ups, plain sheets | Studio-trained models can collapse on field photos (PlantVillage: 99.35% → 31.4%, EVIDENCE E51) | White balance from the sheet; augmentation; quality gate; abstention |
| **No rooftop sun, cheap Android cameras, sack backgrounds** | Unknown accuracy in Noor's real conditions | Desktop Chromium tests only so far; iPhone test pending (`docs/RESPONSIBLE_AI.md` §9); Android untested |
| **Hulled sample only; no whole lots.** In Yemen drying is "followed by selection, hulling" (UNDP p. 23 / PDF p. 37), so farmers sort before hulling | Farz needs a hulled sample; we have not measured how long hand-hulling 100 beans takes, or whether farmers will do it | Use it at the cooperative when the lot is hulled (some cooperatives have donor-provided processing machines, UNDP p. 68 / PDF p. 82), or hand-hull a sample at home |
| **Not every village has a cooperative.** UNDP recommends supporting "the creation of coffee cooperative organizations" (p. 46 / PDF p. 60) | "Take it to the cooperative" may have no one to send her to | A pilot must name another decision-maker (processing unit, association) or limit itself to cooperative members. Open |
| **The "insect" class may be the wrong pest.** UNVERIFIED reading: "broca" in J4ckDev and Loja is probably damage by the coffee berry borer, which EPPO does not list for Yemen; UNDP names the Coffee Berry Moth as Yemen's major pest (p. 39 / PDF p. 53) | Holed Yemeni beans may look different | No pest name is ever said; defect types are never stated as facts |
| **No moisture, density or taste defects** | Some quality problems are invisible to a camera | Farz never says "specialty" or "grade" |
| **No labels from Yemeni graders; no public Yemeni coffee price series found** (WFP's Yemen price data has no coffee, EVIDENCE E21) | No ground truth for Yemen, and no price reference | In a pilot, with farmers' consent, a grader's labels could become the first open set of Yemeni bean photos. The AI never quotes a price |
| **No yield diagnosis:** leaf disease and the berry moth are out of scope | Noor's "why did yields drop" question is not answered | Out of scope; Farz checks a bean sample only |
| **No data on Yemeni women's phones.** Findex 2025 excludes Yemen; we could not check GSMA's country list (gsma.com returned 403); MICS counts phones per household | We cannot show Noor owns a smartphone | Designed for the brief's two-phone household: a smartphone runs Farz, and Noor's own basic phone receives a plain SMS slip |
| **No voice yet; then one voice:** the default build is silent; once recorded, one speaker, one dialect, no female voice, no Tihami, Hadhrami, Mehri or Soqotri | Women who cannot read get no spoken result today; comprehension may vary by region | Record the message list; comprehension test with Yemeni speakers (not run yet). A new voice pack is a set of recordings, not a model |
| **Black and roasted beans: almost no real test data.** One real all-black photo (J4ckDev Negros); roasted beans and hands only as synthetic recolours | The dark-lot rule and the colour gate may misjudge a real dark lot or a light roast | Both outcomes give no band ("not sure" or "not green coffee"); test real roasted beans and hands on the phone |
| **No counting data from beans like Noor's.** The count test uses Indian light-box beans | Count accuracy on Yemeni beans on a cloth is unknown | Touching and count gates trigger a retake |

## 6. Provenance and reproducibility

- **Raw data is not in the repository.** The v1 sources take 6.3 GB (`du -sh data/raw/`, 4 Oct), and each dataset keeps its own licence. The v2 downloads are in `data/raw2/` and the crops in `data/crops_v2/` (both local only). Download URLs are above and in `DATA_SOURCES_V2.md` §2.
- **v2 crops:** `python src/beans_v2/data_build_crops.py --sources <names> --workers 8`, then `--merge`, then `python src/beans_v2/data_report.py`.
- **Shipped model and its evaluation:** `src/beans_v2/train_folds.py licensed`, then `clean_folds.py`, `clean_eval.py`, `clean_export.py`, `clean_simulate.py`, `clean_report.py` (`reports_v2/clean_model.md` §12).
- **Count test:** `.venv/bin/python src/beans/eval_count_cbd.py` → `reports/count_cbd.json`.
- **v1 history:** `src/beans/make_crops.py` → `train_beans.py` → `eval_lopo.py` / `eval_ood_cbd.py`. The two prediction files (`reports/beans_heldout_preds.npz`, `reports/beans_ood_cbd_preds.npz`) are included in the public repository.
