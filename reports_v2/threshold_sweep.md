PRE-REGISTERED 2026-10-04 04:53 BST (Sun), before any sweep number was computed. The rule below is fixed; results are appended under "## Results" by `src/beans_v2/threshold_sweep.py`, and the block above that heading is hashed into `reports_v2/threshold_sweep.json`.

# Farz: "sound" threshold sweep (t_good), shipped model, shipped temperature

## Pre-registration (fixed before any result)

**Why.** Judge B (World Bank) and REGRESSION_B noted that the shipped `t_good` = 0.50 sits at the floor of the threshold grid, so a bean with P(good) >= 0.5 counts as sound with no abstention margin, and that with the shipped thresholds the held-out samruddh trays got 11 wrong bands out of 42. This sweep picks `t_good` by a rule written down now.

**Candidates.** `t_good` in {0.50, 0.55, 0.60, 0.65, 0.70, 0.75}. `t_defect` is fixed at the shipped **0.91**. Temperature is the shipped **T = 2.0459** (`clean_eval.json → deployed.temperature`). Nothing is refit: no temperature, no `t_defect`, no model.

**Per-bean call (the app's `callBean`).** Non-touching crops only. good if P(good) >= t_good; else defect if 1 - P(good) >= 0.91; else unsure.

**Photo rule (the app's `decide()`, via its Python mirror `src/beans/pipeline.decide`, unchanged).** Nothing answered → not sure; > 60% of answered beans defective → not sure (implausible); > 15% of beans unsure → not sure; otherwise the band of the point estimate (clean < 5% <= some <= 20% < many). An "about" band counts as a band.

**Tray sets (same trays for every candidate; common random numbers).**
- **A. Licensed sources, held out.** Each of the 6 licensed groups (loja_yolo, lojano, afiyah = afiyah_deteksi + afiyah_bijikopi held out together, j4ckdev, samruddh_grading, vicanadya16) scored by its own leave-one-source-out fold model `clean_loso_<group>` (the fold never saw that group). Logits from `$FARZ_SCRATCH/v2train/folds/clean_loso_<group>.npz`, re-tempered with the shipped T.
- **B. The 5 unlicensed test-only sources** (mfu17, usk_coffee, daffa_defect, mindforge_doubleside, notplying_defects) scored by the shipped model (`licensed.pt` logits, `clean_emb_final_unlicensed.npz`).
- For A and B: tray sizes **50 and 100 beans**; defect shares {0, 2, 10, 15, 30, 45, 100}% (the auditor's design), **200 trays per (size, share)**; beans drawn without replacement within a tray from that source's non-touching good / defect beans; a (size, share) cell is skipped when the source lacks the beans (sources with no good beans get only 100% trays). numpy `default_rng(20261004)`, drawn once in a fixed order (source, size, share).
- **C. The auditor's 1,200 J4ckDev trays** (AUDIT_V2 (3d), as re-run in `clean_simulate.py`): 80-bean trays from the spatially held-out J4ckDev half, 200 per (fold, truth) at 0 / 10 / 24 defects, `clean_half_1/2` models, `default_rng(2026)`, identical draws to `clean_simulate.py`.

**Truth band** of a tray = band of its true defect share (clean < 5% <= some <= 20% < many). **Wrong** = a band that differs from the truth band. **Dangerous** = a clean tray called "many" or a "many" tray called "clean".

**Units for criterion (b).** One unit per source: each of the 6 licensed groups (50- and 100-bean trays pooled), each of the 5 unlicensed sources (pooled), and the J4ckDev 1,200-tray set as its own unit (it uses different models). Per-size cells are reported but do not decide.

**Decision rule.** Choose the **smallest** `t_good` such that
- (a) dangerous trays = **0** in every unit (A, B and C together), and
- (b) wrong bands are **<= 5% of the bands given** in every unit that gets **>= 10 bands**.

If no candidate qualifies, choose **0.75** and say so. The chosen value is then shipped in the app (`rules.ts V2_THRESHOLDS.good` and `public/models/farz_beans_labels.json`), and the 464 CBD photos, the 7 demo trays and these held-out tray sets are re-measured.

**Not decided by this sweep.** CBD, the demo trays and the browser runs are re-measurements after the choice, not inputs to it.

## Results (appended 04:54 BST; the pre-registration block above is unchanged)

**Integrity.** The block above (everything before the blank line that precedes this heading, trailing newlines stripped) has SHA-256 `d8d95c87d9be8f733c99b71130098d8766d1a29a17bcb2a2e12c1d87f6305e39`. The same hash was written to a scratch file at 04:53:38, 10 s before the run started (04:53:48), and `threshold_sweep.py` recorded it in `threshold_sweep.json → prereg_block_sha256`. Run time 7 s. **Check:** at t_good = 0.50 the J4ckDev 1,200-tray unit gives 241 bands / 0 wrong / 0 dangerous, exactly `clean_deploy.json`, so the auditor's protocol is reproduced.

### Verdict: **t_good = 0.55** (smallest candidate meeting (a) and (b))

| t_good | dangerous (all units) | units failing (b) | bands given (all units) | wrong bands | qualifies |
|---|---|---|---|---|---|
| 0.50 (shipped until now) | 0 | **samruddh_grading: 13 wrong of 52 (25%)** | 293 | 13 | no |
| **0.55** | **0** | none (no unit has >= 10 bands with > 5% wrong) | 16 | 1 | **yes → chosen** |
| 0.60 | 0 | none | 2 | 0 | yes |
| 0.65 | 0 | none | 0 | 0 | yes |
| 0.70 | 0 | none | 0 | 0 | yes |
| 0.75 | 0 | none | 0 | 0 | yes |

### Bands / wrong per unit (50- and 100-bean trays pooled; J4ckDev 1,200 = 80-bean trays)

| Unit | model | trays | 0.50 | **0.55** | 0.60 | 0.65–0.75 |
|---|---|---|---|---|---|---|
| loja_yolo | clean_loso_loja_yolo | 2,800 | 0 | 0 | 0 | 0 |
| lojano | clean_loso_lojano | 2,800 | 0 | 0 | 0 | 0 |
| afiyah (both twins) | clean_loso_afiyah | 2,800 | 0 | 0 | 0 | 0 |
| j4ckdev | clean_loso_j4ckdev | 2,800 | 0 | 0 | 0 | 0 |
| samruddh_grading | clean_loso_samruddh_grading | 2,800 | 52 / **13 wrong** (50-bean 45/10, 100-bean 7/3) | **5 / 1 wrong** (a clean tray called "some") | 1 / 0 | 0 |
| vicanadya16 (100% trays only) | clean_loso_vicanadya16 | 400 | 0 | 0 | 0 | 0 |
| mfu17, notplying (100% trays only) | shipped | 400 each | 0 | 0 | 0 | 0 |
| usk_coffee | shipped | 2,800 | 0 | 0 | 0 | 0 |
| daffa_defect | shipped | 2,800 | 0 | 0 | 0 | 0 |
| mindforge_doubleside | shipped | 2,800 | 0 | 0 | 0 | 0 |
| J4ckDev 1,200 (auditor) | clean_half_1/2 | 1,200 | 241 / 0 (40 clean, 85 some, 116 many, all right) | **11 / 0 (all "many" trays called "many")** | 1 / 0 | 0 |

Dangerous trays (clean → many, many → clean): **0 at every candidate in every unit.**

### What the choice costs (read this before quoting it)
- **The app almost never gives a band on held-out data now.** At 0.55, 16 of 24,800 trays get a band (0.06%); at 0.50 it was 293 (1.2%). On the J4ckDev 1,200 set the bands drop from 241 to 11, and **no clean or "some" tray gets a band at all** – only 11 of the 400 "many" trays. Raising t_good turns more beans "unsure", so the 15% unsure rule and the 60% implausible rule fire more often.
- **What it buys:** the only source with >= 10 bands at 0.50 (samruddh) had 25% wrong bands (one band off, never dangerous). At 0.55 the wrong-band count on held-out trays falls from 13 to 1 **in this one draw**. Over 20 further tray draws (`threshold_seeds.json`) samruddh still gets 26 wrong of 105 bands at 0.55 (24.8%, all one band off) vs 217 of 993 at 0.50 (21.9%): 0.55 mostly makes it answer less often there, not more accurately. 0 dangerous trays in every draw.
- **The sweep could not tell 0.55 from 0.60–0.75 on safety** (all have 0 dangerous); the pre-registered rule picks the smallest, which keeps the most answers.
- Same limits as `clean_model.md`: crop trays, not new photos; one seed per fold; no Yemeni beans.

Re-measurements after shipping 0.55 (CBD 464, the 7 demo trays, browser) are in `reports_v2/threshold_ship.md`.
