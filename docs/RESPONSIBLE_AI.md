# Farz: Responsible AI (the brief's pass/fail criterion)

The brief (p. 11) asks every entry to include a fail-safe where the AI points to "a decision-maker when the data it is acting on is not enough", e.g. "not sure — ask a person", and to give a credible account of privacy, consent, bias and human oversight. This file is that account.

**What it describes.**
- The build of 4 Oct 2026, 05:18–05:25 BST: sound threshold t_good 0.55, the dark-lot rule, the bean ledger, the quality-check calibration demo and no audio by default (`reports_v2/threshold_ship.md`).
- The shipped model is the v2 licence-stated classifier (`app/public/models/farz_beans.onnx`, 3,082,642 bytes; `docs/MODEL_CARD.md`).
- The rules are in `app/src/lib/rules.ts` and `touching.ts`.

**Where the desktop results come from.**
- The Playwright tests in `app/e2e/offline.spec.ts`: Chromium, Pixel 7 emulation, **14/14 pass**, 4 Oct 05:25 BST (`app/reports/e2e-results.json`, `e2e-measurements.json`). e2e test numbers below follow the order in that file.
- The built-app run on all 464 CBD photos (`app/reports/browser_measure.json`, 05:18 BST).
- Unit tests: vitest 126 passed, 5 opt-in measurements skipped (05:25 BST).

Nothing has been run on a phone yet. The device results go in the last column of §9.

---

## 1. Fail-safes: fixed rules trigger them, not the model

The model never decides whether to answer. Plain rules check the photo first. A second set of rules, run after the model, checks whether the answer is plausible and how firm it is. Each trigger shows one fixed message from `docs/CLIPS.md`, addressed to Noor in the feminine form. **The default build has no audio**: the messages are on screen, and Home says "Voice coming soon". Yemeni-Arabic recordings of the same messages, added as an audio pack, would play them (`app/public/audio/README.md`).

Retake checks, in the shipped order (`rules.ts gate()`):

| Trigger (rule) | Threshold (source) | Message | English meaning |
|---|---|---|---|
| Photo too dark | sheet brightness < 90 (`beanfinder.py`) | `c03_dark` «الصورة مظلمة. صوّري في مكان فيه نور.» | Too dark. Take it somewhere with light. |
| Too few beans (nothing touching, nothing off-colour) | fewer than 20 beans | `c05_count` «حطّي حوالي مية حبة، وصوّري ثاني.» | Use about 100 beans and retake. |
| Photo blurred | Laplacian variance < 15 (`beanfinder.py`) | `c02_blur` «الصورة مش واضحة. ثبّتي التلفون وصوّري ثاني.» | Photo not clear. Hold the phone still and take it again. |
| Not green coffee (hand, roasted beans, rice…) | colour check on the bean pixels: hue, saturation and dark-bean limits (`app/src/lib/colourgate.ts`). All 464 CBD photos and 9 of 11 J4ckDev photos pass (`app/reports/colourgate_all_photos.json`, 3 Oct 22:41; the gate does not use the model); hands, roasted beans and leaves tested only as synthetic images. A lot of black beans is the exception below | `c06_not_green` «هذا مش بن أخضر. صوّري البن بعد التقشير وقبل التحميص.» | This isn't green coffee. Photograph beans after hulling, before roasting. |
| Beans touching | `touching.ts` flags a blob that is > 1.8× the median blob, > 1.6× a one-bean area, or not bean-shaped (solidity < 0.8 or fill < 0.85). It fires when flagged blobs are more than max(2, 8% of blobs), or when more than max(2, 8%) of the beans are hidden in bean-coloured clumps. A big bean-coloured blob while many touch also gives this clip | `c04_spread` «الحبوب لاصقة ببعض. فرّقيها وصوّري ثاني.» | Beans are touching. Spread them out and retake. |
| Too many beans | count above 160 | `c05_count` | Use about 100 beans and retake. |

After the photo checks (the black-bean check runs before the model, the rest after it):

| Trigger (rule) | Threshold (source) | Message | English meaning |
|---|---|---|---|
| **A lot of black beans** (checked before the model) | the colour gate says "too dark beans", but ≥ 80% of the blobs look like single beans and the dark blobs have the colour of black green coffee (median hue 26–62°, saturation ≤ 0.42; `app/src/lib/darklot.ts`). Limits from one real photo (J4ckDev Negros) and synthetic roast | `c07_unsure` | **Not sure. Take the sample to the cooperative for a specialist to check.** |
| **Implausible sample** | more than 60% of answered beans called defective (`IMPLAUSIBLE_DEFECT_FRAC`) | `c07_unsure` | **Not sure. Take the sample to the cooperative for a specialist to check.** |
| **Model unsure on too many beans** | more than 15% of beans "?" (`MAX_UNSURE_FRAC`), or none answered | `c07_unsure` «مش متأكد. ودّي العينة للجمعية يشوفها المختص.» | **Not sure. Take the sample to the cooperative for a specialist to check.** |
| Borderline count (sampling, not the AI) | the 95% interval crosses a band edge | band marked "about" + `c18_another` «النتيجة قريبة. صوّري حفنة ثانية من نفس البن عشان نتأكد.» | The result is close. Photograph another handful from the same coffee so we can be sure. |
| Single uncertain bean | P(good) < 0.55 and P(defect) < 0.91, or touching | yellow "?" on that bean, no message | — |

**What these rules achieve, measured on the shipped model.**

| Set | Confident bands |
|---|---|
| 464 real CBD photos from India, built app in Chromium | **0** (421 "not sure: implausible", 38 "not sure: too many unsure", 5 blur retakes) |
| The 50 top-grade (AAA) photos among them | 0 called "many" |
| 777 software re-shoots | **0** |
| Real held-out tray photos: lojano 36, loja_yolo 315, J4ckDev 11 | **0** |
| **Held-out trays at the shipped thresholds**: 24,800 trays (6 licensed groups, each scored by a model trained without it; the 5 unlicensed test sets; the auditor's 1,200 J4ckDev trays) | **16 bands, 1 wrong, 0 dangerous**. The wrong one: a clean samruddh tray called "some". That is one random draw: on 20 further draws (not pre-registered, `reports_v2/threshold_seeds.json`) 0 dangerous in every draw, but **on samruddh about 1 band in 4 is one band off** (26 of 105); 0.55 mainly makes it answer less often there |
| The same trays at the earlier t_good 0.50 | 293 bands, 13 wrong, 0 dangerous; **samruddh: 13 of its 52 bands wrong (25%)**. This is why t_good moved to 0.55 (`docs/MODEL_CARD.md` §2.2a). The 0.55 result is in-sample for that choice |
| The auditor's 1,200 J4ckDev trays | 11 bands, **0 wrong, 0 dangerous** (241 / 0 / 0 at 0.50; v1 under the same test: 793 bands, 57 wrong, 10 clean trays called "many") |

Sources: `app/reports/browser_measure.json`; `reports_v2/clean_model.json`; `reports_v2/threshold_sweep.md`. The re-shoot and real-photo rows were measured at t_good 0.50; raising t_good cannot turn a "not sure" into a band, so they hold at 0.55.

Why the photo rules exist:
- Under the 3 Oct 22:42 rules, the first model gave a confident "many defects, sort the whole lot before selling" on 167 of 464 real Indian photos, including 27 of 50 AAA photos (`reports/beans_ood_cbd_preds.npz`; EVIDENCE M9).
- **The safety comes from these rules, not from the model's accuracy.** Per bean the shipped model is weak on new datasets (`docs/MODEL_CARD.md` §2.2).
- The rules are **not** a general unfamiliar-sample detector: an unfamiliar sample that the model confidently calls mostly sound would still get a band.

**Remaining risk: touching beans** (measured on synthetic trays only, `reports/REGRESSION_A2.md` check 3c; the stricter touching rule has shipped and is measured there).
- It catches pairs, clumps and chains whenever at least 5 single beans lie apart.
- **R1.** A tray where every bean sits in a tight side-by-side pair is not caught by the touching rule. With the v1 model some all-sound trays then got "about many". With the shipped model (t_good 0.50) all 36 such trays ended in a spread retake or "not sure" (`reports/REGRESSION_B_app.md` F5).
- **R2.** A broken piece pressed against a whole bean merges into one blob, so the result can look cleaner than it is.
- **R3.** Lots with widely mixed bean sizes can get a false "spread the beans out" retake. This fails safe, but such a lot may never get a result.
- Spreading the beans out (`c01_welcome`, `c04_spread`) is the defence. Real touching beans and mixed-size lots have not been photographed.

### Example outputs (100 beans, photo OK)

| Situation | Screen | Message (clip) |
|---|---|---|
| 12 defects | Red circles on the 12; Defect / Not sure / Sound counts; band "some" (interval 7.0–19.8%); the possible defect types only under "Possible defect type — not certain" | `c09_some` "Some defects. Remove the beans marked red and photograph again." |
| 18 defects | Counts and circles; "about some" (interval 11.7–26.7% crosses 20%) | `c09_some` + `c18_another` "Photograph another handful…" |
| 61 or more defects, or more than 15 "?" | "Not sure"; the bean count only | `c07_unsure` "Not sure. Take the sample to the cooperative…" |
| A hand, or roasted beans | No count | `c06_not_green` |
| A lot of black beans | "Not sure"; the bean count only | `c07_unsure` |
| Blurry photo | No count | `c02_blur` |

The type clips (`c11`–`c14`, e.g. "there are holed beans") are **not played** since 4 Oct: the shipped model cannot name a defect type better than always guessing the commonest one (30.0% vs 38.2% on J4ckDev photos it never saw; `reports_v2/clean_model.json`).

## 2. Fixed answer list (no generated text)

- Farz can only show the messages in `docs/CLIPS.md` (and play them, once recordings are added; the default build ships no audio files). Its screen shows only counts, circles, the bean ledger, the interval, "possibly …" types under a "not certain" heading, and those same sentences. There is no language model, no text-to-speech at runtime and no free text. The brief's glossary explains why this matters: "If it can say anything, it cannot be checked for safety."
- **It never says:**
  - a price (we found no public coffee price series for Yemen; WFP's Yemen price data has no coffee, EVIDENCE E21);
  - a pesticide;
  - a pest species ("borer" in particular: EPPO does not list the coffee berry borer for Yemen, E23);
  - "specialty";
  - "grade";
  - a defect type as fact.
- A unit test checks that no clip mentions a price, pest species or pesticide (`app/tests/rules.test.ts`).

## 3. Human in the loop

1. **Noor decides.** Farz tells her what it counted. She decides what to re-sort and whether to sell.
2. **She sends the slip herself.** The slip is an `sms:` link: it opens her phone's own Messages app with the text already typed, and nothing is sent until she presses send. Farz has **no SMS service, no server and no account**.
3. **The slip carries totals only, never a type** (measured in the final build, `app/reports/e2e-measurements.json`):
   - Banded result, ~12% demo tray (64 characters, one Arabic SMS of 70): «فرز ٤/١٠/٢٦: ١٠٠ حبة، ١١ فيها عيب، ٧ مش واضحة. فحص ذاتي مش تصنيف» ("Farz 4/10/26: 100 beans, 11 with a defect, 7 unclear. Self-check, not a grade.").
   - "Not sure" result, real AAA photo (48 characters): «فرز ٤/١٠/٢٦: مش متأكد، ٥٠ حبة. فحص ذاتي مش تصنيف» ("Farz 4/10/26: not sure, 50 beans. Self-check, not a grade."). No defect count, no pictures on the slip sheet, and History keeps only the date, the total and the band (e2e test 6). The issue raised in `reports/REGRESSION_A.md` (I-1), that a "not sure" slip listed per-class counts, is **fixed**.
4. **The cooperative grader is the decision-maker.**
   - The brief's Noor has been "a member of the Ondera Coffee Cooperative for eleven years" (brief p. 5).
   - In Yemen some cooperatives have donor-provided processing machines, e.g. the Talok Women's Coffee Association in Taiz (UNDP 2022, p. 68 / PDF p. 82).
   - In our design the grader checks the physical sample before the price is set, and Farz's count is never the grade.
   - The grader can also calibrate Farz for the cooperative (pilot, §6): only the grader, never the farmer's flow, and only after a quality check.
   - The grader process has **not been tested with real graders** (roadmap).
5. **Where there is no cooperative.** UNDP recommends supporting "the creation of coffee cooperative organizations" (p. 46 / PDF p. 60), so many growers may have none. Then "take it to the cooperative" has no one to send her to. A pilot must name another decision-maker (a processing unit or association) or limit itself to cooperative members. **Open.**

## 4. Consent

- **Nothing is sent without her tap.** Every send is a fresh choice in her own Messages app. Nothing is sent automatically, in the background or later.
- **What she is told before sending.**
  - The exact text is shown on screen.
  - With a recorded voice pack, a clip explains it first: `c15_slip` for a banded result ("The message has the bean and defect counts and today's date, and says 'self-check, not a grade'. Send it?") or `c15b_slip_unsure` for "not sure". The default build has no voice, so today only the screen explains.
- **Known gap.** Only 41.0% of rural women aged 15–49 in Yemen are literate (MICS 2022–23, EVIDENCE E30). With no voice, a woman who cannot read cannot check the message; even with a voice pack the clip does **not** read out the numbers. Fix (not built): recorded clips, including number clips played in sequence.
- If grader labels are ever collected as training data in a pilot, each needs the farmer's spoken consent, and carries counts and the photo of the sample only, never a name or location. Nothing like this is built. A calibration stores no photos.

## 5. Privacy: where the data sits, who can read it, and what happens if the phone is lost or shared

The health annex asks entries to state this directly; we apply the same test.

| Data | Where it sits | Who can read it | Kept for |
|---|---|---|---|
| The photo | Phone memory only, while counting. Never uploaded | Nobody else | Not stored; the in-memory copy is released when she leaves the result. Clip `c16_privacy` says "The photo is deleted" |
| Check records | Browser storage (IndexedDB) on that phone. A banded result keeps date, total, band, the 95% range and the per-type counts (History shows only the defect total; `app/src/lib/history.ts`); a "not sure" result keeps only date, total and band | Whoever holds the phone | **Last 5 only**. One tap wipes them (`c17_wiped`); the e2e test confirms 0 records after the wipe |
| A cooperative calibration (pilot) | IndexedDB `farz-localcal` on that phone: two average vectors of 1,024 numbers, a margin, leave-one-out errors, the cooperative or place name the grader typed, counts. No photos, no per-bean data. Never put in the slip | Whoever holds the phone | Until the grader resets it. The farmer's "Erase all" does not remove it; Reset does (checked in a regression run on the 03:42 build, `reports/REGRESSION_B_app.md` check 4; no e2e test covers Reset in the current build) |
| Model (and voice clips, if a pack is added) | Cached on the phone at first visit (service worker) | Public anyway | Until the browser cache is cleared |
| The SMS slip | Her phone's Messages app, then the recipient's phone | Noor and whoever she chooses to send it to | Under her control |
| Anything on a server | **Nothing.** The app is static files with no backend, no account and no analytics script. In a review run with the server killed and the browser offline, the app made no requests to any other host (desktop Chromium; not stored in the repo) | — | — |

- **Lost or shared phone.** The daughter's phone is shared by design (the brief's two-phone household). Anyone holding it sees at most 5 dated records: no photos, no personal name, no location, no price. If a grader calibrated the phone, the cooperative name is shown too. One tap wipes the records, and there is no account to break into.
- **Why so little is stored.**
  - Yemen scores 7 on the ITU Global Cybersecurity Index, the region's worst (E36).
  - The country is split between two authorities with two exchange rates (E21).
  - A central database of farmers' harvests would be a target.

## 6. The calibration pilot: extra safeguards

The calibration is the one place where local data changes the AI's answers, so it gets its own safeguards (`docs/MODEL_CARD.md` §3; `reports_v2/localcal.md`):
- **Pre-registered.** The pass bar was written before any result and hashed. It passed on one dataset (loja_yolo) and failed on the other three. On lojano, 11 clean trays were called "many" at K = 20.
- **Off by default.** It is a separate "PILOT" flow for the grader. It does nothing until a calibration is saved.
- **Refused when it looks wrong.** A calibration is refused when more than 10% of its 20 beans are mis-sorted in leave-one-out (our rule, not validated). Deliberately wrong labels gave 7 of 20 errors and were refused (e2e test 11).
- **Visible.** Every calibrated result and the Home screen say "Calibrated for: <name> (N beans)".
- **Honest demo.** The in-app demo is the quality check only. Labels from the public loja_yolo dataset stand in for a grader's: correct labels give 20/20 and are accepted; the same beans with 6 deliberately wrong labels give 13/20 and are refused. It never saves or installs a calibration, and a grader's saved calibration is unchanged after it runs (e2e tests 10–11). The shipped model was trained on loja_yolo, so the demo shows the check working, not a gain in accuracy.
- **Not enforced: the acceptance test.** A real pilot must check the calibrated app on the cooperative's own labelled trays that were not used for calibration. The app asks for this in words but does not enforce it.

## 7. Bias, and who could be harmed

| Bias | Who it hurts | Mitigation |
|---|---|---|
| **Trained on non-Yemeni beans** (7 datasets from Ecuador, India and unstated countries; no Yemeni varieties, no naturals) and **measured to call many sound beans defective on new datasets** (e.g. lojano 31.4% with thresholds fitted without it; USK, a test-only set, 14.7% at the shipped defect threshold) | Yemeni beans may be flagged defective, so a farmer under-grades her own coffee and loses money | Counts not grades; no types as facts; colour gate; per-bean "?"; the 15% and 60% rules; "not sure" → grader |
| **Does not generalise per bean**: sound vs defect 0.644 balanced accuracy on held-out datasets (coin flip 0.5); defect type 30.0% on photos never seen, below the 38.2% commonest-type baseline | Anyone who trusts a per-bean call | The README and model card lead with these numbers; the app never states a type as fact |
| **The calibration pilot can mislead** (dangerous trays on 2 of 4 datasets) | A cooperative that trusts a bad calibration | Off by default; refusal rule; visible "Calibrated for"; acceptance test required for a pilot |
| **No voice yet; then one voice, one dialect** (the default build is silent until recordings are added) | Women who cannot read; speakers of other dialects; women who might trust a female voice more | Record the messages; comprehension test with Yemeni speakers (not run); a new voice pack is a set of recordings, not a model |
| **Smartphone access is gendered** (account ownership: women 5.4%, men 18.3%, E34; no Yemeni phone-ownership data by gender, E35) | Women without a smartphone | The check runs on the household's smartphone or one at the cooperative; the result goes to her own basic phone as SMS. It will be spoken aloud only once a voice pack is recorded |
| **Literacy** (41.0% rural women 15–49, E30) | Women who cannot read the screen | Today: red circles, the bean ledger and icons. Planned: a recorded voice for every message (none in the default build); the slip read-back would still not speak the numbers (§4) |

## 8. The brief's hard rules, and the limits Farz respects

| Rule (brief §06) | Farz |
|---|---|
| Runs on a device the user already has | A smartphone browser (the daughter's, or one at the cooperative), nothing to install; Noor's basic phone receives an SMS. Tested in desktop Chromium only; iPhone test pending; Android untested. Needs WebAssembly SIMD (minimum browser versions not verified) |
| Core feature works offline | After the first visit the whole check ran with the network off in desktop Chromium (e2e test 1; 14/14 pass, 4 Oct 05:25 BST). On a phone in airplane mode: pending (§9) |
| Model small enough to side-load or send over a weak connection | Classifier **3.1 MB** (3,082,642 bytes, fp16). Whole app on first visit about **21.7 MB** (default build: precache of 57 entries, 21,226.13 KiB, Workbox build log of the 4 Oct 06:31 build), most of it the 14.2 MB browser runtime |
| At least one interaction in a local language; name it | **Yemeni Arabic, written**: every message and screen. Spoken Yemeni Arabic needs the recordings; the default build has no voice |
| "How would it fare in a less-supported language?" | Farz uses **no speech recognition and no translation model**, only fixed clips a local speaker records. On Yemeni dialect, all 12 speech-recognition models tested in one 2026 benchmark got 54–82% of words wrong (E40), and Mehri and Soqotri have no speech model at all (E41). A new language costs a set of recordings, not a new model |
| A person makes the final call; avoid hallucination | §3; fixed answer list (§2) |

**Limits Farz respects** (the brief asks "are the limits respected?"; Annex B has no separate limits section, so we take the §06 rules above plus our own): never a price, a pesticide or a pest species; no diagnosis of why yields dropped; no grade or "specialty"; no defect type as fact; nothing sent without the farmer's tap; no photo stored.

## 9. Before submitting, run these and fill in the results

Replace "pending" with what the phone did; never estimate.

| Test | Expected | Desktop Chromium (e2e, 4 Oct 05:25) | iPhone, airplane mode |
|---|---|---|---|
| Demo tray, network off | counts, circles, bean ledger | pass (e2e 1) | pending |
| Photo of a hand | `c06_not_green` | synthetic images only (unit tests; `reports/REGRESSION_B_app.md` 6e) | pending |
| Roasted beans | `c06_not_green` | synthetic only: a dark-roast tray gives `c06` (e2e 13), and so did 2 dark and 2 medium roast recolours in a regression run. Light-roast recolours: at t_good 0.50, 3 of 36 got "clean" (REGRESSION_B F2); at 0.55, 0 bands on 38 (`reports_v2/threshold_ship.md` §5) | pending |
| A lot of black beans | `c07_unsure` | pass on a synthetic black-bean tray (e2e 13); the real J4ckDev Negros photo gives `c07` in the built app (`reports_v2/threshold_ship.md` §5) | pending |
| Empty cloth | `c05_count` | pass: an empty sheet gives `c05` (e2e 3) | pending |
| Deliberately blurred photo | `c02_blur` | — | pending |
| Dark photo | `c03_dark` | pass (e2e 3) | pending |
| Beans in a pile | `c04_spread` | pass on synthetic touching trays (e2e 8) | pending |
| Real AAA photo from India (demo button) | `c07_unsure` | pass: "not sure (implausible)" (e2e 4) | pending |
| ~3% demo tray | "about clean" + `c18_another` | pass: 3/99 "about clean"; 3 handfuls pool to 9/297 (e2e 4–5) | pending |
| "Send slip" | Messages opens with the totals-only text; nothing sent until she taps | `sms:` link and ≤ 70 characters checked; "not sure" slip 48 characters (e2e 1, 6) | pending |
| Wipe | history empty | pass (e2e 3) | pending |
| Calibration pilot | quality check, save, "Calibrated for"; wrong labels refused; the demo saves nothing | pass (e2e 10–11) | pending |
| No voice in the default build | no Listen buttons, "Voice coming soon", no clip requested | pass (e2e 14) | pending |
| Portrait photo | rings sit on the beans | — | pending |

## 10. Licences and clean room

- **Training data.** The shipped classifier is trained only on datasets that state a licence. Two of them are non-commercial: J4ckDev, **CC BY-NC-SA 4.0**, and lojano, **CC BY-NC 4.0**. So the model is a **non-commercial, ShareAlike research prototype** (`MODEL_LICENSE.md`).
- **Test-only data.** The 5 datasets that state no licence were used only as test data. They are not in the shipped weights and are not redistributed.
- **What a deployable version needs:** Yemeni beans labelled under an open licence, which a consented cooperative pilot could produce.
- **Clean room:** built in a personal repository. No employer code or data was used.
