# Fact-check: docs/RECORD_NOW.md, English blocks (A1–A4, VO-D, VO-T, VO-W3, VO-W4)

Checked 4 Oct 2026 against: `docs/sources/undp_yemen_qat_coffee_2022.txt`, `docs/sources/worldbank_challenge4_brief.txt`, `docs/sources/known_facts_and_decisions.md`, `docs/EVIDENCE.md` (including its section F, "claims we deliberately do NOT make"), `docs/DATA_SOURCES_V2.md`, `reports/count_cbd.json`, `reports_v2/clean_model.{md,json}`, `reports_v2/threshold_sweep.md`, `reports_v2/localcal.md`, `app/reports/*.json`, `app/src/lib/*.ts`, the Bulbul paper text (the 2608.21950 copy in the scratchpad), and the live app at https://farz-green.vercel.app.

Each fix is the smallest wording change that keeps the line natural to say. The fixes are in priority order: fix 1–7 before recording.

## Must fix (wrong, or breaks a rule in EVIDENCE.md §F)

| # | Block · line | Script says | Problem | Fix (smallest change) |
|---|---|---|---|---|
| 1 | VO-T · 80 | "On twelve hundred test trays, our first model made ten dangerous calls. This one made none." | The numbers are right (v1: 10 clean trays called "many"; shipped model at t_good 0.55: 0 dangerous). But it only gave an answer on **11 of the 1,200** trays (M27), and EVIDENCE §F says this result must never be quoted without that. The trays are also simulated from bean crops, not photographed. | "On twelve hundred simulated test trays, our first model made ten dangerous calls. This one made none — though it only answered on eleven." |
| 2 | VO-T · 78 | "a threshold we locked in before testing" | Wrong. The **rule** for choosing the threshold was hashed 10 s before the sweep. The value 0.55 was the sweep's output, chosen on the same held-out trays (threshold_sweep.md; EVIDENCE §F: "t_good 0.55 is validated" is banned). The defect threshold 0.91 was fitted on pooled predictions. | "a threshold picked by a rule we locked in before testing" |
| 3 | A4 · 48 | "Until a cooperative has checked it on its own beans, Farz stays honest: it says "not sure", and a person decides." | Overstated. Farz without calibration still gives answers (the ~12% demo tray in this same video gets "some defects"). EVIDENCE §F also bans "any unfamiliar sample gets 'not sure'". | "Until a cooperative has checked it on its own beans, Farz stays cautious: when it isn't sure, it says "not sure", and a person decides." |
| 4 | A4 · 46 | "It plays fixed sentences that a local speaker records." | Not true of the build. The live app has no audio (`/audio/pack.json` returns 404; the screen says "Voice coming soon. Everything is written on screen."). EVIDENCE §F bans "recorded voice" until recordings are in the build. | "It uses fixed sentences — on screen for now, and recorded by a local speaker next." Keep the original only if the CLIPS recordings are actually shipped in the build before submission. |
| 5 | VO-D · 58 | "Farz finds every bean, counts them…" | "Finds every bean" is on the §F banned list: the finder is within ±2 on 87.5% of 464 photos (M1). | "Farz finds the beans, counts them, and circles the ones that look defective." |
| 6 | VO-D · 54 | "This is Noor's coffee… a hundred beans go on a white cloth." | Per SHOOT_SHEET B2, this plays over the ~12% demo tray. The app itself labels that tray **SYNTHETIC**: J4ckDev bean crops pasted on a **grey** sheet. It is not Noor's (Yemeni) coffee, and it is not on a white cloth. | "Picture Noor's coffee, after it's been hulled at the cooperative. Before anyone talks price, a hundred beans go on a white cloth — here, a demo tray stands in." |
| 7 | VO-W3 · 88 | "Eleven beans look defective, out of the ninety-three Farz was sure about. The rest, it marks as "not sure"." | The figures are verified live (11 of 93 sure beans, 82 sound, range 6.7–20%). But "the rest" sounds like the other 82 of the 93, and those are the **sound** beans. Only 7 beans are "not sure". | "…out of the ninety-three Farz was sure about. The other seven, it marks as "not sure"." |
| 8 | A2 · 28 | "Noor will know how many bad beans are in her coffee… — something she'd otherwise never know." | Overstated. Farz counts a ~100-bean **sample**, gives the beans that *look* defective among the ones it is sure about, or says "not sure". On every real held-out photo set it gave 0 answers (M12, M21). Women already hand-sort (E5), so "never know" goes too far. | "Noor will have a count of the beans that look defective in a sample of her coffee, before the price is set this harvest — a number she doesn't have today." |
| 9 | A1 · 16 | "at a price agreed on the spot, not set by any market" | Overstated. UNDP (E2, p. 35) says prices depend on "the understanding between farmers and market players, not exactly on market rates". The same UNDP page 24 also describes weekly markets with open auctions. | "at a price agreed on the spot, not exactly at market rates" |

## Should fix (imprecise against the evidence register)

| # | Block · line | Script says | Problem | Fix |
|---|---|---|---|---|
| 10 | A1 · 14 | "Today, more than a hundred thousand Yemeni families still grow it." | The 104,000 **families** figure comes from a 2009 study (E6). For "today/still", EVIDENCE says to use FAO 2025 (E16: about 120,000 smallholder **farmers**). | "Today, more than a hundred thousand Yemeni farmers still grow it." |
| 11 | A1 · 14 | "Coffee is believed to have first been farmed commercially in Yemen" | UNDP (E12b) says **Arabica**, and EVIDENCE §F says to use that wording. | "Arabica coffee is believed to have first been farmed commercially in Yemen…" |
| 12 | VO-T · 74 | "and the JavaScript port matches Python exactly" | Counts are identical (21/21 photos compared; same 280/406 result on all 464 CBD photos in the browser). But the Otsu threshold differs by 1 grey level on 8 of the 21 photos, and the boxes differ by up to 1 px (`beanfinder_parity_report.json`), so the port is not identical in every detail. | "and the JavaScript port gets the same counts as Python." |
| 13 | VO-T · 76 | "from seven licensed datasets" | EVIDENCE §F says to say "licence-stated", not "licence-clean". Two of the seven are non-commercial, and three are CC0 uploads with unverified provenance. VO-W4 already uses the correct wording. | "from seven public datasets that state a licence" |
| 14 | VO-D · 66 | "Offline, on the phone she already has." | In the brief, Noor's own phone is for calls, messages and mobile money; the smartphone belongs to her daughter. A1 and VO-W4 already say this correctly. | "Offline, on the phone her family already has." |
| 15 | A4 · 48 | "there isn't a single Yemeni coffee bean in any public dataset we could find" | Slightly too strong. No dataset says it is from Yemen, but six state no country of origin (DATA_SOURCES_V2 §7), so we can't prove they contain no Yemeni beans. | "And we couldn't find a single public dataset of Yemeni coffee beans." |

## Optional polish (true, but could be tighter)

| # | Block · line | Note | Fix |
|---|---|---|---|
| 16 | A1 · 16 | "twelve times more" can be read as 13×. A2 already says "twelve times the usual rate" (E4: $6 instead of $0.50/kg). | "…pays farmers twelve times the usual rate when their coffee is carefully picked and sorted." |
| 17 | A1 · 18 | "No internet needed": the app has to be opened online once (about 21.7 MB precache) before it works offline. | "No internet needed once it's on the phone." |
| 18 | A4 · 46 | Verified: all 12 Bulbul systems have word error rates of 54.5–82.3% on Yemeni dialect. "More than half of Yemeni dialect wrong" is loose. | "…got more than half the words of Yemeni dialect wrong." |
| 19 | VO-T · 74 | "every bean pulled out as a connected component": beans that touch merge into one component, which is flagged and treated as unsure or asks for a retake. | "…and the beans pulled out as connected components." |
| 20 | VO-T · 78 | "a confidence range for every sample": a "not sure" result shows no range. | "a confidence range on every answer" |
| 21 | VO-W4 · 100 | UNDP p. 23 says hulling is normally done by processors after purchase, so a check before sale needs a hulled sample. VO-D covers this; this line doesn't. | "Farz checks a hundred hulled beans from Noor's lot…" |

## Checked and correct (no change)

- **A1:** Port Mokha pays 12× ($6 vs $0.50/kg) for strict picking and sorting (UNDP p. 35, citing Bloomberg). Women mostly do the drying and sorting (UNDP p. 20). Farz = "sorting". The script says "one weekend".
- **A2:** Coffee is mostly sold at the farm gate (UNDP p. 24). The 12× rate is the "usual rate". "Nothing to check it against" is supported (FAO 2021: no effective market information system; E20, E21).
- **A3:** Farz never gives a price: the slip and the UI have none. It never names a pest: only "possibly insect holes" appears, and no species is ever named. "Not sure" → "Take the sample to the cooperative" (c07).
- **A4:** "One 2026 study, twelve speech models, all above 50% on Yemeni dialect" is verified against the paper's tables (Bulbul, Aug 2026, "12 evaluated systems").
- **VO-D:** The defect marker is a red ring with ×. "Remove the beans marked red and photograph again" (c09). The SMS slip is dated, holds counts only, never a price, and says «فحص ذاتي مش تصنيف».
- **VO-T:**
  - Stage one: white balance from the sheet, Otsu, open/close, 4-connected components (`beanfinder.ts`).
  - "Within two beans almost nine times out of ten": 406 of 464 = 87.5%, in Python and in the browser.
  - MobileNetV3-Small, 92,690 crops, 7 datasets, leave-one-dataset-out.
  - Temperature-calibrated (T = 2.046).
  - 3,082,642 bytes, also checked on the live site.
  - 0.644 mean balanced accuracy.
  - The implausible veto is the 60% rule.
  - v1 made 10 dangerous calls on the 1,200 trays.
- **VO-W3:**
  - 11 of 93, verified live.
  - The slip says "self-check, not a grade".
  - The real CBD AAA photo (never used in training) → "not sure", take it to the cooperative.
  - Keeps 5 records (`KEEP = 5`), counts only, never photos; "Erase all" is one tap.
  - The model is 3 MB.
  - "Airplane mode" is valid only if the B2 iPhone recording actually shows the result with Airplane Mode on.
- **VO-W4:**
  - Drying takes 12–21 days, "followed by selection, hulling" (UNDP p. 23).
  - The daughter's phone matches the brief.
  - PyTorch → ONNX → onnxruntime-web, precached by Workbox.
  - No server: only same-origin static files are fetched. No account.
  - Trained only on licence-stated data.
  - "A little better than a coin flip" (0.64 vs 0.50) is consistent.
