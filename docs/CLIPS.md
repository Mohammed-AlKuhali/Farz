# Farz voice clips: the recording list (Yemeni Arabic)

Canonical list. The app shows these lines on screen and, once recordings exist, plays them by **filename** (`app/public/audio/<file>.m4a` or `.mp3`). **Shipped now: no audio.** Since 4 Oct 05:18 the default build contains no voice files and says "Voice coming soon". To switch a voice on, put the recordings in `app/public/audio/` under these file names, add a `pack.json` there (set `"ext"`, describe the voice, `"placeholder": false`) and rebuild (`app/public/audio/README.md`). macOS text-to-speech placeholders (voice "Majed", Modern Standard Arabic, not Yemeni Arabic) exist for local testing only and are not in the public repo.
**19 clips.** Speaks to Noor in the **feminine** form (women do the drying and sorting — UNDP 2022, p34).
**The recorder may rewrite any line so it sounds natural in the dialect.** Keep each clip under ~6 seconds. Record in a quiet corner, phone 20 cm from the mouth.

**Since 4 Oct (v2 model): the type clips c11–c14 are not played.** The shipped model cannot name a defect type reliably (30.0% on photos it never saw, below the 38.2% commonest-type baseline; `reports_v2/clean_model.json`), so a result plays only its band clip (c07–c10, plus c18 when borderline). The c11–c14 lines stay in this list and in the app's text table, but no screen or rule triggers them.

Rules baked into this list (Responsible-AI gate): no prices, no pesticide names, no pest species, no causes we can't source. "Not sure" always sends her to a person/place.

| # | File | Plays when | Yemeni Arabic (edit freely) | Meaning |
|---|---|---|---|---|
| 1 | `c01_welcome` | first open | أهلين. صوّري حوالي مية حبة بن أخضر على قماشة بيضا، مفرّقة عن بعض. | Welcome. Photograph about 100 green coffee beans on a white cloth, spread apart. |
| 2 | `c02_blur` | photo blurred | الصورة مش واضحة. ثبّتي التلفون وصوّري ثاني. | Photo not clear. Hold the phone still and take it again. |
| 3 | `c03_dark` | photo too dark | الصورة مظلمة. صوّري في مكان فيه نور. | Too dark. Take it somewhere with light. |
| 4 | `c04_spread` | beans touching | الحبوب لاصقة ببعض. فرّقيها وصوّري ثاني. | Beans are touching. Spread them out and retake. |
| 5 | `c05_count` | <20 or >160 beans | حطّي حوالي مية حبة، وصوّري ثاني. | Use about 100 beans and retake. |
| 6 | `c06_not_green` | colours not green coffee (hand, roasted beans…) | هذا مش بن أخضر. صوّري البن بعد التقشير وقبل التحميص. | This isn't green coffee. Photograph beans after hulling, before roasting. |
| 7 | `c07_unsure` | doubt about the model or the sample only: the model is unsure of more than 15% of the beans (or answers none), or the sample is implausible (more than 60% of answered beans called defective), or the photo is a lot of black, bean-shaped beans (the dark-lot rule, since 4 Oct). A borderline 95% range no longer plays this; it gets `c18_another` | مش متأكد. ودّي العينة للجمعية يشوفها المختص. | Not sure. Take the sample to the cooperative for a specialist to check. |
| 8 | `c08_clean` | defect rate (point estimate) under 5%; plus `c18_another` if the 95% range crosses 5% | البن نظيف، العيوب قليلة. | The coffee is clean; few defects. |
| 9 | `c09_some` | defect rate (point estimate) 5–20%; plus `c18_another` if the 95% range crosses a band edge | فيه عيوب. شيلي الحبوب المعلّمة بالأحمر وصوّري ثاني. | Some defects. Remove the beans marked red and photograph again. |
| 10 | `c10_many` | defect rate (point estimate) over 20%, up to 60% (above 60% → `c07_unsure`); plus `c18_another` if the 95% range crosses 20% | العيوب كثيرة. فرّزي البن كله قبل البيع. | Many defects. Sort the whole lot before selling. |
| 11 | `c11_dark_beans` | **not played since 4 Oct** (was: black/sour beans found) | فيه حبوب سودا أو حامضة. شيليها قبل البيع. | There are black or sour beans. Remove them before selling. |
| 12 | `c12_insect` | **not played since 4 Oct** (was: insect-damaged beans found) | فيه حبوب مخرّمة. ورّيها الجمعية. | There are holed beans. Show them to the cooperative. |
| 13 | `c13_broken` | **not played since 4 Oct** (was: broken beans found) | فيه حبوب مكسّرة. شيليها قبل البيع. | There are broken beans. Remove them before selling. |
| 14 | `c14_unhulled` | **not played since 4 Oct** (was: parchment / dried cherry found) | فيه حب لسه بقشره. قشّري العينة قبل الفحص. | Some beans still have their husk. Hull the sample before checking. |
| 15 | `c15_slip` | before sending the slip of a banded result (the slip holds the bean count, the defect total and the unsure count, never a type) | الرسالة فيها عدد الحبوب والعيوب وتاريخ اليوم، ومكتوب فيها: فحص ذاتي مش تصنيف. ترسليها؟ | The message has the bean and defect counts and today's date, and says "self-check, not a grade". Send it? |
| 15b | `c15b_slip_unsure` | before sending a slip for a "not sure" result | الرسالة فيها عدد الحبوب وتاريخ اليوم، ومكتوب فيها: مش متأكد، فحص ذاتي مش تصنيف. ترسليها؟ | The message has the bean count and today's date, and says "not sure, self-check, not a grade". Send it? |
| 16 | `c16_privacy` | after a check | الصورة انمسحت. ما يطلع من التلفون إلا الرسالة اللي ترسليها أنتي. | The photo is deleted. Nothing leaves the phone except the message you send. |
| 17 | `c17_wiped` | after wipe | انمسح كل شي. | Everything is erased. |
| 18 | `c18_another` | result is borderline between two bands (sampling, not the AI) | النتيجة قريبة. صوّري حفنة ثانية من نفس البن عشان نتأكد. | The result is close. Photograph another handful from the same coffee so we can be sure. |

**ElevenLabs (not used in this build):** clips may be generated at build time with a voice clone, each checked by a native speaker; anything that does not sound Yemeni is recorded instead. The app never calls ElevenLabs at runtime. If it is used, the voice, model and date go into `pack.json` → `voice_en`.
