# SCRIPT_FLOW — spoken-flow review of docs/RECORD_NOW.md (4 Oct 2026)

Source reviewed: `docs/RECORD_NOW.md` (English blocks A1–A4, VO-D, VO-T, VO-W3, VO-W4; the Arabic clips were not reviewed).
How the timings were worked out: 2.6 words/s. Numbers are counted as the words you say ("twelve hundred" = 2, "2026" = 3, "MobileNet-V3 Small" = 4, "sixty-four" = 2). "+pauses" adds about 0.2–0.35 s per sentence break after the edit tightens the gaps. Slot lengths come from `docs/PRODUCTION_PLAN.md`: in V1, A1 runs from 4 to 58 s, which gives 54 s. V2 and V3 are each ≤ 58 s, with a 60 s hard cap. V4 sections are 35 / 45 / 60 / 35 / 40 s, and V4 itself is flexible at 2–5 min.

| Block | Target | Before (words / pure speech) | After (words / pure speech) | Verdict |
|---|---|---|---|---|
| A1 | ~45 s, slot 54 s | 145 / 55.8 s (with the optional line), 135 / 51.9 s (without) | 127 / 48.8 s (without), 131 / 50.4 s (with) | **Too long. Edit.** |
| A2 | ~30 s, slot 35 s | 89 / 34.2 s | 88 / 33.8 s | OK on length (V4 has no hard cap). Fix the misleading phrase and the long sentence. |
| A3 | ~30 s, slot 45 s | 81 / 31.2 s | 82 / 31.5 s | Good. One optional tweak. |
| A4 | ~40 s | 108 / 41.5 s | 107 / 41.2 s | Good length. Fix the "records" stress and the pile-up of "one"s. |
| VO-D | ~45 s, cap 58 s | 93 / 35.8 s | 95 / 36.5 s | **Good. Fits.** One fix for clarity. |
| VO-T | ~58 s, cap 60 s | **155 / 59.6 s (about 65 s with pauses)** | 137 / 52.7 s (about 56 s cut tight) | **Over the hard cap. Must edit.** |
| VO-W3 | ~45 s, slot 60 s | 80 / 30.8 s | 82 / 31.5 s | Good. Two small fixes. |
| VO-W4 | ~35 s, slot 35 s | 108 / 41.5 s | 106 / 40.8 s | Long, but V4 isn't capped. **Stretch V4 §4 to about 0:45** (V4 becomes about 3:40, still inside 2–5 min). Fix the flow. |

Realism check: a non-native speaker reading technical lines usually runs nearer 2.4 words/s. At that pace the original VO-T is 64.6 s and the revised one is 57.1 s, so the VO-T cut is necessary, not optional.

---

## A1 · Intro — EDIT

Problems:
- The script is about 52 s of pure speech without the optional line and 56 s with it. Natural pauses take it past the 54 s slot.
- "believed to have first been farmed" is a cluster of b/f/v sounds, and it reads like a document.
- "checks a sample of coffee on the phone" sounds as if the coffee is sitting on the phone.
- "this weekend" and "one weekend" say the same thing twice.
- There are two "here's" lines.

Replacement lines (shown without the optional line):

> Hi, I'm Mohammed — an AI engineer and researcher, based in London.
>
> Here's something most people don't know. It's believed coffee was first farmed commercially in Yemen, back in the fourteen hundreds. Today, more than a hundred thousand Yemeni families still grow it.
>
> But most of them sell at the farm gate, at a price agreed on the spot — not set by any market. Yet one exporter pays farmers twelve times the usual rate when their coffee is carefully picked and sorted. In Yemen, that sorting is mostly done by women.
>
> So I built Farz — it means "sorting" in Arabic. It's a small AI that checks a sample of coffee — on the phone a family already has. No internet needed.
>
> Solo team, one weekend. Let me show you. *(smile)*

If you keep the optional line, put "I'm also Yemeni, and that's why I picked this challenge." in place of "Here's something most people don't know." Your Yemeni line then leads straight into the Yemen coffee line. That version runs about 50 s, so the globe opening may need to drop from 4 s to about 2 s.
"twelve times the usual rate" now matches the wording in A2 and in the source. The number is unchanged.
Delivery: say "the FOUR-teen hundreds", then take a breath before "a hundred thousand" so the two "hundreds" don't run together.

## A2 · The problem — minor edits

Problems:
- "before the price is set this harvest" can be heard as "set this harvest".
- The last sentence is 26 words, too long for one breath.
- "And we know that matters" sounds stiff.

> Meet Noor. She grows coffee in the highlands of Yemen. At harvest, a buyer turns up and names a price — and she has nothing to check it against.
>
> So here's our goal. This harvest, because of Farz, Noor will know how many bad beans are in her coffee before the price is set — something she'd otherwise never know.
>
> Why does that matter? In Yemen, most coffee is sold at the farm gate. And one exporter pays twelve times the usual rate for coffee that's carefully picked and sorted.

## A3 · Why AI — good (one optional change)

This already sounds like someone talking, and the "can't do that / can't do that / you need eyes" rhythm works well. The only optional change avoids the five-syllable "deliberately":

> Everything else is simple rules — on purpose.

(This replaces "Everything else is deliberately simple rules.")

## A4 · Your take — small edits

Problems:
- "one 2026 study, every one of twelve" piles up three number words.
- "records" has unclear stress: it's the verb (re-CORDS), but it's easy to say the noun (REC-ords).
- "checked it on its own beans" trips the tongue.

> In a 2026 study, all twelve speech models tested got more than half of Yemeni dialect wrong. So Farz doesn't try to listen. It plays fixed sentences, recorded by a local speaker.

> Until a cooperative has checked it on their own beans, Farz stays honest: it says "not sure", and a person decides.

Everything else stays as written.

## VO-D · Product demo — good, fits (about 36 s against a 45 s budget)

This sounds natural, with short, clean lines. One clarity fix: "take out the red ones" sounds like red beans.

> Then it tells her what to do: take out the beans marked in red, and check again.

## VO-T · Technical walkthrough — MUST EDIT (59.6 s of pure speech, about 65 s with pauses, against the 60 s cap)

Problems:
- It's too long.
- The stage-one and stage-two sentences are 27 and 32 words, so you can't say either in one breath.
- "Calibrated, three megabytes, running right in the browser" reads like a spec sheet.

No facts or numbers are dropped. The words cut are the "here's what's under the hood" opener, "right", "whole", "it's never seen" (now "unseen", which echoes stage two), "every bean pulled out as", and "we locked in" (now "set").

> Three stages: rules, then AI, then rules again.
>
> Stage one is classic computer vision: white balance off the cloth, an Otsu threshold, a cleaned-up mask, and connected components. It counts within two beans almost nine times out of ten, and the JavaScript port matches Python exactly.
>
> Stage two is the AI: a MobileNet-V3 Small. Trained on over ninety thousand bean images from seven licensed datasets, and tested on one unseen dataset at a time. Calibrated, three megabytes, running in the browser.
>
> Honestly — on unseen datasets, its balanced accuracy is only sixty-four per cent. So stage three refuses rather than guesses: a threshold set before testing, a confidence range for every sample, and a veto on anything implausible.
>
> On twelve hundred test trays, our first model made ten dangerous calls. This one made none.

That's 137 words: 52.7 s at 2.6 words/s and about 56 s once the gaps are tightened. If your take comes in at 55 s or less, you can open with "Farz runs in three stages:" (+3 words, about 1.2 s).
Pronunciation:
- Otsu = "OHT-soo"
- MobileNet-V3 Small = "mobile-net vee-three small"
- threshold = "THRESH-ohld"
- calibrated = "KAL-ih-bray-tid"
- veto = "VEE-toh"
- implausible = "im-PLAW-zih-bul"

## VO-W3 · World Bank demo — good (about 31 s against 45 s)

- "out of the ninety-three Farz" runs the number straight into the name. Adding "that" fixes it.
- The last line jumps topic without a link.

> Eleven beans look defective, out of the ninety-three that Farz was sure about. The rest, it marks as "not sure".

> It keeps just five dated records, no photos — and one tap wipes them. And the whole model is three megabytes.

("records" here is the noun: REC-ords.)

## VO-W4 · Noor's day + tech — fix the flow, extend the slot (about 41 s against 35 s)

Problems:
- The aside in the middle of the Farz sentence ("— at the cooperative, or at home… —") breaks the breath.
- "it's PyTorch and a small MobileNet" sounds like a shopping list.
- The last sentence is 30 words.

You can't reach 35 s without cutting facts, so let V4 §4 run to about 45 s instead.

> In Yemen, coffee dries for up to three weeks, then it's sorted, then hulled. Farz checks a hundred beans from Noor's lot before the price is set — at the cooperative, or at home on her daughter's phone. If it isn't sure, the cooperative's grader decides.
>
> Under the hood, it's a small MobileNet built in PyTorch, exported to ONNX, and running in the browser — cached for offline use. No server, no account. It's trained only on public datasets that state a licence. On unseen data, it's only a little better than a coin flip. So it counts rather than grades — and says "not sure" instead of guessing.

Pronunciation:
- ONNX = "ON-ix"
- PyTorch = "PIE-torch"
- hulled rhymes with "dulled"
- cooperative = "koh-OP-er-uh-tiv"
