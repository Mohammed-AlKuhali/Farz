# Farz voice clips (audio pack)

This folder is empty in the published app on purpose: **the default build ships no voice audio.** The app then shows
every message as text and pictures, and says "Voice coming soon" where a Listen button would be.

## Switching a voice on (no code change)

1. Record the 19 clips below in the local dialect, one file per clip, **under exactly these names** (`.mp3` or `.m4a`).
   The words, and when each clip plays, are in `docs/CLIPS.md`.
2. Put the files in this folder (`app/public/audio/`).
3. Add `pack.json` here (copy `app/audio-placeholder/pack.json`), then set `"ext"` to `"mp3"` or `"m4a"`,
   `"placeholder": false`, and describe the voice in `"voice_en"` / `"voice_ar"` (the About screen shows this text).
4. Rebuild: `npm run build`. The service worker then precaches the clips, so they play offline.

Suggested export: mono, 22–44.1 kHz, 48–64 kbps, under about 6 s each.

| # | file (+ .mp3 or .m4a) | plays when |
|---|---|---|
| 1 | c01_welcome | first open (after the first tap) |
| 2 | c02_blur | photo blurred |
| 3 | c03_dark | photo too dark |
| 4 | c04_spread | too many beans touching |
| 5 | c05_count | fewer than 20 or more than 160 beans |
| 6 | c06_not_green | colours are not green coffee (hand, roasted beans, leaves …) |
| 7 | c07_unsure | "not sure": the model is unsure of > 15% of beans, the sample is implausible (> 60% of answered beans defective), or a lot of black beans |
| 8 | c08_clean | defect share under 5% |
| 9 | c09_some | defect share 5–20% |
| 10 | c10_many | defect share over 20%, up to 60% |
| 11 | c11_dark_beans | not played (defect types are only shown as "possible …") |
| 12 | c12_insect | not played |
| 13 | c13_broken | not played |
| 14 | c14_unhulled | not played |
| 15 | c15_slip | read-back before the SMS app opens (a result with a band) |
| 16 | c15b_slip_unsure | read-back before the SMS app opens for a "not sure" result |
| 17 | c16_privacy | after the first check of a session |
| 18 | c17_wiped | after "Erase all" |
| 19 | c18_another | the 95% range crosses a band edge: "photograph another handful" |

## Build switch `VITE_AUDIO`

| value | what the build contains |
|---|---|
| `pack` (default) | whatever is in this folder; with no `pack.json` here, no voice |
| `placeholder` | adds the text-to-speech placeholders from `app/audio-placeholder/` (made with the macOS voice "Majed", Modern Standard Arabic). For local testing and screen recordings only; not for a public deploy, because Apple's licence restricts sharing system-voice output |
| `none` | no audio files at all, even if this folder holds a pack |

Example: `VITE_AUDIO=placeholder npm run build`.
