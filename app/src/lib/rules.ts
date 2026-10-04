/**
 * Fixed fail-safe rules and the result decision (docs/PLAN.md §5, docs/CLIPS.md). NOT AI.
 * Order: too dark -> very few beans -> blurred -> not green coffee -> touching -> bean count -> (classifier) ->
 *        implausible sample (> 60% defects) -> too many unsure (> 15%) -> band of the point estimate
 *        (+ "about" and "add another handful" when the 95% interval crosses a band edge).
 * The app never shows a price, a pest species or a pesticide: there is no code path that could.
 */
import { MAX_BEANS, MIN_BEANS, type Checks } from "./beanfinder";
import { COLOUR_RULE, type ColourGate } from "./colourgate";
import type { ClipId } from "./clips";
import { bandOfRate, crossesBandEdge, wilson, type Band, type Interval } from "./wilson";

/** The 6 outputs of the v2 licence-stated model (models_v2/farz_beans_v2_clean_labels.json), in output order. */
export const CLASSES = ["good", "dark", "insect", "broken", "unhulled", "other_defect"] as const;
export type ClassName = (typeof CLASSES)[number];
/** Every non-good class is a defect; `other_defect` counts as a defect. The TYPE is never shown as a fact ("possible …"). */
export const DEFECTS = ["dark", "insect", "broken", "unhulled", "other_defect"] as const;
export type DefectName = (typeof DEFECTS)[number];
export type BeanCall = ClassName | "unsure";

/** More than this fraction of unsure beans -> "not sure, take the sample to the cooperative" (model uncertainty). */
export const MAX_UNSURE_FRAC = 0.15;
/**
 * Sample-plausibility fail-safe (fixed rule, not AI): if more than this share of the ANSWERED beans are called
 * defective, the photo is unlike anything Farz was built for (measured: real Indian CBD photos, legume/skin blobs)
 * -> "not sure, take the sample to the cooperative". A fail-safe tuned on out-of-distribution photos; it also
 * means Farz never says "many defects" above 60% — those samples go to a person.
 */
export const IMPLAUSIBLE_DEFECT_FRAC = 0.6;
/** "Add another handful": up to this many photos of the same coffee are pooled into one count and one interval. */
export const MAX_HANDFULS = 3;

export type RetakeReason = "dark" | "blur" | "not_green" | "count" | "spread";

export interface Retake {
  kind: "retake";
  reason: RetakeReason;
  clip: ClipId;
}

/**
 * Why the answer is c07 "not sure": the model is unsure of too many beans, the sample is implausible, or (no model
 * call at all) the photo is a lot of very dark, bean-shaped beans (darklot.ts) — black green beans go to a person.
 */
export type UnsureReason = "" | "too_many_unsure" | "implausible" | "dark_lot";

export interface Result {
  kind: "result";
  /** Band of the POINT estimate; "unsure" (c07) only for model or sample-plausibility uncertainty. */
  band: Band;
  /** Sampling uncertainty: the 95% interval crosses a band edge -> show "about", suggest another handful. */
  about: boolean;
  unsureReason: UnsureReason;
  clips: ClipId[];
  counts: Record<BeanCall, number>;
  total: number; // beans found (all handfuls)
  answered: number; // beans the model answered (total - unsure)
  defects: number;
  interval: Interval; // defects among confidently classified beans
  unsureFrac: number;
  handfuls: number; // photos pooled into this result
  /** true when "Add another handful" should be offered (about, and fewer than MAX_HANDFULS photos so far) */
  canAddHandful: boolean;
}

export type Outcome = Retake | Result;

const RETAKE_CLIP: Record<RetakeReason, ClipId> = {
  dark: "c03_dark",
  blur: "c02_blur",
  not_green: "c06_not_green",
  count: "c05_count",
  spread: "c04_spread",
};

/**
 * Gate that runs BEFORE the classifier. Returns null when the photo may be classified. Order:
 *  1. too dark                                                                  -> c03
 *  2. fewer than MIN_BEANS beans, nothing touching, nothing off-colour           -> c05 "use about 100 beans"
 *     (an empty or near-empty sheet has almost no texture, so the blur measure below would wrongly say "blurred")
 *  3. blurred                                                                    -> c02
 *  4. not green coffee                                                           -> c06
 *     except a "big blob" of bean-coloured pixels while many blobs are touching: that is beans in a clump -> c04
 *  5. too many touching / merged blobs (src/lib/touching.ts)                     -> c04, BEFORE the count range,
 *     because beans pushed together are counted as fewer, bigger blobs
 *  6. bean count outside MIN_BEANS..MAX_BEANS                                    -> c05
 */
export function gate(checks: Checks, colour: ColourGate, opts: { minBeans?: number } = {}): Retake | null {
  let reason: RetakeReason | null = null;
  // minBeans < MIN_BEANS only for the cooperative grader's calibration photos and the labelled calibration demo
  const fewBeans = checks.count < (opts.minBeans ?? MIN_BEANS);
  const beanColoured = colour.frac.off + colour.frac.white <= COLOUR_RULE.maxOffFrac && colour.frac.dark <= COLOUR_RULE.maxDarkFrac;
  if (checks.too_dark) reason = "dark";
  else if (fewBeans && !checks.too_many_touching && !colour.notGreen) reason = "count";
  else if (checks.too_blurry) reason = "blur";
  else if (colour.notGreen) reason = colour.reason === "big_blob" && beanColoured && checks.too_many_touching ? "spread" : "not_green";
  else if (checks.too_many_touching) reason = "spread";
  else if (fewBeans || checks.count > MAX_BEANS) reason = "count";
  return reason ? { kind: "retake", reason, clip: RETAKE_CLIP[reason] } : null;
}

/**
 * A lot of very dark BEANS (bean shape and the colour of black green coffee, darklot.ts): "not sure, take the sample
 * to the cooperative" (c07) instead of "not green coffee" (c06). Never a band, never counts in the slip.
 */
export function darkLotResult(total: number): Result {
  const counts: Record<BeanCall, number> = { good: 0, dark: 0, insect: 0, broken: 0, unhulled: 0, other_defect: 0, unsure: total };
  return {
    kind: "result", band: "unsure", about: false, unsureReason: "dark_lot", clips: ["c07_unsure"], counts, total, answered: 0, defects: 0,
    interval: wilson(0, 0), unsureFrac: 1, handfuls: 1, canAddHandful: false,
  };
}

/**
 * Everything decided BEFORE the classifier: the retake gate, except that a "too dark beans" photo whose blobs look like
 * a lot of black green beans (colour.darkLot) is "not sure" (c07), not "not green coffee" (c06). The photo must still pass
 * the photo checks that come first (too dark, blurred) and the bean count / touching checks that come after.
 */
export function screen(checks: Checks, colour: ColourGate, opts: { minBeans?: number } = {}): Retake | Result | null {
  const g = gate(checks, colour, opts);
  if (g?.reason === "not_green" && colour.reason === "too_dark_beans" && colour.darkLot?.isDarkLot
    && !checks.too_many_touching && checks.count >= (opts.minBeans ?? MIN_BEANS) && checks.count <= MAX_BEANS) return darkLotResult(checks.count);
  return g;
}

/** The two thresholds of the v2 model (labels json: good_at_or_above / defect_at_or_above). */
export interface Thresholds {
  good: number;
  defect: number;
}
/**
 * defect 0.91 = reports_v2/clean_model.json pair_thresholds (fitted on held-out predictions; afiyah twins held out together).
 * good 0.55 = the pre-registered "sound" threshold sweep (reports_v2/threshold_sweep.md): the smallest t_good in 0.50..0.75
 * with 0 dangerous trays and <= 5% wrong bands on every held-out source with >= 10 bands (0.50 gave 13 wrong of 52 on samruddh).
 */
export const V2_THRESHOLDS: Thresholds = { good: 0.55, defect: 0.91 };

/** The most likely defect type (argmax of probs[1..]) — for display as "possible …" only, never as a fact. */
export function likelyDefect(probs: ArrayLike<number>): DefectName {
  let best = 1;
  for (let i = 2; i < probs.length && i < CLASSES.length; i++) if (probs[i] > probs[best]) best = i;
  return CLASSES[best] as DefectName;
}

/**
 * Turn one bean's class probabilities into a call (two-threshold rule of the v2 model). Touching blobs are never trusted.
 *   P(good) >= thr.good            -> good
 *   1 - P(good) >= thr.defect      -> defect (its possible type = argmax of the defect classes)
 *   otherwise                      -> unsure
 */
export function callBean(probs: ArrayLike<number> | null, touching: boolean, thr: Thresholds): BeanCall {
  if (touching || !probs) return "unsure";
  const pGood = probs[0];
  if (pGood >= thr.good) return "good";
  if (1 - pGood >= thr.defect) return likelyDefect(probs);
  return "unsure";
}

export const isDefect = (c: BeanCall | null | undefined): boolean => !!c && c !== "good" && c !== "unsure";

const BAND_CLIP: Record<Exclude<Band, "unsure">, ClipId> = { clean: "c08_clean", some: "c09_some", many: "c10_many" };

/**
 * One result from the bean calls of one photo (or several handfuls of the same coffee, pooled). Only the defect TOTAL
 * matters for the band; the per-type split is shown as "possible …" and never spoken.
 * Order: nothing answered -> implausible (> 60% defects among answered) -> > 15% unsure -> band of the point
 * estimate, marked "about" when the 95% interval crosses a band edge.
 */
export function decide(calls: BeanCall[], handfuls = 1): Result {
  const counts: Record<BeanCall, number> = { good: 0, dark: 0, insect: 0, broken: 0, unhulled: 0, other_defect: 0, unsure: 0 };
  for (const c of calls) counts[c]++;
  const total = calls.length;
  const defects = DEFECTS.reduce((a, d) => a + counts[d], 0);
  const answered = total - counts.unsure;
  const interval = wilson(defects, answered);
  const unsureFrac = total ? counts.unsure / total : 1;
  let band: Band;
  let unsureReason: UnsureReason = "";
  let about = false;
  if (answered === 0) {
    band = "unsure";
    unsureReason = "too_many_unsure";
  } else if (defects / answered > IMPLAUSIBLE_DEFECT_FRAC) {
    band = "unsure";
    unsureReason = "implausible";
  } else if (unsureFrac > MAX_UNSURE_FRAC) {
    band = "unsure";
    unsureReason = "too_many_unsure";
  } else {
    band = bandOfRate(interval.p);
    about = crossesBandEdge(interval);
  }
  const canAddHandful = band !== "unsure" && about && handfuls < MAX_HANDFULS;
  const clips: ClipId[] = [];
  if (band === "unsure") clips.push("c07_unsure");
  else {
    clips.push(BAND_CLIP[band]);
    if (canAddHandful) clips.push("c18_another");
    // No type clips (c11-c14): the v2 model's defect TYPE is not reliable, so it is never spoken as a fact.
  }
  return { kind: "result", band, about, unsureReason, clips, counts, total, answered, defects, interval, unsureFrac, handfuls, canAddHandful };
}

/**
 * Several handfuls of the SAME coffee: one pooled count and one interval. Safety first — if any single handful is
 * "not sure" on its own (implausible or too many unsure), the whole lot is "not sure"; pooling never dilutes it.
 */
export function decideLot(handfuls: BeanCall[][]): Result {
  const pooled = decide(handfuls.flat(), handfuls.length);
  if (pooled.band === "unsure") return pooled;
  const bad = handfuls.map((h) => decide(h)).find((r) => r.band === "unsure");
  if (!bad) return pooled;
  return { ...pooled, band: "unsure", about: false, unsureReason: bad.unsureReason, clips: ["c07_unsure"], canAddHandful: false };
}
