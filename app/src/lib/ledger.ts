/**
 * The result's bean ledger: every number on the result screen comes from here, so no "N of M" is ever shown next to
 * a different total (JUDGE_B_hacknation issue 2: "11 of 94" above a "100 beans" card).
 *   found  = every bean the photo rules counted (the slip's total)
 *   found  = sure + notSure          (notSure = touching blobs + beans the model would not call)
 *   sure   = defects + sound         (the beans the band is computed from; the 95% range is defects / sure)
 */
import { DEFECTS, type Result } from "./rules";

export interface Ledger {
  found: number;
  sure: number;
  defects: number;
  sound: number;
  notSure: number;
}

export function ledgerOf(o: Pick<Result, "counts" | "total">): Ledger {
  const defects = DEFECTS.reduce((a, d) => a + (o.counts[d] ?? 0), 0);
  const sound = o.counts.good ?? 0;
  const notSure = o.counts.unsure ?? 0;
  return { found: o.total, sure: defects + sound, defects, sound, notSure };
}

/** true when the ledger adds up (found = sure + notSure, sure = defects + sound) */
export const ledgerAddsUp = (l: Ledger) => l.found === l.sure + l.notSure && l.sure === l.defects + l.sound;

export type DotKind = "defect" | "sound" | "unsure";
/** One dot per bean, in reading order: the sure beans (defects first, then sound), then the not-sure beans. */
export function ledgerDots(l: Ledger): { sure: DotKind[]; notSure: DotKind[] } {
  return {
    sure: [...Array<DotKind>(l.defects).fill("defect"), ...Array<DotKind>(l.sound).fill("sound")],
    notSure: Array<DotKind>(l.notSure).fill("unsure"),
  };
}
