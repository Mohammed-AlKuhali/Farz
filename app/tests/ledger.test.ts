/**
 * Result clarity (JUDGE_B_hacknation issue 2): the result screen never shows "N of M" next to a different total.
 * Every number comes from lib/ledger.ts: found = sure + not sure; sure = defects + sound; the verdict's denominator is
 * the "sure" count and its text says so, in Arabic and English.
 */
import { describe, expect, it } from "vitest";
import { decide, decideLot, type BeanCall } from "../src/lib/rules";
import { ledgerAddsUp, ledgerDots, ledgerOf } from "../src/lib/ledger";
import { num, t } from "../src/lib/i18n";

const calls = (o: Partial<Record<BeanCall, number>>): BeanCall[] => Object.entries(o).flatMap(([k, v]) => Array(v).fill(k as BeanCall));

describe("bean ledger", () => {
  it("the judge's case: 100 found = 94 sure (11 defect + 83 sound) + 6 not sure", () => {
    const r = decide(calls({ good: 83, dark: 4, insect: 3, broken: 1, unhulled: 3, unsure: 6 }));
    const l = ledgerOf(r);
    expect(l).toEqual({ found: 100, sure: 94, defects: 11, sound: 83, notSure: 6 });
    expect(ledgerAddsUp(l)).toBe(true);
    expect(l.sure).toBe(r.answered); // the 95% range and the band use exactly these beans
  });
  it("adds up on 2,000 random results, single handfuls and pooled lots", () => {
    let seed = 7;
    const rnd = (n: number) => { seed = (seed * 1103515245 + 12345) % 2 ** 31; return seed % n; };
    for (let k = 0; k < 2000; k++) {
      const hand = () => calls({ good: rnd(120), dark: rnd(8), insect: rnd(8), broken: rnd(8), unhulled: rnd(5), other_defect: rnd(5), unsure: rnd(20) });
      const r = k % 3 ? decide(hand()) : decideLot([hand(), hand()]);
      const l = ledgerOf(r);
      expect(ledgerAddsUp(l)).toBe(true);
      expect(l.found).toBe(r.total);
      expect(l.sure).toBe(r.answered);
      expect(l.defects).toBe(r.defects);
      const d = ledgerDots(l);
      expect(d.sure.length + d.notSure.length).toBe(l.found);
      expect(d.sure.filter((x) => x === "defect").length).toBe(l.defects);
    }
  });
  it("the verdict line names its denominator as the beans Farz was sure about (both languages)", () => {
    const en = t("en", "defectsOf", { d: 11, n: 94 });
    const ar = t("ar", "defectsOf", { d: num("ar", 11), n: num("ar", 94) });
    expect(en).toBe("11 with a defect, out of the 94 beans Farz was sure about");
    expect(en).toContain(t("en", "sureBeans")); // "Farz was sure" — the same words as the ledger's group heading
    expect(ar).toContain("١١");
    expect(ar).toContain("٩٤");
    expect(ar).toContain(t("ar", "sureBeans")); // «فرز متأكد منها» — the same words as the ledger's group heading
  });
  it("the not-sure line counts against the beans FOUND, and says so", () => {
    expect(t("en", "whyUnsure", { u: 30, n: 50 })).toBe("Farz could not tell 30 of the 50 beans it found.");
    expect(t("ar", "whyUnsure", { u: "٣٠", n: "٥٠" })).toContain("لقاها فرز");
    expect(t("ar", "beansFound")).toContain("لقاها فرز");
  });
});
