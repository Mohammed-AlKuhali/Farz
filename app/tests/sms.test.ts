import { describe, expect, it } from "vitest";
import { arNum, slipBody, smsHref, SMS_MAX, SLIP_TAG, SLIP_UNSURE } from "../src/lib/sms";
import type { BeanCall } from "../src/lib/rules";

const c = (o: Partial<Record<BeanCall, number>>): Record<BeanCall, number> => ({ good: 0, dark: 0, insect: 0, broken: 0, unhulled: 0, other_defect: 0, unsure: 0, ...o });

describe("sms slip", () => {
  const date = new Date(2026, 9, 3);
  it("typical slip fits one SMS and carries the bean count, the defect total, date and the self-check tag", () => {
    const s = slipBody(c({ good: 76, dark: 12, insect: 5, broken: 7 }), 100, date);
    expect(s).toBe(`فرز ٣/١٠/٢٦: ١٠٠ حبة، ٢٤ فيها عيب. ${SLIP_TAG}`);
    expect(s.length).toBeLessThanOrEqual(SMS_MAX);
  });
  it("never exceeds 70 UTF-16 units for any counts up to 160 beans and any date", () => {
    let worst = 0;
    const vals = [0, 1, 9, 10, 99, 100, 160];
    for (const total of [20, 99, 100, 160])
      for (const dark of vals) for (const insect of vals) for (const broken of vals) for (const unhulled of [0, 9, 99]) for (const other_defect of [0, 9, 160]) for (const unsure of [0, 9, 24]) {
        for (const d of [new Date(2026, 9, 3), new Date(2026, 11, 31)]) for (const notSure of [false, true]) {
          const s = slipBody(c({ dark, insect, broken, unhulled, other_defect, unsure }), total, d, { unsure: notSure });
          worst = Math.max(worst, s.length);
          expect(s.length).toBeLessThanOrEqual(SMS_MAX);
          expect(s).toContain(SLIP_TAG);
          if (notSure) expect(s).toContain(SLIP_UNSURE);
          // the date is never followed directly by a space and a number (reads as one long number)
          expect(s).not.toMatch(/\d+\/\d+(\/\d+)? [٠-٩]/u);
          expect(s).not.toMatch(/[٠-٩]+\/[٠-٩]+(\/[٠-٩]+)? [٠-٩]/u);
          // never a middle dot next to Arabic-Indic digits (it looks like ٠)
          expect(s).not.toContain("·");
        }
      }
    expect(worst).toBeLessThanOrEqual(70);
  });
  it("never names a defect TYPE (the type is only a 'possible' guess on screen); other_defect is counted", () => {
    const s = slipBody(c({ good: 87, dark: 4, insect: 3, broken: 2, unhulled: 2, other_defect: 1, unsure: 1 }), 100, date);
    expect(s).toBe(`فرز ٣/١٠/٢٦: ١٠٠ حبة، ١٢ فيها عيب، ١ مش واضحة. ${SLIP_TAG}`);
    for (const word of ["سود", "مخرم", "مكسر", "بقشر", "حامض"]) expect(s).not.toContain(word);
    expect(slipBody(c({ good: 100 }), 100, date)).toBe(`فرز ٣/١٠/٢٦: ١٠٠ حبة، ما فيه عيوب. ${SLIP_TAG}`);
  });
  it("a not-sure result is marked «مش متأكد» on the slip", () => {
    const s = slipBody(c({ broken: 28, unhulled: 5, unsure: 17 }), 50, date, { unsure: true });
    expect(s).toMatch(new RegExp(`^فرز ٣/١٠(/٢٦)?: ${SLIP_UNSURE}، ٥٠ حبة`));
    expect(s.length).toBeLessThanOrEqual(SMS_MAX);
    expect(slipBody(c({ good: 99, dark: 1 }), 100, date)).not.toContain(SLIP_UNSURE);
  });
  it("PRIVACY (I-1): a not-sure slip carries ONLY the date, the total and «مش متأكد» — no per-type or defect counts", () => {
    // the REGRESSION_A case: cbd_aaa_2 (50 beans, 39 broken, 8 unhulled, 3 unsure) used to leak «٣٩ مكسر، ٨ بقشر، ٣؟»
    const s = slipBody(c({ broken: 39, unhulled: 8, unsure: 3 }), 50, date, { unsure: true });
    expect(s).toBe(`فرز ٣/١٠/٢٦: ${SLIP_UNSURE}، ٥٠ حبة. ${SLIP_TAG}`);
    // every possible count mix and both reasons give the same, count-free slip
    for (const total of [20, 50, 100, 160]) for (const k of [0, 1, 7, 39, 99]) {
      const t = slipBody(c({ good: k, dark: k, insect: k, broken: k, unhulled: k, other_defect: k, unsure: k }), total, new Date(2026, 11, 31), { unsure: true });
      expect(t).toBe(`فرز ٣١/١٢/٢٦: ${SLIP_UNSURE}، ${arNum(total)} حبة. ${SLIP_TAG}`);
      expect(t.length).toBeLessThanOrEqual(SMS_MAX);
      for (const word of ["سود", "مخرم", "مكسر", "بقشر", "عيوب", "عيب", "مش واضحة", "؟", "ما فيه"]) expect(t).not.toContain(word);
      // exactly two numbers: the date and the total
      expect(t.match(/[٠-٩]+(\/[٠-٩]+)*/gu)).toEqual(["٣١/١٢/٢٦", arNum(total)]);
    }
  });
  it("never contains a price, currency or grade word", () => {
    const s = slipBody(c({ dark: 3 }), 100, date);
    for (const bad of ["ريال", "دولار", "$", "سعر", "درجة", "ممتاز"]) expect(s).not.toContain(bad);
  });
  it("uses &body= on iOS and ?body= elsewhere", () => {
    expect(smsHref("x", "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)")).toBe("sms:&body=x");
    expect(smsHref("x", "Mozilla/5.0 (Linux; Android 14)")).toBe("sms:?body=x");
  });
});
