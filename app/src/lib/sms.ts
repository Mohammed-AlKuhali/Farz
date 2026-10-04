/**
 * The SMS slip: bean and defect counts only + date + «فحص ذاتي مش تصنيف» ("self-check, not a grade"), Arabic,
 * at most 70 UTF-16 code units = one UCS-2 SMS segment. No price, no grade, no name, no place.
 * A "not sure" result (c07, any reason) carries only the date, the total bean count and «مش متأكد».
 */
import { DEFECTS, type BeanCall } from "./rules";

export const SMS_MAX = 70;
export const SLIP_TAG = "فحص ذاتي مش تصنيف";

const AR_DIGITS = "٠١٢٣٤٥٦٧٨٩";
export const arNum = (n: number | string) => String(n).replace(/[0-9]/g, (d) => AR_DIGITS[+d]);

/**
 * The slip says how many beans and how many have a defect — never the defect TYPE: the v2 model's type is a guess
 * ("possible …" on screen), so it must not leave the phone as if it were a finding.
 * A "not sure" result (c07, any reason) carries only the date, the total bean count and «مش متأكد».
 */
export const SLIP_UNSURE = "مش متأكد";

export function slipBody(counts: Record<BeanCall, number>, total: number, date: Date, opts: { unsure?: boolean } = {}): string {
  const d = date.getDate(), m = date.getMonth() + 1, y = date.getFullYear() % 100;
  const dLong = arNum(`${d}/${m}/${String(y).padStart(2, "0")}`);
  const dShort = arNum(`${d}/${m}`);
  // PRIVACY (REGRESSION_A I-1): a "not sure" result carries ONLY the date, the bean count and «مش متأكد».
  // Farz has just said it cannot judge this sample, so no defect counts may leave the phone:
  // "39 defects of 50" for top-grade beans could be used to under-grade the seller.
  if (opts.unsure) {
    const a = `فرز ${dLong}: ${SLIP_UNSURE}، ${arNum(total)} حبة. ${SLIP_TAG}`;
    return a.length <= SMS_MAX ? a : `فرز ${dShort}: ${SLIP_UNSURE}، ${arNum(total)} حبة. ${SLIP_TAG}`;
  }
  const defects = DEFECTS.reduce((a, k) => a + (counts[k] ?? 0), 0);
  const T = arNum(total), D = arNum(defects), U = counts.unsure ?? 0;
  const defectPart = defects > 0 ? `${D} فيها عيب` : "ما فيه عيوب";
  const candidates = [
    `فرز ${dLong}: ${T} حبة، ${defectPart}${U > 0 ? `، ${arNum(U)} مش واضحة` : ""}. ${SLIP_TAG}`,
    `فرز ${dShort}: ${T} حبة، ${defectPart}${U > 0 ? `، ${arNum(U)} مش واضحة` : ""}. ${SLIP_TAG}`,
    `فرز ${dShort}: ${T} حبة، ${defectPart}. ${SLIP_TAG}`,
  ];
  for (const c of candidates) if (c.length <= SMS_MAX) return c;
  return candidates[candidates.length - 1];
}

/** iOS wants `sms:&body=`, Android `sms:?body=` (docs/PLAN.md §3 row 8 asks to support both). */
export function smsHref(body: string, ua: string = typeof navigator !== "undefined" ? navigator.userAgent : ""): string {
  const ios = /iPhone|iPad|iPod/i.test(ua) || (/Macintosh/.test(ua) && typeof document !== "undefined" && "ontouchend" in document);
  return `sms:${ios ? "&" : "?"}body=${encodeURIComponent(body)}`;
}
