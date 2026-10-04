/**
 * The bean ledger (result screen): one dot per bean, so a farmer who does not read numbers sees the same thing a judge
 * reads in the numbers. Two groups, never one number over another group's total:
 *   [ 100 beans found ]  =  [ Farz was sure: 94 ]  +  [ Farz was not sure: 6 ]
 *   Farz was sure: 94    =  defect 11 (red dot with ×)  +  sound 83 (solid green dot)
 * Not-sure beans are hollow dashed yellow rings, as on the photo. Numbers come from lib/ledger.ts.
 */
import type { ReactNode } from "react";
import { CALL_NAME, num, t, type Lang } from "../lib/i18n";
import { ledgerDots, ledgerOf, type DotKind } from "../lib/ledger";
import type { Result } from "../lib/rules";
import { BeanGlyph } from "./Icons";

const COLS = 20;
const P = 16; // dot pitch (viewBox units)

function Dots({ kinds, rtl, testId }: { kinds: DotKind[]; rtl: boolean; testId: string }) {
  if (!kinds.length) return null;
  const rows = Math.ceil(kinds.length / COLS);
  const cols = Math.min(COLS, kinds.length);
  // the same dot size in every group: a short group is a narrower picture, not smaller dots
  const W = cols * P, H = rows * P;
  return (
    <svg className="ledger-dots" viewBox={`0 0 ${W} ${H}`} style={{ width: `${(cols / COLS) * 100}%`, marginInlineStart: 0 }}
      aria-hidden="true" data-testid={testId} data-n={kinds.length}>
      {kinds.map((k, i) => {
        const c = i % COLS, r = Math.floor(i / COLS);
        const x = (rtl ? cols - 1 - c : c) * P + P / 2, y = r * P + P / 2;
        if (k === "sound") return <circle key={i} cx={x} cy={y} r={6} fill="var(--color-bean-deep)" />;
        if (k === "defect")
          return (
            <g key={i}>
              <circle cx={x} cy={y} r={6.6} fill="var(--color-cherry-deep)" />
              <path d={`M${x - 2.6} ${y - 2.6}l5.2 5.2M${x + 2.6} ${y - 2.6}l-5.2 5.2`} stroke="#fff" strokeWidth="1.7" strokeLinecap="round" />
            </g>
          );
        return <circle key={i} cx={x} cy={y} r={5.4} fill="none" stroke="#a07a10" strokeWidth="2" strokeDasharray="3 2.2" />;
      })}
    </svg>
  );
}

/** "Has a defect" without naming a type: a plain bean with the red × badge used on the photo. */
export function DefectGlyph({ size = 40 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true">
      <g transform="rotate(-28 20 21)">
        <ellipse cx="20" cy="21" rx="9" ry="13" fill="var(--color-bean)" stroke="#55613a" strokeWidth="1.5" />
        <path d="M20 9c-3 4 3 8 0 12s3 8 0 12" fill="none" stroke="#55613a" strokeWidth="1.6" strokeLinecap="round" />
      </g>
      <circle cx="30" cy="10" r="7.5" fill="var(--color-cherry-deep)" />
      <path d="M27 7l6 6M33 7l-6 6" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function Ledger({ lang, rtl, o, children }: { lang: Lang; rtl: boolean; o: Result; children?: ReactNode }) {
  const l = ledgerOf(o);
  const dots = ledgerDots(l);
  const n = (x: number) => num(lang, x);
  return (
    <section className="ledger card-light p-4" aria-label={t(lang, "ledgerAria", { found: n(l.found), sure: n(l.sure), d: n(l.defects), s: n(l.sound), u: n(l.notSure) })}
      data-testid="ledger" data-found={l.found} data-sure={l.sure} data-defects={l.defects} data-sound={l.sound} data-unsure={l.notSure}>
      <p className="ledger-total" data-testid="count-total">
        <span className="font-display ledger-big tabular-nums">{n(l.found)}</span>
        <span className="ledger-total-label">{t(lang, "beansFound")}</span>
      </p>
      <p className="ledger-sum" data-testid="ledger-sum" aria-hidden="true">
        <span className="tabular-nums">{n(l.found)}</span>
        <span className="op">=</span>
        <span className="ledger-sum-term"><span className="sure-chip" /> <span className="tabular-nums">{n(l.sure)}</span></span>
        <span className="op">+</span>
        <span className="ledger-sum-term"><span className="unsure-chip" /> <span className="tabular-nums">{n(l.notSure)}</span></span>
      </p>

      <div className="ledger-group ledger-group-sure" data-testid="count-sure">
        <p className="ledger-head">
          <span className="sure-chip" aria-hidden="true" />
          <span className="flex-1">{t(lang, "sureBeans")}</span>
          <span className="font-display ledger-num tabular-nums">{n(l.sure)}</span>
        </p>
        <Dots kinds={dots.sure} rtl={rtl} testId="dots-sure" />
        <ul className="ledger-split">
          <li className={`count-row ${l.defects === 0 ? "is-zero" : ""}`} data-testid="count-defect">
            <DefectGlyph size={34} />
            <span className="flex-1">{t(lang, "defectBeans")}</span>
            <span className="font-display ledger-num tabular-nums">{n(l.defects)}</span>
          </li>
          <li className={`count-row ${l.sound === 0 ? "is-zero" : ""}`} data-testid="count-good">
            <BeanGlyph call="good" size={34} />
            <span className="flex-1">{CALL_NAME.good[lang]}</span>
            <span className="font-display ledger-num tabular-nums">{n(l.sound)}</span>
          </li>
        </ul>
      </div>

      <div className={`ledger-group ledger-group-unsure ${l.notSure === 0 ? "is-zero" : ""}`} data-testid="count-unsure">
        <p className="ledger-head">
          <BeanGlyph call="unsure" size={34} />
          <span className="flex-1">{t(lang, "notSureBeans")}</span>
          <span className="font-display ledger-num tabular-nums">{n(l.notSure)}</span>
        </p>
        <Dots kinds={dots.notSure} rtl={rtl} testId="dots-unsure" />
      </div>
      {children}
    </section>
  );
}
