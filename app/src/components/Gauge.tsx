/**
 * The qamariya gauge — the half-moon stained-glass window over Yemeni highland doorways, used as the verdict
 * dial. Three glass panes = our three bands (<5%, 5–20%, >20%). The pane of the POINT estimate lights; the 95%
 * Wilson interval is drawn as a bracket around the arch, and any neighbouring pane the bracket reaches glows
 * faintly ("about"). For "not sure" (model / sample uncertainty) no pane lights, no bracket is drawn (no
 * confident-looking numbers) and the hub shows "?".
 * Scale: 0–5% fills the first pane, 5–20% the second, 20–100% the third.
 */
import type { Band, Interval } from "../lib/wilson";

const CX = 150, CY = 158, R_IN = 74, R_OUT = 132, R_BR = 146;

export function tOf(v: number): number {
  const x = Math.max(0, Math.min(1, v));
  let t: number;
  if (x <= 0.05) t = (x / 0.05) / 3;
  else if (x <= 0.2) t = 1 / 3 + ((x - 0.05) / 0.15) / 3;
  else t = 2 / 3 + ((x - 0.2) / 0.8) / 3;
  return Math.max(0.005, Math.min(0.995, t));
}
const MIN_ARC = 6 / 180; // the bracket is always at least ~6 degrees long, so it never collapses to a dot
const pt = (theta: number, r: number) => [CX + r * Math.cos(theta), CY - r * Math.sin(theta)] as const;

export function Gauge({ interval, band, rtl, label }: { interval: Interval; band: Band; rtl: boolean; label: string }) {
  // RTL: low defect rate on the right (start of reading), high on the left.
  const theta = (t: number) => (rtl ? t * Math.PI : Math.PI - t * Math.PI);
  const sector = (t0: number, t1: number, r0: number, r1: number) => {
    const a0 = theta(t0), a1 = theta(t1);
    const [x0, y0] = pt(a0, r1), [x1, y1] = pt(a1, r1), [x2, y2] = pt(a1, r0), [x3, y3] = pt(a0, r0);
    const sweepOuter = rtl ? 0 : 1;
    const sweepInner = rtl ? 1 : 0;
    return `M${x0} ${y0} A${r1} ${r1} 0 0 ${sweepOuter} ${x1} ${y1} L${x2} ${y2} A${r0} ${r0} 0 0 ${sweepInner} ${x3} ${y3}Z`;
  };
  const panes: { band: Exclude<Band, "unsure">; t0: number; t1: number; color: string }[] = [
    { band: "clean", t0: 0, t1: 1 / 3, color: "var(--color-bean)" },
    { band: "some", t0: 1 / 3, t1: 2 / 3, color: "var(--color-qishr)" },
    { band: "many", t0: 2 / 3, t1: 1, color: "var(--color-cherry)" },
  ];
  const ok = Number.isFinite(interval.p) && band !== "unsure";
  let tl = ok ? tOf(interval.lo) : 0, th = ok ? tOf(interval.hi) : 1;
  const tp = ok ? tOf(interval.p) : 0.5;
  if (th - tl < MIN_ARC) {
    const mid = Math.min(1 - MIN_ARC / 2, Math.max(MIN_ARC / 2, (tl + th) / 2));
    tl = mid - MIN_ARC / 2; th = mid + MIN_ARC / 2;
  }
  // a pane is touched by the bracket if the interval reaches into it (edges inclusive, so 100% still touches "many")
  const touched = (p: { t0: number; t1: number }) => ok && th > p.t0 && tl <= p.t1;
  const [bx0, by0] = pt(theta(tl), R_BR), [bx1, by1] = pt(theta(th), R_BR);
  const large = 0;
  const brSweep = rtl ? 0 : 1;
  const [dx, dy] = pt(theta(tp), R_BR);
  const tick = (t: number) => {
    const [a, b] = pt(theta(t), R_BR - 9), [c, d] = pt(theta(t), R_BR + 9);
    return `M${a} ${b} L${c} ${d}`;
  };
  return (
    <svg viewBox="-16 -2 332 176" className="gauge w-full h-auto" role="img" aria-label={label}>
      <defs>
        <radialGradient id="glassGlow" cx="50%" cy="100%" r="80%">
          <stop offset="0%" stopColor="#fff" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#fff" stopOpacity="0" />
        </radialGradient>
      </defs>
      {panes.map((p, i) => {
        const lit = band === p.band;
        const faint = !lit && touched(p); // "about": the interval spills into this neighbouring band
        return (
          <g key={p.band} className="pane" style={{ animationDelay: `${120 * i}ms` }}>
            <path d={sector(p.t0, p.t1, R_IN, R_OUT)} fill={p.color} opacity={lit ? 1 : faint ? 0.42 : 0.16} />
            {lit && <path d={sector(p.t0, p.t1, R_IN, R_OUT)} fill="url(#glassGlow)" />}
            {/* lattice: two muntins + a mid ring per pane, like the gypsum tracery of a qamariya */}
            {[1 / 3, 2 / 3].map((f) => {
              const t = p.t0 + (p.t1 - p.t0) * f;
              const [a, b] = pt(theta(t), R_IN), [c, d] = pt(theta(t), R_OUT);
              return <path key={f} d={`M${a} ${b} L${c} ${d}`} stroke="var(--color-qudad)" strokeWidth="1.6" opacity="0.85" />;
            })}
            <path d={sector(p.t0, p.t1, (R_IN + R_OUT) / 2 - 0.8, (R_IN + R_OUT) / 2 + 0.8)} fill="var(--color-qudad)" opacity="0.85" />
          </g>
        );
      })}
      {/* gypsum frame: outer arch, inner arch, pane dividers, sill */}
      <path d={`M${CX - R_OUT} ${CY} A${R_OUT} ${R_OUT} 0 0 1 ${CX + R_OUT} ${CY}`} stroke="var(--color-qudad)" strokeWidth="5" fill="none" />
      <path d={`M${CX - R_IN} ${CY} A${R_IN} ${R_IN} 0 0 1 ${CX + R_IN} ${CY}`} stroke="var(--color-qudad)" strokeWidth="4" fill="var(--color-husk-2)" />
      {[1 / 3, 2 / 3].map((t) => {
        const [a, b] = pt(theta(t), R_IN), [c, d] = pt(theta(t), R_OUT);
        return <path key={t} d={`M${a} ${b} L${c} ${d}`} stroke="var(--color-qudad)" strokeWidth="5" />;
      })}
      <rect x={CX - R_OUT - 8} y={CY - 1} width={2 * R_OUT + 16} height="8" rx="2" fill="var(--color-qudad)" />
      {/* the evidence: 95% interval bracket around the arch */}
      {ok && (
        <g className="bracket">
          <path d={`M${bx0} ${by0} A${R_BR} ${R_BR} 0 ${large} ${brSweep} ${bx1} ${by1}`} stroke="var(--color-qudad)" strokeWidth="6" fill="none" strokeLinecap="round" pathLength={1} className="bracket-arc" />
          <path d={tick(tl)} stroke="var(--color-qudad)" strokeWidth="3" strokeLinecap="round" />
          <path d={tick(th)} stroke="var(--color-qudad)" strokeWidth="3" strokeLinecap="round" />
          <circle cx={dx} cy={dy} r="5.5" fill="var(--color-husk)" stroke="var(--color-qudad)" strokeWidth="3" />
        </g>
      )}
      {band === "unsure" ? (
        <text x={CX} y={CY - 18} textAnchor="middle" fontSize="54" fontWeight="700" fill="var(--color-saffron)" fontFamily="var(--font-display)">?</text>
      ) : (
        <g transform={`translate(${CX - 20} ${CY - 58})`}>
          <ellipse cx="20" cy="26" rx="10" ry="15" transform="rotate(-28 20 26)" fill={band === "clean" ? "var(--color-bean)" : band === "some" ? "var(--color-qishr)" : "var(--color-cherry)"} />
          <path d="M20 13c-3 4 3 9 0 13s3 9 0 13" transform="rotate(-28 20 26)" stroke="var(--color-husk)" strokeWidth="2" fill="none" />
        </g>
      )}
    </svg>
  );
}
