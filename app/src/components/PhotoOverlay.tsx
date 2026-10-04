import type { BeanCall } from "../lib/rules";

export interface OverlayBean { x0: number; y0: number; x1: number; y1: number; call: BeanCall | null; touching: boolean }

/** How a bean is marked. Shape carries the meaning, colour only repeats it (colour-blind safe, WCAG 1.4.1/1.4.11). */
export type RingKind = "defect" | "unsure" | "sound" | "counted";

const HALO = "#1a100b"; // dark halo under every ring so it shows on a white cloth, a grey sheet or a light box

/**
 * One ring, drawn in the photo's own pixel space. `u` = photo pixels per screen pixel (approx.), so the badge and
 * the "?" keep a readable on-screen size whatever the photo resolution.
 *  defect  = solid red ring on a dark halo + a red badge with a white ×
 *  unsure  = dashed yellow ring on a dark halo + "?"
 *  sound   = thin white ring
 *  counted = thin dotted white ring (retake screens: the bean was counted, never classified)
 */
export function Ring({ kind, cx, cy, r, u }: { kind: RingKind; cx: number; cy: number; r: number; u: number }) {
  if (kind === "sound")
    return (
      <g>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={HALO} strokeOpacity="0.55" strokeWidth="3" vectorEffect="non-scaling-stroke" />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#fff" strokeOpacity="0.9" strokeWidth="1.25" vectorEffect="non-scaling-stroke" />
      </g>
    );
  if (kind === "counted")
    return (
      <g>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={HALO} strokeOpacity="0.55" strokeWidth="3" vectorEffect="non-scaling-stroke" />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#fff" strokeWidth="1.5" strokeDasharray="1.5 3.5" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
      </g>
    );
  if (kind === "unsure")
    return (
      <g>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={HALO} strokeWidth="6" vectorEffect="non-scaling-stroke" />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--color-saffron)" strokeWidth="3" strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />
        <text x={cx} y={cy + r * 0.36} textAnchor="middle" fontSize={Math.max(r * 1.05, 11 * u)} fontWeight="800" fill="var(--color-saffron)" stroke={HALO} strokeWidth={Math.max(r * 0.08, 1.6 * u)} paintOrder="stroke" fontFamily="system-ui">?</text>
      </g>
    );
  // defect: badge at the ring's upper outer corner (mirrors nothing: it is a mark, not text)
  const br = Math.max(5.5 * u, r * 0.34);
  const bx = cx + r * 0.72, by = cy - r * 0.72;
  const k = br * 0.45;
  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={HALO} strokeWidth="6.5" vectorEffect="non-scaling-stroke" />
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--color-cherry)" strokeWidth="3.5" vectorEffect="non-scaling-stroke" />
      <circle cx={bx} cy={by} r={br} fill="var(--color-cherry)" stroke={HALO} strokeWidth="2" vectorEffect="non-scaling-stroke" />
      <path d={`M${bx - k} ${by - k}L${bx + k} ${by + k}M${bx + k} ${by - k}L${bx - k} ${by + k}`} stroke="#fff" strokeWidth="2" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </g>
  );
}

/** Legend swatch: the same ring drawn small, on a light patch like the cloth. */
export function RingSwatch({ kind, size = 30 }: { kind: RingKind; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 30 30" aria-hidden="true" className="shrink-0">
      <rect x="0" y="0" width="30" height="30" rx="7" fill="#e9e4da" />
      <ellipse cx="15" cy="15.5" rx="5" ry="7" transform="rotate(-25 15 15.5)" fill="#a9b07e" />
      <Ring kind={kind} cx={15} cy={15.5} r={9} u={0.62} />
    </svg>
  );
}

function kindOf(b: OverlayBean, classified: boolean): RingKind | null {
  const call = b.call ?? (b.touching ? "unsure" : null);
  if (call === "unsure") return "unsure";
  if (call && call !== "good") return "defect";
  if (call === "good") return "sound";
  return classified ? null : "counted";
}

/** The photo with a ring around every counted bean. Height is capped so the verdict stays above the fold. */
export function PhotoOverlay({ src, w, h, beans, alt, classified = true, caption }: {
  src: string; w: number; h: number; beans: OverlayBean[]; alt: string; classified?: boolean; caption?: string;
}) {
  const u = w / 360;
  return (
    <figure className="photo-frame" style={{ ["--ar" as string]: `${w / h}` }}>
      <div className="photo relative overflow-hidden rounded-[18px] bg-husk-2" style={{ aspectRatio: `${w} / ${h}` }}>
        <img src={src} alt={alt} className="absolute inset-0 h-full w-full object-fill" />
        <svg viewBox={`0 0 ${w} ${h}`} className="absolute inset-0 h-full w-full" aria-hidden="true">
          {beans.map((b, i) => {
            const kind = kindOf(b, classified);
            if (!kind) return null;
            const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
            const r = Math.max(b.x1 - b.x0, b.y1 - b.y0) / 2 + Math.max(4, w / 300);
            return (
              <g key={i} className="ring" style={{ animationDelay: `${Math.min(i * 12, 900)}ms` }}>
                <Ring kind={kind} cx={cx} cy={cy} r={r} u={u} />
              </g>
            );
          })}
        </svg>
        {caption && <figcaption className="photo-cap">{caption}</figcaption>}
      </div>
    </figure>
  );
}
