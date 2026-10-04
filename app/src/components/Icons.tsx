import type { SVGProps } from "react";
import type { BeanCall } from "../lib/rules";

type P = SVGProps<SVGSVGElement> & { size?: number };
const base = (size = 28): SVGProps<SVGSVGElement> => ({
  width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
  strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true, focusable: false,
});

export const IconCamera = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z" /><circle cx="12" cy="13.5" r="4" /></svg>
);
export const IconSpeaker = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" fill="currentColor" stroke="none" /><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11" /></svg>
);
export const IconStop = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><rect x="6.5" y="6.5" width="11" height="11" rx="1.5" fill="currentColor" stroke="none" /></svg>
);
export const IconMessage = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><path d="M4 5h16v11H9l-5 4z" /><path d="M8 9.5h8M8 12.5h5" /></svg>
);
export const IconClock = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></svg>
);
export const IconInfo = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><circle cx="12" cy="12" r="8.5" /><path d="M12 11v5.5" /><circle cx="12" cy="7.8" r="0.6" fill="currentColor" /></svg>
);
export const IconErase = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><path d="M5 7h14M10 7V4.5h4V7M7 7l1 13h8l1-13" /><path d="M10.5 11v5.5M13.5 11v5.5" /></svg>
);
export const IconGallery = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><rect x="3.5" y="5" width="17" height="14" rx="1.5" /><circle cx="9" cy="10" r="1.6" /><path d="M4 17l5-4.5 3.5 3 3-2.5 4.5 4" /></svg>
);
export const IconBack = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><path d="M15 5l-7 7 7 7" /></svg>
);
export const IconHome = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><path d="M4 11l8-6.5 8 6.5M6.5 9.5V19h11V9.5" /></svg>
);
export const IconCoop = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><path d="M3.5 10L12 4l8.5 6" /><path d="M5.5 9v10.5h13V9" /><circle cx="9.5" cy="13.5" r="1.6" /><circle cx="14.5" cy="13.5" r="1.6" /><path d="M7.5 19.5c0-2 1-3 2-3s2 1 2 3M12.5 19.5c0-2 1-3 2-3s2 1 2 3" /></svg>
);
export const IconTray = ({ size, ...p }: P) => (
  <svg {...base(size)} {...p}><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="6" strokeDasharray="1.5 2" /><ellipse cx="10" cy="11" rx="1.4" ry="2" transform="rotate(-25 10 11)" fill="currentColor" stroke="none" /><ellipse cx="14" cy="13.5" rx="1.4" ry="2" transform="rotate(30 14 13.5)" fill="currentColor" stroke="none" /></svg>
);

/** A single coffee bean glyph: oval + the centre-cut crease. Variants draw each call type. */
export function BeanGlyph({ call, size = 36, title }: { call: BeanCall; size?: number; title?: string }) {
  const fill: Record<BeanCall, string> = {
    good: "var(--color-bean)", dark: "#2a1d16", insect: "var(--color-bean)", broken: "var(--color-bean)",
    unhulled: "#c99a5b", other_defect: "var(--color-bean)", unsure: "transparent",
  };
  const ring = call === "good" ? "var(--color-bean-deep)" : call === "unsure" ? "var(--color-saffron)" : "var(--color-cherry)";
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" role={title ? "img" : undefined} aria-label={title} aria-hidden={title ? undefined : true}>
      {call === "unsure" ? (
        <>
          <circle cx="20" cy="20" r="17.5" fill="#2b1b14" />
          <circle cx="20" cy="20" r="15" fill="none" stroke={ring} strokeWidth="3" strokeDasharray="5 4" />
          <text x="20" y="27" textAnchor="middle" fontSize="20" fontWeight="700" fill="var(--color-saffron)" fontFamily="system-ui">?</text>
        </>
      ) : (
        <g transform="rotate(-28 20 20)">
          {call === "unhulled" && <ellipse cx="20" cy="20" rx="12.5" ry="16.5" fill="#e6c995" stroke="#9a7140" strokeWidth="1.5" />}
          {call === "broken" ? (
            <path d="M11 20c0-7 4-13 9-13s9 6 9 13l-3 1.5-2.5-2-3 2.5-2.5-2-2.5 2z" fill={fill[call]} stroke="#55613a" strokeWidth="1.5" />
          ) : (
            <ellipse cx="20" cy="20" rx="9" ry="13" fill={fill[call]} stroke={call === "dark" ? "#000" : "#55613a"} strokeWidth="1.5" />
          )}
          {call !== "broken" && <path d="M20 8c-3 4 3 8 0 12s3 8 0 12" fill="none" stroke={call === "dark" ? "#5a4638" : "#55613a"} strokeWidth="1.6" strokeLinecap="round" />}
          {call === "other_defect" && (
            <path d="M14 13.5l3 2.5-2.5 3 3.5 2.5-2 3.5" fill="none" stroke="var(--color-cherry-deep)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          )}
          {call === "insect" && (
            <g fill="#2a1d16"><circle cx="16" cy="15" r="1.8" /><circle cx="23.5" cy="22" r="1.6" /><circle cx="17" cy="26" r="1.3" /></g>
          )}
        </g>
      )}
    </svg>
  );
}
