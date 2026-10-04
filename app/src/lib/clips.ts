/** The fixed answer list (docs/CLIPS.md). The app never generates text; it only picks one of these. */
export const CLIP_IDS = [
  "c01_welcome", "c02_blur", "c03_dark", "c04_spread", "c05_count", "c06_not_green", "c07_unsure",
  "c08_clean", "c09_some", "c10_many", "c11_dark_beans", "c12_insect", "c13_broken", "c14_unhulled",
  "c15_slip", "c15b_slip_unsure", "c16_privacy", "c17_wiped", "c18_another",
] as const;
export type ClipId = (typeof CLIP_IDS)[number];

/**
 * The AUDIO PACK (public/audio/pack.json) says which voice is in the app and which file extension the clips use.
 * Swapping the voice needs no code: drop the new recordings into public/audio/ under the same names, edit
 * pack.json ("ext", "voice_en", "voice_ar", "placeholder": false), rebuild. The About screen shows pack.json's text.
 */
export interface AudioPack {
  ext: string;
  placeholder: boolean;
  voice_en: string;
  voice_ar: string;
  files?: Partial<Record<ClipId, string>>;
}

const BASE = () => import.meta.env?.BASE_URL ?? "/";
let pack: AudioPack | null = null;

/**
 * Build-time voice switch (vite.config.ts, env VITE_AUDIO):
 *   "pack" (default) — the voice is on only if public/audio/pack.json and its clips are in the build (recordings);
 *                      the default build ships NO text-to-speech audio, so it plays nothing and shows text + icons
 *   "placeholder"   — the build adds the macOS text-to-speech placeholders from app/audio-placeholder/ (local demos only)
 *   "none"          — no voice at all, even if a pack is present
 */
export const AUDIO_MODE: "pack" | "placeholder" | "none" =
  ((import.meta.env?.VITE_AUDIO as string | undefined) ?? "pack") === "none" ? "none"
    : (import.meta.env?.VITE_AUDIO as string | undefined) === "placeholder" ? "placeholder" : "pack";

export function setAudioPack(p: AudioPack | null) { pack = p; }
export function audioPack(): AudioPack | null { return pack; }

export async function loadAudioPack(): Promise<AudioPack | null> {
  if (AUDIO_MODE === "none") return null;
  try {
    const r = await fetch(`${BASE()}audio/pack.json`);
    if (!r.ok || (r.headers.get("content-type") ?? "").includes("text/html")) return null;
    const j = (await r.json()) as AudioPack;
    if (typeof j?.ext !== "string") return null;
    pack = j;
    return j;
  } catch {
    return null;
  }
}

export const clipUrl = (id: ClipId) => {
  const file = pack?.files?.[id] ?? `${id}.${(pack?.ext ?? "mp3").replace(/^\./, "")}`;
  return `${BASE()}audio/${file}`;
};
