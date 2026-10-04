/**
 * One shared <audio> element. iOS only lets a page play sound after a tap, so the first tap "unlocks" this
 * element; later clips reuse it. Clips are local files (precached), so this works in airplane mode.
 */
import { audioPack, clipUrl, type ClipId } from "./clips";

let el: HTMLAudioElement | null = null;
let queue: ClipId[] = [];
let listeners = new Set<(id: ClipId | null) => void>();
let current: ClipId | null = null;

function audio(): HTMLAudioElement {
  if (!el) {
    el = new Audio();
    el.preload = "auto";
    el.addEventListener("ended", next);
    el.addEventListener("error", next);
  }
  return el;
}

function emit() { listeners.forEach((l) => l(current)); }

function next() {
  const id = queue.shift() ?? null;
  current = id;
  emit();
  if (!id) return;
  const a = audio();
  a.src = clipUrl(id);
  a.play().catch(() => { current = null; queue = []; emit(); });
}

/** 0.05 s of silence as a WAV data URI (8 kHz, 8-bit mono). */
function silentWav(): string {
  const n = 400;
  const b = new Uint8Array(44 + n);
  const dv = new DataView(b.buffer);
  const str = (o: number, s: string) => { for (let i = 0; i < s.length; i++) b[o + i] = s.charCodeAt(i); };
  str(0, "RIFF"); dv.setUint32(4, 36 + n, true); str(8, "WAVE"); str(12, "fmt ");
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, 8000, true); dv.setUint32(28, 8000, true); dv.setUint16(32, 1, true); dv.setUint16(34, 8, true);
  str(36, "data"); dv.setUint32(40, n, true); b.fill(128, 44);
  let bin = ""; b.forEach((x) => (bin += String.fromCharCode(x)));
  return `data:audio/wav;base64,${btoa(bin)}`;
}

/**
 * Call synchronously inside a tap handler (before any await). Playing a short silent sound, unmuted, from a real
 * tap is what lets iOS Safari play the result clip later on the same element after the async analysis.
 */
export function unlockAudio() {
  const a = audio();
  if (a.dataset.unlocked || current) return;
  a.dataset.unlocked = "1";
  a.src = silentWav();
  a.play().catch(() => { delete a.dataset.unlocked; });
}

export function play(ids: ClipId | ClipId[]) {
  stop();
  if (!audioPack()) return; // no voice in this build (no audio pack): never request clips that are not there
  queue = Array.isArray(ids) ? [...ids] : [ids];
  const a = audio();
  a.muted = false;
  next();
}

export function stop() {
  queue = [];
  const a = audio();
  a.pause();
  current = null;
  emit();
}

export function onClip(fn: (id: ClipId | null) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Resolves when the queue finishes (or after maxMs). */
export function whenIdle(maxMs = 15000): Promise<void> {
  return new Promise((resolve) => {
    if (!current) return resolve();
    const t = setTimeout(done, maxMs);
    const off = onClip((id) => { if (!id) done(); });
    function done() { clearTimeout(t); off(); resolve(); }
  });
}
