import { createContext, useCallback, useContext, useEffect, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import { Engine, EngineStartError, type Analysis, type AnalyzeOptions, type ReadyInfo } from "./lib/engine";
import { play, stop, unlockAudio, onClip, whenIdle } from "./lib/audio";
import { loadAudioPack, type AudioPack, type ClipId } from "./lib/clips";
import { addHistory, historyRecordFor, listHistory, shownCounts, wipeHistory, type HistoryRecord } from "./lib/history";
import { loadDemoTrays, type DemoTray } from "./lib/demo";
import { BAND_NAME, CALL_NAME, CLIP_TEXT, POSSIBLE_NAME, num, pct, t, type Lang } from "./lib/i18n";
import { DEFECTS, MAX_HANDFULS, decideLot, type BeanCall, type Result, type RetakeReason } from "./lib/rules";
import { loadStoredHead, toHead, type StoredHead } from "./lib/localcal";
import { Calibrate, CalibratedChip } from "./components/Calibrate";
import { slipBody, smsHref } from "./lib/sms";
import { Gauge } from "./components/Gauge";
import { PhotoOverlay, RingSwatch } from "./components/PhotoOverlay";
import {
  BeanGlyph, IconBack, IconCamera, IconClock, IconCoop, IconErase, IconGallery, IconInfo, IconMessage, IconSpeaker, IconStop, IconTray,
} from "./components/Icons";
import { About } from "./components/About";
import { DefectGlyph, Ledger } from "./components/Ledger";

/** true when this build has a voice (an audio pack was found and VITE_AUDIO is not "none"); otherwise text + icons only */
const VoiceOn = createContext(false);

type Screen = "home" | "working" | "result" | "history" | "about" | "calibrate";

declare global {
  interface Window {
    __farz?: {
      last?: { kind: string; ms: Analysis["ms"]; total: number; stub: boolean; band?: string; about?: boolean; reason?: string; handfuls?: number; tapToResultMs?: number; calibrated?: boolean; counts?: Record<string, number> };
      /** the cooperative head in use ("calibrate for your cooperative" pilot), null = base model thresholds */
      head?: { name: string; n: number; demo: boolean } | null;
      ready?: ReadyInfo;
      swReady?: boolean;
      startFailed?: boolean;
    };
  }
}

/** localStorage can throw (Safari "Block All Cookies", private modes): never let that blank the app. */
const store = {
  get(k: string): string | null { try { return localStorage.getItem(k); } catch { return null; } },
  set(k: string, v: string) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};

export default function App() {
  const [lang, setLang] = useState<Lang>(() => (store.get("farz.lang") === "en" ? "en" : "ar"));
  const rtl = lang === "ar";
  const [screen, setScreen] = useState<Screen>("home");
  const [info, setInfo] = useState<ReadyInfo | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [error, setError] = useState(false);
  /** the inference worker did not start within 15 s: show "Try again" instead of "Counting…" forever */
  const [startFailed, setStartFailed] = useState(false);
  const [demos, setDemos] = useState<DemoTray[]>([]);
  const [demoOpen, setDemoOpen] = useState(false);
  const [slipOpen, setSlipOpen] = useState(false);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [swReady, setSwReady] = useState(false);
  const [playing, setPlaying] = useState<ClipId | null>(null);
  const [activeDemo, setActiveDemo] = useState<DemoTray | null>(null);
  const [pack, setPack] = useState<AudioPack | null>(null);
  /** the retake screen belongs to an "add another handful" step: retaking continues the same lot */
  const [retakeAdds, setRetakeAdds] = useState(false);
  /** the stored cooperative head and whether it is valid for the loaded model (pilot; null = none) */
  const [storedHead, setStoredHead] = useState<StoredHead | null>(null);
  const [headActive, setHeadActive] = useState(false);
  const engine = useRef<Engine | null>(null);
  const firstCheck = useRef(true);
  const runId = useRef(0); // a result that arrives after the user navigated away is dropped
  const mainRef = useRef<HTMLElement>(null);
  /** handfuls of the SAME coffee (bean calls only, never photos), pooled into one count and one interval */
  const lot = useRef<BeanCall[][]>([]);
  const lotHistoryId = useRef<number | undefined>(undefined);
  const demoRef = useRef<DemoTray | null>(null);

  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = rtl ? "rtl" : "ltr";
    store.set("farz.lang", lang);
  }, [lang, rtl]);

  /** Install (or clear) the cooperative head on the engine. A head made with another model file is never used. */
  const applyHead = useCallback((h: StoredHead | null, r: ReadyInfo | null) => {
    const head = r && !r.stub && r.embedDim ? toHead(h, r.modelSha) : null;
    if (engine.current) engine.current.head = head;
    setStoredHead(h);
    setHeadActive(!!head);
    window.__farz = { ...(window.__farz ?? {}), head: head && h ? { name: h.name, n: h.n_good + h.n_defect, demo: !!h.demo } : null };
  }, []);

  const startEngine = useCallback(() => {
    engine.current?.dispose();
    const e = new Engine();
    engine.current = e;
    e.ready.then(async (r) => {
      if (engine.current !== e) return;
      setInfo(r);
      setStartFailed(false);
      window.__farz = { ...(window.__farz ?? {}), ready: r, startFailed: false };
      applyHead(await loadStoredHead(), r);
    });
    e.started.catch(() => {
      if (engine.current !== e) return;
      setStartFailed(true);
      window.__farz = { ...(window.__farz ?? {}), startFailed: true };
    });
  }, [applyHead]);

  useEffect(() => {
    startEngine();
    loadDemoTrays(import.meta.env.BASE_URL).then(setDemos);
    loadAudioPack().then(setPack);
    listHistory().then(setHistory).catch(() => setHistory([]));
    const off = onClip(setPlaying);
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.ready.then(() => {
        setSwReady(true);
        window.__farz = { ...(window.__farz ?? {}), swReady: true };
      });
    }
    return () => { off(); };
  }, [startEngine]);

  useEffect(() => {
    mainRef.current?.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  }, [screen]);

  const go = useCallback((s: Screen) => {
    runId.current++;
    stop();
    setSlipOpen(false);
    setDemoOpen(false);
    if (s !== "result" && photoUrl) {
      URL.revokeObjectURL(photoUrl); // the photo is never kept
      setPhotoUrl(null);
      setAnalysis(null);
    }
    setScreen(s);
  }, [photoUrl]);

  const run = useCallback(async (blob: Blob, tTap: number = performance.now(), adding = false, opts: AnalyzeOptions = {}) => {
    if (!engine.current) return;
    const myRun = ++runId.current;
    stop();
    setError(false);
    setDemoOpen(false);
    setSlipOpen(false);
    if (photoUrl) URL.revokeObjectURL(photoUrl);
    const url = URL.createObjectURL(blob);
    setPhotoUrl(url);
    setAnalysis(null);
    setScreen("working");
    let a: Analysis;
    try {
      a = await engine.current.analyze(blob, opts);
    } catch (e) {
      console.error(e);
      if (myRun !== runId.current) return;
      if (e instanceof EngineStartError) setStartFailed(true);
      else setError(true);
      setScreen("home");
      URL.revokeObjectURL(url);
      setPhotoUrl(null);
      return;
    }
    if (myRun !== runId.current) return;
    const keepLot = adding && lot.current.length > 0 && lot.current.length < MAX_HANDFULS;
    if (a.outcome.kind === "result" && a.outcome.unsureReason === "dark_lot") {
      // a lot of black beans (no model call): "not sure" on its own, never pooled with other handfuls
      lot.current = [];
      lotHistoryId.current = undefined;
      setRetakeAdds(false);
    } else if (a.outcome.kind === "result") {
      const calls = a.beans.map((b) => b.call as BeanCall);
      lot.current = keepLot ? [...lot.current, calls] : [calls];
      if (!keepLot) lotHistoryId.current = undefined;
      a = { ...a, outcome: decideLot(lot.current) };
      setRetakeAdds(false);
    } else {
      // a retake inside "add another handful" keeps the lot; a retake of a fresh check starts a new one
      if (!keepLot) { lot.current = []; lotHistoryId.current = undefined; }
      setRetakeAdds(keepLot);
    }
    setAnalysis(a);
    setScreen("result");
    const o = a.outcome;
    const clips: ClipId[] = o.kind === "retake" ? [o.clip] : [...o.clips];
    if (o.kind === "result" && firstCheck.current) {
      clips.push("c16_privacy");
      firstCheck.current = false;
    }
    play(clips);
    window.__farz = {
      ...(window.__farz ?? {}),
      last: {
        kind: o.kind, ms: a.ms, total: a.checks.count, stub: a.stub,
        band: o.kind === "result" ? o.band : undefined, about: o.kind === "result" ? o.about : undefined,
        reason: o.kind === "retake" ? o.reason : o.unsureReason || undefined, handfuls: o.kind === "result" ? o.handfuls : undefined,
        calibrated: a.calibrated, counts: o.kind === "result" ? { ...o.counts } : undefined,
      },
    };
    // tap -> result painted (two frames after React commits the result screen)
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (window.__farz?.last) window.__farz.last.tapToResultMs = performance.now() - tTap;
    }));
    if (o.kind === "result") {
      // a storage failure must never throw away a valid result
      try {
        const rec: HistoryRecord = historyRecordFor(o, Date.now(), demoRef.current ? (demoRef.current.synthetic ? "synthetic" : "real") : opts.minBeans ? "real" : undefined);
        if (o.handfuls > 1 && lotHistoryId.current !== undefined) rec.id = lotHistoryId.current; // same coffee: update its record
        lotHistoryId.current = await addHistory(rec);
        setHistory(await listHistory());
      } catch (e) {
        console.warn("[Farz] could not save to history", e);
      }
    }
  }, [photoUrl]);

  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    unlockAudio();
    setActiveDemo(null);
    demoRef.current = null;
    const f = e.target.files?.[0];
    e.target.value = "";
    if (f) void run(f);
  };

  /** "Add another handful": the next photo is pooled with this lot (same coffee). */
  const onAddFile = (e: ChangeEvent<HTMLInputElement>) => {
    unlockAudio();
    const f = e.target.files?.[0];
    e.target.value = "";
    if (f) void run(f, performance.now(), true);
  };

  const onDemo = async (tray: DemoTray) => {
    const tTap = performance.now();
    unlockAudio();
    setActiveDemo(tray);
    demoRef.current = tray;
    try {
      const r = await fetch(tray.src);
      if (!r.ok) throw new Error(String(r.status));
      await run(await r.blob(), tTap);
    } catch {
      setError(true);
    }
  };

  const say = (ids: ClipId | ClipId[]) => {
    unlockAudio();
    if (playing) stop();
    else play(ids);
  };

  const result = analysis?.outcome.kind === "result" ? analysis.outcome : null;

  return (
    <VoiceOn.Provider value={!!pack}>
    <div className="app mx-auto flex min-h-dvh w-full max-w-[480px] flex-col">
      <header className="topbar sticky top-0 z-20 flex items-center gap-2 px-4 pb-2 pt-[max(env(safe-area-inset-top),10px)]">
        {screen !== "home" ? (
          <button className="iconbtn" onClick={() => go("home")} aria-label={t(lang, "back")}>
            <IconBack className="flip-rtl" />
          </button>
        ) : (
          <span className="w-12" aria-hidden="true" />
        )}
        <button className="wordmark flex min-h-12 flex-1 items-center justify-center" onClick={() => go("home")} aria-label={`فرز Farz — ${t(lang, "home")}`}>
          <span className="font-display text-[34px] leading-none text-qudad">فرز</span>
          <span className="ms-2 font-display text-[15px] tracking-[0.2em] text-parchment">FARZ</span>
        </button>
        <button className="iconbtn text-[17px] font-bold" onClick={() => setLang(lang === "ar" ? "en" : "ar")} aria-label={lang === "ar" ? "English" : "العربية"} lang={lang === "ar" ? "en" : "ar"}>
          {lang === "ar" ? "EN" : "ع"}
        </button>
      </header>

      <main ref={mainRef} tabIndex={-1} className="flex flex-1 flex-col px-4 pb-[max(env(safe-area-inset-bottom),16px)] outline-none">
        {info?.stub && (
          <p role="alert" className="mb-3 rounded-xl border-2 border-saffron bg-saffron/15 px-3 py-2 text-[15px] text-saffron">
            ⚠ {t(lang, "stub")}
          </p>
        )}
        {error && (
          <p role="alert" className="mb-3 rounded-xl border-2 border-cherry bg-cherry/15 px-3 py-2 text-[17px] text-qudad">{t(lang, "error")}</p>
        )}
        {startFailed && (
          <div role="alert" className="mb-3 flex flex-col gap-2 rounded-xl border-2 border-cherry bg-cherry/15 px-3 py-2 text-[17px] text-qudad" data-testid="start-failed">
            <p>{t(lang, "startFailed")}</p>
            <button className="btn btn-primary" data-testid="start-retry" onClick={() => { setStartFailed(false); setError(false); startEngine(); }}>
              {t(lang, "tryAgain")}
            </button>
          </div>
        )}

        {screen === "home" && (
          <Home lang={lang} demos={demos} onFile={onFile} onDemo={() => setDemoOpen(true)} playing={playing} say={say} swReady={swReady}
            onHistory={() => go("history")} onAbout={() => go("about")} onCalibrate={() => go("calibrate")}
            head={headActive ? storedHead : null} />
        )}
        {screen === "working" && <Working lang={lang} photoUrl={photoUrl} />}
        {screen === "result" && analysis && photoUrl && (
          analysis.outcome.kind === "retake" ? (
            <Retake lang={lang} a={analysis} photoUrl={photoUrl} reason={analysis.outcome.reason} clip={analysis.outcome.clip}
              onFile={retakeAdds ? onAddFile : onFile} playing={playing} say={say} />
          ) : (
            <ResultView lang={lang} rtl={rtl} a={analysis} photoUrl={photoUrl} onFile={onFile} onAddFile={onAddFile} playing={playing} say={say}
              onSlip={() => { unlockAudio(); setSlipOpen(true); play(result?.band === "unsure" ? "c15b_slip_unsure" : "c15_slip"); }}
              demo={activeDemo} head={a_calibrated(analysis) ? storedHead : null} />
          )
        )}
        {screen === "history" && (
          <HistoryView lang={lang} items={history} playing={playing} say={say} onWipe={async () => {
            unlockAudio();
            try { await wipeHistory(); } catch (e) { console.warn("[Farz] wipe failed", e); }
            setHistory(await listHistory().catch(() => []));
            play("c17_wiped");
          }} />
        )}
        {screen === "about" && <About lang={lang} info={info} swReady={swReady} pack={pack} playing={playing} say={say} head={headActive ? storedHead : null} />}
        {screen === "calibrate" && engine.current && (
          <Calibrate lang={lang} engine={engine.current} info={info} stored={storedHead} active={headActive}
            onSaved={(h) => applyHead(h, info)} onReset={() => applyHead(null, info)} />
        )}
      </main>

      {demoOpen && (
        <Sheet onClose={() => setDemoOpen(false)} label={t(lang, "demoTitle")}>
          <h2 className="mb-3 font-display text-[24px] text-basalt" tabIndex={-1} data-autofocus>{t(lang, "demoTitle")}</h2>
          {([true, false] as const).map((syn) => {
            const group = demos.filter((d) => d.synthetic === syn);
            if (!group.length) return null;
            return (
              <section key={String(syn)} className="mb-4">
                <h3 className="mb-2 text-[16px] font-bold text-basalt">{t(lang, syn ? "demoGroupSynthetic" : "demoGroupReal")}</h3>
                <ul className="grid grid-cols-2 gap-3">
                  {group.map((d) => (
                    <li key={d.src}>
                      <button className="demo-tile w-full" onClick={() => onDemo(d)} data-testid="demo-tray" data-file={d.src.split("/").pop()}>
                        <img src={d.src} alt="" loading="lazy" className="aspect-square w-full rounded-xl object-cover" />
                        <span className={`tag mt-1.5 inline-block ${d.synthetic ? "" : "tag-real"}`}>{d.synthetic ? t(lang, "synthetic") : t(lang, "realPhoto")}</span>
                        <span className="mt-1 block text-[15px] font-bold leading-tight text-basalt">{lang === "ar" && d.titleAr ? d.titleAr : d.title}</span>
                        {!d.synthetic && (d.noteAr || d.noteEn) && (
                          <span className="mt-1 block text-[14px] leading-snug text-basalt/80">{lang === "ar" ? d.noteAr ?? d.noteEn : d.noteEn ?? d.noteAr}</span>
                        )}
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            );
          })}
          <button className="btn btn-quiet mt-1 w-full" onClick={() => setDemoOpen(false)}>{t(lang, "cancel")}</button>
        </Sheet>
      )}

      {slipOpen && result && (
        <SlipSheet lang={lang} counts={result.band === "unsure" ? null : result.counts}
          body={slipBody(result.counts, result.total, new Date(), { unsure: result.band === "unsure" })} playing={playing}
          onClose={() => { stop(); setSlipOpen(false); }} />
      )}
    </div>
    </VoiceOn.Provider>
  );
}

const a_calibrated = (a: Analysis | null) => !!a?.calibrated;

/* ---------- shared pieces ---------- */

function CameraButton({ lang, onFile, big = false, label, quiet = false, testId }: {
  lang: Lang; onFile: (e: ChangeEvent<HTMLInputElement>) => void; big?: boolean; label?: string; quiet?: boolean; testId?: string;
}) {
  return (
    <label className={big ? "tray-btn" : `btn ${quiet ? "btn-quiet" : "btn-primary"} w-full`} data-testid={big ? "camera-main" : testId} onClick={unlockAudio}>
      <input type="file" accept="image/*" capture="environment" className="sr-only" onChange={onFile} data-testid={testId ? `${testId}-input` : "photo-input"} />
      {big ? (
        <span className="tray-face">
          <IconCamera size={64} />
          <span className="mt-2 block font-display text-[26px] leading-tight">{label ?? t(lang, "takePhoto")}</span>
        </span>
      ) : (
        <>
          <IconCamera />
          <span>{label ?? t(lang, "takePhoto")}</span>
        </>
      )}
    </label>
  );
}

function SayButton({ lang, ids, playing, say, className = "", label }: { lang: Lang; ids: ClipId | ClipId[]; playing: ClipId | null; say: (ids: ClipId | ClipId[]) => void; className?: string; label?: string }) {
  const voice = useContext(VoiceOn);
  if (!voice) return null; // no audio pack in this build: the same words are on screen
  const on = playing !== null;
  return (
    <button className={`btn btn-voice ${on ? "is-playing" : ""} ${className}`} onClick={() => say(ids)} aria-pressed={on}>
      {on ? <IconStop /> : <IconSpeaker />}
      <span>{on ? t(lang, "stop") : label ?? t(lang, "listen")}</span>
    </button>
  );
}

/** Bottom sheet: modal dialog that takes focus, keeps Tab inside, closes on Escape and gives focus back. */
function Sheet({ children, onClose, label }: { children: ReactNode; onClose: () => void; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const back = document.activeElement as HTMLElement | null;
    const box = ref.current;
    (box?.querySelector<HTMLElement>("[data-autofocus]") ?? box)?.focus();
    const k = (e: KeyboardEvent) => {
      if (e.key === "Escape") return onClose();
      if (e.key !== "Tab" || !box) return;
      const f = [...box.querySelectorAll<HTMLElement>('button, a[href], input, [tabindex]:not([tabindex="-1"])')].filter((x) => !x.hasAttribute("disabled"));
      if (!f.length) return;
      const first = f[0], last = f[f.length - 1];
      if (e.shiftKey && (document.activeElement === first || !box.contains(document.activeElement))) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    window.addEventListener("keydown", k);
    return () => { window.removeEventListener("keydown", k); back?.focus?.(); };
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-40 flex items-end justify-center bg-black/55" onClick={onClose}>
      <div ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-label={label} className="sheet max-h-[90dvh] w-full max-w-[480px] overflow-y-auto overscroll-contain rounded-t-[26px] bg-qudad p-5 pb-[max(env(safe-area-inset-bottom),20px)] outline-none" onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
}

/* ---------- screens ---------- */

function Home({ lang, demos, onFile, onDemo, playing, say, swReady, onHistory, onAbout, onCalibrate, head }: {
  lang: Lang; demos: DemoTray[]; onFile: (e: ChangeEvent<HTMLInputElement>) => void; onDemo: () => void; playing: ClipId | null;
  say: (ids: ClipId | ClipId[]) => void; swReady: boolean; onHistory: () => void; onAbout: () => void; onCalibrate: () => void;
  head: StoredHead | null;
}) {
  return (
    <div className="flex flex-1 flex-col items-center">
      <p className="mt-1 text-center text-[17px] text-parchment">{t(lang, "tagline")}</p>
      {head && <CalibratedChip lang={lang} head={head} className="mt-2" onClick={onCalibrate} />}
      <SayButton lang={lang} ids="c01_welcome" playing={playing} say={say} className="mt-4" />
      <VoiceComing lang={lang} />
      <div className="my-6">
        <CameraButton lang={lang} onFile={onFile} big />
      </div>
      <ol className="steps grid w-full grid-cols-3 gap-2 text-center" aria-label={CLIP_TEXT.c01_welcome[lang]}>
        <li><StepCloth /><span>{t(lang, "stepCloth")}</span></li>
        <li><StepBeans /><span>{t(lang, "stepBeans")}</span></li>
        <li><StepPhone /><span>{t(lang, "stepPhone")}</span></li>
      </ol>
      <div className="mt-6 grid w-full grid-cols-2 gap-3">
        {demos.length > 0 && (
          <button className="btn btn-quiet col-span-2" onClick={onDemo} data-testid="demo-open">
            <IconTray /> <span>{t(lang, "demo")}</span>
          </button>
        )}
        <label className="btn btn-quiet" onClick={unlockAudio}>
          <input type="file" accept="image/*" className="sr-only" onChange={onFile} />
          <IconGallery /> <span>{t(lang, "gallery")}</span>
        </label>
        <button className="btn btn-quiet" onClick={onHistory}>
          <IconClock /> <span>{t(lang, "history")}</span>
        </button>
        <button className="btn btn-quiet col-span-2 min-h-[52px] text-[17px]" onClick={onCalibrate} data-testid="calibrate-open">
          <IconCoop /> <span>{t(lang, "calibrate")}</span> <span className="tag">{t(lang, "calibratedPilot")}</span>
        </button>
      </div>
      <div className="mt-auto flex w-full items-center justify-between pt-6 text-[14px] text-parchment">
        <span className="flex items-center gap-2" data-testid="offline-status">
          <span className={`inline-block h-2.5 w-2.5 rounded-full ${swReady ? "bg-bean" : "bg-parchment/50"}`} aria-hidden="true" />
          {swReady ? t(lang, "offlineReady") : t(lang, "offlinePending")}
        </span>
        <button className="flex min-h-12 items-center gap-1.5 px-2 underline-offset-4 hover:underline" onClick={onAbout}>
          <IconInfo size={22} /> {t(lang, "about")}
        </button>
      </div>
    </div>
  );
}

/** Shown once on Home when the build has no voice: says so plainly instead of a button that does nothing. */
function VoiceComing({ lang }: { lang: Lang }) {
  const voice = useContext(VoiceOn);
  if (voice) return null;
  return (
    <p className="voice-coming mt-4" data-testid="voice-coming">
      <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" fill="currentColor" /><path d="M15.5 9.5l5 5M20.5 9.5l-5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
      <span>{t(lang, "voiceComing")}</span>
    </p>
  );
}

function Working({ lang, photoUrl }: { lang: Lang; photoUrl: string | null }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-6" role="status" aria-live="polite">
      <div className="working-tray">
        {photoUrl && <img src={photoUrl} alt="" className="h-full w-full rounded-full object-cover opacity-60" />}
      </div>
      <p className="font-display text-[26px] text-qudad">{t(lang, "counting")}</p>
    </div>
  );
}

const RETAKE_ICON: Record<RetakeReason, ReactNode> = {
  dark: <PictDark />, blur: <PictBlur />, not_green: <PictNotGreen />, count: <PictCount />, spread: <PictSpread />,
};

function Retake({ lang, a, photoUrl, reason, clip, onFile, playing, say }: {
  lang: Lang; a: Analysis; photoUrl: string; reason: RetakeReason; clip: ClipId; onFile: (e: ChangeEvent<HTMLInputElement>) => void;
  playing: ClipId | null; say: (ids: ClipId | ClipId[]) => void;
}) {
  // beans were counted, never classified: neutral dotted rings (touching blobs keep the "?" ring)
  const showRings = reason === "spread" || reason === "count";
  return (
    <div className="flex flex-col gap-4" data-testid="retake" data-reason={reason}>
      <div className="card-dark flex flex-col items-center gap-3 p-5 text-center">
        <div className="pict">{RETAKE_ICON[reason]}</div>
        <h1 className="text-[26px] font-bold leading-snug text-qudad">{CLIP_TEXT[clip][lang]}</h1>
        <SayButton lang={lang} ids={clip} playing={playing} say={say} />
      </div>
      <CameraButton lang={lang} onFile={onFile} label={t(lang, "retake")} />
      <PhotoOverlay src={photoUrl} w={a.workW} h={a.workH} alt="" classified={false}
        beans={showRings ? a.beans.map((b) => ({ ...b, call: reason === "spread" && b.touching ? "unsure" : null })) : []} />
      {showRings && <p className="text-center text-[17px] text-qudad" data-testid="retake-count">{num(lang, a.checks.count)} {t(lang, "beans")}</p>}
      <p className="text-center text-[14px] text-parchment">{t(lang, "photoGone")}</p>
    </div>
  );
}

const CALL_NAME_SHORT = { defect: { ar: "عيب", en: "Defect" } } as const;

function ResultView({ lang, rtl, a, photoUrl, onFile, onAddFile, playing, say, onSlip, demo, head }: {
  lang: Lang; rtl: boolean; a: Analysis; photoUrl: string; onFile: (e: ChangeEvent<HTMLInputElement>) => void;
  onAddFile: (e: ChangeEvent<HTMLInputElement>) => void;
  playing: ClipId | null; say: (ids: ClipId | ClipId[]) => void; onSlip: () => void; demo: DemoTray | null;
  head: StoredHead | null;
}) {
  const o = a.outcome;
  if (o.kind !== "result") return null;
  const unsure = o.band === "unsure";
  const lo = pct(lang, o.interval.lo), hi = pct(lang, o.interval.hi);
  const bandName = BAND_NAME[o.band][lang];
  const gaugeLabel = unsure ? bandName : `${o.about ? `${t(lang, "aboutTag")} ` : ""}${bandName} — ${t(lang, "rangeLine", { lo, hi })}`;
  return (
    <div className="flex flex-col gap-4" data-testid="result" data-band={o.band} data-about={o.about ? "1" : "0"} data-reason={o.unsureReason} data-handfuls={o.handfuls} data-calibrated={a.calibrated ? "1" : "0"}>
      {head && <CalibratedChip lang={lang} head={head} className="self-center" />}
      <PhotoOverlay src={photoUrl} w={a.workW} h={a.workH} classified={!unsure}
        beans={unsure ? a.beans.map((b) => ({ ...b, call: null, touching: false })) : a.beans}
        alt={`${num(lang, a.checks.count)} ${t(lang, "beans")}`}
        caption={o.handfuls > 1 ? t(lang, "handfulOf", { i: num(lang, o.handfuls), n: num(lang, MAX_HANDFULS) }) : undefined} />
      {!unsure && (
        <ul className="legend">
          <li><RingSwatch kind="defect" /><span aria-hidden="true">{CALL_NAME_SHORT.defect[lang]}</span><span className="sr-only">{t(lang, "legendDefect")}</span></li>
          <li><RingSwatch kind="unsure" /><span aria-hidden="true">{CALL_NAME.unsure[lang]}</span><span className="sr-only">{t(lang, "legendUnsure")}</span></li>
          <li><RingSwatch kind="sound" /><span aria-hidden="true">{CALL_NAME.good[lang]}</span><span className="sr-only">{t(lang, "legendSound")}</span></li>
        </ul>
      )}

      <section className="card-dark px-4 pb-5 pt-4 text-center" aria-labelledby="verdict">
        <Gauge interval={o.interval} band={o.band} rtl={rtl} label={gaugeLabel} />
        <h1 id="verdict" className={`font-display text-[34px] leading-tight band-${o.band}`}>
          {unsure && <IconCoop size={34} className="me-2 inline align-[-4px]" />}
          {o.about && <span className="about-tag" data-testid="about-tag">{t(lang, "aboutTag")}</span>}
          {bandName}
        </h1>
        {unsure ? (
          <p className="mx-auto mt-2 max-w-[32ch] text-[18px] leading-relaxed text-qudad" data-testid="unsure-why">
            {o.unsureReason === "implausible"
              ? t(lang, "whyImplausible")
              : o.unsureReason === "dark_lot"
                ? t(lang, "whyDarkLot")
                : t(lang, "whyUnsure", { u: num(lang, o.counts.unsure), n: num(lang, o.total) })}
          </p>
        ) : (
          <div className="mt-2 text-[16px] leading-snug text-parchment">
            <p className="text-[18px] text-qudad" data-testid="defects-of">{t(lang, "defectsOf", { d: num(lang, o.defects), n: num(lang, o.answered) })}</p>
            <p className="mt-0.5">{t(lang, "rangeLine", { lo, hi })}</p>
            {o.handfuls > 1 && <p className="mt-0.5" data-testid="pooled">{handfulsText(lang, o.handfuls)}</p>}
          </div>
        )}
        <p className="mx-auto mt-3 max-w-[30ch] text-[20px] leading-relaxed text-qudad">{CLIP_TEXT[o.clips[0]][lang]}</p>
        {o.canAddHandful && <p className="mx-auto mt-1 max-w-[30ch] text-[18px] leading-relaxed text-saffron">{CLIP_TEXT.c18_another[lang]}</p>}
        {o.about && !o.canAddHandful && <p className="mx-auto mt-1 max-w-[30ch] text-[16px] text-parchment">{t(lang, "lotMax")}</p>}
        <SayButton lang={lang} ids={o.clips} playing={playing} say={say} className="mt-3" />
        {!unsure && <p className="mx-auto mt-3 max-w-[36ch] text-[13.5px] leading-snug text-parchment">{t(lang, "ruleOfThumb")}</p>}
      </section>

      <ResultActions lang={lang} o={o} onFile={onFile} onAddFile={onAddFile} onSlip={onSlip} />

      {unsure ? (
        <p className="text-center text-[16px] text-parchment" data-testid="counted-only">{num(lang, o.total)} {t(lang, "beansFound")}</p>
      ) : (
        <Ledger lang={lang} rtl={rtl} o={o}>
          {o.defects > 0 && (
            <div className="possible mt-3" data-testid="possible-types">
              <p className="possible-head">{t(lang, "possibleTypes")}</p>
              <ul>
                {DEFECTS.filter((d) => o.counts[d] > 0).map((d) => (
                  <li key={d} data-testid={`possible-${d}`}>
                    <BeanGlyph call={d} size={26} />
                    <span className="flex-1">{POSSIBLE_NAME[d][lang]}</span>
                    <span className="tabular-nums">{num(lang, o.counts[d])}</span>
                  </li>
                ))}
              </ul>
              <p className="possible-note">{t(lang, "possibleNote")}</p>
            </div>
          )}
        </Ledger>
      )}

      {demo && <DemoProvenance lang={lang} demo={demo} counts={o.handfuls === 1 && !unsure ? o.counts : null} />}
      <div className="text-center text-[14px] text-parchment">
        <p>{t(lang, "selfCheck")}</p>
        <p>{t(lang, "photoGone")}</p>
        {a.stub && <p>{t(lang, "stub")}</p>}
      </div>
    </div>
  );
}

function handfulsText(lang: Lang, h: number) {
  if (lang === "ar") return h === 2 ? "حفنتين من نفس البن، محسوبة مع بعض" : `${num(lang, h)} حفنات من نفس البن، محسوبة مع بعض`;
  return t(lang, "handfulsPooled", { h });
}

/** One clear primary action per outcome. */
function ResultActions({ lang, o, onFile, onAddFile, onSlip }: {
  lang: Lang; o: Result; onFile: (e: ChangeEvent<HTMLInputElement>) => void; onAddFile: (e: ChangeEvent<HTMLInputElement>) => void; onSlip: () => void;
}) {
  const slip = (primary: boolean) => (
    <button className={`btn ${primary ? "btn-primary" : "btn-quiet"}`} onClick={onSlip} data-testid="send-slip">
      <IconMessage /> <span>{t(lang, "sendSlip")}</span>
    </button>
  );
  if (o.band === "unsure")
    return (
      <div className="grid grid-cols-1 gap-3">
        <CameraButton lang={lang} onFile={onFile} label={t(lang, "newPhoto")} testId="new-check" />
        {slip(false)}
      </div>
    );
  return (
    <div className="grid grid-cols-1 gap-3">
      {o.canAddHandful && <CameraButton lang={lang} onFile={onAddFile} label={t(lang, "addHandful")} testId="add-handful" />}
      {slip(!o.canAddHandful)}
      <CameraButton lang={lang} onFile={onFile} label={t(lang, "newCheck")} quiet testId="new-check" />
    </div>
  );
}

/** For demo trays: where the picture comes from and, for SYNTHETIC trays, the known answer next to Farz's. */
function DemoProvenance({ lang, demo, counts }: { lang: Lang; demo: DemoTray; counts: Record<BeanCall, number> | null }) {
  // good / defect / not sure only: defect TYPES are never compared as facts
  const known = demo.trueCounts ? { good: demo.trueCounts.good ?? 0, defect: DEFECTS.reduce((a, d) => a + (demo.trueCounts?.[d] ?? 0), 0), unsure: null as number | null } : null;
  const farz = counts ? { good: counts.good, defect: DEFECTS.reduce((a, d) => a + counts[d], 0), unsure: counts.unsure as number | null } : null;
  const cols = ["good", "defect", "unsure"] as const;
  const note = lang === "ar" ? demo.noteAr ?? demo.noteEn : demo.noteEn ?? demo.noteAr;
  return (
    <section className="card-dark p-4 text-[15px] text-parchment" data-testid="demo-provenance">
      <p className="mb-1 font-bold text-qudad">
        <span className={`tag me-2 ${demo.synthetic ? "" : "tag-real"}`}>{demo.synthetic ? t(lang, "synthetic") : t(lang, "realPhoto")}</span>
        {lang === "ar" && demo.titleAr ? demo.titleAr : demo.title}
      </p>
      {note && <p className="mt-1 text-[16px] leading-snug text-qudad" data-testid="demo-note">{note}</p>}
      {known && farz && (
        <table className="my-2 w-full text-center text-[15px]" data-testid="demo-table">
          <thead><tr><th className="text-start font-normal">{t(lang, "demoTruth")}</th>{cols.map((k) => <th key={k} className="px-0.5 font-normal">{k === "defect" ? t(lang, "defectBeans") : CALL_NAME[k][lang]}</th>)}</tr></thead>
          <tbody>
            <tr><td className="text-start">{t(lang, "demoKnown")}</td>{cols.map((k) => <td key={k}>{known[k] === null ? "–" : num(lang, known[k]!)}</td>)}</tr>
            <tr className="text-qudad"><td className="text-start">{t(lang, "demoFarz")}</td>{cols.map((k) => <td key={k}>{num(lang, farz[k] ?? 0)}</td>)}</tr>
          </tbody>
        </table>
      )}
      {demo.credit && <p lang="en" dir="ltr" className="credit mt-1 text-[12.5px] leading-snug">{demo.credit}</p>}
    </section>
  );
}

/** `counts` is null for a "not sure" result: the sheet then shows no per-class pictures (REGRESSION_A I-1). */
function SlipSheet({ lang, body, counts, playing, onClose }: { lang: Lang; body: string; counts: Record<BeanCall, number> | null; playing: ClipId | null; onClose: () => void }) {
  const [busy, setBusy] = useState(false);
  const defects = counts ? DEFECTS.reduce((a, d) => a + counts[d], 0) : 0;
  return (
    <Sheet onClose={onClose} label={t(lang, "slipTitle")}>
      <h2 className="mb-2 font-display text-[22px] text-basalt" tabIndex={-1} data-autofocus>{t(lang, "slipTitle")}</h2>
      <div className="sms-bubble" dir="rtl" lang="ar" data-testid="slip-body" data-length={body.length}>{body}</div>
      {counts && (defects > 0 || counts.unsure > 0) && (
        <div className="mt-3" data-testid="slip-pictures">
          <p className="text-[15px] text-basalt/80">{t(lang, "slipPictures")}</p>
          <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1">
            {defects > 0 && <li className="flex items-center gap-1 text-[22px] font-bold text-basalt"><DefectGlyph size={34} />{num(lang, defects)}</li>}
            {counts.unsure > 0 && <li className="flex items-center gap-1 text-[22px] font-bold text-basalt"><BeanGlyph call="unsure" size={34} title={CALL_NAME.unsure[lang]} />{num(lang, counts.unsure)}</li>}
          </ul>
        </div>
      )}
      <p className="mt-3 text-[17px] leading-relaxed text-basalt">{CLIP_TEXT[counts ? "c15_slip" : "c15b_slip_unsure"][lang]}</p>
      <div className="mt-4 grid grid-cols-1 gap-3">
        <a className="btn btn-primary" href={smsHref(body)} aria-disabled={busy}
          onClick={async (e) => {
            // the read-back clip (c15) is played when this sheet opens; let it finish before leaving the app
            if (playing) {
              e.preventDefault();
              setBusy(true);
              await whenIdle(12000);
              setBusy(false);
              window.location.href = smsHref(body);
            }
          }}>
          <IconMessage /> <span>{t(lang, "openSms")}</span>
        </a>
        <button className="btn btn-quiet" onClick={onClose}>{t(lang, "cancel")}</button>
      </div>
    </Sheet>
  );
}

function HistoryView({ lang, items, onWipe, playing, say }: { lang: Lang; items: HistoryRecord[]; onWipe: () => void; playing: ClipId | null; say: (ids: ClipId | ClipId[]) => void }) {
  // date on one line, time on its own line: never digits run together
  const fmtDate = new Intl.DateTimeFormat(lang === "ar" ? "ar-YE" : "en-GB", { day: "numeric", month: "long" });
  const fmtTime = new Intl.DateTimeFormat(lang === "ar" ? "ar-YE" : "en-GB", { hour: "2-digit", minute: "2-digit" });
  return (
    <div className="flex flex-col gap-4">
      <h1 className="font-display text-[30px] text-qudad">{t(lang, "history")}</h1>
      <p className="text-[16px] text-parchment">{t(lang, "historyNote")}</p>
      <SayButton lang={lang} ids="c16_privacy" playing={playing} say={say} label={t(lang, "listenPrivacy")} className="self-center" />
      {items.length === 0 ? (
        <p className="card-dark p-5 text-center text-[19px] text-qudad">{t(lang, "historyEmpty")}</p>
      ) : (
        <ul className="flex flex-col gap-3" data-testid="history-list">
          {items.map((h) => (
            <li key={h.id} className="card-light flex items-center gap-3 p-3" data-band={h.band}>
              <div className="min-w-0 flex-1">
                <p className="flex flex-wrap items-center gap-x-2 text-[14px] text-basalt/80">
                  <span>{fmtDate.format(h.ts)}</span>
                  <span className="opacity-90">{fmtTime.format(h.ts)}</span>
                  {h.demo && <span className={`tag ${h.demo === "real" ? "tag-real" : ""}`}>{h.demo === "real" ? t(lang, "demoReal") : t(lang, "synthetic")}</span>}
                </p>
                <p className="text-[18px] text-basalt">
                  <span className="font-display text-[26px]">{num(lang, h.total)}</span> {t(lang, "beans")}
                  {(h.handfuls ?? 1) > 1 && <span className="ms-1 text-[15px]">({handfulsText(lang, h.handfuls!)})</span>}
                </p>
                {(() => {
                  // "not sure": only the total is shown (and stored) — no per-class counts (REGRESSION_A I-1)
                  const hc = shownCounts(h);
                  if (!hc) return null;
                  return (
                    <p className="flex flex-wrap gap-x-3 text-[18px] text-basalt" data-testid="history-counts">
                      {(() => { const d = DEFECTS.reduce((a, k) => a + (hc[k] ?? 0), 0); return d > 0 ? (
                        <span className="inline-flex items-center gap-0.5" title={t(lang, "defectBeans")}><DefectGlyph size={22} />{num(lang, d)}</span>
                      ) : null; })()}
                      {hc.unsure > 0 && <span className="inline-flex items-center gap-0.5"><BeanGlyph call="unsure" size={22} title={CALL_NAME.unsure[lang]} />{num(lang, hc.unsure)}</span>}
                    </p>
                  );
                })()}
              </div>
              <span className={`chip band-chip-${h.band}`}>{h.about ? `${t(lang, "aboutTag")} ` : ""}{BAND_NAME[h.band][lang]}</span>
            </li>
          ))}
        </ul>
      )}
      <button className="btn btn-danger" onClick={onWipe} data-testid="wipe">
        <IconErase /> <span>{t(lang, "wipe")}</span>
      </button>
    </div>
  );
}

/* ---------- pictograms for the retake screens (no reading needed) ---------- */

function Pict({ children }: { children: ReactNode }) {
  return <svg width="132" height="104" viewBox="0 0 132 104" aria-hidden="true">{children}</svg>;
}
const Bean = ({ x, y, r = -25, fill = "var(--color-bean)" }: { x: number; y: number; r?: number; fill?: string }) => (
  <g transform={`translate(${x} ${y}) rotate(${r})`}>
    <ellipse rx="9" ry="13" fill={fill} stroke="#55613a" strokeWidth="1.5" />
    <path d="M0 -12c-3 4 3 8 0 12s3 8 0 12" stroke="#55613a" strokeWidth="1.6" fill="none" />
  </g>
);
function PictDark() {
  return (<Pict><rect x="8" y="10" width="116" height="84" rx="12" fill="#120c09" /><Bean x={50} y={56} fill="#3b3326" /><Bean x={80} y={52} r={20} fill="#3b3326" />
    <circle cx="104" cy="28" r="10" fill="none" stroke="var(--color-saffron)" strokeWidth="3" />{[0, 45, 90, 135, 180, 225, 270, 315].map((d) => <path key={d} d="M104 12v-6" stroke="var(--color-saffron)" strokeWidth="3" strokeLinecap="round" transform={`rotate(${d} 104 28)`} />)}</Pict>);
}
function PictBlur() {
  return (<Pict><rect x="8" y="10" width="116" height="84" rx="12" fill="var(--color-qudad)" /><g opacity="0.45"><Bean x={58} y={54} /><Bean x={64} y={52} /></g><Bean x={61} y={53} />
    <path d="M24 30h12M20 52h12M24 74h12M96 30h12M100 52h12M96 74h12" stroke="var(--color-husk)" strokeWidth="3" strokeLinecap="round" /></Pict>);
}
function PictSpread() {
  return (<Pict><rect x="8" y="10" width="116" height="84" rx="12" fill="var(--color-qudad)" /><Bean x={56} y={52} /><Bean x={74} y={52} r={15} />
    <path d="M38 52H20m0 0 7-7m-7 7 7 7M94 52h18m0 0-7-7m7 7-7 7" stroke="var(--color-cherry)" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" fill="none" /></Pict>);
}
function PictCount() {
  const pts = Array.from({ length: 18 }, (_, i) => [24 + (i % 6) * 17, 26 + Math.floor(i / 6) * 24, (i * 37) % 60 - 30]);
  return (<Pict><rect x="8" y="10" width="116" height="84" rx="12" fill="var(--color-qudad)" />{pts.map(([x, y, r], i) => <Bean key={i} x={x} y={y} r={r} />)}
    <rect x="40" y="38" width="52" height="28" rx="8" fill="var(--color-husk)" /><text x="66" y="59" textAnchor="middle" fontSize="20" fontWeight="700" fill="var(--color-qudad)" fontFamily="var(--font-display)">١٠٠</text></Pict>);
}
function PictNotGreen() {
  return (<Pict><rect x="8" y="10" width="116" height="84" rx="12" fill="var(--color-qudad)" /><Bean x={46} y={52} fill="#5b3a26" /><Bean x={72} y={50} r={20} fill="#4a2e1f" /><Bean x={92} y={60} r={-50} fill="#5b3a26" />
    <circle cx="66" cy="52" r="38" fill="none" stroke="var(--color-cherry)" strokeWidth="5" /><path d="M39 79 93 25" stroke="var(--color-cherry)" strokeWidth="5" /></Pict>);
}
function StepCloth() {
  return <svg width="64" height="52" viewBox="0 0 64 52" aria-hidden="true"><path d="M6 10h52l-4 34H10z" fill="var(--color-qudad)" /><path d="M6 10h52" stroke="var(--color-parchment)" strokeWidth="2" /></svg>;
}
function StepBeans() {
  return <svg width="64" height="52" viewBox="0 0 64 52" aria-hidden="true">{[[14, 16], [30, 12], [46, 18], [20, 34], [38, 32], [52, 38]].map(([x, y], i) => (
    <ellipse key={i} cx={x} cy={y} rx="5" ry="7" transform={`rotate(${i * 40 - 30} ${x} ${y})`} fill="var(--color-bean)" />))}</svg>;
}
function StepPhone() {
  return <svg width="64" height="52" viewBox="0 0 64 52" aria-hidden="true"><rect x="22" y="2" width="20" height="30" rx="4" fill="none" stroke="var(--color-qudad)" strokeWidth="2.5" />
    <path d="M32 36v8m0 0-4-4m4 4 4-4" stroke="var(--color-bean)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" /><path d="M10 48h44" stroke="var(--color-qudad)" strokeWidth="2.5" strokeLinecap="round" /></svg>;
}
