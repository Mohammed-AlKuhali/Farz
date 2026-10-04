/**
 * "Calibrate for your cooperative" — PILOT, for the cooperative's grader (not the farmer).
 * The grader photographs 2–3 trays of the cooperative's own beans on the usual sheet and light, taps beans to mark
 * them sound or defect (or marks a whole tray), and Farz builds the prototype head of reports_v2/localcal_spec.md from
 * the model's `embed` output (src/lib/localcal.ts). Only the head (two 1,024-number averages + a margin) is stored, in
 * IndexedDB on this phone; the photos are never stored and nothing leaves the phone.
 * Demo: the QUALITY CHECK on 20 bean crops from loja_yolo (CC BY 4.0) whose labels come from the public dataset, standing in
 * for a grader: the correct labels are accepted and the same beans with 6 deliberately wrong labels are refused (measured:
 * tests/localcal.test.ts). The demo runs in its own context: it never saves, never touches the engine's head, so the
 * user's saved calibration is unchanged. (The earlier before/after demo on 3 test photos was removed: it gave the same
 * answer before and after on all 3 photos, because the shipped model was trained on this dataset.)
 */
import { useEffect, useRef, useState, type ChangeEvent, type KeyboardEvent, type MouseEvent } from "react";
import type { Engine, ReadyInfo } from "../lib/engine";
import { CLIP_TEXT, num, type Lang } from "../lib/i18n";
import {
  fitHead, MAX_LOO_ERROR_FRAC, N_PER_CLASS, resetStoredHead, saveStoredHead, toStored, type FitReport, type Label, type StoredHead,
} from "../lib/localcal";
import { unlockAudio } from "../lib/audio";
import { Ring } from "./PhotoOverlay";
import { BeanGlyph, IconCamera, IconCoop, IconErase } from "./Icons";

interface DemoManifest {
  credit: string;
  calibration: { file: string; label: Label; source_label: string; source_photo: string }[];
}

/** The demo's deliberately wrong labels: the first WRONG_EACH sound and the first WRONG_EACH defect beans are swapped. */
export const WRONG_EACH = 3;
export function wrongDemoLabels<T extends { label: Label }>(items: T[]): (T & { flipped: boolean })[] {
  let g = 0, d = 0;
  return items.map((it) => {
    const flip = it.label === "good" ? g++ < WRONG_EACH : d++ < WRONG_EACH;
    return { ...it, label: flip ? (it.label === "good" ? "defect" : "good") : it.label, flipped: flip };
  });
}

const BASE = () => import.meta.env?.BASE_URL ?? "/";
const L = (lang: Lang, ar: string, en: string) => (lang === "ar" ? ar : en);

/* ---------- small shared pieces (also used by App.tsx) ---------- */

export function CalibratedChip({ lang, head, className = "", onClick }: { lang: Lang; head: StoredHead; className?: string; onClick?: () => void }) {
  const n = head.n_good + head.n_defect;
  const body = (
    <>
      <IconCoop size={22} className="shrink-0" />
      <span>{L(lang, `مضبوط على: ${head.name} (${num(lang, n)} حبة)`, `Calibrated for: ${head.name} (${n} beans)`)}</span>
      <span className="tag">{head.demo ? L(lang, "عرض تجريبي", "DEMO") : L(lang, "تجربة", "PILOT")}</span>
    </>
  );
  return onClick ? (
    <button className={`cal-chip ${className}`} onClick={onClick} data-testid="calibrated-chip">{body}</button>
  ) : (
    <p className={`cal-chip ${className}`} data-testid="calibrated-chip">{body}</p>
  );
}

/* ---------- the two piles: the grader's tally (the screen's one signature element) ---------- */

function Piles({ lang, good, defect }: { lang: Lang; good: number; defect: number }) {
  const pile = (kind: Label, n: number) => (
    <div className={`pile pile-${kind}`} data-testid={`pile-${kind}`} data-count={n}>
      <p className="pile-label">
        <span>{kind === "good" ? L(lang, "سليمة", "Sound") : L(lang, "فيها عيب", "Defect")}</span>
        <span className="font-display tabular-nums">{num(lang, Math.min(n, 99))}{n < N_PER_CLASS ? <span className="pile-of">/{num(lang, N_PER_CLASS)}</span> : " ✓"}</span>
      </p>
      <div className="pile-slots" aria-hidden="true">
        {Array.from({ length: N_PER_CLASS }, (_, i) => (
          <span key={i} className={`slot ${i < n ? "is-full" : ""}`}>
            {i < n && (kind === "good" ? <BeanGlyph call="good" size={22} /> : <DefectMini />)}
          </span>
        ))}
      </div>
    </div>
  );
  return (
    <div className="piles" role="status" aria-live="polite"
      aria-label={L(lang, `${num(lang, good)} سليمة، ${num(lang, defect)} فيها عيب. المطلوب ${num(lang, N_PER_CLASS)} من كل نوع.`, `${good} sound, ${defect} defect labelled. ${N_PER_CLASS} of each are needed.`)}>
      {pile("good", good)}
      {pile("defect", defect)}
    </div>
  );
}
function DefectMini() {
  return (
    <svg width="22" height="22" viewBox="0 0 40 40" aria-hidden="true">
      <g transform="rotate(-28 20 21)"><ellipse cx="20" cy="21" rx="9" ry="13" fill="var(--color-bean)" stroke="#55613a" strokeWidth="1.5" /></g>
      <circle cx="29" cy="11" r="8" fill="var(--color-cherry-deep)" /><path d="M26 8l6 6M32 8l-6 6" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" />
    </svg>
  );
}

/* ---------- labelling a tray photo ---------- */

interface TrayBean { x0: number; y0: number; x1: number; y1: number; touching: boolean }
interface Tray { id: number; url: string; w: number; h: number; beans: TrayBean[]; embeds: Float32Array; D: number; labels: (Label | null)[] }

function LabelPhoto({ lang, tray, index, pen, onLabel }: { lang: Lang; tray: Tray; index: number; pen: Label; onLabel: (i: number, l: Label | null) => void }) {
  const { w, h } = tray;
  const u = w / 360;
  const toggle = (i: number) => {
    if (tray.beans[i].touching) return;
    onLabel(i, tray.labels[i] === pen ? null : pen);
  };
  const onClick = (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * w, y = ((e.clientY - r.top) / r.height) * h;
    let best = -1, bd = Infinity;
    tray.beans.forEach((b, i) => {
      const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
      const rad = Math.max(b.x1 - b.x0, b.y1 - b.y0) / 2 + 14 * u;
      const d = Math.hypot(x - cx, y - cy);
      if (d <= rad && d < bd) { bd = d; best = i; }
    });
    if (best >= 0) toggle(best);
  };
  return (
    <figure className="label-frame" style={{ ["--ar" as string]: `${w / h}` }}>
      <div className="relative overflow-hidden rounded-[18px] bg-husk-2" style={{ aspectRatio: `${w} / ${h}` }}>
        <img src={tray.url} alt="" className="absolute inset-0 h-full w-full object-fill" />
        <svg viewBox={`0 0 ${w} ${h}`} className="absolute inset-0 h-full w-full cursor-pointer touch-manipulation" onClick={onClick}
          role="group" aria-label={L(lang, "اضغطي على الحبة لتعليمها", "Tap a bean to label it")} data-testid="label-photo" data-tray={index + 1}>
          {tray.beans.map((b, i) => {
            const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
            const r = Math.max(b.x1 - b.x0, b.y1 - b.y0) / 2 + Math.max(4, w / 300);
            const l = tray.labels[i];
            const kind = b.touching ? "unsure" : l === "defect" ? "defect" : l === "good" ? null : "counted";
            const name = b.touching ? L(lang, "حبة لاصقة، ما تنفع", "touching bean, cannot be used")
              : l === "good" ? L(lang, "سليمة", "sound") : l === "defect" ? L(lang, "فيها عيب", "defect") : L(lang, "مش معلّمة", "not labelled");
            return (
              <g key={i} data-testid="label-bean" data-label={b.touching ? "touching" : l ?? "none"}
                role="button" tabIndex={b.touching ? -1 : 0} aria-label={`${L(lang, "حبة", "Bean")} ${num(lang, i + 1)}: ${name}`}
                onKeyDown={(e: KeyboardEvent) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(i); } }}>
                <circle cx={cx} cy={cy} r={r + 6 * u} fill="transparent" />
                {kind && <Ring kind={kind} cx={cx} cy={cy} r={r} u={u} />}
                {l === "good" && !b.touching && <SoundMark cx={cx} cy={cy} r={r} u={u} />}
              </g>
            );
          })}
        </svg>
      </div>
    </figure>
  );
}
/** Labelled sound: a thick green ring with a check badge (shape differs from the red ×, colour only repeats it). */
function SoundMark({ cx, cy, r, u }: { cx: number; cy: number; r: number; u: number }) {
  const br = Math.max(5.5 * u, r * 0.34), bx = cx + r * 0.72, by = cy - r * 0.72, k = br * 0.5;
  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#1a100b" strokeWidth="6.5" vectorEffect="non-scaling-stroke" />
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--color-bean)" strokeWidth="3.5" vectorEffect="non-scaling-stroke" />
      <circle cx={bx} cy={by} r={br} fill="var(--color-bean)" stroke="#1a100b" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      <path d={`M${bx - k} ${by}l${k * 0.7} ${k * 0.7}l${k * 1.2} ${-k * 1.3}`} fill="none" stroke="#1a100b" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
    </g>
  );
}

/* ---------- the screen ---------- */

type Phase = "start" | "label" | "demo" | "check";

export function Calibrate({ lang, engine, info, stored, active, onSaved, onReset }: {
  lang: Lang; engine: Engine; info: ReadyInfo | null; stored: StoredHead | null; active: boolean;
  onSaved: (h: StoredHead) => void; onReset: () => void;
}) {
  const [phase, setPhase] = useState<Phase>("start");
  const [name, setName] = useState("");
  const [trays, setTrays] = useState<Tray[]>([]);
  const [cur, setCur] = useState(0);
  const [pen, setPen] = useState<Label>("good");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [fit, setFit] = useState<{ r: FitReport; trays: number } | null>(null);
  const [demo, setDemo] = useState<DemoManifest | null>(null);
  const [demoEmb, setDemoEmb] = useState<{ embed: Float32Array; label: Label }[] | null>(null);
  /** the demo's quality checks, in memory only (never saved, never installed on the engine) */
  const [demoCheck, setDemoCheck] = useState<{ right: FitReport; wrong: FitReport | null } | null>(null);
  const [demoWrong, setDemoWrong] = useState(false);
  const urls = useRef<string[]>([]);
  const ready = !!info && !info.stub && info.embedDim > 0;

  useEffect(() => {
    fetch(`${BASE()}calib-demo/manifest.json`).then((r) => (r.ok ? r.json() : null)).then((j) => {
      if (!j?.calibration) return;
      setDemo({ credit: j.credit, calibration: j.calibration });
    }).catch(() => setDemo(null));
    // the grader's photos are never kept: object URLs die with this screen
    return () => { urls.current.forEach((u) => URL.revokeObjectURL(u)); urls.current = []; };
  }, []);

  // a new tray becomes the one being labelled
  useEffect(() => { if (trays.length) setCur(trays.length - 1); }, [trays.length]);

  const all = trays.flatMap((t) => t.labels.map((l, i) => ({ l, t, i }))).filter((x) => x.l && !x.t.beans[x.i].touching);
  const nGood = all.filter((x) => x.l === "good").length, nDefect = all.filter((x) => x.l === "defect").length;

  const onPhoto = async (e: ChangeEvent<HTMLInputElement>) => {
    unlockAudio();
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    setBusy(true); setProblem(null);
    try {
      const a = await engine.analyze(f, { minBeans: 1, noHead: true });
      if (a.outcome.kind === "retake") { setProblem(CLIP_TEXT[a.outcome.clip][lang]); return; }
      if (!a.embeds) { setProblem(L(lang, "المصنّف مش موجود، ما ينفع الضبط.", "The classifier is missing, so calibration is not possible.")); return; }
      const url = URL.createObjectURL(f);
      urls.current.push(url);
      const tray: Tray = { id: Date.now(), url, w: a.workW, h: a.workH, beans: a.beans.map(({ x0, y0, x1, y1, touching }) => ({ x0, y0, x1, y1, touching })),
        embeds: a.embeds, D: a.embedDim, labels: a.beans.map(() => null) };
      setTrays((ts) => [...ts, tray]);
      setPhase("label");
    } catch (err) {
      console.error(err);
      setProblem(L(lang, "ما قدرت أقرا الصورة. صوّري ثاني.", "Could not read that photo. Take it again."));
    } finally {
      setBusy(false);
    }
  };

  const setLabel = (ti: number, i: number, l: Label | null) =>
    setTrays((ts) => ts.map((t, k) => (k !== ti ? t : { ...t, labels: t.labels.map((x, j) => (j === i ? l : x)) })));
  const markAll = (ti: number, l: Label | null) =>
    setTrays((ts) => ts.map((t, k) => (k !== ti ? t : { ...t, labels: t.beans.map((b) => (b.touching ? null : l)) })));

  const build = () => {
    const items = trays.flatMap((t) => t.labels.flatMap((l, i) => (l && !t.beans[i].touching ? [{ embed: t.embeds.subarray(i * t.D, (i + 1) * t.D), label: l }] : [])));
    setFit({ r: fitHead(items), trays: trays.length });
    setPhase("check");
  };

  const startDemo = async () => {
    if (!demo) return;
    unlockAudio();
    setBusy(true); setProblem(null);
    try {
      const crops = await Promise.all(demo.calibration.map((c) => loadCrop(`${BASE()}calib-demo/${c.file}`)));
      const res = await engine.embedCrops(crops);
      if (!res.embeds) throw new Error("no embeddings");
      setDemoEmb(demo.calibration.map((c, i) => ({ embed: res.embeds!.subarray(i * res.embedDim, (i + 1) * res.embedDim), label: c.label })));
      setDemoCheck(null); setDemoWrong(false);
      setPhase("demo");
    } catch (err) {
      console.error(err);
      setProblem(L(lang, "ما قدرت أجهّز التجربة.", "Could not prepare the demo."));
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    if (!fit?.r.ok || !info) return;
    const h = toStored(fit.r, { modelSha: info.modelSha, name: name.trim() || L(lang, "جمعيتي", "My cooperative"), trays: fit.trays });
    try {
      await saveStoredHead(h);
    } catch (err) {
      console.warn("[Farz] could not save the calibration", err);
      setProblem(L(lang, "ما قدرت أحفظ الضبط في هذا التلفون.", "Could not save the calibration on this phone."));
      return;
    }
    onSaved(h);
    setTrays([]); setFit(null); setPhase("start");
  };

  const reset = async () => {
    try { await resetStoredHead(); } catch (err) { console.warn("[Farz] could not reset the calibration", err); }
    onReset();
  };

  const tray = trays[cur];

  return (
    <div className="calibrate flex flex-col gap-4" data-testid="calibrate" data-phase={phase}>
      <header className="mt-2">
        <p className="eyebrow">{L(lang, "للمختص في الجمعية", "For the cooperative's grader")} <span className="tag">{L(lang, "تجربة", "PILOT")}</span></p>
        <h1 className="font-display text-[30px] leading-tight text-qudad">{L(lang, "اضبطي فرز على بن جمعيتك", "Calibrate for your cooperative")}</h1>
      </header>

      {!ready && <p className="card-dark p-4 text-[17px]" role="alert">{L(lang, "المصنّف لسه ما جهز.", "The classifier is not ready yet.")}</p>}

      {phase === "start" && (
        <>
          <section className="card-dark p-4" data-testid="cal-status">
            {active && stored ? (
              <>
                <CalibratedChip lang={lang} head={stored} />
                <p className="mt-2 text-[16px] leading-snug text-parchment">
                  {L(lang,
                    `${num(lang, stored.n_good)} سليمة و${num(lang, stored.n_defect)} فيها عيب من ${num(lang, stored.trays)} صواني. الفحص: ${num(lang, stored.n_good + stored.n_defect - stored.loo_errors)} من ${num(lang, stored.n_good + stored.n_defect)} طلعت مثل تعليمك.`,
                    `${stored.n_good} sound and ${stored.n_defect} defect beans from ${stored.trays} tray${stored.trays === 1 ? "" : "s"}. Check: ${stored.n_good + stored.n_defect - stored.loo_errors} of ${stored.n_good + stored.n_defect} sorted the way you labelled them.`)}
                </p>
                <button className="btn btn-danger mt-3 w-full" onClick={reset} data-testid="cal-reset">
                  <IconErase /> <span>{L(lang, "امسحي الضبط", "Reset calibration")}</span>
                </button>
              </>
            ) : (
              <p className="text-[17px] text-qudad" data-testid="cal-none">
                {stored && !active
                  ? L(lang, "الضبط المحفوظ لنسخة ثانية من المصنّف، فما ينستخدم. اضبطي من جديد.", "The saved calibration was made with a different classifier file, so it is not used. Calibrate again.")
                  : L(lang, "فرز يشتغل الآن بدون ضبط.", "Farz is not calibrated: it uses its own built-in thresholds.")}
              </p>
            )}
          </section>

          <section className="card-dark p-4 text-[17px] leading-relaxed">
            <ol className="cal-steps">
              <li>{L(lang, "صوّري ٢–٣ صواني من بن الجمعية، على نفس القماشة ونفس الضوء اللي تستخدموه.", "Photograph 2–3 trays of your cooperative's beans, on your usual sheet and light.")}</li>
              <li>{L(lang, `اضغطي على كل حبة: سليمة أو فيها عيب. المطلوب ${num(lang, N_PER_CLASS)} من كل نوع على الأقل.`, `Tap each bean: sound or defect. At least ${N_PER_CLASS} of each are needed.`)}</li>
              <li>{L(lang, "فرز يحفظ ملخص صغير في هذا التلفون بس. الصور ما تنحفظ، وما يطلع شي من التلفون.", "Farz keeps a small summary on this phone only. The photos are not kept and nothing leaves the phone.")}</li>
            </ol>
            <p className="mt-2 text-[15px] text-parchment">
              {L(lang,
                "تجربة: جرّبناها على أربع مجموعات صور ونجحت على وحدة بس (من الإكوادور)، وعلى مجموعتين قالت عن صواني نظيفة «عيوب كثيرة». جرّبيها على صواني تعرفين جوابها قبل ما تستخدمينها مع المزارعات.",
                "Pilot: tried on four photo sets, it passed on one (Ecuador); on two others it called clean trays “many defects”. Try it on trays whose answer you know before using it with farmers.")}
            </p>
          </section>

          <label className="flex flex-col gap-1 text-[16px] text-parchment">
            {L(lang, "اسم الجمعية أو المكان", "Cooperative or place name")}
            <input className="field" value={name} onChange={(e) => setName(e.target.value.slice(0, 40))} maxLength={40}
              placeholder={L(lang, "مثلاً: جمعية حراز", "e.g. Haraz cooperative")} data-testid="cal-name" />
          </label>
          <CamButton lang={lang} onFile={onPhoto} disabled={!ready || busy} label={L(lang, "صوّري صينية للضبط", "Photograph a tray to label")} testId="cal-photo" />

          {demo && (
            <section className="card-dark p-4" data-testid="cal-demo">
              <h2 className="font-display text-[22px] text-qudad">{L(lang, "تجربة: كيف يشتغل فحص الجودة", "Demo: how the quality check works")}</h2>
              <p className="mt-1 text-[15px] leading-snug text-parchment" data-testid="cal-demo-intro">
                {L(lang,
                  "٢٠ حبة من مجموعة صور عامة من لوخا في الإكوادور (١٠ سليمة، ١٠ فيها عيب). التعليم جاي من المجموعة نفسها، مكان تعليم المختص. فرز يفحص التعليم: التعليم الصحيح ينقبل، والتعليم الغلط عمداً ينرفض.",
                  "20 beans from a public photo set from Loja, Ecuador (10 sound, 10 defect). Their labels come from the dataset itself, standing in for a cooperative grader. Farz checks the labels: correct labels are accepted, deliberately wrong ones are refused.")}
              </p>
              <p className="mt-1 text-[15px] leading-snug text-qudad">
                {L(lang, "تجربة بس: ما ينحفظ شي، والضبط حق جمعيتك ما يتغير. فرز تعلّم من هذي المجموعة، فالتجربة تورّي الفحص، مش الدقة.",
                  "Demo only: nothing is saved and your cooperative's calibration does not change. Farz was trained on this dataset, so this shows the check, not accuracy.")}
              </p>
              <button className="btn btn-quiet mt-3 w-full" onClick={startDemo} disabled={!ready || busy} data-testid="cal-demo-start">
                {L(lang, "حمّلي الحبوب المعلّمة", "Load the labelled beans")}
              </button>
              <p lang="en" dir="ltr" className="credit mt-2 text-[12px] leading-snug">{demo.credit} Crops cut by Farz's bean finder (cropped, white-balanced, resized to 128 px).</p>
            </section>
          )}
        </>
      )}

      {problem && <p role="alert" className="rounded-xl border-2 border-saffron bg-saffron/15 px-3 py-2 text-[17px] text-qudad" data-testid="cal-problem">{problem}</p>}
      {busy && <p className="text-center text-[17px] text-parchment" role="status">{L(lang, "أعدّ الحبوب…", "Counting the beans…")}</p>}

      {phase === "label" && tray && (
        <>
          <Piles lang={lang} good={nGood} defect={nDefect} />
          <div className="pen" role="radiogroup" aria-label={L(lang, "اللي تعلّمينه الآن", "What a tap marks")}>
            {(["good", "defect"] as const).map((k) => (
              <button key={k} role="radio" aria-checked={pen === k} className={`pen-opt pen-${k} ${pen === k ? "is-on" : ""}`} onClick={() => setPen(k)} data-testid={`pen-${k}`}>
                {k === "good" ? <BeanGlyph call="good" size={26} /> : <DefectMini />}
                <span>{k === "good" ? L(lang, "سليمة", "Sound") : L(lang, "فيها عيب", "Defect")}</span>
              </button>
            ))}
          </div>
          <LabelPhoto key={tray.id} lang={lang} tray={tray} index={cur} pen={pen} onLabel={(i, l) => setLabel(cur, i, l)} />
          <p className="text-center text-[15px] text-parchment">
            {L(lang,
              `صينية ${num(lang, cur + 1)} من ${num(lang, trays.length)}، فيها ${num(lang, tray.beans.length)} حبة${tray.beans.some((b) => b.touching) ? "، الحبوب اللاصقة (؟) ما تنفع" : ""}`,
              `Tray ${cur + 1} of ${trays.length} · ${tray.beans.length} beans${tray.beans.some((b) => b.touching) ? "; touching beans (?) cannot be used" : ""}`)}
          </p>
          <div className="grid grid-cols-2 gap-2">
            <button className="btn btn-quiet min-h-[52px] text-[16px]" onClick={() => markAll(cur, "good")} data-testid="mark-all-good">{L(lang, "كل الصينية سليمة", "Whole tray sound")}</button>
            <button className="btn btn-quiet min-h-[52px] text-[16px]" onClick={() => markAll(cur, "defect")} data-testid="mark-all-defect">{L(lang, "كل الصينية عيب", "Whole tray defect")}</button>
            <button className="btn btn-quiet col-span-2 min-h-[48px] text-[15px]" onClick={() => markAll(cur, null)}>{L(lang, "امسحي تعليم هذي الصينية", "Clear this tray's labels")}</button>
          </div>
          {trays.length < 3 && <CamButton lang={lang} onFile={onPhoto} disabled={busy} label={L(lang, "صوّري صينية ثانية", "Photograph another tray")} quiet testId="cal-photo-more" />}
          <button className="btn btn-primary" disabled={nGood < N_PER_CLASS || nDefect < N_PER_CLASS} onClick={build} data-testid="cal-build">
            {L(lang, "جهّزي الضبط", "Build calibration")}
          </button>
          {(nGood < N_PER_CLASS || nDefect < N_PER_CLASS) && (
            <p className="text-center text-[15px] text-parchment">
              {L(lang, `باقي ${num(lang, Math.max(0, N_PER_CLASS - nGood))} سليمة و${num(lang, Math.max(0, N_PER_CLASS - nDefect))} فيها عيب.`,
                `${Math.max(0, N_PER_CLASS - nGood)} more sound and ${Math.max(0, N_PER_CLASS - nDefect)} more defect needed.`)}
            </p>
          )}
        </>
      )}

      {phase === "demo" && demo && demoEmb && (
        <DemoQualityCheck lang={lang} demo={demo} emb={demoEmb} check={demoCheck} wrong={demoWrong}
          onCheck={() => { setDemoWrong(false); setDemoCheck({ right: fitHead(demoEmb), wrong: null }); }}
          onWrong={() => { setDemoWrong(true); setDemoCheck((c) => ({ right: c?.right ?? fitHead(demoEmb), wrong: fitHead(wrongDemoLabels(demoEmb)) })); }}
          onBack={() => { setDemoCheck(null); setDemoWrong(false); setDemoEmb(null); setPhase("start"); }} />
      )}

      {phase === "check" && fit && (
        <section className={`card-dark p-4 check ${fit.r.ok ? "is-ok" : "is-bad"}`} data-testid="cal-check" data-ok={fit.r.ok ? "1" : "0"}
          data-loo-errors={fit.r.looErrors} data-mhi={Number.isFinite(fit.r.mHi) ? fit.r.mHi.toFixed(4) : ""}>
          <h2 className="font-display text-[22px] text-qudad">{L(lang, "فحص الجودة", "Quality check")}</h2>
          {fit.r.refused === "too_few" ? (
            <p className="text-[17px]">{L(lang, `يحتاج ${num(lang, N_PER_CLASS)} من كل نوع.`, `Needs ${N_PER_CLASS} of each kind.`)}</p>
          ) : (
            <>
              <p className="check-score font-display">{num(lang, fit.r.looN - fit.r.looErrors)}<span>/{num(lang, fit.r.looN)}</span></p>
              <p className="text-[17px] leading-relaxed text-qudad">
                {L(lang,
                  `فرز خبّى كل حبة من الـ${num(lang, fit.r.looN)} بدورها وفرزها بالباقي: ${num(lang, fit.r.looN - fit.r.looErrors)} طلعت مثل تعليمك. الحد: أخطاء لا تزيد عن ${num(lang, Math.floor(fit.r.looN * MAX_LOO_ERROR_FRAC))}.`,
                  `Farz hid each of the ${fit.r.looN} beans in turn and sorted it using the others: ${fit.r.looN - fit.r.looErrors} matched your label. The limit is ${Math.floor(fit.r.looN * MAX_LOO_ERROR_FRAC)} mistakes.`)}
              </p>
              <p className="mt-1 text-[14px] text-parchment" lang="en" dir="ltr">margin m_hi = {fit.r.mHi.toFixed(4)} · {N_PER_CLASS}+{N_PER_CLASS} of {fit.r.nGood}+{fit.r.nDefect} labelled</p>
              {fit.r.ok ? (
                <button className="btn btn-primary mt-3 w-full" onClick={save} data-testid="cal-save">{L(lang, "احفظي واستخدمي الضبط", "Save and use this calibration")}</button>
              ) : (
                <p className="mt-2 text-[17px] text-cherry-light" role="alert" data-testid="cal-refused">
                  {L(lang, "التعليم ما يفرّق كفاية بين السليمة والعيب، فما انحفظ شي. راجعي التعليم أو صوّري في نفس الضوء.", "These labels do not separate sound from defect well enough, so nothing was saved. Check the labels, or photograph in the same light.")}
                </p>
              )}
            </>
          )}
          <button className="btn btn-quiet mt-2 w-full" onClick={() => setPhase("label")}>{L(lang, "رجوع", "Back")}</button>
        </section>
      )}
    </div>
  );
}

/** One quality-check score card (the grader's check and the demo's two checks look the same). */
function Score({ lang, r, testId, label }: { lang: Lang; r: FitReport; testId: string; label: string }) {
  return (
    <div className={`check demo-check ${r.ok ? "is-ok" : "is-bad"}`} data-testid={testId} data-ok={r.ok ? "1" : "0"} data-loo-errors={r.looErrors}
      data-mhi={Number.isFinite(r.mHi) ? r.mHi.toFixed(4) : ""}>
      <p className="text-[15px] font-bold text-parchment">{label}</p>
      <p className="check-score font-display">{num(lang, r.looN - r.looErrors)}<span>/{num(lang, r.looN)}</span></p>
      <p className={`text-[17px] font-bold ${r.ok ? "text-bean" : "text-cherry-light"}`}>
        {r.ok ? L(lang, "✓ مقبول", "✓ Accepted") : L(lang, "✗ مرفوض", "✗ Refused")}
      </p>
      <p className="text-[15px] leading-snug text-qudad">
        {L(lang,
          `${num(lang, r.looN - r.looErrors)} من ${num(lang, r.looN)} طلعت مثل التعليم. الحد: أخطاء لا تزيد عن ${num(lang, Math.floor(r.looN * MAX_LOO_ERROR_FRAC))}.`,
          `${r.looN - r.looErrors} of ${r.looN} matched their label. The limit is ${Math.floor(r.looN * MAX_LOO_ERROR_FRAC)} mistakes.`)}
      </p>
    </div>
  );
}

/** The calibration demo: the quality check on dataset labels, right and deliberately wrong. Nothing is saved. */
function DemoQualityCheck({ lang, demo, emb, check, wrong, onCheck, onWrong, onBack }: {
  lang: Lang; demo: DemoManifest; emb: { embed: Float32Array; label: Label }[]; check: { right: FitReport; wrong: FitReport | null } | null;
  wrong: boolean; onCheck: () => void; onWrong: () => void; onBack: () => void;
}) {
  const shown = wrong ? wrongDemoLabels(demo.calibration) : demo.calibration.map((c) => ({ ...c, flipped: false }));
  return (
    <section className="flex flex-col gap-4" data-testid="cal-demo-panel" data-wrong={wrong ? "1" : "0"}>
      <p className="demo-only" data-testid="cal-demo-only">
        {L(lang, "تجربة بس — ما ينحفظ شي، والضبط حق جمعيتك ما يتغير.", "Demo only — nothing is saved; your cooperative's calibration does not change.")}
      </p>
      <Piles lang={lang} good={emb.filter((d) => d.label === "good").length} defect={emb.filter((d) => d.label === "defect").length} />
      <ul className="crop-grid" data-testid="cal-demo-crops">
        {shown.map((c) => (
          <li key={c.file} className={`crop crop-${c.label} ${c.flipped ? "is-flipped" : ""}`} data-flipped={c.flipped ? "1" : "0"}>
            <img src={`${BASE()}calib-demo/${c.file}`} alt={`${c.label}: ${c.source_label}${c.flipped ? " (wrong label on purpose)" : ""}`} />
            <span>{c.flipped ? "✗ " : ""}{c.label === "good" ? L(lang, "سليمة", "Sound") : L(lang, "عيب", "Defect")}</span>
          </li>
        ))}
      </ul>
      <p className="text-[14px] text-parchment">
        {wrong
          ? L(lang, `الحبوب المعلّمة بـ✗ تعليمها غلط عمداً: ${num(lang, WRONG_EACH)} سليمة صارت «عيب» و${num(lang, WRONG_EACH)} فيها عيب صارت «سليمة».`,
            `Beans marked ✗ carry a wrong label on purpose: ${WRONG_EACH} sound beans labelled “defect” and ${WRONG_EACH} defect beans labelled “sound”.`)
          : L(lang, "التعليم من المجموعة نفسها (YOLO)، مكان تعليم المختص. الحبوب من ٣ صواني: IMG_4758، IMG_5488، IMG_5749.",
            "Labels come from the dataset itself (YOLO polygons), standing in for a grader. Beans from 3 tray photos: IMG_4758, IMG_5488, IMG_5749.")}
      </p>
      {!check && (
        <button className="btn btn-primary" onClick={onCheck} data-testid="cal-demo-check">{L(lang, "شغّلي فحص الجودة", "Run the quality check")}</button>
      )}
      {check && (
        <section className="card-dark p-4" aria-live="polite">
          <h2 className="font-display text-[22px] text-qudad">{L(lang, "فحص الجودة", "Quality check")}</h2>
          <p className="text-[15px] leading-snug text-parchment">
            {L(lang, "فرز يخبّي كل حبة بدورها ويفرزها بالباقي، ويقارن بالتعليم.", "Farz hides each bean in turn, sorts it using the others, and compares with its label.")}
          </p>
          <div className="demo-checks mt-3">
            <Score lang={lang} r={check.right} testId="cal-demo-right" label={L(lang, "التعليم الصحيح", "Correct labels")} />
            {check.wrong && <Score lang={lang} r={check.wrong} testId="cal-demo-wrong-result" label={L(lang, "٦ تعليمات غلط عمداً", "6 wrong labels on purpose")} />}
          </div>
          {check.wrong && !check.wrong.ok && (
            <p className="mt-2 text-[16px] leading-snug text-qudad" data-testid="cal-demo-refused">
              {L(lang, "مع التعليم الغلط فرز يرفض الضبط وما يستخدمه. هذا اللي يصير لو المختص علّم غلط.", "With the wrong labels Farz refuses the calibration and does not use it. That is what happens when a grader's labels are wrong.")}
            </p>
          )}
          {!check.wrong && (
            <button className="btn btn-primary mt-3 w-full" onClick={onWrong} data-testid="cal-demo-wrong">{L(lang, "جرّبي تعليم غلط عمداً", "Now try deliberately wrong labels")}</button>
          )}
        </section>
      )}
      <button className="btn btn-quiet" onClick={onBack} data-testid="cal-demo-back">{L(lang, "رجوع", "Back")}</button>
    </section>
  );
}

function CamButton({ lang, onFile, disabled, label, quiet = false, testId }: { lang: Lang; onFile: (e: ChangeEvent<HTMLInputElement>) => void; disabled?: boolean; label: string; quiet?: boolean; testId: string }) {
  void lang;
  return (
    <label className={`btn ${quiet ? "btn-quiet" : "btn-primary"} w-full ${disabled ? "is-disabled" : ""}`} data-testid={testId} onClick={unlockAudio} aria-disabled={disabled}>
      <input type="file" accept="image/*" capture="environment" className="sr-only" onChange={onFile} disabled={disabled} data-testid={`${testId}-input`} />
      <IconCamera /> <span>{label}</span>
    </label>
  );
}

/** A 128x128 PNG crop -> RGB bytes, no colour management (the crops are untagged sRGB, decoded as stored). */
async function loadCrop(url: string): Promise<Uint8Array> {
  const blob = await (await fetch(url)).blob();
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const c = document.createElement("canvas");
  c.width = 128; c.height = 128;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const d = ctx.getImageData(0, 0, 128, 128).data;
  const rgb = new Uint8Array(128 * 128 * 3);
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { rgb[j] = d[i]; rgb[j + 1] = d[i + 1]; rgb[j + 2] = d[i + 2]; }
  return rgb;
}
