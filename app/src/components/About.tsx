import type { ReadyInfo } from "../lib/engine";
import type { Lang } from "../lib/i18n";
import { num, t } from "../lib/i18n";
import type { AudioPack, ClipId } from "../lib/clips";
import type { StoredHead } from "../lib/localcal";
import { IconSpeaker, IconStop } from "./Icons";

const MB = (b: number) => (b / 1e6).toFixed(2);

/** The 7 datasets the shipped weights were trained on (reports_v2/clean_model.json "training_sources"), as each states its licence. */
const TRAINED_ON: { name: string; where: string; licence: string; nc?: boolean }[] = [
  { name: "Green Coffee Beans (J4ckDev)", where: "github.com/J4ckDev/GreenCoffeeBeansDataset", licence: "CC BY-NC-SA 4.0", nc: true },
  { name: "Lojano Arabica Coffee", where: "kaggle.com/datasets/patopucho/lojano-arabica-coffee", licence: "CC BY-NC 4.0", nc: true },
  { name: "Detección de defectos del grano (Loja, Ecuador)", where: "universe.roboflow.com/tesis-kmw54 (Kaggle: cristianyagg)", licence: "CC BY 4.0" },
  { name: "Coffee bean grading", where: "huggingface.co/datasets/SamruddhK/coffee-bean-grading-dataset", licence: "MIT" },
  { name: "Coffee defect 16 classes", where: "kaggle.com/datasets/vicanadya/coffee-defect-16-classes", licence: "CC0" },
  { name: "Deteksi biji kopi + Biji kopi deteksi (one uploader, same phone)", where: "kaggle.com/datasets/afiyahmusrah", licence: "CC0" },
];

export function About({ lang, info, swReady, pack, playing, say, head }: {
  lang: Lang; info: ReadyInfo | null; swReady: boolean; pack: AudioPack | null; playing: ClipId | null; say: (ids: ClipId | ClipId[]) => void;
  head: StoredHead | null;
}) {
  const ar = lang === "ar";
  const voice = pack ? (ar ? pack.voice_ar : pack.voice_en)
    : ar ? "هذي النسخة بدون صوت: كل رسالة مكتوبة على الشاشة ومعها صورة. الصوت المسجّل ينضاف بدون تغيير في الكود."
      : "This build has no voice: every message is on screen as text and pictures. A recorded voice pack can be added with no code change.";
  const model = info
    ? info.stub
      ? ar ? "المصنّف مش موجود — نسخة بديلة (كل الحبوب «مش متأكد»)." : "Classifier file missing — STUB in use (every bean is “not sure”)."
      : ar ? `المصنّف: ${num(lang, MB(info.modelBytes))} ميغابايت، يشتغل في التلفون.` : `Classifier: ${MB(info.modelBytes)} MB, runs on this phone.`
    : "…";
  return (
    <article className="about flex flex-col gap-4 text-qudad" data-testid="about">
      <h1 className="font-display text-[30px]">{ar ? "عن فرز" : "About Farz"}</h1>
      {pack && (
        <button className={`btn btn-voice self-center ${playing ? "is-playing" : ""}`} onClick={() => say("c16_privacy")} aria-pressed={playing !== null}>
          {playing ? <IconStop /> : <IconSpeaker />}
          <span>{playing ? t(lang, "stop") : t(lang, "listenPrivacy")}</span>
        </button>
      )}

      <section className="card-dark status-card p-4" data-testid="status">
        <h2>{ar ? "وين وصلنا (بصراحة)" : "Where this prototype stands"}</h2>
        <ul>
          <li><b>{ar ? "نموذج أولي للبحث، غير تجاري." : "A non-commercial research prototype."}</b>{" "}
            {ar ? "مش أداة تصنيف رسمية." : "Not a grading tool."}</li>
          <li data-testid="voice-status">{voice}</li>
          <li>{ar
            ? "جرّبناه على متصفح كروميوم في الكمبيوتر فقط. تجربة الآيفون لسه، والأندرويد ما انجرّب."
            : "Tested in desktop Chromium only (automated, phone-sized screen). iPhone test pending; Android untested."}</li>
          <li>{ar
            ? "على صور من مزرعة ما شافها قبل، حكمه على الحبة الوحدة ضعيف (دقة متوازنة ٠٫٦٤ بالمتوسط، بين ٠٫٥٦ و٠٫٨٠). عشان كذا قواعد الصورة ترسل أغلب الصور الجديدة لـ«مش متأكد»."
            : "On photos from a farm it has never seen, its call on a single bean is weak: good-vs-defect balanced accuracy 0.64 on average (0.56–0.80) across held-out datasets. That is why the photo rules send most new photos to “not sure”."}</li>
          <li>{ar
            ? "على ٤٦٤ صورة حقيقية من الهند ما أعطى ولا حكم (كلها «مش متأكد» أو «صوّري ثاني»)، وما قال عن ولا صورة من الـ٥٠ صورة درجة AAA «عيوب كثيرة»."
            : "On 464 real photos from India it gave no verdict at all (all “not sure” or “take it again”), and called none of the 50 top-grade (AAA) photos “many defects”."}</li>
          <li>{ar
            ? "ما فيه بن يمني في بيانات التدريب. التجربة الحقيقية تحتاج صور محلية يعلّمها مختص الجمعية."
            : "No Yemeni beans in the training data. A real pilot needs local labelled photos: a cooperative grader's own trays, labelled bean by bean, and a check on trays not used for calibration."}</li>
        </ul>
      </section>

      <section className="card-dark p-4">
        <h2>{ar ? "الخصوصية" : "Privacy"}</h2>
        <ul>
          <li>{ar ? "الصورة تنمسح بعد العدّ، وما تنحفظ أبداً." : "The photo is discarded after counting; it is never saved."}</li>
          <li>{ar ? "يحفظ آخر ٥ فحوصات (أرقام فقط) في التلفون، وزر واحد يمسحها." : "The last 5 checks (numbers only) stay on this phone; one tap erases them."}</li>
          <li>{ar ? "ضبط الجمعية (إذا انعمل) ملخص أرقام في التلفون بس، بدون صور، وله زر مسح." : "A cooperative calibration (if made) is a summary of numbers on this phone only — no photos — with its own reset button."}</li>
          <li>{ar ? "ما فيه حساب، ولا اسم شخص، ولا موقع. اسم الجمعية اللي يكتبه المختص في الضبط يبقى في هذا التلفون بس." : "No account, no personal name, no location. A cooperative name typed by the grader for calibration stays on this phone only."}</li>
          <li>{ar ? "ما يطلع من التلفون إلا الرسالة اللي ترسليها أنتي بيدك." : "Nothing leaves the phone except the SMS you choose to send yourself."}</li>
          <li>{ar ? "يشتغل بدون نت بعد أول فتح." : "Works with no internet after the first visit."} {swReady ? "✓" : ""}</li>
        </ul>
      </section>

      <section className="card-dark p-4">
        <h2>{ar ? "وين الذكاء الاصطناعي، ووين لا" : "What is AI — and what deliberately is not"}</h2>
        <p className="font-bold text-bean">{ar ? "ذكاء اصطناعي (شي واحد فقط):" : "AI (one part only):"}</p>
        <ul>
          <li>{ar
            ? <>مصنّف لكل حبة: سليمة إذا ثقته ٥٥٪ أو أكثر، فيها عيب إذا ثقته في العيب ٩١٪ أو أكثر، وغير كذا «مش متأكد». نوع العيب يظهر بس كـ«يمكن…»، مش كحقيقة.<span className="mt-1 block text-[15px] text-parchment" dir="ltr" lang="en">MobileNetV3-Small, fp16 ONNX, 6 outputs</span></>
            : "A per-bean classifier (MobileNetV3-Small, fp16 ONNX, run by onnxruntime-web in a background thread). Sound if P(sound) ≥ 0.55 (chosen by a pre-registered sweep); defect if P(defect) ≥ 0.91; otherwise “not sure”. The defect kind is shown only as “possibly …”, never as a fact, and never spoken or sent."}</li>
          <li>{model}</li>
          <li data-testid="about-calibration">{ar
            ? <>ضبط الجمعية (تجربة): المختص يعلّم ٢٠ حبة من بنهم، وفرز يقارن كل حبة بمتوسطين (سليمة/عيب) من مخرجات المصنّف نفسه. في دراسة مسجّلة مسبقاً جرّبناه على ٤ مجموعات صور ونجح على وحدة بس (من الإكوادور)، وعلى مجموعتين ثانيتين قال عن صواني نظيفة «عيوب كثيرة».{head ? ` الآن: مضبوط على ${head.name}.` : " الآن: غير مضبوط."}</>
            : <>Cooperative calibration (pilot): a grader labels 20 of their own beans and Farz compares each bean with two averages (sound / defect) of the classifier's own embedding. In a pre-registered test it was tried on 4 datasets and passed on 1 (Ecuador); on 2 others it called clean trays “many”.{head ? ` Now: calibrated for ${head.name}.` : " Now: not calibrated."}</>}</li>
        </ul>
        <p className="mt-2 font-bold text-parchment">{ar ? "مش ذكاء اصطناعي (عن قصد):" : "Not AI (on purpose):"}</p>
        <ul>
          <li>{ar ? "إيجاد الحبوب وعدّها: معالجة صور عادية." : "Finding and counting beans: classic image processing, identical in Python and in this app."}</li>
          <li>{ar ? "فحص الصورة: الضوء، الوضوح، اللون، الحبوب اللاصقة، العدد." : "Photo checks: light, blur, colour, touching beans, bean count — fixed rules."}</li>
          <li>{ar ? "الحكم: نسبة العيوب من الحبوب المعدودة، ومعها مدى ثقة ٩٥٪ (ويلسون). إذا المدى يقطع حد بين قسمين يقول «تقريباً» ويطلب حفنة ثانية من نفس البن (لحد ٣ حفنات)." : "The verdict: the defect share of the counted beans, with its 95% Wilson range. If the range crosses a band edge it says “about” and asks for another handful of the same coffee (up to 3, counted together)."}</li>
          <li>{ar ? "«مش متأكد، ودّيها الجمعية» لما المصنّف ما يعرف أكثر من ١٥٪ من الحبوب، أو لما أكثر من ٦٠٪ منها تطلع عيوب (عينة غريبة عليه)." : "“Not sure — take it to the cooperative” when the classifier cannot tell more than 15% of the beans, or when over 60% come out defective (a sample unlike anything it knows)."}</li>
          <li>{ar ? "الحدود (أقل من ٥٪، ٥–٢٠٪، أكثر من ٢٠٪) قاعدة تقريبية منّا، مش تصنيف رسمي." : "The bands (<5%, 5–20%, >20%) are our rule of thumb, not an official grade."}</li>
          <li>{ar ? "الكلام: قائمة رسائل ثابتة مكتوبة من قبل (والصوت بعد ما ينسجّل). ما فيه نص مولّد." : "What it says: a fixed list of pre-written messages (voice once recorded). No generated text."}</li>
        </ul>
        <p className="mt-2 font-bold text-cherry-light">{ar ? "ما يقوله أبداً:" : "Never says:"}</p>
        <ul>
          <li>{ar ? "سعر، اسم حشرة، أو مبيد. ونوع العيب ما ينقال كحقيقة." : "A price, a pest species or a pesticide — and never a defect type as a fact."}</li>
        </ul>
      </section>

      <section className="card-dark p-4" lang="en" dir="ltr" data-testid="credits">
        <h2>Model, data &amp; licences</h2>
        <ul>
          <li>Classifier <b>farz_beans_v2_clean</b> (MobileNetV3-Small, fp16, 3,082,642 bytes). Trained on <b>92,690 bean crops from the 7 public datasets that state a licence</b>:
            <ul className="mt-1">
              {TRAINED_ON.map((d) => (
                <li key={d.name}>{d.name} — <span className="credit">{d.where}</span> — <b>{d.licence}</b>{d.nc ? " (non-commercial)" : ""}</li>
              ))}
            </ul>
          </li>
          <li><b>Licence of these weights: non-commercial research prototype, ShareAlike — CC BY-NC-SA 4.0</b> (inherited from J4ckDev; Lojano adds CC BY-NC 4.0). Not for commercial use.</li>
          <li>Five datasets that state <b>no licence</b> (mfu17, a USK-Coffee mirror, daffa, Mindforge, Notplying) were used <b>only to test</b> the model. They were never trained on and are not in this app.</li>
          <li>Real test photos: <b>CBD Coffee Bean Dataset</b> (Mendeley Data 52877z55vr, India), CC BY 4.0 — test only, never trained on.</li>
          <li>Calibration demo: 20 bean crops from the Loja (Ecuador) dataset above, CC BY 4.0, cut and resized by Farz; their labels come from the dataset, standing in for a cooperative grader. The demo shows the quality check (correct labels accepted, deliberately wrong labels refused) and saves nothing. The model was trained on that dataset, so it shows the check, not accuracy.</li>
          <li>Demo trays marked SYNTHETIC are composites of single-bean photos from J4ckDev, which the model was trained on: they show the flow, not accuracy.</li>
          <li>Known gaps: no Yemeni beans (Udaini, Dawairi, Tufahi, Bura'ai) or Yemeni naturals in the training data — Farz may mark darker Yemeni naturals as defects. That is why it gives counts, not grades, and sends “not sure” to the cooperative.</li>
          <li>{pack?.voice_en ?? "Voice: none in this build (text and pictures only)."} Voice clips are an “audio pack” (public/audio/ + pack.json): recordings under the 19 fixed file names switch the voice on with no code change. The app never calls any online service at runtime.</li>
          <li>Software: onnxruntime-web (MIT), React (MIT), Workbox (MIT); fonts Reem Kufi and Noto Naskh Arabic (SIL OFL 1.1).</li>
        </ul>
      </section>

      <section className="card-dark p-4">
        <h2>{ar ? "فحص ذاتي مش تصنيف" : "Self-check, not a grade"}</h2>
        <p>{ar
          ? "فرز يساعدك تفرزي قبل ما يجي المشتري. الحكم الأخير للمختص في الجمعية بعد ما يشوف العينة نفسها."
          : "Farz helps you sort before the collector comes. The final word belongs to the cooperative's grader, who checks the actual sample."}</p>
      </section>
    </article>
  );
}
