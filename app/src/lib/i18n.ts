import type { ClipId } from "./clips";
import type { BeanCall } from "./rules";
import type { Band } from "./wilson";

export type Lang = "ar" | "en";

/** On-screen text for each clip = what the voice says (docs/CLIPS.md), so screen and voice never disagree. */
export const CLIP_TEXT: Record<ClipId, { ar: string; en: string }> = {
  c01_welcome: { ar: "أهلين. صوّري حوالي مية حبة بن أخضر على قماشة بيضا، مفرّقة عن بعض.", en: "Welcome. Photograph about 100 green coffee beans on a white cloth, spread apart." },
  c02_blur: { ar: "الصورة مش واضحة. ثبّتي التلفون وصوّري ثاني.", en: "The photo is not clear. Hold the phone still and take it again." },
  c03_dark: { ar: "الصورة مظلمة. صوّري في مكان فيه نور.", en: "Too dark. Take it somewhere with light." },
  c04_spread: { ar: "الحبوب لاصقة ببعض. فرّقيها وصوّري ثاني.", en: "Beans are touching. Spread them out and take it again." },
  c05_count: { ar: "حطّي حوالي مية حبة، وصوّري ثاني.", en: "Use about 100 beans and take it again." },
  c06_not_green: { ar: "هذا مش بن أخضر. صوّري البن بعد التقشير وقبل التحميص.", en: "This isn't green coffee. Photograph beans after hulling, before roasting." },
  c07_unsure: { ar: "مش متأكد. ودّي العينة للجمعية يشوفها المختص.", en: "Not sure. Take the sample to the cooperative for a specialist to check." },
  c08_clean: { ar: "البن نظيف، العيوب قليلة.", en: "The coffee is clean; few defects." },
  c09_some: { ar: "فيه عيوب. شيلي الحبوب المعلّمة بالأحمر وصوّري ثاني.", en: "Some defects. Remove the beans marked red and photograph again." },
  c10_many: { ar: "العيوب كثيرة. فرّزي البن كله قبل البيع.", en: "Many defects. Sort the whole lot before selling." },
  c11_dark_beans: { ar: "فيه حبوب سودا أو حامضة. شيليها قبل البيع.", en: "There are black or sour beans. Remove them before selling." },
  c12_insect: { ar: "فيه حبوب مخرّمة. ورّيها الجمعية.", en: "There are holed beans. Show them to the cooperative." },
  c13_broken: { ar: "فيه حبوب مكسّرة. شيليها قبل البيع.", en: "There are broken beans. Remove them before selling." },
  c14_unhulled: { ar: "فيه حب لسه بقشره. قشّري العينة قبل الفحص.", en: "Some beans still have their husk. Hull the sample before checking." },
  c15_slip: { ar: "الرسالة فيها عدد الحبوب والعيوب وتاريخ اليوم، ومكتوب فيها: فحص ذاتي مش تصنيف. ترسليها؟", en: "The message has the bean and defect counts and today's date, and says “self-check, not a grade”. Send it?" },
  c15b_slip_unsure: { ar: "الرسالة فيها عدد الحبوب وتاريخ اليوم، ومكتوب فيها: مش متأكد، فحص ذاتي مش تصنيف. ترسليها؟", en: "The message has the bean count and today's date, and says “not sure, self-check, not a grade”. Send it?" },
  c16_privacy: { ar: "الصورة انمسحت. ما يطلع من التلفون إلا الرسالة اللي ترسليها أنتي.", en: "The photo is deleted. Nothing leaves the phone except the message you send." },
  c17_wiped: { ar: "انمسح كل شي.", en: "Everything is erased." },
  c18_another: { ar: "النتيجة قريبة. صوّري حفنة ثانية من نفس البن عشان نتأكد.", en: "The result is close. Photograph another handful from the same coffee so we can be sure." },
};

export const CALL_NAME: Record<BeanCall, { ar: string; en: string }> = {
  good: { ar: "سليمة", en: "Sound" },
  dark: { ar: "سودا أو حامضة", en: "Black or sour" },
  insect: { ar: "مخرّمة", en: "Insect holes" },
  broken: { ar: "مكسّرة", en: "Broken" },
  unhulled: { ar: "بقشرها", en: "Still in husk" },
  other_defect: { ar: "عيب ثاني", en: "Other defect" },
  unsure: { ar: "مش متأكد", en: "Not sure" },
};

/** Defect TYPES are never facts (the v2 model's type is unreliable): every type is said as "possibly …". */
export const POSSIBLE_NAME: Record<Exclude<BeanCall, "good" | "unsure">, { ar: string; en: string }> = {
  dark: { ar: "يمكن سودا أو حامضة", en: "Possibly black or sour" },
  insect: { ar: "يمكن مخرّمة", en: "Possibly insect holes" },
  broken: { ar: "يمكن مكسّرة", en: "Possibly broken" },
  unhulled: { ar: "يمكن بقشرها", en: "Possibly still in husk" },
  other_defect: { ar: "عيب ثاني", en: "Another kind of defect" },
};

export const BAND_NAME: Record<Band, { ar: string; en: string }> = {
  clean: { ar: "نظيف", en: "Clean" },
  some: { ar: "فيه عيوب", en: "Some defects" },
  many: { ar: "عيوب كثيرة", en: "Many defects" },
  unsure: { ar: "مش متأكد", en: "Not sure" },
};

const S = {
  tagline: { ar: "فحص البن الأخضر بالتلفون", en: "Green-coffee self-check" },
  listen: { ar: "اسمعي", en: "Listen" },
  stop: { ar: "وقّفي", en: "Stop" },
  takePhoto: { ar: "صوّري البن", en: "Photograph the beans" },
  gallery: { ar: "صورة محفوظة", en: "Saved photo" },
  demo: { ar: "صينية تجريبية", en: "Try a demo tray" },
  demoTitle: { ar: "اختاري صينية", en: "Pick a demo tray" },
  counting: { ar: "أعدّ الحبوب…", en: "Counting the beans…" },
  stepCloth: { ar: "قماشة بيضا", en: "White cloth" },
  stepBeans: { ar: "حوالي ١٠٠ حبة، مفرّقة", en: "~100 beans, apart" },
  stepPhone: { ar: "التلفون من فوق", en: "Phone straight above" },
  beans: { ar: "حبة", en: "beans" },
  defectsBetween: { ar: "العيوب بين {lo}٪ و{hi}٪", en: "Defects between {lo}% and {hi}%" },
  /** the denominator is ALWAYS the beans Farz was sure about, and says so (never next to a bare total) */
  defectsOf: { ar: "{d} فيها عيب، من {n} حبة فرز متأكد منها", en: "{d} with a defect, out of the {n} beans Farz was sure about" },
  beansFound: { ar: "حبة لقاها فرز في الصورة", en: "beans found in the photo" },
  sureBeans: { ar: "فرز متأكد منها", en: "Farz was sure" },
  notSureBeans: { ar: "فرز مش متأكد منها", en: "Farz was not sure" },
  ledgerAria: { ar: "{found} حبة لقاها فرز: {sure} متأكد منها ({d} فيها عيب و{s} سليمة)، و{u} مش متأكد منها.", en: "{found} beans found: Farz was sure about {sure} ({d} with a defect, {s} sound) and not sure about {u}." },
  whyDarkLot: { ar: "أغلب الحبوب سودا. فرز ما يقدر يحكم على بن زي هذا، والمختص يقدر.", en: "Most of these beans are black. Farz cannot judge a lot like this; a specialist can." },
  voiceComing: { ar: "الصوت جاي قريب. الكلام مكتوب على الشاشة.", en: "Voice coming soon. Everything is written on screen." },
  rangeLine: { ar: "غالباً بين {lo}٪ و{hi}٪ (ثقة ٩٥٪)", en: "Likely between {lo}% and {hi}% (95% range)" },
  aboutTag: { ar: "تقريباً", en: "About" },
  ruleOfThumb: { ar: "الحدود قاعدة تقريبية منّا، مش تصنيف رسمي: أقل من ٥٪ نظيف، من ٥ إلى ٢٠٪ فيه عيوب، فوق ٢٠٪ كثيرة.", en: "Bands are our rule of thumb, not an official grade: under 5% clean, 5–20% some, over 20% many." },
  addHandful: { ar: "أضيفي حفنة ثانية", en: "Add another handful" },
  handfulOf: { ar: "حفنة {i} من {n}", en: "Handful {i} of {n}" },
  handfulsPooled: { ar: "{h} حفنات من نفس البن، محسوبة مع بعض", en: "{h} handfuls of the same coffee, counted together" },
  lotMax: { ar: "هذا أحسن تقدير بعد ٣ حفنات.", en: "This is the best estimate after 3 handfuls." },
  whyImplausible: { ar: "العينة مش مثل البن اللي يعرفه فرز، فما يقدر يحكم عليها.", en: "This sample is not like the beans Farz knows (implausible or unfamiliar sample), so it will not judge it." },
  whyUnsure: { ar: "ما عرفت {u} من {n} حبة لقاها فرز.", en: "Farz could not tell {u} of the {n} beans it found." },
  newCheck: { ar: "فحص جديد", en: "New check" },
  sendSlip: { ar: "أرسلي الرسالة", en: "Send the slip" },
  slipTitle: { ar: "هذي الرسالة بالضبط", en: "This is exactly what is sent" },
  openSms: { ar: "افتحي الرسائل", en: "Open Messages" },
  cancel: { ar: "رجوع", en: "Cancel" },
  newPhoto: { ar: "صورة جديدة", en: "New photo" },
  retake: { ar: "صوّري ثاني", en: "Take it again" },
  history: { ar: "آخر الفحوصات", en: "Recent checks" },
  historyEmpty: { ar: "ما فيه فحوصات محفوظة.", en: "No saved checks." },
  historyNote: { ar: "يحفظ آخر ٥ فحوصات: أرقام فقط، بدون صور.", en: "Keeps the last 5 checks: numbers only, never photos." },
  wipe: { ar: "امسحي الكل", en: "Erase all" },
  selfCheck: { ar: "فحص ذاتي مش تصنيف", en: "Self-check, not a grade" },
  about: { ar: "عن فرز", en: "About Farz" },
  home: { ar: "الرئيسية", en: "Home" },
  back: { ar: "رجوع", en: "Back" },
  stub: { ar: "المصنّف مش موجود: كل الحبوب «مش متأكد».", en: "Classifier missing — stub in use, every bean is “not sure”." },
  offlineReady: { ar: "يشتغل بدون نت", en: "Works offline" },
  offlinePending: { ar: "يجهّز للعمل بدون نت…", en: "Preparing offline use…" },
  photoGone: { ar: "الصورة ما تنحفظ", en: "Photo is not saved" },
  error: { ar: "ما قدرت أقرا الصورة. صوّري ثاني.", en: "Could not read that photo. Take it again." },
  startFailed: { ar: "فرز ما قدر يجهّز عدّاد الحبوب. اضغطي «حاولي ثاني».", en: "Farz could not start its bean counter. Tap “Try again”." },
  tryAgain: { ar: "حاولي ثاني", en: "Try again" },
  legendDefect: { ar: "حلقة حمرا بعلامة × = عيب", en: "Red ring with × = defect" },
  legendUnsure: { ar: "حلقة صفرا متقطعة ؟ = مش متأكد", en: "Yellow dashed ring ? = not sure" },
  legendSound: { ar: "حلقة بيضا رفيعة = سليمة", en: "Thin white ring = sound" },
  listenPrivacy: { ar: "اسمعي عن الخصوصية", en: "Listen: privacy" },
  slipPictures: { ar: "نفس الأرقام بالصور:", en: "The same numbers as pictures:" },
  demoGroupSynthetic: { ar: "صواني مركّبة (تُظهر الطريقة، مش الدقة)", en: "Synthetic trays (show the flow, not accuracy)" },
  demoGroupReal: { ar: "صور حقيقية (حدود فرز)", en: "Real photos (Farz's limits)" },
  synthetic: { ar: "مركّبة SYNTHETIC", en: "SYNTHETIC" },
  realPhoto: { ar: "صورة حقيقية", en: "Real photo" },
  demoReal: { ar: "تجربة: صورة حقيقية", en: "DEMO: real photo" },
  demoTruth: { ar: "الصينية", en: "Tray" },
  demoKnown: { ar: "الجواب الصحيح", en: "Known answer" },
  demoFarz: { ar: "فرز قال", en: "Farz found" },
  defectBeans: { ar: "فيها عيب", en: "Defect" },
  possibleTypes: { ar: "نوع العيب المحتمل — مش أكيد", en: "Possible defect type — not certain" },
  possibleNote: { ar: "النوع تخمين من فرز. المختص في الجمعية يحدد النوع.", en: "The type is Farz's guess. The cooperative's grader decides the type." },
  calibratedFor: { ar: "مضبوط على: {name} ({n} حبة)", en: "Calibrated for: {name} ({n} beans)" },
  calibratedPilot: { ar: "تجربة", en: "Pilot" },
  calibrate: { ar: "اضبطي فرز على بن جمعيتك", en: "Calibrate for your cooperative" },
  calibrateShort: { ar: "ضبط الجمعية", en: "Cooperative calibration" },
} as const;

export type StrKey = keyof typeof S;
export function t(lang: Lang, key: StrKey, vars: Record<string, string | number> = {}): string {
  let s: string = S[key][lang];
  for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, String(v));
  return s;
}

const AR_DIGITS = "٠١٢٣٤٥٦٧٨٩";
export function num(lang: Lang, n: number | string): string {
  const s = String(n);
  return lang === "ar" ? s.replace(/[0-9]/g, (d) => AR_DIGITS[+d]).replace(".", "٫") : s;
}
export const pct = (lang: Lang, x: number) => num(lang, Math.round(x * 1000) / 10);
