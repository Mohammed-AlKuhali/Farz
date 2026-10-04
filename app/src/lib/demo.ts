/** Demo trays: /demo/manifest.json is written by the model agent. Format is read tolerantly; missing => no button. */
export interface DemoTray {
  src: string;
  title: string;
  titleAr?: string;
  credit?: string;
  note?: string;
  synthetic: boolean;
  /** user-facing caption in each language (falls back to `note`) */
  noteAr?: string;
  noteEn?: string;
  /** known answer for SYNTHETIC trays (from the manifest), shown next to Farz's own counts */
  trueCounts?: Record<string, number>;
  /** the outcome recorded (and verified) for this tray in the manifest */
  expected?: { band: string; about: boolean; reason: string };
}

type Raw = string | Record<string, unknown>;

export function parseDemoManifest(json: unknown, base: string): DemoTray[] {
  let list: Raw[] = [];
  if (Array.isArray(json)) list = json as Raw[];
  else if (json && typeof json === "object") {
    const o = json as Record<string, unknown>;
    const arr = o.trays ?? o.items ?? o.demos ?? o.images ?? o.photos;
    if (Array.isArray(arr)) list = arr as Raw[];
  }
  const out: DemoTray[] = [];
  list.forEach((it, i) => {
    const o: Record<string, unknown> = typeof it === "string" ? { src: it } : it ?? {};
    const src = (o.src ?? o.file ?? o.image ?? o.path ?? o.url ?? o.png ?? o.jpg) as string | undefined;
    if (!src || typeof src !== "string") return;
    const url = /^(https?:)?\//.test(src) ? src : `${base}demo/${src.replace(/^\.?\/?(demo\/)?/, "")}`;
    const title = String(o.title_en ?? o.titleEn ?? o.title ?? o.name ?? o.label ?? o.id ?? `Tray ${i + 1}`);
    const str = (v: unknown) => (typeof v === "string" && v.trim() ? v : undefined);
    const credit = str(o.credit) ?? str(o.source) ?? str(o.licence) ?? str(o.license);
    const note = str(o.note) ?? str(o.description);
    const synthetic = Boolean(o.synthetic ?? /synthetic/i.test(`${title} ${credit ?? ""} ${src}`));
    const tc = o.true_counts ?? o.trueCounts;
    const trueCounts = tc && typeof tc === "object" ? (tc as Record<string, number>) : undefined;
    const ex = o.expected as DemoTray["expected"] | undefined;
    out.push({ src: url, title, titleAr: str(o.title_ar) ?? str(o.titleAr) ?? str(o.name_ar), credit, note, synthetic, trueCounts,
      noteAr: str(o.note_ar), noteEn: str(o.note_en), expected: ex && typeof ex === "object" && typeof ex.band === "string" ? ex : undefined });
  });
  return out;
}

export async function loadDemoTrays(base: string): Promise<DemoTray[]> {
  try {
    const r = await fetch(`${base}demo/manifest.json`);
    if (!r.ok) return [];
    const ct = r.headers.get("content-type") ?? "";
    if (ct.includes("text/html")) return [];
    return parseDemoManifest(await r.json(), base);
  } catch {
    return [];
  }
}
