"""Farz v2 data hunter: every external green-bean dataset, where it lives in data/raw2/, its licence (verbatim
from the source page / file), and how each source label maps onto the Farz v2 classes.

Farz v2 classes: good, dark (black/sour/fungus), insect, broken (broken/chipped/cut/immature/shell),
unhulled (parchment/dried cherry/pod/husk), other_defect (defect of unknown or other type).
`source_class` is always kept in the CSV so a trainer can re-map or drop (e.g. foreign material, peaberry).
"""
import glob, os, re
from data_common import ROOT

RAW2 = os.path.join(ROOT, "data", "raw2")

# SCA-style defect names (lower-cased, typos as spelled in the downloads) -> Farz class
SCA_MAP = {
    "full black": "dark", "partial black": "dark", "full sour": "dark", "partial sour": "dark",
    "fungus": "dark", "fungus damange": "dark", "fungus damage": "dark",
    "slight insect": "insect", "severe insect": "insect", "slight insect damage": "insect",
    "severe insect damange": "insect", "severe insect damage": "insect",
    "broken": "broken", "cut": "broken", "shell": "broken", "immature": "broken",
    "parchment": "unhulled", "dry cherry": "unhulled", "cherry pod": "unhulled", "hull": "unhulled", "husk": "unhulled",
    "floater": "other_defect", "withered": "other_defect", "fade": "other_defect",
    "foreign material": "other_defect",
}
LOJA_NAMES = ['agrio_parcial', 'broca_leve_severa', 'cereza_seca', 'concha', 'cortado', 'grano_negro',
              'negro_parcial', 'normal', 'por_hongo']
LOJA_MAP = {'agrio_parcial': 'dark', 'broca_leve_severa': 'insect', 'cereza_seca': 'unhulled', 'concha': 'broken',
            'cortado': 'broken', 'grano_negro': 'dark', 'negro_parcial': 'dark', 'normal': 'good', 'por_hongo': 'dark'}

SOURCES = {
    "mfu17": dict(
        url="https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original",
        licence="Kaggle licence field: 'Unknown' (metadata API: 'unknown'). No licence text on the page.",
        country="Thailand (Mae Fah Luang University; paper: Smart Agricultural Technology 9 (2024) 100680)",
        mode="single"),
    "usk_coffee": dict(
        url="https://www.kaggle.com/datasets/mfaisalriftiarrasyid/duardata (mirror of USK-Coffee, https://coffee.comvislab-usk.org/)",
        licence="Kaggle mirror licence field: 'Unknown'. The authors' page states no licence; official download is behind a Google Form (not used).",
        country="Indonesia (Banda Aceh, KNT Coffee)", mode="single"),
    "vicanadya16": dict(
        url="https://www.kaggle.com/datasets/vicanadya/coffee-defect-16-classes",
        licence="Kaggle licence field: 'CC0: Public Domain' (metadata API: 'CC0-1.0'). No description or provenance on the page.",
        country="unknown (no provenance given)", mode="precrop"),
    "loja_yolo": dict(
        url="https://www.kaggle.com/datasets/cristianyagg/defectos-grano-de-cafe-de-la-provincia-de-loja (Roboflow export of https://universe.roboflow.com/tesis-kmw54/deteccion-de-defectos-del-grano v14)",
        licence="Kaggle licence field: 'Apache 2.0'; README.roboflow.txt inside the download: 'License: CC BY 4.0'. Treated as CC BY 4.0 (the stricter, original).",
        country="Ecuador (Loja province, per dataset title)", mode="tray_polygons"),
    "lojano": dict(
        url="https://www.kaggle.com/datasets/patopucho/lojano-arabica-coffee",
        licence="Kaggle licence field: 'Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)'",
        country="Ecuador (Loja), Coffea arabica", mode="tray_single_class"),
    "afiyah_deteksi": dict(
        url="https://www.kaggle.com/datasets/afiyahmusrah/deteksi-biji-kopi",
        licence="Kaggle licence field: 'CC0: Public Domain' (metadata API: 'CC0-1.0'). No description on the page.",
        country="unknown (Indonesian-language labels: biji bagus = good bean, biji rusak = damaged bean)", mode="single"),
    "mindforge_doubleside": dict(
        url="https://github.com/Mindforge-inc/GreenCoffeeBeanDoubleSide",
        licence="No licence file and no licence in the README; GitHub API license field: null (all rights reserved by default).",
        country="unknown; README in Korean (labels 정상두 etc.); likely the public 12-class set of the 'Dual-Sided Green Coffee Bean Defect Inspection' paper (Agriculture 16:1796), NOT verified",
        mode="mask_chroma"),
    "notplying_defects": dict(
        url="https://github.com/Notplying/DatasetCoffeeBeanDefects",
        licence="No licence file; GitHub API license field: null (all rights reserved by default). Repo description: 'Dataset Cacat Biji Kopi'.",
        country="unknown (Indonesian-language description); phone photos dated 2024-10-10 by file name",
        mode="box"),
    "afiyah_bijikopi": dict(
        url="https://www.kaggle.com/datasets/afiyahmusrah/bijikopideteksi",
        licence="Kaggle licence field: 'CC0: Public Domain' (metadata API: 'CC0-1.0'). No description on the page.",
        country="unknown (Indonesian-language labels); sample EXIF: samsung Galaxy A05s, 6112x6112",
        mode="single"),
    "daffa_defect": dict(
        url="https://github.com/daffakurnia11/Coffee-Beans-Defect-Classification (folder Source/)",
        licence="No licence file; GitHub API license field: null (all rights reserved by default). No README.",
        country="unknown; EXIF camera Canon EOS 60D, dates Feb 2023",
        mode="single"),
    "samruddh_grading": dict(
        url="https://huggingface.co/datasets/SamruddhK/coffee-bean-grading-dataset",
        licence="Hugging Face card: 'license: mit'",
        country="India (Coorg/Kodagu, Karnataka); card says both 'Arabica' and 'Coffea robusta'", mode="single"),
}


def items_mfu17():
    for f in sorted(glob.glob(os.path.join(RAW2, "mfu17", "*", "*"))):
        sc = os.path.basename(os.path.dirname(f))
        yield dict(source="mfu17", path=f, source_class=sc, farz_class=SCA_MAP[sc.lower()],
                   group=os.path.relpath(f, RAW2), split_hint="")


def items_usk():
    m = {"premium": "good", "longberry": "good", "peaberry": "good", "defect": "other_defect"}
    for f in sorted(glob.glob(os.path.join(RAW2, "usk_coffee", "USK-Coffee", "*", "*", "*.jpg"))):
        parts = f.split(os.sep); sc = parts[-2]; split = parts[-3]
        yield dict(source="usk_coffee", path=f, source_class=sc, farz_class=m[sc],
                   group=os.path.relpath(f, RAW2), split_hint=split)


def items_vicanadya():
    for f in sorted(glob.glob(os.path.join(RAW2, "vicanadya16", "coffee_classification", "*", "*", "*"))):
        parts = f.split(os.sep); sc = parts[-2]; split = parts[-3]
        mm = re.match(r"(.+?)_(?:JPG|jpg|jpeg|png|PNG)_(\d+)\.\w+$", parts[-1])
        yield dict(source="vicanadya16", path=f, source_class=sc, farz_class=SCA_MAP[sc.lower()],
                   group="vicanadya16/" + (mm.group(1) if mm else parts[-1]), split_hint=split)


def items_loja():
    for split in ("train", "valid", "test"):
        for f in sorted(glob.glob(os.path.join(RAW2, "loja_yolo", split, "images", "*"))):
            lab = os.path.join(RAW2, "loja_yolo", split, "labels", os.path.splitext(os.path.basename(f))[0] + ".txt")
            polys = []
            if os.path.exists(lab):
                for line in open(lab):
                    v = line.split()
                    if len(v) < 5: continue
                    k = int(v[0]); pts = list(map(float, v[1:]))
                    if len(pts) == 4:  # plain YOLO box
                        x, y, w, h = pts; box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
                    else:              # YOLO polygon
                        xs, ys = pts[0::2], pts[1::2]; box = (min(xs), min(ys), max(xs), max(ys))
                    polys.append((LOJA_NAMES[k], box))
            if not polys: continue
            base = re.sub(r"_(?:JPG|jpg|jpeg|png)\.rf\..*$", "", os.path.basename(f))
            yield dict(source="loja_yolo", path=f, polys=polys, group="loja_yolo/" + base, split_hint=split)


def items_lojano():
    m = {"bueno": "good", "defectuoso": "other_defect"}
    pats = [("dataset-lojano/fotos en bruto/jpg", "main"), ("img-generalizacon/completas", "generalisation")]
    for sub, tag in pats:
        for f in sorted(glob.glob(os.path.join(RAW2, "lojano", sub, "*", "*"))):
            sc = os.path.basename(os.path.dirname(f))
            yield dict(source="lojano", path=f, source_class=sc, farz_class=m[sc],
                       group="lojano/" + os.path.basename(f), split_hint=tag)


def items_afiyah():
    m = {"bijibagus": "good", "bijirusak": "other_defect"}
    for f in sorted(glob.glob(os.path.join(RAW2, "afiyah_deteksi", "**", "*.*"), recursive=True)):
        sc = os.path.basename(os.path.dirname(f)).lower()
        if sc not in m: continue
        split = os.path.basename(os.path.dirname(os.path.dirname(f)))
        yield dict(source="afiyah_deteksi", path=f, source_class=sc, farz_class=m[sc],
                   group=os.path.relpath(f, RAW2), split_hint=split)


def items_afiyah2():
    m = {"bijikopibagus": "good", "bijikopirusak": "other_defect"}
    for f in sorted(glob.glob(os.path.join(RAW2, "afiyah_bijikopi", "**", "*.*"), recursive=True)):
        sc = os.path.basename(os.path.dirname(f)).lower()
        if sc not in m: continue
        split = os.path.basename(os.path.dirname(os.path.dirname(f)))
        yield dict(source="afiyah_bijikopi", path=f, source_class=sc, farz_class=m[sc],
                   group=os.path.relpath(f, RAW2), split_hint=split)


def items_samruddh():
    m = {"CGA": "good", "CGD": "other_defect"}
    for g in ("CGA", "CGD"):
        for f in sorted(glob.glob(os.path.join(RAW2, "samruddh_grading", g, "images", "*.jp*g"))):
            if os.path.getsize(f) < 10000: continue  # partial / failed download
            yield dict(source="samruddh_grading", path=f, source_class=g, farz_class=m[g],
                       group=os.path.relpath(f, RAW2), split_hint="")


NOTPLYING_NAMES = ["Broken/Chipped/Cut", "Dried Cherry/Pod", "Floater", "Foreign Matter", "Full Black", "Full Sour",
                  "Fungus Damaged", "Hull/Husk", "Immature/Unripe", "Parchment/Pergamino", "Partial Black",
                  "Partial Sour", "Severe Insect Damage", "Shell", "Slight Insect Damage", "Withered"]  # notes.json
NOTPLYING_MAP = {"Broken/Chipped/Cut": "broken", "Dried Cherry/Pod": "unhulled", "Floater": "other_defect",
                 "Foreign Matter": "other_defect", "Full Black": "dark", "Full Sour": "dark", "Fungus Damaged": "dark",
                 "Hull/Husk": "unhulled", "Immature/Unripe": "broken", "Parchment/Pergamino": "unhulled",
                 "Partial Black": "dark", "Partial Sour": "dark", "Severe Insect Damage": "insect", "Shell": "broken",
                 "Slight Insect Damage": "insect", "Withered": "other_defect"}
MINDFORGE_MAP = {"Normal": "good", "Black": "dark", "Sour": "dark", "Fungus-damaged": "dark",
                 "Inspect-damaged": "insect", "Broken": "broken", "Shell": "broken", "Immature": "broken",
                 "Fissure": "other_defect", "Over-dried": "other_defect", "Floater": "other_defect",
                 "Foreign matter": "other_defect"}


def items_mindforge():
    base = os.path.join(RAW2, "mindforge_doubleside", "GreenCoffeeBeanDoubleSide-main")
    for rnd in ("1st", "2nd"):
        for lab in sorted(glob.glob(os.path.join(base, rnd, "*.txt"))):
            cls = os.path.splitext(os.path.basename(lab))[0]
            for line in open(lab, encoding="utf-8", errors="replace"):
                if ":" not in line: continue
                fn, rest = line.split(":", 1)
                stem = os.path.splitext(fn.strip())[0]
                sides = dict(t.strip().split("-") for t in rest.split(",") if "-" in t)
                for side in ("U", "D"):
                    f = os.path.join(base, rnd, cls, side, stem + ".jpg")
                    if not os.path.exists(f): continue
                    if sides.get(side) != "1": continue   # this face does not show the class: not used
                    yield dict(source="mindforge_doubleside", path=f, source_class=cls, farz_class=MINDFORGE_MAP[cls],
                               group=f"mindforge/{rnd}/{cls}/{stem}", split_hint=rnd)


def items_notplying():
    base = os.path.join(RAW2, "notplying_defects", "DatasetCoffeeBeanDefects-main")
    imgs = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(base, "Grade*", "*.jpg"))}
    for lab in sorted(glob.glob(os.path.join(base, "labels", "*.txt"))):
        stem = os.path.splitext(os.path.basename(lab))[0]
        if stem not in imgs: continue
        boxes = []
        for line in open(lab):
            v = line.split()
            if len(v) != 5: continue
            k = int(v[0]); x, y, w, h = map(float, v[1:])
            boxes.append((NOTPLYING_NAMES[k], (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
        f = imgs[stem]
        yield dict(source="notplying_defects", path=f, polys=boxes, group="notplying/" + stem,
                   split_hint=os.path.basename(os.path.dirname(f)))


def items_daffa():
    m = {"Defected": "other_defect", "Undefected": "good"}
    for g in ("Defected", "Undefected"):
        for f in sorted(glob.glob(os.path.join(RAW2, "daffa_defect", "Source", g, "*"))):
            if not f.lower().endswith((".jpg", ".jpeg")) or os.path.getsize(f) < 100000: continue
            yield dict(source="daffa_defect", path=f, source_class=g, farz_class=m[g],
                       group=os.path.relpath(f, RAW2), split_hint="")


ITEMS = {"afiyah_bijikopi": items_afiyah2, "daffa_defect": items_daffa, "mindforge_doubleside": items_mindforge, "notplying_defects": items_notplying,"mfu17": items_mfu17, "usk_coffee": items_usk, "vicanadya16": items_vicanadya, "loja_yolo": items_loja,
         "lojano": items_lojano, "afiyah_deteksi": items_afiyah, "samruddh_grading": items_samruddh}
