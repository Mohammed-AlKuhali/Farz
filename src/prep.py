"""Build a clean, resized, de-duplicated image manifest from the raw coffee-leaf datasets.

Every source keeps its country tag so we can evaluate on a country the model never saw.
Classes: healthy, rust, miner, phoma, cercospora.
"""
import csv, hashlib, os, random, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/proc"
CLASSES = ["healthy", "rust", "miner", "phoma", "cercospora"]
MAX_SIDE = 320
CAP_PER_CLASS = int(os.environ.get("CAP", 1500))  # stops 18k augmented JMuBEN crops drowning everything
random.seed(13)

# folder-name keyword -> class
KEYWORDS = [("health", "healthy"), ("rust", "rust"), ("roya", "rust"), ("miner", "miner"),
            ("minador", "miner"), ("phoma", "phoma"), ("cerc", "cercospora"), ("cersc", "cercospora")]

# source folder (under data/raw) -> (source id, country, variety, conditions)
SOURCES = {
    "ug": ("uganda", "Uganda", "coffee (variety n/s)", "field"),
    "jmuben": ("jmuben", "Kenya", "Arabica", "field crops 128px"),
    "jmuben2": ("jmuben2", "Kenya", "Arabica", "field crops 128px"),
    "rocole": ("rocole", "Ecuador", "Robusta", "field"),
    "decafia_cls": ("decafia", "Colombia", "Arabica", "field, smartphone, 1,500 m"),
}

def label_from_path(p: Path):
    s = str(p.parent).lower()
    for kw, cls in KEYWORDS:
        if kw in s:
            return cls
    return None

def collect():
    items = []  # (src_path, cls, source, country, variety, cond)
    # BRACOL: labels in csv; 0 healthy 1 miner 2 rust 3 phoma 4 cercospora 5 excluded
    bdir = RAW / "bracol/coffee-datasets/coffee-datasets/leaf"
    if (bdir / "dataset.csv").exists():
        m = {"0": "healthy", "1": "miner", "2": "rust", "3": "phoma", "4": "cercospora"}
        for r in csv.DictReader(open(bdir / "dataset.csv")):
            f = bdir / "images" / f"{r['id']}.jpg"
            if r["predominant_stress"] in m and f.exists():
                items.append((f, m[r["predominant_stress"]], "bracol", "Brazil", "Arabica", "detached leaf, white background"))
    for folder, (src, country, variety, cond) in SOURCES.items():
        d = RAW / folder
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and not f.name.startswith("._"):
                cls = label_from_path(f)
                if cls:
                    items.append((f, cls, src, country, variety, cond))
    return items

def process(item):
    src, cls, source, *_ = item
    try:
        im = Image.open(src); im.load(); im = im.convert("RGB")
    except Exception:
        return None
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    h = hashlib.md5(im.tobytes()).hexdigest()
    out = PROC / source / cls / f"{h}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        im.save(out, quality=90)
    return (str(out.relative_to(ROOT)), h)

def file_md5(item):
    return hashlib.md5(open(item[0], "rb").read()).hexdigest()

def main():
    items = collect()
    with ProcessPoolExecutor() as ex:
        digests = list(ex.map(file_md5, items, chunksize=256))
    uniq, seen_f = [], set()
    for it, d in zip(items, digests):
        if d not in seen_f:
            seen_f.add(d); uniq.append(it)
    print(f"raw files {len(items)}, byte-identical duplicates removed {len(items)-len(uniq)}")
    raw_dups = {}
    for it, d in zip(items, digests):
        raw_dups.setdefault((it[2], it[1]), [0, set()]); raw_dups[(it[2], it[1])][0] += 1; raw_dups[(it[2], it[1])][1].add(d)
    for k, (n, u) in sorted(raw_dups.items()):
        print(f"  {k[0]:10s} {k[1]:11s} files={n:6d} unique={len(u):6d}")
    items = uniq
    by = {}
    for it in items:
        by.setdefault((it[2], it[1]), []).append(it)
    kept = []
    for k, v in sorted(by.items()):
        random.shuffle(v)
        kept += v[: CAP_PER_CLASS]
    print(f"found {len(items)} images, keeping {len(kept)} after per-class cap {CAP_PER_CLASS}")
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(process, kept, chunksize=64))
    rows, seen, bad, dup = [], set(), 0, 0
    for it, res in zip(kept, results):
        if res is None:
            bad += 1; continue
        path, h = res
        if h in seen:  # exact duplicate pixels -> drop
            dup += 1; continue
        seen.add(h)
        rows.append({"path": path, "label": it[1], "source": it[2], "country": it[3], "variety": it[4], "conditions": it[5]})
    with open(ROOT / "data/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"undecodable dropped: {bad}, exact duplicates dropped: {dup}, manifest rows: {len(rows)}")
    tab = {}
    for r in rows:
        tab.setdefault(r["source"], {}).setdefault(r["label"], 0); tab[r["source"]][r["label"]] += 1
    for s, c in tab.items():
        print(f"  {s:10s} " + "  ".join(f"{k}={c.get(k,0)}" for k in CLASSES))

if __name__ == "__main__":
    main()
