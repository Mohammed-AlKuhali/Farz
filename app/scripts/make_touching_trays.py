"""SYNTHETIC "beans pushed together" trays for measuring the touching rule (app/src/lib/touching.ts).

Built exactly like the shipped demo trays (src/beans/make_demo.py: J4ckDev bean crops cut out with the bean
finder's own mask, size jitter 0.88-1.12, random rotation, flat 235-grey sheet, JPEG q90) but with the beans
pushed together. Every pasted bean's class and centre is recorded, so each detected blob can be scored as
"one bean" or "merged" (how many pasted centres fall inside it).

Layouts (100 beans each; same defect mix as the ~12% demo tray unless the name says otherwise):
  spread_108  control: the demo-tray grid (108 px cells, beans ~60 px)            -> beans apart
  cell72 / cell64 / cell58   the same grid squeezed to 72 / 64 / 58 px cells        -> beans touching
  pairs       demo grid, every second column pushed against its left neighbour     -> 50 touching pairs
  clump30     70 beans on the demo grid + 30 beans pushed into a clump (52 px)      -> one touching patch
  cell64_03 / cell64_30      cell64 with the ~3% and ~30% defect mixes

HONESTY: these beans are in the deployed model's training set (like the demo trays). They measure the bean
finder and the rules, not classifier accuracy.

usage: .venv/bin/python app/scripts/make_touching_trays.py   -> app/tests/touching/*.jpg + truth.json
"""
import csv, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/beans"))
import numpy as np
from PIL import Image
from common import CLASSES, CROPS_CSV
from make_demo import bean_rgba, BEAN_PX, MARGIN, GRID
from beanfinder import SHEET_TARGET

OUT = ROOT / "app/tests/touching"
SPEC = {
    "03": {"dark": 1, "insect": 1, "broken": 1, "unhulled": 0},
    "12": {"dark": 4, "insect": 3, "broken": 3, "unhulled": 2},
    "30": {"dark": 10, "insect": 7, "broken": 8, "unhulled": 5},
}


def positions(layout, rng):
    """100 bean centres for a layout (grid order) and the canvas side."""
    pts = []
    if layout.startswith("cell") or layout.startswith("spread"):
        cell = int(layout.split("_")[0].replace("cell", "").replace("spread", "") or 108) if not layout.startswith("spread") else 108
        side = 2 * MARGIN + GRID * cell
        jit = 10 if cell >= 100 else max(2, cell // 12)
        for i in range(100):
            gy, gx = divmod(i, GRID)
            pts.append((MARGIN + gx * cell + cell // 2 + rng.randint(-jit, jit), MARGIN + gy * cell + cell // 2 + rng.randint(-jit, jit)))
        return pts, side
    side = 2 * MARGIN + GRID * 108
    if layout == "pairs":
        for i in range(100):
            gy, gx = divmod(i, GRID)
            cx = MARGIN + gx * 108 + 54 + (-60 if gx % 2 else 0)  # odd columns pushed left: centre gap 48 px
            pts.append((cx + rng.randint(-3, 3), MARGIN + gy * 108 + 54 + rng.randint(-3, 3)))
        return pts, side
    if layout == "clump30":
        for i in range(70):
            gy, gx = divmod(i, GRID)
            pts.append((MARGIN + gx * 108 + 54 + rng.randint(-10, 10), MARGIN + gy * 108 + 54 + rng.randint(-10, 10)))
        x0, y0 = MARGIN + 60, MARGIN + 7 * 108 + 20
        for i in range(30):
            gy, gx = divmod(i, 10)
            pts.append((x0 + gx * 52 + rng.randint(-3, 3), y0 + gy * 52 + rng.randint(-3, 3)))
        return pts, side
    raise ValueError(layout)


def make(pool, spec, layout, seed):
    rng = random.Random(seed)
    good_n = 100 - sum(spec.values())
    chosen = [("good", r) for r in rng.sample(pool["good"], good_n)]
    for c, k in spec.items(): chosen += [(c, r) for r in rng.sample(pool[c], k)]
    rng.shuffle(chosen)
    pts, side = positions(layout, rng)
    nprng = np.random.default_rng(seed)
    canvas = Image.fromarray(np.clip(SHEET_TARGET + nprng.normal(0, 1.2, (side, side, 3)), 0, 255).astype(np.uint8), "RGB")
    placed = []
    for (c, r), (cx, cy) in zip(chosen, pts):
        b = bean_rgba(r["img"])
        s = BEAN_PX * rng.uniform(0.88, 1.12) / max(b.size)
        b = b.resize((max(8, round(b.width * s)), max(8, round(b.height * s))), Image.BICUBIC)
        b = b.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)
        canvas.paste(b, (cx - b.width // 2, cy - b.height // 2), b)
        placed.append({"cls": c, "cx": cx, "cy": cy, "crop": r["path"]})
    return canvas, placed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(CROPS_CSV)) if r["touching"] == "0"]
    for r in rows: r["img"] = np.asarray(Image.open(ROOT / r["path"]).convert("RGB"))
    pool = {c: [r for r in rows if r["cls"] == c] for c in CLASSES}
    trays = [("spread_108", "12", 11), ("cell72", "12", 12), ("cell64", "12", 13), ("cell58", "12", 14),
             ("pairs", "12", 15), ("clump30", "12", 16), ("cell64", "03", 17), ("cell64", "30", 18)]
    truth = []
    for layout, mix, seed in trays:
        name = f"touch_{layout}_{mix}.jpg"
        img, placed = make(pool, SPEC[mix], layout, seed)
        img.save(OUT / name, quality=90)
        counts = {c: sum(p["cls"] == c for p in placed) for c in CLASSES}
        truth.append({"file": name, "layout": layout, "mix": mix, "seed": seed, "size": img.size[0], "true_counts": counts,
                      "true_defects": 100 - counts["good"], "beans": [{k: p[k] for k in ("cls", "cx", "cy")} for p in placed]})
        print(name, img.size, counts)
    json.dump({"what": "SYNTHETIC touching-bean trays (app/scripts/make_touching_trays.py); beans pasted, classes known",
               "trays": truth}, open(OUT / "truth.json", "w"), indent=0)


if __name__ == "__main__":
    main()
