"""Step 1: cut every J4ckDev photo into per-bean crops with the FROZEN bean finder.

Writes data/crops/<class>/<photo>_<idx>.png and data/crops/crops.csv with the source photo,
bean box and centre (work-image px and normalised), and the fold each bean is held out in:
  fold 1 tests on beans whose centre x is in the RIGHT half of their photo (cx_norm >= 0.5)
  fold 2 tests on beans whose centre x is in the LEFT half  (cx_norm <  0.5)
"""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from PIL import Image
from beanfinder import find_beans, PAD
from common import PHOTO_CLASS, J4CK_DIR, CROPS_DIR, CROPS_CSV, ROOT

def main():
    rows = []
    for stem, cls in sorted(PHOTO_CLASS.items()):
        r = find_beans(Image.open(J4CK_DIR / f"{stem}.jpg"))
        h, w = r.work.shape[:2]
        (CROPS_DIR / cls).mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(r.beans):
            cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
            side = max(b.x1 - b.x0, b.y1 - b.y0)
            crop_src_px = 2 * (side // 2 + int(round(PAD * side)))  # source px covered by the 128 px crop
            p = CROPS_DIR / cls / f"{stem}_{i:03d}.png"
            Image.fromarray(b.crop).save(p)
            rows.append({"path": str(p.relative_to(ROOT)), "cls": cls, "photo": stem, "idx": i,
                         "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1, "cx": cx, "cy": cy,
                         "work_w": w, "work_h": h, "cx_norm": round(cx / w, 4), "cy_norm": round(cy / h, 4),
                         "area": b.area, "touching": int(b.touching), "bean_side_px": side, "crop_src_px": crop_src_px,
                         "test_fold": 1 if cx / w >= 0.5 else 2})
        print(f"{stem:16s} -> {cls:9s} {len(r.beans):4d} beans (touching {r.checks['touching']})")
    with open(CROPS_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("wrote", len(rows), "crops ->", CROPS_CSV)

if __name__ == "__main__":
    main()
