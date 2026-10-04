"""Parity fixtures for app/src/lib/beanfinder.ts.

Runs the FROZEN Python reference (src/beans/beanfinder.py) on 11 J4ckDev + 10 CBD photos and writes
app/tests/fixtures/beanfinder.json with, per photo, two runs:
  "pillow": input decoded by Pillow (what the Python pipeline really sees). The JS test decodes the same
            JPEG with jpeg-js (a different decoder), so here we expect equal counts and bboxes within 3 px.
  "jpegjs": input = the exact RGB pixels jpeg-js produces (decoded by tests/decode_jpeg.mjs and handed to
            Python). Same pixels on both sides, so the port must match EXACTLY (bboxes, areas, threshold,
            sheet colour, median area and an md5 of every 128 px crop).
Run from app/:  ../.venv/bin/python tests/make_fixtures.py
"""
import hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "src/beans"))
from beanfinder import find_beans  # noqa: E402  (frozen reference, not modified)

J4CK = ROOT / "data/raw/green/ImageDataset"
CBD = ROOT / "data/raw/cbd/CBD_Coffee Bean Dataset/CBD_Coffee Bean Dataset"


def pick_files():
    j4 = sorted(J4CK.glob("*.jpg"))
    cbd = []
    for grade in sorted(p for p in CBD.iterdir() if p.is_dir()):
        cbd.append(sorted(grade.glob("*.jpg"))[0])
    cbd.append(sorted((CBD / "Bits").glob("*.jpg"))[1])
    return j4, cbd


def summarise(r):
    beans = []
    for b in r.beans:
        crop = np.ascontiguousarray(b.crop, dtype=np.uint8)
        beans.append({"bbox": [int(b.x0), int(b.y0), int(b.x1), int(b.y1)], "area": int(b.area), "touching": bool(b.touching),
                      "crop_md5": hashlib.md5(crop.tobytes()).hexdigest(), "crop_sum": int(crop.sum())})
    ch = {k: (float(v) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else int(v)))
          for k, v in r.checks.items()}
    return {"work_size": [int(r.work.shape[1]), int(r.work.shape[0])], "scale": float(r.scale), "checks": ch,
            "median_area": float(r.median_area), "work_md5": hashlib.md5(np.ascontiguousarray(r.work).tobytes()).hexdigest(),
            "beans": beans}


def main():
    j4, cbd = pick_files()
    files = j4 + cbd
    out = {"generated_by": "app/tests/make_fixtures.py", "reference": "src/beans/beanfinder.py",
           "pillow_version": Image.__version__, "numpy_version": np.__version__, "photos": []}
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["node", str(APP / "tests/decode_jpeg.mjs"), td, *map(str, files)], check=True, cwd=APP)
        for i, f in enumerate(files):
            meta = json.loads(Path(td, f"{i}.json").read_text())
            raw = Path(td, f"{i}.rgb").read_bytes()
            img_js = Image.frombytes("RGB", (meta["width"], meta["height"]), raw)
            r_pil = find_beans(Image.open(f))
            r_js = find_beans(img_js)
            rel = str(f.relative_to(ROOT))
            out["photos"].append({"file": rel, "set": "J4ckDev" if f in j4 else "CBD",
                                  "pillow": summarise(r_pil), "jpegjs": summarise(r_js)})
            print(f"{rel}: pillow count={r_pil.checks['count']} jpegjs count={r_js.checks['count']}", flush=True)
    dst = APP / "tests/fixtures/beanfinder.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out))
    print("wrote", dst, dst.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
