"""Counting test on CBD (Mendeley 52877z55vr, CC BY 4.0): 464 light-box photos.
Reports the count distribution per grade; we do NOT assume 50 — we report what we measure."""
import sys, glob, json, collections
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from beanfinder import find_beans
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
def run(f):
    r = find_beans(Image.open(f)); return f, r.checks["count"], r.checks["touching"], r.checks["blur_var"]
if __name__ == "__main__":
    files = sorted(glob.glob(str(ROOT / "data/raw/cbd/**/*.jpg"), recursive=True))
    with ProcessPoolExecutor() as ex: res = list(ex.map(run, files, chunksize=8))
    by = collections.defaultdict(list)
    for f, n, t, b in res: by[Path(f).parent.name].append(n)
    allc = [n for _, n, _, _ in res]
    out = {"n_images": len(res), "count_hist": dict(collections.Counter(allc).most_common()),
           "exactly_50": sum(c == 50 for c in allc), "within_2_of_50": sum(abs(c - 50) <= 2 for c in allc),
           "mean_abs_err_vs_50": sum(abs(c - 50) for c in allc) / len(allc),
           "per_grade": {g: {"n": len(v), "mean": sum(v)/len(v), "min": min(v), "max": max(v)} for g, v in sorted(by.items())}}
    (ROOT / "reports").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "reports/count_cbd.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["n_images","exactly_50","within_2_of_50","mean_abs_err_vs_50"]}))
    print("most common counts:", list(out["count_hist"].items())[:10])
    for g, s in out["per_grade"].items(): print(f"  {g:6s} {s}")
