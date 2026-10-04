"""Step 8: parity fixtures for the browser (onnxruntime-web must reproduce these from the SAME files).

tests/fixtures/crops/*.png        20 bean crops (4 per class), lossless, exactly the 128x128 crops the model was trained on
tests/fixtures/expected_probs.json  probs from onnxruntime (Python, CPU) on the deployed model app/public/models/farz_beans.onnx
tests/fixtures/photos/*.png       3 whole photos (2 J4ckDev, 1 CBD) already downscaled by PIL to the bean finder's work size,
                                   saved lossless so JPEG-decoder and resize differences cannot hide a bean-finder bug
tests/fixtures/pipeline_expected.json  per photo: bean count, checks, every bean's box, probs, argmax, abstain decision
"""
import csv, hashlib, json, random, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
import onnxruntime as ort
from beanfinder import find_beans, WORK_MAX
from common import ROOT, CLASSES, CROPS_CSV, J4CK_DIR, CBD_DIR
import pipeline

FIX = ROOT / "tests/fixtures"
APP_MODEL = ROOT / "app/public/models/farz_beans.onnx"
PHOTOS = [("j4ck_brocadoleve", J4CK_DIR / "BrocadoLeve.jpg"), ("j4ck_pergamino", J4CK_DIR / "Pergamino.jpg"),
          ("cbd_aaa_1", CBD_DIR / "AAA/1.jpg")]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    model_path, labels = pipeline.deployed(); thr = labels["abstain_below"]
    assert sha(model_path) == sha(APP_MODEL), "app/public/models/farz_beans.onnx is not the deployed model - copy it first"
    s = pipeline.session(APP_MODEL, threads=1)
    (FIX / "crops").mkdir(parents=True, exist_ok=True); (FIX / "photos").mkdir(parents=True, exist_ok=True)
    for old in list((FIX / "crops").glob("*.png")) + list((FIX / "photos").glob("*.png")): old.unlink()
    rows = list(csv.DictReader(open(CROPS_CSV))); rng = random.Random(7)
    items = []
    for c in CLASSES:
        pick = rng.sample([r for r in rows if r["cls"] == c], 4)
        for r in pick:
            name = f"crop_{len(items):02d}_{c}_{Path(r['path']).stem}.png"
            shutil.copyfile(ROOT / r["path"], FIX / "crops" / name)
            img = np.asarray(Image.open(FIX / "crops" / name).convert("RGB"))
            p = pipeline.classify(s, [img])[0]
            items.append({"file": f"crops/{name}", "true_class": c, "source_crop": r["path"], "test_fold": int(r["test_fold"]),
                          "probs": [round(float(v), 7) for v in p], "argmax": CLASSES[int(p.argmax())],
                          "max_prob": round(float(p.max()), 7), "answered": bool(p.max() >= thr)})
    # how much does onnxruntime itself move between graph-optimisation levels / thread counts? (sets a fair tolerance)
    X = np.stack([np.asarray(Image.open(FIX / it["file"]).convert("RGB")) for it in items]).astype(np.float32).transpose(0, 3, 1, 2) / 255
    ref = np.array([it["probs"] for it in items])
    spread = {}
    for name, lvl, th in (("no_graph_opt_1thread", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1), ("all_opt_8threads", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 8)):
        so = ort.SessionOptions(); so.graph_optimization_level = lvl; so.intra_op_num_threads = th
        p = ort.InferenceSession(str(APP_MODEL), so, providers=["CPUExecutionProvider"]).run(["probs"], {"image": X})[0]
        spread[name] = {"max_abs_diff_vs_expected": float(np.abs(p - ref).max()), "argmax_agreement": float((p.argmax(1) == ref.argmax(1)).mean())}
    exp = {"model": "app/public/models/farz_beans.onnx", "model_sha256": sha(APP_MODEL), "classes": CLASSES, "abstain_below": thr,
           "input": "name 'image', float32 [N,3,128,128], RGB, value = png_byte / 255 (no other normalisation; ImageNet mean/std is inside the graph)",
           "output": "name 'probs', float32 [N,5], softmax(logits/T) with T baked in",
           "runtime": f"onnxruntime {ort.__version__} Python, CPUExecutionProvider, intra_op_num_threads=1, default graph optimisation",
           "suggested_tolerance_abs": 0.01, "python_runtime_spread": spread, "items": items}
    json.dump(exp, open(FIX / "expected_probs.json", "w"), indent=1)
    print("crops:", len(items), "argmax==true:", sum(i["argmax"] == i["true_class"] for i in items), "spread:", spread)

    out = {"model": exp["model"], "model_sha256": exp["model_sha256"], "classes": CLASSES, "abstain_below": thr,
           "how_to_use": ("Feed photos/<name>.png to the JS bean finder + model. These PNGs are the PIL-BILINEAR downscale of the source "
                          "photo to <= %d px (the bean finder's own first step), so the bean finder's scale is 1.0 and results must match "
                          "box-for-box. 'from_source_jpeg' shows the same pipeline run on the original JPEG in Python; it must be identical." % WORK_MAX),
           "photos": []}
    for name, src in PHOTOS:
        im = Image.open(src).convert("RGB"); sc = min(1.0, WORK_MAX / max(im.size))
        work_in = im.resize((round(im.width * sc), round(im.height * sc)), Image.BILINEAR) if sc < 1 else im
        png = FIX / "photos" / f"{name}.png"; work_in.save(png)
        r, p, pred = pipeline.run_photo(s, Image.open(png), thr)
        r2, p2, pred2 = pipeline.run_photo(s, Image.open(src), thr)
        same = len(r.beans) == len(r2.beans) and all((a.x0, a.y0, a.x1, a.y1) == (b.x0, b.y0, b.x1, b.y1) for a, b in zip(r.beans, r2.beans)) \
            and (len(p) == 0 or float(np.abs(p - p2).max()) == 0.0)
        beans = [{"box": [int(b.x0), int(b.y0), int(b.x1), int(b.y1)], "area": int(b.area), "touching": bool(b.touching),
                  "probs": [round(float(v), 7) for v in pp], "argmax": CLASSES[int(pp.argmax())], "decision": d}
                 for b, pp, d in zip(r.beans, p, pred)]
        out["photos"].append({"name": name, "fixture_png": f"photos/{name}.png", "source": str(Path(src).relative_to(ROOT)),
                              "source_size": list(Image.open(src).size), "work_size": list(work_in.size),
                              "bean_count": len(r.beans), "checks": r.checks, "median_area": r.median_area,
                              "decision_counts": {k: pred.count(k) for k in CLASSES + ["abstain"]},
                              "from_source_jpeg_identical": bool(same), "beans": beans})
        print(name, "beans", len(r.beans), "identical from jpeg:", same, {k: pred.count(k) for k in CLASSES + ["abstain"]})
    json.dump(out, open(FIX / "pipeline_expected.json", "w"), indent=1)
    # the same crops through onnxruntime-web (WASM) in Node: the app's own runtime
    import subprocess
    r = subprocess.run(["node", str(ROOT / "src/beans/check_wasm_parity.mjs")], capture_output=True, text=True)
    wasm = json.loads(r.stdout.strip().splitlines()[-1]); print("wasm parity:", wasm)
    exp["browser_check_node_wasm"] = wasm; json.dump(exp, open(FIX / "expected_probs.json", "w"), indent=1)
    rep_path = ROOT / "reports/beans_model.json"; rep = json.load(open(rep_path))
    rep["fixtures"] = {"crops": len(items), "photos": [{k: p[k] for k in ("name", "source", "bean_count", "decision_counts", "from_source_jpeg_identical")} for p in out["photos"]],
                       "python_runtime_spread": spread, "wasm_parity": wasm}
    json.dump(rep, open(rep_path, "w"), indent=1)

if __name__ == "__main__":
    main()
