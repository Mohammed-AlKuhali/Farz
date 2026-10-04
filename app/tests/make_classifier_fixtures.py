"""Classifier parity fixtures for the SHIPPED v2 model (app/public/models/farz_beans.onnx), made by Python onnxruntime.

Writes app/tests/fixtures/classifier_v2/:
  crops/*.png         20 J4ckDev crops (copied from tests/fixtures/crops, the v1 parity set, lossless)
  photos/*.png        3 whole photos already at the bean finder's work size (copied from tests/fixtures/photos)
  expected.json       per crop: probs [6], embed (base64 float32 [1024]), two-threshold call; also the 20 calibration
                      crops in app/public/calib-demo/calibration (loja_yolo, CC BY 4.0); per photo: the frozen Python
                      bean finder's boxes + touching flags, probs, calls and call counts
The app test (tests/classifier.parity.test.ts) re-runs the same inputs through onnxruntime-web (wasm) and the app's
own TypeScript bean finder. Run from app/:  ../.venv/bin/python tests/make_classifier_fixtures.py
"""
import base64, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "src/beans"))
from beanfinder import find_beans  # noqa: E402  (frozen reference, imported read-only)

OUT = APP / "tests/fixtures/classifier_v2"
MODEL = APP / "public/models/farz_beans.onnx"
labels = json.load(open(APP / "public/models/farz_beans_labels.json"))
C = labels["classes"]; TG, TD = labels["good_at_or_above"], labels["defect_at_or_above"]


def call(p, touching=False):
    if touching: return "unsure"
    if p[0] >= TG: return "good"
    if 1 - float(p[0]) >= TD: return C[1 + int(np.argmax(p[1:]))]
    return "unsure"


def main():
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
    run = lambda x: s.run(["probs", "embed"], {"image": x})
    (OUT / "crops").mkdir(parents=True, exist_ok=True); (OUT / "photos").mkdir(parents=True, exist_ok=True)
    items = []
    for f in sorted((ROOT / "tests/fixtures/crops").glob("*.png")):
        shutil.copyfile(f, OUT / "crops" / f.name); items.append(("crops/" + f.name, "fixtures"))
    for f in sorted((APP / "public/calib-demo/calibration").glob("*.png")):
        items.append(("public/calib-demo/calibration/" + f.name, "app"))
    def path(rel, where): return (OUT / rel) if where == "fixtures" else (APP / rel)
    X = np.stack([np.asarray(Image.open(path(r, w)).convert("RGB")) for r, w in items]).astype(np.float32).transpose(0, 3, 1, 2) / 255
    P, E = run(X)
    crops = [{"file": r, "base": w, "probs": [round(float(v), 7) for v in p], "call": call(p),
              "embed_b64": base64.b64encode(e.astype(np.float32).tobytes()).decode()} for (r, w), p, e in zip(items, P, E)]
    photos = []
    for f in sorted((ROOT / "tests/fixtures/photos").glob("*.png")):
        shutil.copyfile(f, OUT / "photos" / f.name)
        r = find_beans(Image.open(OUT / "photos" / f.name))
        x = np.stack([b.crop for b in r.beans]).astype(np.float32).transpose(0, 3, 1, 2) / 255
        p = np.concatenate([run(x[i:i + 64])[0] for i in range(0, len(x), 64)])
        calls = [call(pp, b.touching) for b, pp in zip(r.beans, p)]
        photos.append({"file": "photos/" + f.name, "bean_count": len(r.beans),
                       "beans": [{"box": [int(b.x0), int(b.y0), int(b.x1), int(b.y1)], "touching": bool(b.touching),
                                  "probs": [round(float(v), 7) for v in pp], "call": c} for b, pp, c in zip(r.beans, p, calls)],
                       "call_counts": {k: calls.count(k) for k in C + ["unsure"]}})
        print(f.name, len(r.beans), photos[-1]["call_counts"])
    out = {"model": "app/public/models/farz_beans.onnx", "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
           "classes": C, "thresholds": {"good": TG, "defect": TD},
           "runtime": f"onnxruntime {ort.__version__} Python, CPUExecutionProvider, 1 thread",
           "crops": crops, "photos": photos}
    json.dump(out, open(OUT / "expected.json", "w"))
    print("crops", len(crops), "photos", len(photos), "calls", {k: sum(c["call"] == k for c in crops) for k in C + ["unsure"]})


if __name__ == "__main__":
    main()
