"""Python reference counts for the demo trays with the SHIPPED v2 model (app/public/models/farz_beans.onnx).

Pillow decode + EXIF orientation -> frozen src/beans/beanfinder.py (imported read-only) -> onnxruntime (CPU) ->
two-threshold rule from app/public/models/farz_beans_labels.json (good if P(good) >= t_good; defect if 1 - P(good) >=
t_defect, type = argmax of the defect classes; otherwise abstain; touching -> abstain). Writes `farz_python_counts`
into app/public/demo/manifest.json. The app's extra touching rule (touching.ts) is TS-only; on these trays it flags
nothing (tests/touching.test.ts). The e2e test compares the browser's per-class counts with these.
Run from app/:  ../.venv/bin/python scripts/demo_python_counts.py
"""
import json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "src/beans"))
from beanfinder import find_beans  # noqa: E402

labels = json.load(open(APP / "public/models/farz_beans_labels.json"))
C = labels["classes"]; tg, td = labels["good_at_or_above"], labels["defect_at_or_above"]
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(str(APP / "public/models/farz_beans.onnx"), so, providers=["CPUExecutionProvider"])
man_p = APP / "public/demo/manifest.json"
man = json.load(open(man_p))
for tray in man:
    img = ImageOps.exif_transpose(Image.open(APP / "public/demo" / tray["file"])).convert("RGB")
    r = find_beans(img)
    x = np.stack([b.crop for b in r.beans]).astype(np.float32).transpose(0, 3, 1, 2) / 255
    p = np.concatenate([sess.run(["probs"], {"image": x[i:i + 64]})[0] for i in range(0, len(x), 64)])
    counts = {k: 0 for k in C + ["abstain"]}
    for b, pp in zip(r.beans, p):
        if b.touching: counts["abstain"] += 1
        elif pp[0] >= tg: counts["good"] += 1
        elif 1 - float(pp[0]) >= td: counts[C[1 + int(np.argmax(pp[1:]))]] += 1
        else: counts["abstain"] += 1
    tray["farz_python_counts"] = counts
    print(tray["file"], len(r.beans), counts)
json.dump(man, open(man_p, "w"), indent=1, ensure_ascii=False)
