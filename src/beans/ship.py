"""Step 9a: copy the deployed model + labels into the app and record what shipped (sha256, size, batch latency).
Run after train_beans.py, before make_fixtures.py / make_demo.py / eval_ood_cbd.py."""
import hashlib, json, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
from common import ROOT, CROPS_CSV
import csv, pipeline

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    model_path, labels = pipeline.deployed()
    dst = ROOT / "app/public/models"; dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, dst / "farz_beans.onnx")
    shutil.copyfile(ROOT / "models/farz_beans_labels.json", dst / "farz_beans_labels.json")
    assert sha(model_path) == sha(dst / "farz_beans.onnx")
    rows = list(csv.DictReader(open(CROPS_CSV)))[:100]
    x = np.stack([np.asarray(Image.open(ROOT / r["path"]).convert("RGB")) for r in rows]).astype(np.float32).transpose(0, 3, 1, 2) / 255
    lat = {}
    for fmt in ("fp32", "int8"):
        s = pipeline.session(ROOT / f"models/farz_beans_{fmt}.onnx", threads=1)
        for _ in range(3): s.run(None, {"image": x})
        ts = []
        for _ in range(15):
            t0 = time.perf_counter(); s.run(None, {"image": x}); ts.append((time.perf_counter() - t0) * 1000)
        lat[fmt] = round(float(np.median(ts)), 1)
    rep_path = ROOT / "reports/beans_model.json"; rep = json.load(open(rep_path))
    rep["shipped"] = {"app_model": "app/public/models/farz_beans.onnx", "from": str(model_path.relative_to(ROOT)), "sha256": sha(model_path),
                      "bytes": model_path.stat().st_size, "labels": labels,
                      "median_cpu_ms_batch100_1thread": lat,
                      "latency_machine": "MacBook M5 Pro CPU, onnxruntime 1.30.0 Python CPUExecutionProvider, 1 thread (NOT a phone, NOT wasm)"}
    json.dump(rep, open(rep_path, "w"), indent=1)
    print(json.dumps(rep["shipped"], indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
