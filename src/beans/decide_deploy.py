"""Step 4b: final deploy decision, adding the BROWSER parity gate that train_beans.py cannot check on its own.

Rule (train_beans.py): INT8 only if its mean held-out accuracy is within 1.5 points of fp32.
Added rule (measured 3 Oct 2026): INT8 must ALSO reproduce Python onnxruntime in onnxruntime-web 1.30.0 (WASM, the
runtime the app uses) within 0.01 max abs probability (PLAN gate (b)). Otherwise every Python-measured number
(threshold, coverage, OOD table) would not describe what the phone actually computes. If INT8 fails, ship fp32 and
re-choose the abstain threshold on the fp32 fold predictions.
Runs node src/beans/check_wasm_parity.mjs on 60 training crops (12 per class) for each final model.
"""
import csv, json, random, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
from common import ROOT, CLASSES, CROPS_CSV
import pipeline
from train_beans import coverage_curve, pick_threshold, report

TOL = 0.01

def wasm_parity(model):
    rows = list(csv.DictReader(open(CROPS_CSV))); rng = random.Random(11)
    pick = [r for c in CLASSES for r in rng.sample([r for r in rows if r["cls"] == c], 12)]
    imgs = [np.asarray(Image.open(ROOT / r["path"]).convert("RGB")) for r in pick]
    p = pipeline.classify(pipeline.session(model), imgs)
    fx = {"model": str(model), "classes": CLASSES, "suggested_tolerance_abs": TOL,
          "items": [{"file": str(Path("../..") / r["path"]), "probs": [float(v) for v in pp], "argmax": CLASSES[int(pp.argmax())]} for r, pp in zip(pick, p)]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump(fx, f)
    out = subprocess.run(["node", str(ROOT / "src/beans/check_wasm_parity.mjs"), f.name, str(model)], capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])

def main():
    rep_path = ROOT / "reports/beans_model.json"; rep = json.load(open(rep_path))
    par = {fmt: wasm_parity(ROOT / f"models/farz_beans_{fmt}.onnx") for fmt in ("fp32", "int8")}
    for fmt, r in par.items(): print(fmt, r)
    acc_gap = rep["deploy_decision"]["fp32_minus_int8_points"]
    int8_ok = acc_gap <= 1.5 and par["int8"]["max_abs_prob_diff_vs_python"] <= TOL and par["int8"]["argmax_agreement"] == 1.0
    deploy = "int8" if int8_ok else "fp32"
    rep["deploy_decision"].update({"python_rule_alone_would_deploy": rep["deploy_decision"]["deployed"], "deployed": deploy,
                                   "browser_parity_rule": f"INT8 also needs onnxruntime-web wasm vs Python max abs prob diff <= {TOL} (60 crops)",
                                   "browser_parity": par})
    d = np.load(ROOT / "reports/beans_heldout_preds.npz"); y = d["y"]; fold = d["fold"]; P = d[deploy]
    curve = coverage_curve(P, y); chosen = pick_threshold(curve)
    rep["selective"].update({"on": f"pooled held-out halves, fold models, {deploy} ONNX, calibrated", "chosen": chosen, "curve": curve,
                             "per_fold_at_chosen": {str(fd): {"coverage": float(((fold == fd) & (P.max(1) >= chosen["threshold"])).sum() / (fold == fd).sum()),
                                                              "accuracy_answered": float((P[(fold == fd) & (P.max(1) >= chosen["threshold"])].argmax(1) == y[(fold == fd) & (P.max(1) >= chosen["threshold"])]).mean())}
                                                    for fd in (1, 2)}})
    rep["final"]["deployed_file"] = f"models/farz_beans_{deploy}.onnx"
    t_final = rep["final"]["temperature"]; thr = round(chosen["threshold"], 2)
    labels = {"classes": CLASSES, "abstain_below": thr, "input": 128,
              "note": (f"Farz per-bean classifier, {deploy} ONNX. input 'image' float32 [N,3,128,128] RGB in [0,1] (crop contract of src/beans/beanfinder.py); "
                       f"output 'probs' [N,5] = softmax(logits/T), T={t_final:.3f} baked in. Abstain (yellow '?') when max prob < abstain_below. "
                       f"Threshold chosen on within-photo spatially held-out J4ckDev beans for >=90% answered accuracy - an UPPER BOUND, not field accuracy; "
                       f"on CBD photos from India the model calls most beans 'broken' (reports/beans_model.json cbd_ood). "
                       f"Trained on J4ckDev Green Coffee Beans (CC BY-NC-SA 4.0); the model inherits NC-SA, prototype only.")}
    json.dump(labels, open(ROOT / "models/farz_beans_labels.json", "w"), indent=1, ensure_ascii=False)
    rep["labels_json"] = labels
    json.dump(rep, open(rep_path, "w"), indent=1)
    print("deploy", deploy, "threshold", thr, "coverage", round(chosen["coverage"], 4), "answered acc", round(chosen["accuracy_answered"], 4))

if __name__ == "__main__":
    main()
