"""End-to-end Farz bean pipeline in Python (the reference the browser must match):
photo -> beanfinder.find_beans (frozen, no ML) -> per-bean crops -> deployed ONNX classifier -> abstain rule.

The deployed model and threshold are read from reports/beans_model.json + models/farz_beans_labels.json, so every
downstream script (CBD OOD check, fixtures, demo trays) uses exactly the file that ships to the app.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from beanfinder import find_beans
from common import ROOT, CLASSES

def deployed():
    rep = json.load(open(ROOT / "reports/beans_model.json"))
    labels = json.load(open(ROOT / "models/farz_beans_labels.json"))
    return ROOT / rep["final"]["deployed_file"], labels

def session(path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

def classify(sess, crops_u8):
    """crops_u8: list of 128x128x3 uint8 -> probs [N,5] (input 'image' float32 NCHW RGB in [0,1])."""
    if not len(crops_u8): return np.zeros((0, len(CLASSES)), np.float32)
    x = np.stack(crops_u8).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return np.concatenate([sess.run(["probs"], {"image": x[b:b + 64]})[0] for b in range(0, len(x), 64)])

def run_photo(sess, img, abstain_below):
    r = find_beans(img)
    p = classify(sess, [b.crop for b in r.beans])
    conf = p.max(1) if len(p) else np.zeros(0)
    pred = [CLASSES[i] if c >= abstain_below else "abstain" for i, c in zip(p.argmax(1), conf)]
    return r, p, pred

# ---------------------------------------------------------------------------------------------------------------
# Result rules: Python MIRROR of app/src/lib/rules.ts decide() + app/src/lib/wilson.ts (fixed rules, not AI).
# Keep in sync with the app. Used by app/scripts/measure_rules.py and src/beans/make_demo_extra.py.
MAX_UNSURE_FRAC = 0.15          # > 15% of beans unsure -> "not sure" (c07), model uncertainty
IMPLAUSIBLE_DEFECT_FRAC = 0.6   # > 60% of ANSWERED beans defective -> "not sure" (c07), implausible / unfamiliar sample
CLEAN_MAX, MANY_MIN = 0.05, 0.2 # our rule of thumb, not an official grade: clean <5%, some 5-20%, many >20%
MAX_HANDFULS = 3

def wilson(k, n, z=1.96):
    if n <= 0 or k < 0 or k > n: return float("nan"), 0.0, 1.0
    p = k / n; z2 = z * z; den = 1 + z2 / n
    c = (p + z2 / (2 * n)) / den; h = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)

def band_of_rate(p):
    if p != p: return "unsure"
    return "clean" if p < CLEAN_MAX else "some" if p <= MANY_MIN else "many"

def app_calls(result, pred):
    """The app's per-bean call: touching blobs are never trusted; 'abstain' is shown as 'unsure'."""
    return ["unsure" if b.touching or q == "abstain" else q for b, q in zip(result.beans, pred)]

def decide(calls, handfuls=1):
    """Mirror of rules.ts decide(). calls: class names or 'unsure'. Returns band/about/reason + counts + interval."""
    counts = {k: 0 for k in CLASSES + ["unsure"]}
    for c in calls: counts["unsure" if c == "abstain" else c] += 1
    total = len(calls); defects = sum(counts[k] for k in CLASSES[1:]); answered = total - counts["unsure"]
    p, lo, hi = wilson(defects, answered)
    unsure_frac = counts["unsure"] / total if total else 1.0
    about = False
    if answered == 0: band, reason = "unsure", "too_many_unsure"
    elif defects / answered > IMPLAUSIBLE_DEFECT_FRAC: band, reason = "unsure", "implausible"
    elif unsure_frac > MAX_UNSURE_FRAC: band, reason = "unsure", "too_many_unsure"
    else:
        band, reason = band_of_rate(p), ""
        about = band_of_rate(lo) != band or band_of_rate(hi) != band
    return {"band": band, "about": about, "reason": reason, "counts": counts, "total": total, "answered": answered,
            "defects": defects, "p": p, "lo": lo, "hi": hi, "unsure_frac": unsure_frac,
            "can_add_handful": band != "unsure" and about and handfuls < MAX_HANDFULS}
