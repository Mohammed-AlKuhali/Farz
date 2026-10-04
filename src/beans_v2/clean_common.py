"""Shared helpers for the licence-stated ("clean") v2 model: clean_eval.py, clean_export.py, clean_simulate.py, localcal.py.

The clean model = train_common recipe trained only on train_folds.LICENSED (CACHE/folds/licensed.pt). Fold models are
CACHE/folds/clean_*.pt (clean_folds.py). The embedding used for on-device calibration is the PENULTIMATE feature of
MobileNetV3-Small: the 1,024-d Hardswish output that feeds the last Linear layer (eval mode, so dropout is off),
L2-normalised. The ONNX file exports it as output "embed"; embed_and_logits() below computes the same thing in PyTorch.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import train_common as tc
from train_folds import LICENSED

UNLICENSED = ["mfu17", "usk_coffee", "daffa_defect", "mindforge_doubleside", "notplying_defects"]
GROUP_OF = {s: ("afiyah" if s.startswith("afiyah") else s) for s in tc.SOURCES}  # the afiyah twins are ONE capture set-up
FOLDS = tc.CACHE / "folds"
ROBUST = tc.SCRATCH / "robust_cache"
AUDIT = Path("/tmp/audit_v2")
FINAL_PT = FOLDS / "licensed.pt"
ONNX32 = tc.MODELS_V2 / "farz_beans_v2_clean_fp32.onnx"
ONNX16 = tc.MODELS_V2 / "farz_beans_v2_clean_fp16.onnx"
LABELS = tc.MODELS_V2 / "farz_beans_v2_clean_labels.json"
EVAL_JSON = tc.REPORTS_V2 / "clean_eval.json"


def load_torch(name_or_path, dev=None):
    import torch
    p = Path(name_or_path)
    if not p.suffix: p = FOLDS / f"{name_or_path}.pt"
    net = tc.Norm(tc.make_net(pretrained=False)); net.load_state_dict(torch.load(p, map_location="cpu")); net.eval()
    return net.to(dev or tc.device())


def penultimate(model, x):
    """model = train_common.Norm(MobileNetV3-Small). x float [N,3,128,128] in [0,1] -> (logits [N,6], h [N,1024] raw)."""
    net = model.net
    z = net.features((x - model.mean) / model.std)
    z = net.avgpool(z).flatten(1)
    h = net.classifier[2](net.classifier[1](net.classifier[0](z)))   # Linear(576,1024) -> Hardswish -> Dropout(eval: identity)
    return net.classifier[3](h), h


def embed_and_logits(model, crops_u8, bs=512):
    """crops uint8 [N,128,128,3] -> (logits [N,6] float32, embed [N,1024] float32 L2-normalised)."""
    import torch
    dev = next(model.parameters()).device; L, E = [], []
    with torch.no_grad():
        for b in range(0, len(crops_u8), bs):
            x = tc.to_float(np.asarray(crops_u8[b:b + bs]), dev)
            lg, h = penultimate(model, x)
            E.append(torch.nn.functional.normalize(h, dim=1).float().cpu().numpy()); L.append(lg.float().cpu().numpy())
    if not L: return np.zeros((0, 6), np.float32), np.zeros((0, 1024), np.float32)
    return np.concatenate(L), np.concatenate(E)


def ort_session(path, threads=0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads: so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_run(sess, crops_u8, bs=256):
    """-> (probs [N,6], embed [N,D]) from the two-output clean ONNX file."""
    P, E = [], []
    for b in range(0, len(crops_u8), bs):
        x = np.ascontiguousarray(np.asarray(crops_u8[b:b + bs])).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        p, e = sess.run(["probs", "embed"], {"image": x}); P.append(p); E.append(e)
    if not P: return np.zeros((0, 6), np.float32), np.zeros((0, 1024), np.float32)
    return np.concatenate(P), np.concatenate(E)


def calls3(p, tg, td, touching=None):
    """good / defect / unsure codes 0 / 1 / 2 (two-threshold rule); touching -> unsure."""
    pg = p[:, 0]
    c = np.where(pg >= tg, 0, np.where(1 - pg >= td, 1, 2))
    if touching is not None: c = np.where(np.asarray(touching, bool), 2, c)
    return c


def decide_codes(codes):
    """codes 0 good / 1 defect / 2 unsure -> app band, via src/beans/pipeline.decide (the Python mirror of
    app/src/lib/rules.ts decide()). A defect is passed as 'dark' (only the defect TOTAL matters for the band)."""
    import pipeline  # src/beans (on sys.path via train_common)
    names = np.array(["good", "dark", "unsure"])[np.asarray(codes, int)]
    r = pipeline.decide(list(names))
    if r["band"] == "unsure": return "notsure:" + r["reason"], r
    return r["band"], r


def app_gate(checks, cg=None):
    """Python port of app/src/lib/rules.ts gate() ORDER, on the frozen Python bean finder's checks (+ colour gate when
    available). The app's newer touching.ts rules are NOT ported (the Python finder's own touching flag is used), so this is
    an approximation; every safety number is also reported with all gates ignored."""
    from beanfinder import MIN_BEANS, MAX_BEANS
    few = checks["count"] < MIN_BEANS
    ng = bool(cg and cg.get("not_green"))
    if checks["too_dark"]: return "dark"
    if few and not checks["too_many_touching"] and not ng: return "count"
    if checks["too_blurry"]: return "blur"
    if ng:
        fr = cg.get("frac", {})
        bean_col = fr.get("off", 0) + fr.get("white", 0) <= 0.5 and fr.get("dark", 0) <= 0.6
        return "spread" if (cg.get("reason") == "big_blob" and bean_col and checks["too_many_touching"]) else "not_green"
    if checks["too_many_touching"]: return "spread"
    if few or checks["count"] > MAX_BEANS: return "count"
    return ""


def kind(o):
    return "retake" if o.startswith("retake") else ("notsure" if o.startswith("notsure") else "band")


def tally(outs):
    c = {}
    for o in outs: c[o] = c.get(o, 0) + 1
    k = {"band": 0, "notsure": 0, "retake": 0}
    for o in outs: k[kind(o)] += 1
    n = len(outs)
    return {"n": n, "given_band": k["band"], "not_sure": k["notsure"], "retake": k["retake"], "outcomes": dict(sorted(c.items())),
            "pct_band": round(100 * k["band"] / n, 1) if n else None}


def tray_score(truths, outs):
    """truths/outs lists -> correct / wrong / dangerous counts (dangerous = clean -> many or many -> clean)."""
    bands = [(t, o) for t, o in zip(truths, outs) if kind(o) == "band"]
    return {"trays": len(outs), "bands": len(bands), "correct": sum(t == o for t, o in bands), "wrong": sum(t != o for t, o in bands),
            "dangerous_clean_called_many": sum(t == "clean" and o == "many" for t, o in bands),
            "dangerous_many_called_clean": sum(t == "many" and o == "clean" for t, o in bands),
            "by_truth": {tb: dict(sorted({o: sum(1 for t, q in zip(truths, outs) if t == tb and q == o) for o in set(outs)}.items()))
                         for tb in sorted(set(truths))}}


def deployed():
    ev = json.load(open(EVAL_JSON))
    d = ev["deployed"]
    return d["temperature"], d["pair_thresholds"]["t_good"], d["pair_thresholds"]["t_defect"], ev


def cbd_rows():
    """robust-cache CBD bean rows: (crop indices into ROBUST/crops.npy, photo index per bean, photo meta list, touching)."""
    pmeta = json.load(open(ROBUST / "photos.json"))["photos"]; z = np.load(ROBUST / "beans.npz")
    pi = z["photo_idx"]; src = np.array([pmeta[i]["src"] for i in pi])
    b = np.flatnonzero(src == "cbd")
    return b, pi[b], pmeta, z["touching"][b]


def jdump(obj, path):
    def conv(o):
        if hasattr(o, "item"): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    json.dump(obj, open(path, "w"), indent=1, default=conv)
