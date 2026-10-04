"""Shared helpers for the v2 ROBUST-SIGNALS study (colour baseline, unusual-bean detector, OOD gate).

Nothing here modifies v1 files. The frozen bean finder (src/beans/beanfinder.py) is imported read-only; the app's
decision rules (app/src/lib/rules.ts, wilson.ts, colourgate.ts) are PORTED here line for line so the v1 verdict
("clean" / "some" / "many" / "unsure") can be reproduced in Python for every CBD and J4ckDev photo.

Per-crop features work on the bean finder's 128 px crop (white-balanced: sheet -> 235, padding filled with 235).
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/beans"))
from beanfinder import find_beans, SHEET_TARGET, CROP, MIN_BEANS, MAX_BEANS, _erode, _dilate  # noqa: E402  (frozen, read-only)
from common import CLASSES, PHOTO_CLASS, J4CK_DIR, CBD_DIR, CROPS_CSV  # noqa: E402

REPORTS_V2 = ROOT / "reports_v2"
import os, tempfile
# working cache (1.2 GB of crops) lives OUTSIDE the repo; set FARZ_SCRATCH to choose where
SCRATCH = Path(os.environ.get("FARZ_SCRATCH", str(Path(tempfile.gettempdir()) / "farz_robust")))
CACHE = SCRATCH / "robust_cache"
V1_MODEL = ROOT / "models/farz_beans_fp32.onnx"
V1_LABELS = ROOT / "models/farz_beans_labels.json"
CBD_GRADES = ["AAA", "AA", "A", "AB", "PB-I", "PB-II", "C", "Bulk", "Bits"]

def rel(p):
    """Repo-relative path string (never write the home directory into a report)."""
    try: return str(Path(p).resolve().relative_to(ROOT))
    except ValueError: return str(p)

def pixel_md5(arr):
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()

# ------------------------------------------------------------------------------------------------ app rules (port)
# Ported from app/src/lib/rules.ts + wilson.ts AS OF 23:05 BST Sat 3 Oct (band from the POINT estimate, "about" flag when
# the 95% interval crosses a band edge, and the new 'implausible' fail-safe: > 60% defects among answered -> not sure).
# The 22:40 version (band only if the whole interval sits in one band, no implausible rule) is kept as decide_v0.
RULES_VERSION = "app rules.ts/wilson.ts 23:05 BST 3 Oct 2026"
MAX_UNSURE_FRAC = 0.15
IMPLAUSIBLE_DEFECT_FRAC = 0.6
Z95, CLEAN_MAX, MANY_MIN = 1.96, 0.05, 0.20
DEFECTS = ["dark", "insect", "broken", "unhulled"]

def wilson(k, n, z=Z95):
    if n <= 0 or k < 0 or k > n: return dict(k=k, n=n, p=float("nan"), lo=0.0, hi=1.0)
    p = k / n; z2 = z * z; denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return dict(k=k, n=n, p=p, lo=max(0.0, centre - half), hi=min(1.0, centre + half))

def band_of_rate(p):
    if not np.isfinite(p): return "unsure"
    if p < CLEAN_MAX: return "clean"
    if p <= MANY_MIN: return "some"
    return "many"

def band_of(iv):  # whole-interval band (the 22:40 rule)
    a, b = band_of_rate(iv["lo"]), band_of_rate(iv["hi"])
    return a if a == b else "unsure"

def crosses_band_edge(iv):
    b = band_of_rate(iv["p"])
    return b == "unsure" or band_of_rate(iv["lo"]) != b or band_of_rate(iv["hi"]) != b

def call_bean(probs, touching, abstain_below):
    if touching or probs is None: return "unsure"
    b = int(np.argmax(probs))
    return CLASSES[b] if probs[b] >= abstain_below else "unsure"

def _counts(calls):
    counts = {c: 0 for c in CLASSES + ["unsure"]}
    for c in calls: counts[c] += 1
    return counts

def decide(calls):
    counts = _counts(calls); total = len(calls); defects = sum(counts[d] for d in DEFECTS); answered = total - counts["unsure"]
    iv = wilson(defects, answered); unsure_frac = counts["unsure"] / total if total else 1.0; about = False
    if answered == 0: band, why = "unsure", "too_many_unsure"
    elif defects / answered > IMPLAUSIBLE_DEFECT_FRAC: band, why = "unsure", "implausible"
    elif unsure_frac > MAX_UNSURE_FRAC: band, why = "unsure", "too_many_unsure"
    else: band, why = band_of_rate(iv["p"]), ""; about = crosses_band_edge(iv)
    return dict(band=band, why=why, about=about, counts=counts, total=total, lo=iv["lo"], hi=iv["hi"], p=iv["p"], unsure_frac=unsure_frac)

def decide_v0(calls):
    counts = _counts(calls); total = len(calls); defects = sum(counts[d] for d in DEFECTS); answered = total - counts["unsure"]
    iv = wilson(defects, answered); unsure_frac = counts["unsure"] / total if total else 1.0
    if unsure_frac > MAX_UNSURE_FRAC or answered == 0: band, why = "unsure", "too_many_unsure"
    else:
        band = band_of(iv); why = "interval_spans_bands" if band == "unsure" else ""
    return dict(band=band, why=why, counts=counts, total=total, lo=iv["lo"], hi=iv["hi"], p=iv["p"], unsure_frac=unsure_frac)

COLOUR_RULE = dict(darkV=0.3, whiteS=0.15, whiteV=0.6, hueMin=26, hueMax=62, satMax=0.66, maxOffFrac=0.5, maxDarkFrac=0.6, maxBlobFrac=0.08)

def _hsv(r, g, b):
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return h * 360, s, v

def classify_colour(r, g, b):
    h, s, v = _hsv(r, g, b); R = COLOUR_RULE
    if v < R["darkV"]: return "dark"
    if s < R["whiteS"] and v >= R["whiteV"]: return "white"
    if h < R["hueMin"] or h > R["hueMax"] or s > R["satMax"]: return "off"
    return "pale"

def colour_gate(res):
    """Port of app/src/lib/colourgate.ts on the bean finder's white-balanced work image. Mask/components are
    recomputed from res.work with the finder's own threshold and 3x3 morphology; components via scipy (4-connected),
    which gives the same boxes/areas as the finder's flood fill."""
    from scipy import ndimage
    wb = res.work.astype(np.float32)
    luma = 0.299 * wb[..., 0] + 0.587 * wb[..., 1] + 0.114 * wb[..., 2]
    chroma = wb.max(-1) - wb.min(-1)
    mask = np.maximum(SHEET_TARGET - luma, chroma * 1.5) > res.checks["threshold"]
    mask = _dilate(_erode(mask)); mask = _erode(_dilate(mask))
    lab, n = ndimage.label(mask, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    h, w = mask.shape
    tot = dict(pale=0, dark=0, white=0, off=0); largest = 0
    min_area = max(30, 0.25 * res.median_area)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        area = int((lab[sl] == i).sum())
        if area < min_area: continue
        m = mask[sl]  # JS sums every mask pixel in the box (not only this component) - kept identical
        px = res.work[sl][m]
        if not len(px): continue
        tot[classify_colour(*px.mean(0))] += area; largest = max(largest, area)
    A = sum(tot.values()); R = COLOUR_RULE
    frac = {k: (v / A if A else 0.0) for k, v in tot.items()}
    reason = ""
    if A > 0:
        if largest / (w * h) >= R["maxBlobFrac"]: reason = "big_blob"
        elif frac["off"] + frac["white"] > R["maxOffFrac"]: reason = "off_colour"
        elif frac["dark"] > R["maxDarkFrac"]: reason = "too_dark_beans"
    return dict(not_green=reason != "", reason=reason, frac=frac, largest_blob_frac=largest / (w * h))

def gate_reason(checks, cg):
    if checks["too_dark"]: return "dark"
    if checks["too_blurry"]: return "blur"
    if cg["not_green"]: return "not_green"
    if checks["count"] < MIN_BEANS or checks["count"] > MAX_BEANS: return "count"
    if checks["too_many_touching"]: return "spread"
    return ""

def v1_verdict(probs, touching, checks, cg, abstain_below, rules="current"):
    """Full v1 app outcome for one photo: retake reason or band. rules='current' (23:05) or 'v0' (22:40)."""
    g = gate_reason(checks, cg)
    if g: return dict(band="retake:" + g)
    calls = [call_bean(p, t, abstain_below) for p, t in zip(probs, touching)]
    if rules != "current": return decide_v0(calls)
    out = decide(calls); out["band_v0"] = decide_v0(calls)["band"]
    return out

# ------------------------------------------------------------------------------------------------ ONNX v1 with extras
def v1_session(path=V1_MODEL, threads=1):
    """v1 deployed model with two extra graph outputs exposed IN MEMORY (file on disk untouched):
    feat576 = pooled backbone features, logits = raw logits before /T."""
    import onnx, onnxruntime as ort
    m = onnx.load(str(path))
    for name in ["/m/net/Flatten_output_0", "/m/net/classifier/classifier.3/Gemm_output_0"]:
        m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

def v1_run(sess, crops_u8, bs=64):
    """-> probs [N,5] (T baked in), feat [N,576], logits [N,5] (raw, before /T)."""
    if not len(crops_u8): return np.zeros((0, 5), np.float32), np.zeros((0, 576), np.float32), np.zeros((0, 5), np.float32)
    x = np.stack(crops_u8).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    outs = [sess.run(None, {"image": x[b:b + bs]}) for b in range(0, len(x), bs)]
    return tuple(np.concatenate([o[k] for o in outs]) for k in range(3))

# ------------------------------------------------------------------------------------------------ crop features
def bean_mask(crop, thr=25.0):
    """Bean pixels in a white-balanced 128 px crop: same 'different from the sheet' rule as the finder
    (max of darkness and 1.5*chroma), fixed threshold; keep the component nearest the centre; fill holes."""
    from scipy import ndimage
    c = crop.astype(np.float32)
    luma = 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]
    chroma = c.max(-1) - c.min(-1)
    m = np.maximum(SHEET_TARGET - luma, 1.5 * chroma) > thr
    lab, n = ndimage.label(m)
    if n == 0: return m
    h, w = m.shape
    best, bestd = 0, 1e9
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    coms = ndimage.center_of_mass(m, lab, range(1, n + 1))
    for i, (s, (cy, cx)) in enumerate(zip(sizes, coms), start=1):
        if s < 30: continue
        d = np.hypot(cy - h / 2, cx - w / 2) - 0.02 * s  # prefer central, then large
        if d < bestd: best, bestd = i, d
    if best == 0: best = int(np.argmax(sizes)) + 1
    return ndimage.binary_fill_holes(lab == best)

HAND_NAMES = ["luma_med", "luma_p10", "luma_p90", "luma_std", "dark_frac", "sat_mean", "hue_deg", "val_mean",
              "lab_a", "lab_b", "chroma_mean", "fill", "elong", "solidity", "circ", "edge_luma_drop"]

def hand_features(crop, mask=None):
    """16 colour/shape stats of one bean (mask pixels only). Scale-free shape stats: the absolute bean size in the
    photo is NOT in here (it is added per photo as area / photo-median area)."""
    import cv2
    if mask is None: mask = bean_mask(crop)
    px = crop[mask].astype(np.float32)
    if len(px) < 20: return np.full(len(HAND_NAMES), np.nan, np.float32)
    luma = 0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2]
    mx, mn = px.max(1), px.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    mr, mg, mb = px.mean(0)
    h, s, v = _hsv(mr, mg, mb)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[mask].astype(np.float32)
    ys, xs = np.nonzero(mask)
    cov = np.cov(np.stack([xs, ys]).astype(np.float64)) if len(xs) > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))
    elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    area = float(mask.sum()); hull = cv2.contourArea(cv2.convexHull(cnt)); per = cv2.arcLength(cnt, True)
    # inner vs rim luminance (a pale rim / dark centre pattern; crude 'texture' that survives low resolution)
    er = cv2.erode(m8, np.ones((5, 5), np.uint8), iterations=2).astype(bool)
    inner = crop[er].astype(np.float32) if er.sum() > 10 else px
    il = (0.299 * inner[:, 0] + 0.587 * inner[:, 1] + 0.114 * inner[:, 2]).mean()
    return np.array([np.median(luma), np.percentile(luma, 10), np.percentile(luma, 90), luma.std(),
                     (luma < 80).mean(), sat.mean(), h, v,
                     lab[:, 1].mean() - 128, lab[:, 2].mean() - 128, (mx - mn).mean(),
                     area / mask.size, elong, area / max(hull, 1.0), per * per / (4 * np.pi * area),
                     luma.mean() - il], np.float32)

# ------------------------------------------------------------------------------------------------ exposure normalisation
def normalise_crop(crop, res=40, ring=6):
    """Per-crop exposure normalisation used to fight photo-identity leakage in synthetic trays:
    (1) local white balance: the crop's own background ring (pixels near the sheet colour) is mapped to 235 per channel;
    (2) common effective resolution: down to `res` px and back to 128 (bilinear), so sharp close-ups and small, soft
        beans end up with the same detail level. The bean's colour relative to its sheet is kept (dark stays dark)."""
    from PIL import Image
    c = crop.astype(np.float32)
    rim = np.concatenate([c[:ring].reshape(-1, 3), c[-ring:].reshape(-1, 3), c[:, :ring].reshape(-1, 3), c[:, -ring:].reshape(-1, 3)])
    l = 0.299 * rim[:, 0] + 0.587 * rim[:, 1] + 0.114 * rim[:, 2]
    bg = rim[l >= np.percentile(l, 50)]
    ref = np.median(bg, axis=0) if len(bg) else np.full(3, SHEET_TARGET)
    c = np.clip(c * (SHEET_TARGET / np.maximum(ref, 1.0)), 0, 255).astype(np.uint8)
    im = Image.fromarray(c).resize((res, res), Image.BILINEAR, reducing_gap=None).resize((CROP, CROP), Image.BILINEAR)
    return np.asarray(im)

# ------------------------------------------------------------------------------------------------ frozen ImageNet backbone
def imagenet_extractor():
    """torchvision MobileNetV3-Small IMAGENET1K_V1, frozen: features -> global average pool -> 576-d.
    Input RGB [0,1] NCHW 128x128, ImageNet mean/std inside the module (same input contract as v1)."""
    import torch, torch.nn as nn
    from torchvision import models
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
    class Feat(nn.Module):
        def __init__(s):
            super().__init__(); s.f = net.features; s.pool = nn.AdaptiveAvgPool2d(1)
            s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        def forward(s, image): return s.pool(s.f((image - s.mean) / s.std)).flatten(1)
    return Feat().eval()

def embed(model, crops_u8, bs=256, device="cpu"):
    import torch
    out = []
    with torch.no_grad():
        for b in range(0, len(crops_u8), bs):
            x = torch.from_numpy(np.stack(crops_u8[b:b + bs]).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 576), np.float32)

def dump(obj, name):
    REPORTS_V2.mkdir(exist_ok=True)
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        raise TypeError(type(o))
    json.dump(obj, open(REPORTS_V2 / name, "w"), indent=1, default=conv)
