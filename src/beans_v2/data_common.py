"""Farz v2 data hunter: shared helpers for turning external green-bean datasets into crops that follow the
FROZEN crop contract of src/beans/beanfinder.py (which is imported, never modified).

Ways an external image becomes a 128 px crop (also: crop_box for annotated boxes in dense piles, crop_mask_chroma
for close-ups on a black plate; see their docstrings):
  tray      -> beanfinder.find_beans(photo) exactly as the app does; every bean gets the photo's label
               (photos that hold a single class) or the label of the annotation polygon it overlaps (IoU>=0.5).
  single    -> single-bean photo: pad the photo on every side with its own border colour (median of the same 4%
               border band the bean finder uses) so the bean cannot touch the frame, then find_beans(); keep the
               largest bean. Same white balance, padding, square canvas and resize as the app.
  precrop   -> already-tight crop with no sheet visible (vicanadya16): no white balance is possible; the crop is
               pasted centred on a SHEET_TARGET-grey square with the contract's PAD and resized to CROP (bilinear).
"""
import hashlib, os, sys
import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "beans"))
import beanfinder as bf  # FROZEN reference, imported read-only

FARZ_CLASSES = ["good", "dark", "insect", "broken", "unhulled", "other_defect"]
SINGLE_PAD_FRAC = 0.30


def load_rgb(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def pixel_md5(im):
    a = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h = hashlib.md5(); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def dhash64(arr):
    """64-bit difference hash of an RGB uint8 array (for near-duplicate REPORTING only)."""
    g = Image.fromarray(arr).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def border_median(im):
    a = np.asarray(im, dtype=np.float32); h, w, _ = a.shape
    b = max(2, int(0.04 * min(h, w)))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    return np.median(border, axis=0)


def crops_tray(im):
    """Run the frozen bean finder on a whole photo. Returns (result, boxes_in_original_px)."""
    r = bf.find_beans(im)
    s = r.scale if r.scale > 0 else 1.0
    boxes = [(b.x0 / s, b.y0 / s, b.x1 / s, b.y1 / s) for b in r.beans]
    return r, boxes


def locate_window(im, grow=1.0, by="flat"):
    """Large single-bean photo: find the bean with a flat-field difference (cv2, LOCATING ONLY), and return a
    window around it (bean bbox grown by `grow` x its larger side on every side). The crop itself is then made
    by the frozen bean finder on the original pixels of that window."""
    import cv2
    a = np.asarray(im, dtype=np.uint8); h, w, _ = a.shape
    s = 800.0 / max(h, w); small = cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA).astype(np.float32)
    if by == "chroma":   # beans are yellow/green/brown; paper shading and shadows are grey
        ch = small.max(-1) - small.min(-1)
        d = cv2.GaussianBlur(np.clip(ch - np.median(ch), 0, None), (0, 0), 1.5) * 3.0
    else:
        bg = cv2.GaussianBlur(small, (0, 0), sigmaX=0.06 * max(small.shape[:2]))
        d = np.abs(small - bg).sum(-1)
    m = (d > max(30.0, np.percentile(d, 99.0) * 0.5)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return None
    H, Wd = m.shape
    inner = [k for k in range(1, n) if stats[k, 0] > 1 and stats[k, 1] > 1 and
             stats[k, 0] + stats[k, 2] < Wd - 1 and stats[k, 1] + stats[k, 3] < H - 1]
    cand = inner if inner else list(range(1, n))   # prefer blobs that do not touch the frame (tape, table edge)
    i = max(cand, key=lambda k: stats[k, cv2.CC_STAT_AREA])
    x, y, bw, bh = stats[i, :4] / s
    side = max(bw, bh); g = grow * side
    return (int(max(0, x - g)), int(max(0, y - g)), int(min(w, x + bw + g)), int(min(h, y + bh + g))), side


def crop_single(im, window_above=700):
    """Single-bean photo -> (crop uint8 128x128x3 or None, info dict).
    Large photos: locate the bean (cv2, locating only), cut a window around it, run the frozen finder on the window.
    QC: the finder's bean must agree in size with the located blob (0.4x-2x); otherwise retry with a tighter
    window, then give up (flag 'finder_mismatch')."""
    if max(im.size) > window_above:
        for by in ("chroma", "flat"):
            for grow in (1.0, 0.35):
                loc = locate_window(im, grow, by)
                if loc is None: break
                win, L = loc
                crop, info = _crop_single_core(im.crop(win), (win[0], win[1]))
                if crop is None: continue
                bx = info["bbox"]; W0, H0 = im.size
                cut = [(bx[0] <= win[0] + 1 and win[0] > 0), (bx[1] <= win[1] + 1 and win[1] > 0),
                       (bx[2] >= win[2] - 1 and win[2] < W0), (bx[3] >= win[3] - 1 and win[3] < H0)]
                if any(cut): continue        # bean cut by the window, not by the photo: try the next window
                B = max(bx[2] - bx[0], bx[3] - bx[1])
                if 0.4 * L <= B <= 2.0 * L:
                    extra = [x for x in (("tight_window" if grow != 1.0 else ""), ("flat_locator" if by == "flat" else "")) if x]
                    info["flags"] = ";".join([f for f in [info["flags"]] + extra if f])
                    return crop, info
        return None, {"n_found": 0, "flags": "finder_mismatch"}
    return _crop_single_core(im, (0, 0))


def _crop_single_core(im, off):
    w, h = im.size
    p = int(round(SINGLE_PAD_FRAC * max(w, h)))
    col = tuple(int(round(c)) for c in border_median(im))
    canvas = Image.new("RGB", (w + 2 * p, h + 2 * p), col)
    canvas.paste(im, (p, p))
    r = bf.find_beans(canvas)
    if not r.beans:
        return None, {"n_found": 0, "flags": "no_bean"}
    areas = sorted((b.area for b in r.beans), reverse=True)
    best = max(r.beans, key=lambda b: b.area)
    flags = []
    if len(areas) > 1 and areas[1] >= 0.5 * areas[0]:
        flags.append("second_object")
    if r.checks.get("too_dark"): flags.append("too_dark")
    if r.checks.get("too_blurry"): flags.append("too_blurry")
    s = r.scale
    bbox = ((best.x0 / s) - p + off[0], (best.y0 / s) - p + off[1], (best.x1 / s) - p + off[0], (best.y1 / s) - p + off[1])
    return best.crop, {"n_found": len(r.beans), "flags": ";".join(flags), "bbox": bbox,
                       "bean_px_work": int(max(best.x1 - best.x0, best.y1 - best.y0))}


def crop_precrop(im):
    """Tight pre-cut bean (no sheet visible): contract geometry only, no white balance."""
    a = np.asarray(im, dtype=np.uint8); h, w, _ = a.shape
    side = max(w, h); pad = int(round(bf.PAD * side)); half = side // 2 + pad
    canvas = np.full((2 * half, 2 * half, 3), bf.SHEET_TARGET, np.float32)
    y0 = half - h // 2; x0 = half - w // 2
    canvas[y0:y0 + h, x0:x0 + w] = a
    crop = np.asarray(Image.fromarray(canvas.astype(np.uint8)).resize((bf.CROP, bf.CROP), Image.BILINEAR))
    return crop, {"n_found": 1, "flags": "precrop_no_wb", "bbox": (0, 0, w, h), "bean_px_work": side}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def crop_mask_chroma(im):
    """Close-up single bean on a mixed black-plate / white-jig background (Mindforge). The frozen finder cannot
    segment this (it assumes a plain light sheet), so: white balance from the bright low-chroma jig pixels (maps to
    SHEET_TARGET like the contract), bean mask = Otsu on chroma, largest blob, holes filled; the masked bean is
    composited onto a SHEET_TARGET square with the contract's PAD and resized to CROP. Flagged 'mask_composite'."""
    import cv2
    a = np.asarray(im, dtype=np.float32); h, w, _ = a.shape
    lum = a.mean(-1); ch = a.max(-1) - a.min(-1)
    ref = a[(lum > np.percentile(lum, 80)) & (ch < np.percentile(ch, 50))]
    white = np.median(ref, axis=0) if len(ref) > 100 else np.median(a.reshape(-1, 3), axis=0)
    wb = np.clip(a * (bf.SHEET_TARGET / np.maximum(white, 1.0)), 0, 255)
    chroma = wb.max(-1) - wb.min(-1)
    sm = cv2.GaussianBlur(chroma, (0, 0), 2)
    thr = max(bf._otsu(sm), 12.0)
    m = (sm > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None, {"n_found": 0, "flags": "no_bean"}
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, bw, bh, ar = [int(v) for v in st[j]]
    mf = (lab == j).astype(np.uint8)
    cnts, _ = cv2.findContours(mf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mf = np.zeros_like(mf); cv2.drawContours(mf, cnts, -1, 1, -1)
    flags = ["mask_composite"]
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea)); ha = float(cv2.contourArea(hull)); solidity = float(mf.sum()) / ha if ha > 0 else 0.0
    if solidity < 0.93:
        return None, {"n_found": 1, "flags": f"low_solidity={solidity:.2f}"}
    extent = float(mf.sum()) / float(max(1, bw * bh))   # ellipse ~0.79; plate labels / PCB patches are rectangles ~1.0
    if extent > 0.88:
        return None, {"n_found": 1, "flags": f"rectangular_mask_extent={extent:.2f}"}
    if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
        # touching the frame: either the bean is cut off, or (dark beans, low chroma) the mask grabbed a brown
        # patch of the jig at the frame edge instead of the bean. Both are rejected.
        return None, {"n_found": 1, "flags": "cut_by_frame"}
    if ar < 0.03 * h * w: flags.append("tiny_mask")
    side = max(bw, bh); pad = int(round(bf.PAD * side)); half = side // 2 + pad; cx, cy = x + bw // 2, y + bh // 2
    canvas = np.full((2 * half, 2 * half, 3), bf.SHEET_TARGET, np.float32)
    sx0, sy0, sx1, sy1 = max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)
    mm = cv2.GaussianBlur(mf[sy0:sy1, sx0:sx1].astype(np.float32), (3, 3), 0)[..., None]
    sl = (slice(sy0 - (cy - half), sy1 - (cy - half)), slice(sx0 - (cx - half), sx1 - (cx - half)))
    canvas[sl] = wb[sy0:sy1, sx0:sx1] * mm + canvas[sl] * (1 - mm)
    crop = np.asarray(Image.fromarray(canvas.astype(np.uint8)).resize((bf.CROP, bf.CROP), Image.BILINEAR))
    return crop, {"n_found": 1, "flags": ";".join(flags), "bbox": (x, y, x + bw, y + bh), "bean_px_work": side}


def sheet_wb_factor(im):
    """Contract white balance factor (sheet = median of the 4% border band, mapped to SHEET_TARGET)."""
    return bf.SHEET_TARGET / np.maximum(border_median(im), 1.0)


def crop_box(im, box, wbf):
    """Annotated box in a dense pile (Notplying): white balance per the contract, square crop of the box padded by
    PAD, at ORIGINAL resolution (neighbouring beans stay visible; outside the photo = SHEET_TARGET), resized to CROP."""
    a = np.asarray(im, dtype=np.float32); h, w, _ = a.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    side = max(x1 - x0, y1 - y0); pad = int(round(bf.PAD * side)); half = side // 2 + pad
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    canvas = np.full((2 * half, 2 * half, 3), bf.SHEET_TARGET, np.float32)
    sx0, sy0, sx1, sy1 = max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)
    canvas[sy0 - (cy - half):sy1 - (cy - half), sx0 - (cx - half):sx1 - (cx - half)] = np.clip(a[sy0:sy1, sx0:sx1] * wbf, 0, 255)
    crop = np.asarray(Image.fromarray(canvas.astype(np.uint8)).resize((bf.CROP, bf.CROP), Image.BILINEAR))
    return crop, {"n_found": 1, "flags": "box_crop_neighbours_visible", "bbox": (x0, y0, x1, y1), "bean_px_work": side}
