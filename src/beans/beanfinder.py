"""Farz bean finder — deterministic image processing, NO machine learning.

This file is the reference implementation. The phone app runs a line-for-line JavaScript port
(app/src/lib/beanfinder.ts) and parity is tested against this file. Keep every step portable:
downscale, white-point from the sheet, threshold (Otsu), 3x3 morphology, connected components.

Crop contract (shared with the classifier and the app):
  - work image = photo downscaled so the longest side <= WORK_MAX px
  - colours are normalised so the sheet's median colour maps to SHEET_TARGET (white balance from the cloth)
  - each bean crop = its bounding box, padded by PAD of the larger side, made square, filled with SHEET_TARGET,
    resized to CROP px (bilinear). Model input = RGB float in [0,1], NCHW.
"""
from dataclasses import dataclass, field
import numpy as np
from PIL import Image

WORK_MAX = 1200        # longest side of the working image
SHEET_TARGET = 235.0   # sheet colour after white balance (per channel)
PAD = 0.18             # crop padding, fraction of the bean's larger side
CROP = 128             # classifier input size
MIN_BEANS, MAX_BEANS = 20, 160

@dataclass
class Bean:
    x0: int; y0: int; x1: int; y1: int; area: int
    touching: bool = False
    crop: np.ndarray = field(default=None, repr=False)  # CROPxCROPx3 uint8

@dataclass
class Result:
    beans: list
    work: np.ndarray          # white-balanced working image (uint8 HxWx3)
    scale: float              # work px per original px
    checks: dict              # quality-gate outcomes
    median_area: float

def _otsu(values: np.ndarray) -> float:
    hist = np.bincount(np.clip(values, 0, 255).astype(np.int64).ravel(), minlength=256).astype(np.float64)
    total = hist.sum(); sum_all = (np.arange(256) * hist).sum()
    w_b = 0.0; sum_b = 0.0; best, thr = -1.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best: best, thr = between, t
    return float(thr)

def _erode(m):  # 3x3 erosion, border treated as background
    p = np.pad(m, 1, constant_values=False)
    out = np.ones_like(m, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out

def _dilate(m):
    p = np.pad(m, 1, constant_values=False)
    out = np.zeros_like(m, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out

def _components(mask):
    """4-connected labelling, iterative flood fill (same as the JS port)."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    comps = []
    flat = mask.ravel(); lab = labels.ravel()
    nxt = 1
    for start in np.flatnonzero(flat):
        if lab[start]: continue
        stack = [start]; lab[start] = nxt
        xs0 = w; ys0 = h; xs1 = ys1 = -1; area = 0
        while stack:
            i = stack.pop(); y, x = divmod(i, w); area += 1
            if x < xs0: xs0 = x
            if x > xs1: xs1 = x
            if y < ys0: ys0 = y
            if y > ys1: ys1 = y
            for j in (i - 1 if x > 0 else -1, i + 1 if x < w - 1 else -1, i - w if y > 0 else -1, i + w if y < h - 1 else -1):
                if j >= 0 and flat[j] and not lab[j]:
                    lab[j] = nxt; stack.append(j)
        comps.append((xs0, ys0, xs1 + 1, ys1 + 1, area)); nxt += 1
    return comps

def _laplacian_var(gray):
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())

def find_beans(img: Image.Image) -> Result:
    img = img.convert("RGB")
    s = min(1.0, WORK_MAX / max(img.size))
    if s < 1.0:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    a = np.asarray(img).astype(np.float32)
    h, w, _ = a.shape
    # 1) sheet colour = median of a 4% border band (beans are expected in the middle)
    b = max(2, int(0.04 * min(h, w)))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    sheet = np.median(border, axis=0)
    # 2) white balance from the sheet
    wb = np.clip(a * (SHEET_TARGET / np.maximum(sheet, 1.0)), 0, 255)
    # 3) "how different from the sheet": max of darkness and chroma
    luma = 0.299 * wb[..., 0] + 0.587 * wb[..., 1] + 0.114 * wb[..., 2]
    chroma = wb.max(-1) - wb.min(-1)
    diff = np.maximum(SHEET_TARGET - luma, chroma * 1.5)
    thr = max(_otsu(diff), 18.0)
    mask = diff > thr
    # 4) clean: open (remove specks) then close (fill cracks)
    mask = _dilate(_erode(mask)); mask = _erode(_dilate(mask))
    comps = _components(mask)
    areas = np.array([c[4] for c in comps]) if comps else np.array([0])
    big = areas[areas >= max(30, 0.15 * np.percentile(areas, 90))] if comps else areas
    med = float(np.median(big)) if big.size else 0.0
    beans = []
    for (x0, y0, x1, y1, ar) in comps:
        if med == 0 or ar < 0.25 * med: continue          # specks / dust
        if x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h: continue  # cut by the frame edge
        bean = Bean(x0, y0, x1, y1, int(ar), touching=ar > 1.8 * med)
        side = max(x1 - x0, y1 - y0); pad = int(round(PAD * side)); half = side // 2 + pad
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        canvas = np.full((2 * half, 2 * half, 3), SHEET_TARGET, np.float32)
        sx0, sy0, sx1, sy1 = max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)
        canvas[sy0 - (cy - half):sy1 - (cy - half), sx0 - (cx - half):sx1 - (cx - half)] = wb[sy0:sy1, sx0:sx1]
        bean.crop = np.asarray(Image.fromarray(canvas.astype(np.uint8)).resize((CROP, CROP), Image.BILINEAR))
        beans.append(bean)
    gray = luma.astype(np.float32)
    n = len(beans)
    checks = {
        "sheet_luma": float(0.299 * sheet[0] + 0.587 * sheet[1] + 0.114 * sheet[2]),
        "too_dark": bool((0.299 * sheet[0] + 0.587 * sheet[1] + 0.114 * sheet[2]) < 90),
        "blur_var": _laplacian_var(gray),
        "too_blurry": _laplacian_var(gray) < 15.0,
        "count": n,
        "count_ok": MIN_BEANS <= n <= MAX_BEANS,
        "touching": int(sum(b.touching for b in beans)),
        "too_many_touching": sum(b.touching for b in beans) > max(2, 0.08 * max(n, 1)),
        "threshold": thr,
    }
    return Result(beans, wb.astype(np.uint8), s, checks, med)
