"""Shared code for the Farz v2 per-bean classifier (train_*.py).

Data: every kept crop of data/crops_v2/crops_v2.csv (keep == 1) + the 582 v1 J4ckDev crops (data/crops/crops.csv,
source "j4ckdev", group = photo). All crops already follow the crop contract of src/beans/beanfinder.py (frozen,
read-only): 128 px RGB on the 235-grey sheet.

Label kinds (7) -> 6 output classes:
  good, dark, insect, broken, unhulled, other_defect  : typed labels, cross-entropy on that class
  defect_untyped                                       : "some defect, type not given" (USK defect, lojano defectuoso,
                                                         afiyah rusak, Samruddh grade D, daffa Defected) -> partial-label
                                                         loss  -log(sum of the 5 defect-class probabilities)
Typed "other" defects (floater, withered, fade, foreign matter, Mindforge fissure / over-dried) are other_defect.

Sampling (per batch, with replacement): label kind by fixed weights KIND_W -> source uniformly among the training
sources that have that kind -> crop uniformly. So no source can dominate an epoch however many crops it has
(vicanadya16 has 86k), and good vs defect is drawn 40 / 60.

The working cache (all crops as one uint8 array) lives OUTSIDE the repo in $FARZ_SCRATCH/v2train.
"""
import csv, json, os, sys, time, random, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/beans"))
from beanfinder import SHEET_TARGET, CROP  # noqa: E402  frozen, read-only

SCRATCH = Path(os.environ.get("FARZ_SCRATCH", str(Path(tempfile.gettempdir()) / "farz_robust")))
CACHE = SCRATCH / "v2train"
MODELS_V2 = ROOT / "models_v2"
REPORTS_V2 = ROOT / "reports_v2"
LOGF = CACHE / "train_v2.log"  # outside the repo (logs/ is not owned by this step)

CLASSES = ["good", "dark", "insect", "broken", "unhulled", "other_defect"]
KINDS = CLASSES + ["defect_untyped"]
DEFECT_IDX = [1, 2, 3, 4, 5]
KIND_W = {"good": 0.40, "dark": 0.11, "insect": 0.11, "broken": 0.11, "unhulled": 0.11, "other_defect": 0.06, "defect_untyped": 0.10}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# source_class values that mean "defective, type not stated" (data_sources.py maps them to other_defect)
UNTYPED = {("usk_coffee", "defect"), ("lojano", "defectuoso"), ("afiyah_deteksi", "bijirusak"),
           ("afiyah_bijikopi", "bijikopirusak"), ("samruddh_grading", "CGD"), ("daffa_defect", "Defected")}

SOURCES = ["j4ckdev", "mfu17", "loja_yolo", "lojano", "usk_coffee", "samruddh_grading", "afiyah_deteksi",
           "afiyah_bijikopi", "daffa_defect", "mindforge_doubleside", "notplying_defects", "vicanadya16"]


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.parent.mkdir(exist_ok=True)
    with open(LOGF, "a") as f: f.write(s + "\n")


def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except ValueError: return str(p)


# ------------------------------------------------------------------------------------------------ data cache
def _load_png(p):
    from PIL import Image
    a = np.asarray(Image.open(ROOT / p).convert("RGB"))
    assert a.shape == (CROP, CROP, 3), (p, a.shape)
    return a


def build_cache(workers=14):
    """All crops -> CACHE/crops.npy (uint8 N,128,128,3) + CACHE/meta.json (one dict per row, same order)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in csv.DictReader(open(ROOT / "data/crops/crops.csv")):
        rows.append(dict(path=r["path"], source="j4ckdev", source_class=r["photo"], kind=r["cls"], group=f"j4ckdev/{r['photo']}",
                         touching=int(r["touching"]), side=float(r["bean_side_px"]), licence="CC BY-NC-SA 4.0"))
    for r in csv.DictReader(open(ROOT / "data/crops_v2/crops_v2.csv")):
        if r["keep"] != "1": continue
        kind = "defect_untyped" if (r["source"], r["source_class"]) in UNTYPED else r["farz_class"]
        try: side = max(float(r["bbox_x1"]) - float(r["bbox_x0"]), float(r["bbox_y1"]) - float(r["bbox_y0"]))
        except ValueError: side = float("nan")
        rows.append(dict(path=r["crop_path"], source=r["source"], source_class=r["source_class"], kind=kind, group=r["group"],
                         touching=int(r["touching"] or 0), side=side, licence=r["licence"]))
    from concurrent.futures import ProcessPoolExecutor
    arr = np.lib.format.open_memmap(CACHE / "crops_tmp.npy", mode="w+", dtype=np.uint8, shape=(len(rows), CROP, CROP, 3))
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, a in enumerate(ex.map(_load_png, [r["path"] for r in rows], chunksize=256)):
            arr[i] = a
            if i % 20000 == 0: log(f"cache {i}/{len(rows)} {time.time()-t0:.0f}s")
    arr.flush(); del arr
    os.replace(CACHE / "crops_tmp.npy", CACHE / "crops.npy")
    json.dump(rows, open(CACHE / "meta.json", "w"))
    log(f"cache built: {len(rows)} crops in {time.time()-t0:.0f}s")


def load_cache(in_memory=True):
    meta = json.load(open(CACHE / "meta.json"))
    crops = np.load(CACHE / "crops.npy", mmap_mode="r")
    if in_memory: crops = np.ascontiguousarray(crops)
    return meta, crops


class Sampler:
    """kind (KIND_W) -> source uniform among sources having that kind -> crop uniform."""
    def __init__(self, meta, idx, seed=0):
        self.rng = np.random.default_rng(seed)
        self.by = {}
        for i in idx:
            m = meta[i]; self.by.setdefault(m["kind"], {}).setdefault(m["source"], []).append(i)
        self.by = {k: {s: np.array(v) for s, v in d.items()} for k, d in self.by.items()}
        kinds = [k for k in KINDS if k in self.by]
        w = np.array([KIND_W[k] for k in kinds]); self.kinds, self.kw = kinds, w / w.sum()
        self.srcs = {k: sorted(self.by[k]) for k in kinds}

    def draw(self, n):
        ks = self.rng.choice(len(self.kinds), n, p=self.kw); out = np.empty(n, np.int64)
        for j, ki in enumerate(ks):
            k = self.kinds[ki]; s = self.srcs[k][self.rng.integers(len(self.srcs[k]))]; pool = self.by[k][s]
            out[j] = pool[self.rng.integers(len(pool))]
        return out

    def describe(self):
        return {k: {s: int(len(v)) for s, v in self.by[k].items()} for k in self.kinds}


# ------------------------------------------------------------------------------------------------ model
def make_net(n_cls=len(CLASSES), pretrained=True):
    import torch.nn as nn
    from torchvision import models
    w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v3_small(weights=w)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, n_cls)
    return net


def Norm(net):
    import torch, torch.nn as nn
    class _Norm(nn.Module):
        """RGB [0,1] NCHW -> logits; ImageNet mean/std inside the graph."""
        def __init__(s, net):
            super().__init__(); s.net = net
            s.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1)); s.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        def forward(s, x): return s.net((x - s.mean) / s.std)
    return _Norm(net)


def Deploy(norm, temperature):
    import torch, torch.nn as nn
    class _Deploy(nn.Module):
        """image [N,3,128,128] RGB in [0,1] -> probs = softmax(logits / T), T baked in."""
        def __init__(s, m, t):
            super().__init__(); s.m = m; s.register_buffer("t", torch.tensor(float(t)))
        def forward(s, image): return torch.softmax(s.m(image) / s.t, dim=1)
    return _Deploy(norm, temperature)


# ------------------------------------------------------------------------------------------------ GPU augmentation
RES_CHOICES = [24, 28, 32, 40, 48, 56, 64, 80, 96]


def augment_gpu(xu8, gen_seed=None):
    """xu8: uint8 tensor [B,128,128,3] on device -> float [B,3,128,128] in [0,1], augmented (batched).
    Same intent as v1's recipe: resolution degrade (p 0.85, 24-96 px), rotation 0-360 + scale 0.85-1.15 + translate 4%
    (fill = sheet grey), flips, brightness/contrast/saturation, per-channel gain (white-balance error) 0.9-1.1, gamma
    0.8-1.25, p 0.3 gaussian noise sd <= 0.02, p 0.15 small blur."""
    import torch, torch.nn.functional as F
    dev = xu8.device; B = xu8.shape[0]
    x = xu8.permute(0, 3, 1, 2).float() / 255.0
    fill = SHEET_TARGET / 255.0
    # 1) resolution degrade, grouped by target size
    deg = torch.rand(B, device=dev) < 0.85
    res = torch.tensor(np.random.choice(RES_CHOICES, B), device=dev)
    for r in RES_CHOICES:
        m = deg & (res == r)
        if m.any():
            xs = x[m]
            xs = F.interpolate(F.interpolate(xs, size=(r, r), mode="bilinear", antialias=True, align_corners=False),
                               size=(CROP, CROP), mode="bilinear", align_corners=False)
            x[m] = xs
    # 2) affine (rotation, scale, translate, flips) with sheet-grey fill
    ang = torch.rand(B, device=dev) * 2 * np.pi
    sc = 1.0 / (0.85 + 0.30 * torch.rand(B, device=dev))
    fx = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0); fy = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    tx = (torch.rand(B, device=dev) - 0.5) * 0.16; ty = (torch.rand(B, device=dev) - 0.5) * 0.16
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos * fx, -sin * fy, tx], 1), torch.stack([sin * fx, cos * fy, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x - fill, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + fill
    # 3) colour
    def U(lo, hi, shape=(B, 1, 1, 1)): return lo + (hi - lo) * torch.rand(shape, device=dev)
    x = x * U(0.8, 1.2)                                               # brightness
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    mu = gray.mean((2, 3), keepdim=True)
    x = (x - mu) * U(0.75, 1.25) + mu                                 # contrast
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = gray + (x - gray) * U(0.7, 1.3)                               # saturation
    x = x * U(0.9, 1.1, (B, 3, 1, 1))                                 # per-channel gain (white-balance error)
    x = x.clamp(0, 1) ** U(0.8, 1.25)                                 # gamma
    nz = (torch.rand(B, 1, 1, 1, device=dev) < 0.3).float() * U(0.0, 0.02)
    x = x + torch.randn_like(x) * nz
    bl = torch.rand(B, device=dev) < 0.15
    if bl.any():
        k = torch.tensor([1., 2., 1.], device=dev); k = (k[:, None] * k[None, :]); k = (k / k.sum()).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        x[bl] = F.conv2d(F.pad(x[bl], (1, 1, 1, 1), mode="replicate"), k, groups=3)
    return x.clamp(0, 1)


def to_float(xu8_np, dev):
    import torch
    return torch.from_numpy(np.ascontiguousarray(xu8_np)).to(dev).permute(0, 3, 1, 2).float() / 255.0


# ------------------------------------------------------------------------------------------------ training
def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def kind_targets(meta, idx):
    """-> y (class index, or -1 for defect_untyped)"""
    return np.array([CLASSES.index(meta[i]["kind"]) if meta[i]["kind"] in CLASSES else -1 for i in idx])


def loss_fn(logits, y, smooth=0.05):
    import torch, torch.nn.functional as F
    typed = y >= 0
    loss = torch.zeros((), device=logits.device); n = 0
    if typed.any():
        loss = loss + F.cross_entropy(logits[typed], y[typed], label_smoothing=smooth, reduction="sum"); n += int(typed.sum())
    if (~typed).any():
        lp = F.log_softmax(logits[~typed], 1)
        loss = loss - torch.logsumexp(lp[:, DEFECT_IDX], 1).sum(); n += int((~typed).sum())
    return loss / max(n, 1)


def train(meta, crops, train_idx, steps=2500, bs=128, lr=2e-3, seed=0, tag="", init_state=None, warm=0.1):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = device()
    model = Norm(make_net()).to(dev)
    if init_state is not None: model.load_state_dict(init_state)
    smp = Sampler(meta, train_idx, seed=seed)
    y_all = kind_targets(meta, range(len(meta)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=warm)
    t0 = time.time(); run = 0.0
    model.train()
    for st in range(steps):
        idx = np.sort(smp.draw(bs))
        xb = torch.from_numpy(crops[idx]).to(dev)
        x = augment_gpu(xb)
        y = torch.from_numpy(y_all[idx]).to(dev)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if st else loss.item()
        if (st + 1) % 500 == 0: log(f"  [{tag}] step {st+1}/{steps} loss {run:.3f} ({time.time()-t0:.0f}s)")
    model.eval()
    return model, {"steps": steps, "batch": bs, "lr": lr, "seed": seed, "train_seconds": round(time.time() - t0, 1),
                   "sampler_pool_sizes": smp.describe()}


def logits_of(model, crops, idx, bs=512):
    import torch
    dev = next(model.parameters()).device; out = []
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            sel = np.asarray(idx[b:b + bs])
            x = to_float(crops[sel], dev)
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)


def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------------------------------------ metrics
def correct_typed(p, y):
    """typed rows: argmax == y; untyped rows (y == -1): argmax is any defect class."""
    pred = p.argmax(1)
    return np.where(y >= 0, pred == y, pred != 0)


def binary(p, y):
    """good-vs-defect view: truth defect = (y != 0); prob of the predicted side; correct."""
    pd = 1.0 - p[:, 0]; truth = y != 0; pred = pd >= 0.5
    conf = np.maximum(pd, 1 - pd)
    return pred, truth, conf


def ece_binary(p, y, w=None, bins=10):
    pred, truth, conf = binary(p, y)
    ok = (pred == truth).astype(float); w = np.ones(len(y)) if w is None else w
    e = 0.0; W = w.sum()
    edges = np.linspace(0.5, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any(): e += w[m].sum() / W * abs(np.average(ok[m], weights=w[m]) - np.average(conf[m], weights=w[m]))
    return float(e)


def ece_top1(p, y, bins=10):
    conf = p.max(1); ok = correct_typed(p, y).astype(float); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def summary(p, y):
    """Per-class recall (typed rows), untyped-defect recall (called any defect), macro accuracy (mean recall over the
    label kinds present), binary good-vs-defect numbers, ECE, 7x6 confusion (rows = label kinds)."""
    pred = p.argmax(1); out = {"n": int(len(y))}
    cm = np.zeros((len(KINDS), len(CLASSES)), int)
    for t, q in zip(y, pred): cm[t if t >= 0 else len(KINDS) - 1, q] += 1
    rec = {}
    for i, k in enumerate(KINDS):
        n = cm[i].sum()
        if not n: continue
        rec[k] = float(cm[i, i] / n) if i < len(CLASSES) else float(cm[i, 1:].sum() / n)
    out["n_by_kind"] = {k: int(cm[i].sum()) for i, k in enumerate(KINDS) if cm[i].sum()}
    out["per_kind_recall"] = rec
    out["macro_accuracy"] = float(np.mean(list(rec.values()))) if rec else None
    out["accuracy_rows"] = float(correct_typed(p, y).mean()) if len(y) else None
    bp, bt, bc = binary(p, y)
    out["binary"] = {"good_recall": float((~bp[~bt]).mean()) if (~bt).any() else None,
                     "defect_recall": float(bp[bt].mean()) if bt.any() else None,
                     "accuracy": float((bp == bt).mean()) if len(y) else None}
    rb = [v for v in (out["binary"]["good_recall"], out["binary"]["defect_recall"]) if v is not None]
    out["binary"]["balanced_accuracy"] = float(np.mean(rb)) if rb else None
    out["ece_binary"] = ece_binary(p, y) if len(y) else None
    out["ece_top1"] = ece_top1(p, y) if len(y) else None
    out["confusion_rows_kinds_cols_classes"] = {"rows": KINDS, "cols": CLASSES, "matrix": cm.tolist()}
    return out


def coverage_curve(p, y, w=None, mode="binary", thresholds=None):
    """answered accuracy vs coverage. mode 'binary': confidence = max(P(good), 1-P(good)), correct = good-vs-defect;
    mode 'top1': confidence = max prob, correct = correct_typed."""
    if thresholds is None: thresholds = np.round(np.arange(0.50, 1.0001, 0.01), 2)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if mode == "binary":
        bp, bt, conf = binary(p, y); ok = bp == bt
    else:
        conf = p.max(1); ok = correct_typed(p, y)
    out = []
    for t in thresholds:
        m = conf >= t - 1e-12
        out.append({"threshold": float(t), "coverage": float(w[m].sum() / w.sum()) if w.sum() else None, "answered": int(m.sum()),
                    "accuracy_answered": float(np.average(ok[m], weights=w[m])) if m.any() and w[m].sum() > 0 else None})
    return out
