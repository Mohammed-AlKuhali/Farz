"""Steps 2-5: train, measure and export the Farz per-bean classifier (MobileNetV3-Small, 5 classes).

Honesty rules baked in:
  * Each J4ckDev class comes from ONE photo, so we split SPATIALLY inside each photo:
      fold 1 tests on beans whose centre x is in the right half, fold 2 on the left half.
    Within-photo held-out accuracy is an UPPER BOUND (lighting, sheet and camera are shared).
  * Temperature is fitted on a validation split carved out of each fold's TRAINING half, never on its test half.
  * The final model is trained on ALL crops; it has no held-out data, so it uses the mean fold temperature
    and the abstain threshold chosen on the fold models' held-out predictions.
  * fp32 vs INT8 is decided on the fold models' held-out halves (INT8 only if within 1.5 points).

Outputs: models/farz_beans_{fp32,int8}.onnx, models/farz_beans_labels.json, models/farz_fold{1,2}_*.onnx,
         reports/beans_model.json (+ reports/beans_heldout_preds.npz for charts).
"""
import csv, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import v2 as T
from common import ROOT, CLASSES, CROPS_CSV, MEAN, STD

SEED = 0
import os
EPOCHS = int(os.environ.get("FARZ_EPOCHS", 40))  # fixed recipe (no early stopping on any test data)
EXTRA_SEEDS = [int(x) for x in os.environ.get("FARZ_EXTRA_SEEDS", "1,2").split(",") if x]
SAMPLES_PER_EPOCH = 512   # class-balanced draws with replacement
BS = 32
LR = 1e-3
VAL_FRAC = 0.2            # per photo, from the TRAINING half only, for temperature fitting
SHEET = 235               # beanfinder.SHEET_TARGET
PHONE_RES = 56            # "phone-like" effective crop resolution used for a robustness check
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
(ROOT / "logs").mkdir(exist_ok=True)  # logs/ is not in a fresh clone; create it before opening the log
LOG = open(ROOT / "logs/beans_train.log", "a")

def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# ---------------------------------------------------------------- data
def load_rows():
    rows = list(csv.DictReader(open(CROPS_CSV)))
    for r in rows:
        r["img"] = np.asarray(Image.open(ROOT / r["path"]).convert("RGB"))
        r["y"] = CLASSES.index(r["cls"]); r["test_fold"] = int(r["test_fold"])
        r["native"] = min(128, int(r["crop_src_px"]))  # effective resolution of the 128 px crop
    return rows

def degrade(img_u8, res):
    """Down-sample a 128 px crop to `res` px and back: mimics a smaller bean in the photo (phone use)."""
    if res >= 128: return img_u8
    im = Image.fromarray(img_u8).resize((res, res), Image.BILINEAR, reducing_gap=None)
    return np.asarray(im.resize((128, 128), Image.BILINEAR))

AUG = T.Compose([
    T.RandomAffine(degrees=180, translate=(0.04, 0.04), scale=(0.85, 1.15), fill=SHEET / 255.0),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.3, hue=0.02),
])

def augment(r):
    img = r["img"]
    # resolution shortcut control: J4ckDev "good" beans are small in their photo (blurry crops) while most defect
    # photos are close-ups (sharp crops). Push every class into the same low-res band most of the time.
    if random.random() < 0.85:
        img = degrade(img, random.randint(36, 80))
    x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
    x = AUG(x)
    x = x.clamp(0, 1) ** random.uniform(0.8, 1.25)           # gamma jitter
    if random.random() < 0.3: x = (x + torch.randn_like(x) * random.uniform(0.0, 0.02)).clamp(0, 1)
    return x

def to_tensor(imgs):
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0

# ---------------------------------------------------------------- model
def make_net():
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(CLASSES))
    return net

class Norm(nn.Module):
    """Input RGB [0,1] NCHW -> logits. Normalisation is inside the graph."""
    def __init__(s, net):
        super().__init__(); s.net = net
        s.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1)); s.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
    def forward(s, x): return s.net((x - s.mean) / s.std)

class Deploy(nn.Module):
    """image [N,3,128,128] RGB in [0,1] -> probs = softmax(logits / T)."""
    def __init__(s, norm, temperature):
        super().__init__(); s.m = norm; s.register_buffer("t", torch.tensor(float(temperature)))
    def forward(s, image): return torch.softmax(s.m(image) / s.t, dim=1)

def train(rows, tag, seed=SEED):
    seed_all(seed)
    model = Norm(make_net()).to(DEV)
    counts = np.bincount([r["y"] for r in rows], minlength=len(CLASSES))
    w = np.array([1.0 / counts[r["y"]] for r in rows]); w /= w.sum()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * (SAMPLES_PER_EPOCH // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        idx = np.random.choice(len(rows), SAMPLES_PER_EPOCH, p=w)
        for b in range(0, SAMPLES_PER_EPOCH, BS):
            batch = [rows[i] for i in idx[b:b + BS]]
            x = torch.stack([augment(r) for r in batch]).to(DEV)
            y = torch.tensor([r["y"] for r in batch], device=DEV)
            opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if (ep + 1) % 10 == 0: log(f"  [{tag}] epoch {ep+1}/{EPOCHS} loss {tot / (SAMPLES_PER_EPOCH // BS):.3f} ({time.time()-t0:.0f}s)")
    return model.eval()

@torch.no_grad()
def logits_of(model, imgs):
    out = []
    for b in range(0, len(imgs), 128):
        out.append(model(to_tensor(imgs[b:b + 128]).to(DEV)).float().cpu())
    return torch.cat(out)

def fit_temperature(logits, y):
    y = torch.as_tensor(y)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad(); l = F.cross_entropy(logits / log_t.exp(), y); l.backward(); return l
    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0))

# ---------------------------------------------------------------- metrics
def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)

def report(p, y):
    pred = p.argmax(1)
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    for a, b in zip(y, pred): cm[a, b] += 1
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()),
            "per_class_recall": {c: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i, c in enumerate(CLASSES)},
            "confusion": cm.tolist(), "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()), "ece_10bin": ece(p, y)}

def coverage_curve(p, y):
    conf, ok = p.max(1), (p.argmax(1) == y)
    out = []
    for t in np.round(np.arange(0.20, 1.0001, 0.01), 2):
        m = conf >= t
        out.append({"threshold": float(t), "coverage": float(m.mean()), "answered": int(m.sum()),
                    "accuracy_answered": float(ok[m].mean()) if m.any() else None})
    return out

def pick_threshold(curve, target=0.90, min_answered=30):
    """Lowest threshold whose answered accuracy is >= target AND stays >= target for every higher threshold
    that still answers at least `min_answered` beans (so one lucky bin cannot set it)."""
    ok = [c for c in curve if c["answered"] >= min_answered]
    for i, c in enumerate(ok):
        if all(d["accuracy_answered"] >= target for d in ok[i:]): return c
    return None

# ---------------------------------------------------------------- ONNX
def export(model, temperature, path):
    dep = Deploy(model.cpu().eval(), temperature).eval()
    torch.onnx.export(dep, torch.rand(1, 3, 128, 128), str(path), input_names=["image"], output_names=["probs"],
                      dynamic_axes={"image": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
    model.to(DEV)
    return path

QUANT_SKIP_SWEEP = [0, 4, 8, 10, 12, 13, 16, 20, 26]  # number of leading Conv layers kept in fp32
QUANT_MIN_AGREE = 0.99                                 # vs fp32 argmax, on the CALIBRATION (training) crops only

def quantize(fp32, int8, calib_imgs):
    """Static INT8 QDQ, per-channel weights (QInt8), uint8 activations, calibrated (MinMax) on training crops.

    Measured 3 Oct 2026: quantising EVERY op of MobileNetV3-Small collapses held-out accuracy to 23-27% (fold models,
    vs 82-90% fp32): the early layers (first conv output range ~[-160,130], SE-block Mul/Add) have per-channel activation
    ranges that one per-tensor uint8 scale cannot hold. So we quantise only Conv + Gemm, and keep the first k Conv layers
    in fp32, choosing the smallest k in QUANT_SKIP_SWEEP whose argmax agreement with fp32 on the calibration crops is
    >= QUANT_MIN_AGREE. No test labels are used to pick k."""
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = int8.with_name(int8.stem + "_pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    g = onnx.load(str(pre)).graph
    convs = [n.name for n in g.node if n.op_type == "Conv"]
    gemms = [n.name for n in g.node if n.op_type == "Gemm"]
    X = to_tensor(calib_imgs).numpy()
    ref = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).run(["probs"], {"image": X})[0]
    class Calib(CalibrationDataReader):
        def __init__(s): s.it = iter([{"image": X[i:i + 1]} for i in range(len(X))])
        def get_next(s): return next(s.it, None)
    sweep = []
    tmp = int8.with_name(int8.stem + "_try.onnx")
    for k in QUANT_SKIP_SWEEP:
        quantize_static(str(pre), str(tmp), Calib(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, nodes_to_quantize=convs[k:] + gemms)
        p = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"]).run(["probs"], {"image": X})[0]
        agree = float((p.argmax(1) == ref.argmax(1)).mean())
        sweep.append({"fp32_convs": k, "calib_argmax_agreement": agree, "calib_max_abs_prob_diff": float(np.abs(p - ref).max()), "bytes": tmp.stat().st_size})
        log(f"    quant sweep k={k:2d}: agreement {agree:.4f}  size {tmp.stat().st_size/1e6:.2f} MB")
        if agree >= QUANT_MIN_AGREE:
            tmp.replace(int8); break
    else:
        tmp.replace(int8)  # nothing reached the bar: keep the last (most-fp32) attempt; the 1.5-point rule then decides
    pre.unlink()
    info = {"quantised_ops": "Conv + Gemm only (QDQ, per-channel QInt8 weights, QUInt8 activations, MinMax calibration on training crops)",
            "fp32_leading_convs": sweep[-1]["fp32_convs"], "total_convs": len(convs), "sweep": sweep,
            "rule": f"smallest k in {QUANT_SKIP_SWEEP} with argmax agreement >= {QUANT_MIN_AGREE} vs fp32 on calibration crops"}
    return int8, info

def ort_probs(path, imgs, threads=0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads: so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(["probs"], {"image": to_tensor(imgs[b:b + 64]).numpy()})[0] for b in range(0, len(imgs), 64)])

def latency_ms(path, img, n=200):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = to_tensor([img]).numpy()
    for _ in range(20): s.run(None, {"image": x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); s.run(None, {"image": x}); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))

# ---------------------------------------------------------------- main
def main():
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    rows = load_rows()
    log(f"\n=== train_beans {time.strftime('%Y-%m-%d %H:%M:%S')}  device={DEV}  crops={len(rows)}")
    res = {"classes": CLASSES, "recipe": {"arch": "torchvision mobilenet_v3_small IMAGENET1K_V1, 5-class head", "input": "128x128 RGB [0,1], ImageNet mean/std inside the graph",
           "epochs": EPOCHS, "samples_per_epoch": SAMPLES_PER_EPOCH, "batch": BS, "lr": LR, "optimizer": "AdamW wd 1e-4, OneCycle", "label_smoothing": 0.05,
           "sampling": "class-balanced (weights 1/class count, with replacement)",
           "augmentation": "p=0.85 resolution degrade to 36-80 px and back; affine rotation +-180 deg, translate 4%, scale 0.85-1.15 (fill = sheet grey 235); h+v flips; brightness 0.2, contrast 0.25, saturation 0.3, hue 0.02; gamma 0.8-1.25; p=0.3 gaussian noise sd<=0.02",
           "device": str(DEV), "seed": SEED},
           "training_set_counts": {c: sum(r["cls"] == c for r in rows) for c in CLASSES},
           "training_set_counts_by_photo": {p: sum(r["photo"] == p for r in rows) for p in sorted({r["photo"] for r in rows})},
           "split": "spatial within each photo: fold 1 tests on beans with centre x in the right half of their photo, fold 2 on the left half; temperature fitted on a random 20% per-photo validation split of the TRAINING half",
           "upper_bound_warning": "Every class comes from a single J4ckDev photo, so train and test beans share the photo's camera, lighting and sheet. Held-out accuracy here is an UPPER BOUND on accuracy for new photos, new farms or Yemeni beans.",
           "folds": {}}
    held = {"y": [], "fold": [], "torch": [], "fp32": [], "int8": [], "torch_phone": [], "idx": []}
    temps = []
    for fold in (1, 2):
        test = [r for r in rows if r["test_fold"] == fold]
        trainval = [r for r in rows if r["test_fold"] != fold]
        rng = random.Random(SEED + fold); val = []
        for p in sorted({r["photo"] for r in trainval}):
            g = [r for r in trainval if r["photo"] == p]; rng.shuffle(g); val += g[:max(1, round(VAL_FRAC * len(g)))]
        vset = {id(r) for r in val}; train_rows = [r for r in trainval if id(r) not in vset]
        log(f"fold {fold}: train {len(train_rows)}  val {len(val)}  test {len(test)}")
        model = train(train_rows, f"fold{fold}")
        lv = logits_of(model, [r["img"] for r in val]); yv = np.array([r["y"] for r in val])
        t = fit_temperature(lv, yv); temps.append(t)
        lt = logits_of(model, [r["img"] for r in test]); yt = np.array([r["y"] for r in test])
        p_raw = torch.softmax(lt, 1).numpy(); p_cal = torch.softmax(lt / t, 1).numpy()
        lph = logits_of(model, [degrade(r["img"], PHONE_RES) for r in test]); p_ph = torch.softmax(lph / t, 1).numpy()
        fp32 = export(model, t, MODELS / f"farz_fold{fold}_fp32.onnx")
        int8, qinfo = quantize(fp32, MODELS / f"farz_fold{fold}_int8.onnx", [r["img"] for r in train_rows])
        p32 = ort_probs(fp32, [r["img"] for r in test]); p8 = ort_probs(int8, [r["img"] for r in test])
        f = {"train_n": len(train_rows), "val_n": len(val), "test_n": len(test),
             "test_counts": {c: int((yt == i).sum()) for i, c in enumerate(CLASSES)},
             "temperature": t, "val_accuracy_uncalibrated": float((lv.argmax(1).numpy() == yv).mean()),
             "pytorch_uncalibrated": report(p_raw, yt), "pytorch_calibrated": report(p_cal, yt),
             "pytorch_calibrated_phone_res": {"effective_crop_px": PHONE_RES, **report(p_ph, yt)},
             "onnx_fp32": report(p32, yt), "onnx_int8": report(p8, yt),
             "onnx_fp32_vs_torch_max_abs_diff": float(np.abs(p32 - p_cal).max()),
             "int8_quant": qinfo, "val_idx": [rows.index(r) for r in val]}
        res["folds"][str(fold)] = f
        log(f"fold {fold}: T={t:.3f}  acc raw {f['pytorch_uncalibrated']['accuracy']:.3f}  fp32 {f['onnx_fp32']['accuracy']:.3f}  int8 {f['onnx_int8']['accuracy']:.3f}  phone-res {f['pytorch_calibrated_phone_res']['accuracy']:.3f}  ece {f['pytorch_uncalibrated']['ece_10bin']:.3f}->{f['pytorch_calibrated']['ece_10bin']:.3f}  onnx-torch diff {f['onnx_fp32_vs_torch_max_abs_diff']:.2e}")
        held["y"] += yt.tolist(); held["fold"] += [fold] * len(yt); held["torch"].append(p_cal); held["fp32"].append(p32); held["int8"].append(p8); held["torch_phone"].append(p_ph)
        held["idx"] += [rows.index(r) for r in test]
    for k in ("torch", "fp32", "int8", "torch_phone"): held[k] = np.concatenate(held[k])
    y = np.array(held["y"])
    pooled = {k: report(held[k], y) for k in ("fp32", "int8", "torch_phone")}
    res["heldout_pooled"] = pooled
    res["heldout_mean_of_folds"] = {k: float(np.mean([res["folds"][f][n]["accuracy"] for f in ("1", "2")]))
                                    for k, n in (("fp32", "onnx_fp32"), ("int8", "onnx_int8"), ("phone_res", "pytorch_calibrated_phone_res"))}
    gap = 100 * (res["heldout_mean_of_folds"]["fp32"] - res["heldout_mean_of_folds"]["int8"])
    deploy = "int8" if gap <= 1.5 else "fp32"
    res["deploy_decision"] = {"fp32_minus_int8_points": round(gap, 2), "rule": "INT8 only if within 1.5 points of fp32 (mean of held-out folds)", "deployed": deploy}
    log(f"held-out mean: fp32 {res['heldout_mean_of_folds']['fp32']:.4f}  int8 {res['heldout_mean_of_folds']['int8']:.4f}  gap {gap:.2f} pts -> deploy {deploy}")
    curve = coverage_curve(held[deploy], y)
    chosen = pick_threshold(curve)
    res["selective"] = {"on": f"pooled held-out halves, fold models, {deploy} ONNX, calibrated", "target_answered_accuracy": 0.90,
                        "rule": "lowest max-prob threshold whose answered accuracy is >=90% and stays >=90% for every higher threshold answering >=30 beans",
                        "chosen": chosen, "curve": curve,
                        "per_fold_at_chosen": {str(fd): (lambda m: {"coverage": float(m.mean()), "accuracy_answered": float((held[deploy][m].argmax(1) == y[m]).mean()) if m.any() else None})((np.array(held["fold"]) == fd) & (held[deploy].max(1) >= chosen["threshold"])) for fd in (1, 2)} if chosen else None,
                        "curve_phone_res_torch": coverage_curve(held["torch_phone"], y)}
    log("chosen threshold:", chosen)
    np.savez(REPORTS / "beans_heldout_preds.npz", y=y, fold=np.array(held["fold"]), idx=np.array(held["idx"]),
             fp32=held["fp32"], int8=held["int8"], torch=held["torch"], torch_phone=held["torch_phone"])

    # ---------------- seed spread (same recipe, same splits, other seeds; PyTorch calibrated, reported only)
    spread = {}
    for fold in (1, 2):
        test = [r for r in rows if r["test_fold"] == fold]; yt = np.array([r["y"] for r in test])
        vidx = set(res["folds"][str(fold)]["val_idx"]); val = [rows[i] for i in sorted(vidx)]
        train_rows = [r for i, r in enumerate(rows) if r["test_fold"] != fold and i not in vidx]
        accs = [res["folds"][str(fold)]["pytorch_calibrated"]["accuracy"]]
        for sd in EXTRA_SEEDS:
            m = train(train_rows, f"fold{fold}-seed{sd}", seed=sd)
            t = fit_temperature(logits_of(m, [r["img"] for r in val]), np.array([r["y"] for r in val]))
            accs.append(float((logits_of(m, [r["img"] for r in test]).argmax(1).numpy() == yt).mean()))
        spread[str(fold)] = {"seeds": [SEED] + EXTRA_SEEDS, "accuracy": accs, "mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else None}
        log(f"seed spread fold {fold}: {accs}")
    res["seed_spread_pytorch"] = spread

    # ---------------- final model on ALL crops
    t_final = float(np.mean(temps))
    log(f"final: training on all {len(rows)} crops; temperature = mean of fold temperatures {temps} = {t_final:.3f}")
    model = train(rows, "final")
    fp32 = export(model, t_final, MODELS / "farz_beans_fp32.onnx")
    int8, qinfo_final = quantize(fp32, MODELS / "farz_beans_int8.onnx", [r["img"] for r in rows])
    imgs = [r["img"] for r in rows]
    p32, p8 = ort_probs(fp32, imgs), ort_probs(int8, imgs)
    res["final"] = {"train_n": len(rows), "temperature": t_final, "temperature_source": "mean of the two fold temperatures (each fitted on its fold's validation split)",
                    "fold_temperatures": temps,
                    "train_accuracy_fp32_not_a_test": float((p32.argmax(1) == np.array([r["y"] for r in rows])).mean()),
                    "int8_vs_fp32_argmax_agreement_on_train_crops": float((p32.argmax(1) == p8.argmax(1)).mean()),
                    "int8_vs_fp32_max_abs_prob_diff_on_train_crops": float(np.abs(p32 - p8).max()),
                    "sizes_bytes": {"fp32": fp32.stat().st_size, "int8": int8.stat().st_size},
                    "median_cpu_latency_ms_batch1_1thread": {"fp32": latency_ms(fp32, imgs[0]), "int8": latency_ms(int8, imgs[0])},
                    "latency_machine": "MacBook M5 Pro CPU, onnxruntime 1.30.0 CPUExecutionProvider, intra_op_num_threads=1 (NOT a phone)",
                    "int8_quant": qinfo_final,
                    "deployed_file": f"models/farz_beans_{deploy}.onnx"}
    log("final:", json.dumps({k: res["final"][k] for k in ("sizes_bytes", "median_cpu_latency_ms_batch1_1thread", "int8_vs_fp32_argmax_agreement_on_train_crops")}))
    thr = round(chosen["threshold"], 2) if chosen else 0.99
    labels = {"classes": CLASSES, "abstain_below": thr, "input": 128,
              "note": (f"Farz per-bean classifier, {deploy} ONNX. input 'image' float32 [N,3,128,128] RGB in [0,1] (crop contract of src/beans/beanfinder.py); "
                       f"output 'probs' [N,5] = softmax(logits/T), T={t_final:.3f} baked in. Abstain (yellow '?') when max prob < abstain_below. "
                       f"Threshold chosen on within-photo spatially held-out J4ckDev beans for >=90% answered accuracy - an UPPER BOUND, not field accuracy. "
                       f"Trained on J4ckDev Green Coffee Beans (CC BY-NC-SA 4.0); the model inherits NC-SA, prototype only.")}
    json.dump(labels, open(MODELS / "farz_beans_labels.json", "w"), indent=1, ensure_ascii=False)
    res["labels_json"] = labels
    json.dump(res, open(REPORTS / "beans_model.json", "w"), indent=1)
    log("wrote", REPORTS / "beans_model.json")

if __name__ == "__main__":
    main()
