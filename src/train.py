"""Fine-tune a small CNN on coffee leaves, export to ONNX, quantize to INT8, and measure honestly.

Held-out test = whole countries the model never sees in training (default: Uganda + Ecuador).
Outputs land in models/ and reports/.
"""
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms as T

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["healthy", "rust", "miner", "phoma", "cercospora"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class DS(Dataset):
    def __init__(s, rows, tf): s.rows, s.tf = rows, tf
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r = s.rows[i]
        return s.tf(Image.open(ROOT / r["path"]).convert("RGB")), CLASSES.index(r["label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mnv3s", choices=["mnv3s", "mnv2"])
    ap.add_argument("--test-sources", default="uganda,rocole")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--tag", default="heldout")
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    test_src = set(s for s in args.test_sources.split(",") if s)
    test = [r for r in rows if r["source"] in test_src]
    trainval = [r for r in rows if r["source"] not in test_src]
    groups = {}
    for r in trainval:
        groups.setdefault((r["source"], r["label"]), []).append(r)
    train, val = [], []
    for g in groups.values():
        random.shuffle(g); k = max(1, int(len(g) * 0.12)); val += g[:k]; train += g[k:]
    print(f"train {len(train)}  val {len(val)}  test {len(test)} (held-out: {sorted(test_src) or 'none'})")

    train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.45, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                          T.RandomRotation(25), T.ColorJitter(0.35, 0.35, 0.3, 0.04), T.RandomGrayscale(0.05),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.Resize(240), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


    counts = np.bincount([CLASSES.index(r["label"]) for r in train], minlength=len(CLASSES))
    w = [1.0 / counts[CLASSES.index(r["label"])] for r in train]
    dl_train = DataLoader(DS(train, train_tf), batch_size=args.bs, sampler=WeightedRandomSampler(w, len(train)), num_workers=8, persistent_workers=True)
    dl_val = DataLoader(DS(val, eval_tf), batch_size=128, num_workers=8)
    dl_test = DataLoader(DS(test, eval_tf), batch_size=128, num_workers=8) if test else None

    if args.arch == "mnv3s":
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(CLASSES))
    else:
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(CLASSES))
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl_train))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    @torch.no_grad()
    def predict(dl):
        net.eval(); P, Y = [], []
        for x, y in dl:
            P.append(torch.softmax(net(x.to(dev)), 1).cpu()); Y.append(y)
        return torch.cat(P).numpy(), torch.cat(Y).numpy()

    best, ckpt = -1, ROOT / f"models/{args.arch}_{args.tag}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    for ep in range(args.epochs):
        net.train(); t0, tot = time.time(), 0.0
        for x, y in dl_train:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(net(x), y); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        p, y = predict(dl_val); acc = (p.argmax(1) == y).mean()
        msg = f"ep {ep+1}/{args.epochs} loss {tot/len(dl_train):.3f} val {acc:.3f} ({time.time()-t0:.0f}s)"
        if dl_test:
            pt, yt = predict(dl_test); msg += f"  heldout-test {(pt.argmax(1)==yt).mean():.3f}"
        print(msg, flush=True)
        if acc > best:
            best = acc; torch.save(net.state_dict(), ckpt)
    net.load_state_dict(torch.load(ckpt)); net.to(dev)

    def report(p, y, name):
        pred = p.argmax(1); conf = p.max(1)
        cm = np.zeros((len(CLASSES), len(CLASSES)), int)
        for a, b in zip(y, pred): cm[a, b] += 1
        present = sorted(set(y.tolist()))
        sel = []
        for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            m = conf >= t
            sel.append({"threshold": t, "coverage": float(m.mean()), "accuracy_when_answering": float((pred[m] == y[m]).mean()) if m.any() else None})
        return {"set": name, "n": int(len(y)), "accuracy": float((pred == y).mean()),
                "per_class_recall": {CLASSES[c]: float(cm[c, c] / cm[c].sum()) for c in present},
                "confusion_matrix": cm.tolist(), "classes_present": [CLASSES[c] for c in present], "selective": sel}

    res = {"arch": args.arch, "tag": args.tag, "train_n": len(train), "heldout_sources": sorted(test_src), "pytorch": {}}
    res["pytorch"]["val"] = report(*predict(dl_val), "val (same sources as train)")
    if dl_test:
        # restrict scoring to classes the held-out set actually contains
        pt, yt = predict(dl_test); res["pytorch"]["heldout"] = report(pt, yt, "held-out countries")

    # ---- ONNX export with normalisation + softmax baked in, input = RGB float [0,1] NCHW 224 ----
    class Deploy(nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
            s.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1)); s.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        def forward(s, x): return torch.softmax(s.m((x - s.mean) / s.std), 1)
    dep = Deploy(net.cpu().eval()).eval()
    fp32 = ROOT / f"models/coffee_{args.arch}_{args.tag}_fp32.onnx"
    torch.onnx.export(dep, torch.rand(1, 3, 224, 224), str(fp32), input_names=["image"], output_names=["probs"],
                      dynamic_axes={"image": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)

    # ---- INT8 static quantization (QDQ, per-channel) calibrated on real training photos ----
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnxruntime as ort
    raw_tf = T.Compose([T.Resize(240), T.CenterCrop(224), T.ToTensor()])
    class Calib(CalibrationDataReader):
        def __init__(s):
            pick = random.sample(train, min(300, len(train)))
            s.it = iter([{"image": raw_tf(Image.open(ROOT / r["path"]).convert("RGB")).unsqueeze(0).numpy()} for r in pick])
        def get_next(s): return next(s.it, None)
    pre = ROOT / f"models/_pre_{args.arch}.onnx"
    quant_pre_process(str(fp32), str(pre))
    int8 = ROOT / f"models/coffee_{args.arch}_{args.tag}_int8.onnx"
    quantize_static(str(pre), str(int8), Calib(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink()

    def ort_eval(path, rows_):
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        P, Y, t = [], [], []
        for r in rows_:
            x = raw_tf(Image.open(ROOT / r["path"]).convert("RGB")).unsqueeze(0).numpy()
            t0 = time.perf_counter(); P.append(s.run(None, {"image": x})[0][0]); t.append(time.perf_counter() - t0)
            Y.append(CLASSES.index(r["label"]))
        return np.array(P), np.array(Y), float(np.median(t) * 1000)

    eval_rows = test if test else val
    res["onnx"] = {}
    for name, path in [("fp32", fp32), ("int8", int8)]:
        p, y, ms = ort_eval(path, eval_rows)
        res["onnx"][name] = {"size_mb": round(path.stat().st_size / 1e6, 2), "median_ms_cpu": round(ms, 2), **report(p, y, "held-out" if test else "val")}
        print(f"{name}: {res['onnx'][name]['size_mb']} MB  acc {res['onnx'][name]['accuracy']:.3f}  {ms:.1f} ms/img")

    (ROOT / "reports").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / f"reports/metrics_{args.arch}_{args.tag}.json", "w"), indent=1)
    json.dump(CLASSES, open(ROOT / "models/labels.json", "w"))
    print("saved", ROOT / f"reports/metrics_{args.arch}_{args.tag}.json")


if __name__ == "__main__":
    main()
