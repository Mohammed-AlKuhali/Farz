"""Step 9b: demo images for the app (app/public/demo/) + their measured results (reports/beans_demo_trays.json).

* 3 SYNTHETIC trays (~100 beans, ~3% / ~12% / ~30% defects): J4ckDev bean crops cut out with a mask and pasted onto a
  flat 235-grey sheet with gaps, random rotation and size jitter. Every bean's true class and position is recorded.
  HONESTY: the deployed (final) model was trained on ALL crops, so these beans are IN its training set. The demo shows
  the flow, not accuracy. For a held-out number we also run each tray end to end through the two FOLD models and, for
  every bean, keep the prediction of the fold model that never saw that bean ("out-of-fold").
* 2 real CBD photos (Mendeley 52877z55vr, CC BY 4.0), copied unmodified.
"""
import csv, json, random, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
from beanfinder import _otsu, _erode, _dilate, _components, SHEET_TARGET
from common import ROOT, CLASSES, CROPS_CSV, CBD_DIR
import pipeline

DEMO = ROOT / "app/public/demo"
CELL, GRID, MARGIN, BEAN_PX = 108, 10, 60, 60
TRAYS = [  # file, defects per class (dark, insect, broken, unhulled), titles
    ("tray_synthetic_03.jpg", {"dark": 1, "insect": 1, "broken": 1, "unhulled": 0},
     "صينية تجريبية مركّبة: عيوب قليلة (حوالي ٣٪)", "Synthetic demo tray: few defects (~3%)"),
    ("tray_synthetic_12.jpg", {"dark": 4, "insect": 3, "broken": 3, "unhulled": 2},
     "صينية تجريبية مركّبة: بعض العيوب (حوالي ١٢٪)", "Synthetic demo tray: some defects (~12%)"),
    ("tray_synthetic_30.jpg", {"dark": 10, "insect": 7, "broken": 8, "unhulled": 5},
     "صينية تجريبية مركّبة: عيوب كثيرة (حوالي ٣٠٪)", "Synthetic demo tray: many defects (~30%)"),
]
J4CK_CREDIT = ("SYNTHETIC composite made by Farz from bean crops of the J4ckDev Green Coffee Beans dataset "
               "(https://github.com/J4ckDev/GreenCoffeeBeansDataset, CC BY-NC-SA 4.0). Not a real tray photo.")
CBD_CREDIT = ("CBD Coffee Bean Dataset, Mendeley Data 52877z55vr (2024), CC BY 4.0, "
              "https://data.mendeley.com/datasets/52877z55vr/1 - photo copied unmodified. Wayanad, India; size-graded, not defect-labelled.")

def bean_rgba(crop):
    """Cut the bean out of a 128 px crop: the bean finder's own 'differs from sheet' rule, Otsu, open/close, keep the
    component at the centre. Returns an RGBA image cropped to the bean."""
    a = crop.astype(np.float32)
    luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    diff = np.maximum(SHEET_TARGET - luma, (a.max(-1) - a.min(-1)) * 1.5)
    m = diff > max(_otsu(diff), 18.0)
    m = _dilate(_erode(m)); m = _erode(_dilate(m))
    comps = _components(m)
    lab_mask = np.zeros_like(m)
    # keep the largest component (the bean is centred and by far the largest blob in its crop)
    best = max(comps, key=lambda c: c[4])
    # rebuild that component's pixels with a flood fill from inside its box
    x0, y0, x1, y1, _ = best
    sub = m[y0:y1, x0:x1]
    lab_mask[y0:y1, x0:x1] = sub
    # fill holes (cracks / dark centres) so the pasted bean is solid
    inv = ~lab_mask; holes = np.zeros_like(inv); h, w = inv.shape
    from collections import deque
    seen = np.zeros_like(inv)
    border = [(y, x) for y in range(h) for x in (0, w - 1)] + [(y, x) for x in range(w) for y in (0, h - 1)]
    q = deque([(y, x) for y, x in border if inv[y, x]])
    for y, x in q: seen[y, x] = True
    while q:
        y, x = q.popleft()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and inv[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True; q.append((yy, xx))
    solid = ~seen
    alpha = (solid * 255).astype(np.uint8)
    rgba = np.dstack([crop, alpha])
    ys, xs = np.nonzero(solid)
    return Image.fromarray(rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "RGBA")

def make_tray(pool, spec, rng):
    good_n = 100 - sum(spec.values())
    chosen = [("good", r) for r in rng.sample(pool["good"], good_n)]
    for c, k in spec.items(): chosen += [(c, r) for r in rng.sample(pool[c], k)]
    rng.shuffle(chosen)
    side = 2 * MARGIN + GRID * CELL
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    sheet = np.clip(SHEET_TARGET + nprng.normal(0, 1.2, (side, side, 3)), 0, 255).astype(np.uint8)
    canvas = Image.fromarray(sheet, "RGB")
    placed = []
    for i, (c, r) in enumerate(chosen):
        b = bean_rgba(r["img"])
        s = BEAN_PX * rng.uniform(0.88, 1.12) / max(b.size)
        b = b.resize((max(8, round(b.width * s)), max(8, round(b.height * s))), Image.BICUBIC)
        b = b.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)
        gy, gx = divmod(i, GRID)
        cx = MARGIN + gx * CELL + CELL // 2 + rng.randint(-10, 10)
        cy = MARGIN + gy * CELL + CELL // 2 + rng.randint(-10, 10)
        canvas.paste(b, (cx - b.width // 2, cy - b.height // 2), b)
        placed.append({"cls": c, "crop": r["path"], "test_fold": int(r["test_fold"]), "cx": cx, "cy": cy})
    return canvas, placed

def match(beans, placed):
    """Map each detected bean to the pasted bean whose centre is nearest (must be inside the detected box)."""
    out = []
    for b in beans:
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        j = min(range(len(placed)), key=lambda k: (placed[k]["cx"] - cx) ** 2 + (placed[k]["cy"] - cy) ** 2)
        d = ((placed[j]["cx"] - cx) ** 2 + (placed[j]["cy"] - cy) ** 2) ** 0.5
        out.append(j if d < 30 else None)
    return out

def main():
    DEMO.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(CROPS_CSV)) if r["touching"] == "0"]
    for r in rows: r["img"] = np.asarray(Image.open(ROOT / r["path"]).convert("RGB"))
    pool = {c: [r for r in rows if r["cls"] == c] for c in CLASSES}
    model_path, labels = pipeline.deployed(); thr = labels["abstain_below"]
    deploy_fmt = "int8" if model_path.stem.endswith("int8") else "fp32"
    final = pipeline.session(model_path)
    folds = {f: pipeline.session(ROOT / f"models/farz_fold{f}_{deploy_fmt}.onnx") for f in (1, 2)}
    rng = random.Random(2026)
    manifest, report = [], {"deployed_model": str(model_path.relative_to(ROOT)), "abstain_below": thr,
                            "honesty": "Deployed model saw these beans in training (in-sample). 'out_of_fold' uses, per bean, the fold model that never saw it.",
                            "trays": []}
    for fname, spec, t_ar, t_en in TRAYS:
        img, placed = make_tray(pool, spec, rng)
        img.save(DEMO / fname, quality=90)
        img = Image.open(DEMO / fname)  # measure what actually ships (after JPEG)
        r, p_final, pred_final = pipeline.run_photo(final, img, thr)
        crops = [b.crop for b in r.beans]
        p_f = {f: pipeline.classify(s, crops) for f, s in folds.items()}
        m = match(r.beans, placed)
        truth = [placed[j]["cls"] if j is not None else None for j in m]
        oof = []
        for i, j in enumerate(m):
            if j is None: oof.append(None); continue
            pr = p_f[placed[j]["test_fold"]][i]   # the fold model whose TEST half contains this bean
            oof.append(CLASSES[int(pr.argmax())] if pr.max() >= thr else "abstain")
        def summary(pred):
            ans = [(t, q) for t, q in zip(truth, pred) if t is not None and q is not None and q != "abstain"]
            n_def = sum(q != "good" for q in pred if q not in (None, "abstain"))
            return {"counts": {k: sum(q == k for q in pred) for k in CLASSES + ["abstain"]},
                    "answered": len(ans), "accuracy_answered": round(sum(t == q for t, q in ans) / max(1, len(ans)), 4),
                    "defects_flagged": n_def}
        true_def = sum(spec.values())
        entry = {"file": fname, "true_beans": 100, "true_defects": true_def, "true_counts": {"good": 100 - true_def, **spec},
                 "detected_beans": len(r.beans), "matched": sum(j is not None for j in m), "checks": r.checks,
                 "deployed_in_sample": summary(pred_final), "out_of_fold_held_out": summary(oof)}
        report["trays"].append(entry)
        print(fname, "detected", len(r.beans), "| final:", entry["deployed_in_sample"]["counts"], "| oof:", entry["out_of_fold_held_out"]["counts"])
        manifest.append({"file": fname, "title_ar": t_ar, "title_en": t_en, "synthetic": True, "credit": J4CK_CREDIT,
                         "true_counts": entry["true_counts"], "farz_python_counts": entry["deployed_in_sample"]["counts"],
                         "note": "Beans pasted onto a flat grey sheet. The shipped model saw these beans in training: demo of the flow, not evidence of accuracy."})
    # two real CBD photos with an exact bean-finder count of 50 (deterministic pick: first such file by name)
    for grade, t_ar, t_en in (("AAA", "صورة حقيقية من الهند (بيانات CBD): درجة AAA", "Real photo, India (CBD dataset): grade AAA"),
                              ("Bits", "صورة حقيقية من الهند (بيانات CBD): درجة Bits", "Real photo, India (CBD dataset): grade 'Bits'")):
        files = sorted((CBD_DIR / grade).glob("*.jpg"), key=lambda f: int(f.stem) if f.stem.isdigit() else 1 << 30)
        for f in files:
            r, p, pred = pipeline.run_photo(final, Image.open(f), thr)
            if r.checks["count"] == 50: break
        dst = f"cbd_{grade.lower()}_{f.stem}.jpg"
        shutil.copyfile(f, DEMO / dst)
        counts = {k: pred.count(k) for k in CLASSES + ["abstain"]}
        report["trays"].append({"file": dst, "source": str(f.relative_to(ROOT)), "detected_beans": r.checks["count"], "deployed_counts": counts,
                                "note": "real photo; no per-bean defect labels exist"})
        print(dst, "detected", r.checks["count"], counts)
        manifest.append({"file": dst, "title_ar": t_ar, "title_en": t_en, "synthetic": False, "credit": CBD_CREDIT,
                         "farz_python_counts": counts,
                         "note": ("Real light-box photo from India. Size-graded only: no per-bean defect labels. MEASURED: the current model "
                                  "calls most beans in CBD photos 'broken' (out-of-distribution failure, see reports/beans_ood_cbd.json) - "
                                  "show this as the honest limit, not as a correct answer.")})
    json.dump(manifest, open(DEMO / "manifest.json", "w"), indent=1, ensure_ascii=False)
    json.dump(report, open(ROOT / "reports/beans_demo_trays.json", "w"), indent=1, ensure_ascii=False)
    rep_path = ROOT / "reports/beans_model.json"; rep = json.load(open(rep_path)); rep["demo_trays"] = report
    json.dump(rep, open(rep_path, "w"), indent=1)

if __name__ == "__main__":
    main()
