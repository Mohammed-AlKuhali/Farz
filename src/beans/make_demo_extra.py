"""Two more SYNTHETIC demo trays for the app (0 defects, ~40% defects) + a verified outcome for EVERY demo tray.

Built exactly like src/beans/make_demo.py (same code: make_demo.make_tray): J4ckDev bean crops cut out and pasted
onto a flat 235-grey sheet. HONESTY: the deployed model was trained on ALL crops, so these beans are in its training
set — the trays show the flow, not accuracy. As in make_demo.py, every bean is ALSO scored out-of-fold (by the fold
model that never saw it) and both results are recorded.

Then, for every tray in app/public/demo (old and new), the app's own decision rules (Python mirror in pipeline.py,
EXIF orientation applied like the browser) give the outcome the app must show; it is written into manifest.json as
"expected" and asserted in the app's e2e test.

Does NOT touch make_demo.py's outputs for the existing trays (images, reports/beans_demo_trays.json,
reports/beans_model.json). Writes: app/public/demo/tray_synthetic_00.jpg, tray_synthetic_40.jpg, manifest.json and
app/reports/demo_trays_extra.json.
usage: .venv/bin/python src/beans/make_demo_extra.py
"""
import csv, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image, ImageOps
from common import ROOT, CLASSES, CROPS_CSV
import pipeline
import make_demo

DEMO = ROOT / "app/public/demo"
NEW = [  # file, defects per class, seed, titles
    ("tray_synthetic_00.jpg", {"dark": 0, "insect": 0, "broken": 0, "unhulled": 0}, 2027,
     "صينية تجريبية مركّبة: بدون عيوب (٠٪)", "Synthetic demo tray: no defects (0%)"),
    ("tray_synthetic_40.jpg", {"dark": 12, "insect": 8, "broken": 12, "unhulled": 8}, 2028,
     "صينية تجريبية مركّبة: عيوب كثيرة (حوالي ٤٠٪)", "Synthetic demo tray: many defects (~40%)"),
]
NOTE_SYN = {
    "note": "Beans pasted onto a flat grey sheet. The shipped model saw these beans in training: demo of the flow, not evidence of accuracy.",
    "note_en": "Made by us from photos of single beans pasted on a grey sheet. Farz learned from these same beans, so this shows how Farz works, not how accurate it is.",
    "note_ar": "صينية ركّبناها من صور حبوب مفردة على خلفية رمادية. فرز تعلّم من نفس هذي الحبوب، فهي تورّي طريقة الشغل، مش دقته.",
}
NOTE_CBD = {
    "note": "Real light-box photo from India (CBD dataset, size-graded, no per-bean defect labels). Farz was not trained on beans like these; the implausible-sample rule (>60% of answered beans called defective) makes it answer 'not sure' instead of a wrong 'many defects'. See app/reports/rules_measure.json.",
    "note_en": "Real photo from India — Farz has not seen beans like these, so it says not sure.",
    "note_ar": "صورة حقيقية من الهند — فرز ما شاف حبوب مثل هذي قبل، فيقول «مش متأكد».",
}


def outcome(calls):
    d = pipeline.decide(calls)
    return {"band": d["band"], "about": d["about"], "reason": d["reason"], "answered": d["answered"], "defects": d["defects"],
            "p": round(d["p"], 4) if d["p"] == d["p"] else None, "lo": round(d["lo"], 4), "hi": round(d["hi"], 4)}


def main():
    rows = [r for r in csv.DictReader(open(CROPS_CSV)) if r["touching"] == "0"]
    for r in rows: r["img"] = np.asarray(Image.open(ROOT / r["path"]).convert("RGB"))
    pool = {c: [r for r in rows if r["cls"] == c] for c in CLASSES}
    model_path, labels = pipeline.deployed(); thr = labels["abstain_below"]
    fmt = "int8" if model_path.stem.endswith("int8") else "fp32"
    final = pipeline.session(model_path)
    folds = {f: pipeline.session(ROOT / f"models/farz_fold{f}_{fmt}.onnx") for f in (1, 2)}
    report = {"deployed_model": str(model_path.relative_to(ROOT)), "abstain_below": thr,
              "honesty": "SYNTHETIC trays: the deployed model saw these beans in training (in-sample). 'out_of_fold' scores each bean with the fold model that never saw it.",
              "rules": "pipeline.decide() = Python mirror of app/src/lib/rules.ts decide()", "new_trays": [], "all_trays": []}
    for fname, spec, seed, t_ar, t_en in NEW:
        rng = random.Random(seed)
        img, placed = make_demo.make_tray(pool, spec, rng)
        img.save(DEMO / fname, quality=90)
        img = Image.open(DEMO / fname)  # measure what ships (after JPEG)
        r, p_final, pred_final = pipeline.run_photo(final, img, thr)
        crops = [b.crop for b in r.beans]
        p_f = {f: pipeline.classify(s, crops) for f, s in folds.items()}
        m = make_demo.match(r.beans, placed)
        oof = []
        for i, j in enumerate(m):
            if j is None: oof.append("abstain"); continue
            pr = p_f[placed[j]["test_fold"]][i]
            oof.append(CLASSES[int(pr.argmax())] if pr.max() >= thr else "abstain")
        truth = [placed[j]["cls"] if j is not None else None for j in m]
        app_final = pipeline.app_calls(r, pred_final); app_oof = pipeline.app_calls(r, oof)
        def acc(pred):
            ans = [(t, q) for t, q in zip(truth, pred) if t is not None and q != "unsure"]
            return round(sum(t == q for t, q in ans) / max(1, len(ans)), 4)
        true_def = sum(spec.values())
        entry = {"file": fname, "seed": seed, "true_beans": 100, "true_defects": true_def, "true_counts": {"good": 100 - true_def, **spec},
                 "detected_beans": len(r.beans), "matched": sum(j is not None for j in m), "checks": r.checks,
                 "deployed_in_sample": {"counts": {k: app_final.count(k) for k in CLASSES + ["unsure"]}, "accuracy_answered": acc(app_final), "outcome": outcome(app_final)},
                 "out_of_fold_held_out": {"counts": {k: app_oof.count(k) for k in CLASSES + ["unsure"]}, "accuracy_answered": acc(app_oof), "outcome": outcome(app_oof)}}
        report["new_trays"].append(entry)
        print(fname, "detected", len(r.beans), "| deployed:", entry["deployed_in_sample"]["counts"], entry["deployed_in_sample"]["outcome"]["band"],
              "about" if entry["deployed_in_sample"]["outcome"]["about"] else "", "| oof:", entry["out_of_fold_held_out"]["counts"], entry["out_of_fold_held_out"]["outcome"])

    # manifest: keep the existing entries (made by make_demo.py), add the new ones, add honest user-facing notes and
    # the VERIFIED outcome for every tray (deployed model + app rules; EXIF orientation applied as the browser does)
    old = json.load(open(DEMO / "manifest.json"))
    old = [e for e in old if e["file"] not in {n[0] for n in NEW}]
    new_entries = []
    for fname, spec, seed, t_ar, t_en in NEW:
        e = next(x for x in report["new_trays"] if x["file"] == fname)
        new_entries.append({"file": fname, "title_ar": t_ar, "title_en": t_en, "synthetic": True, "credit": make_demo.J4CK_CREDIT,
                            "true_counts": e["true_counts"], **NOTE_SYN})
    order = ["tray_synthetic_00.jpg", "tray_synthetic_03.jpg", "tray_synthetic_12.jpg", "tray_synthetic_30.jpg", "tray_synthetic_40.jpg",
             "cbd_aaa_2.jpg", "cbd_bits_262.jpg"]
    entries = {e["file"]: e for e in old + new_entries}
    manifest = []
    for fname in order + sorted(set(entries) - set(order)):
        if fname not in entries: continue
        e = dict(entries[fname])
        im = ImageOps.exif_transpose(Image.open(DEMO / fname))
        r, p, pred = pipeline.run_photo(final, im, thr)
        calls = pipeline.app_calls(r, pred)
        out = outcome(calls)
        counts = {k: calls.count(k) for k in CLASSES + ["unsure"]}
        e["farz_python_counts"] = {**{k: counts[k] for k in CLASSES}, "abstain": counts["unsure"]}
        e["expected"] = {"band": out["band"], "about": out["about"], "reason": out["reason"],
                         "verified": "Python mirror of the app rules on the shipped JPEG, EXIF orientation applied, deployed model (src/beans/make_demo_extra.py); the app e2e test asserts the same band in Chromium"}
        if not e.get("synthetic"):
            e.update(NOTE_CBD)
            if out["band"] != "unsure":
                raise SystemExit(f"{fname}: real CBD photo gives {out} - remove it from the demo (task rule)")
        else:
            for k, v in NOTE_SYN.items(): e.setdefault(k, v)
        manifest.append(e)
        report["all_trays"].append({"file": fname, "counts": counts, "outcome": out})
        print("manifest", fname, counts, out)
    json.dump(manifest, open(DEMO / "manifest.json", "w"), indent=1, ensure_ascii=False)
    (ROOT / "app/reports").mkdir(exist_ok=True)
    json.dump(report, open(ROOT / "app/reports/demo_trays_extra.json", "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
