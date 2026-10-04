"""Step 6: out-of-distribution sanity check on CBD (Mendeley 52877z55vr, CC BY 4.0; Wayanad, India; light box;
SIZE-graded, NOT defect-labelled). Full pipeline (frozen bean finder -> deployed model -> abstain rule) on every photo.

There is no per-bean ground truth here. What it can show: whether the model calls ordinary graded beans 'good', and
whether the 'Bits' (broken pieces) grade shows more 'broken'. We report what we measure, per grade folder.
Writes reports/beans_ood_cbd.json and merges the table into reports/beans_model.json under "cbd_ood".
"""
import glob, json, sys, collections
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
from common import ROOT, CLASSES, CBD_DIR
import pipeline

_S = None; _T = None
def _init(model_path, thr):
    global _S, _T
    _S = pipeline.session(model_path, threads=1); _T = thr

def _run(f):
    r, p, pred = pipeline.run_photo(_S, Image.open(f), _T)
    return f, r.checks["count"], r.checks["touching"], p.astype(np.float32), pred

def main():
    model_path, labels = pipeline.deployed()
    thr = labels["abstain_below"]
    files = sorted(glob.glob(str(CBD_DIR / "**/*.jpg"), recursive=True))
    with ProcessPoolExecutor(initializer=_init, initargs=(str(model_path), thr)) as ex:
        res = list(ex.map(_run, files, chunksize=4))
    by = collections.defaultdict(lambda: {"photos": 0, "beans": 0, "pred": collections.Counter(), "argmax": collections.Counter(), "conf": []})
    for f, n, t, p, pred in res:
        g = by[Path(f).parent.name]; g["photos"] += 1; g["beans"] += n
        g["pred"].update(pred); g["argmax"].update(CLASSES[i] for i in p.argmax(1)); g["conf"] += p.max(1).tolist()
    order = ["AAA", "AA", "A", "AB", "PB-I", "PB-II", "C", "Bulk", "Bits"]
    table = {}
    for gname in order + sorted(set(by) - set(order)):
        if gname not in by: continue
        g = by[gname]; n = max(g["beans"], 1)
        table[gname] = {"photos": g["photos"], "beans": g["beans"],
                        "share_with_abstain": {k: round(g["pred"][k] / n, 4) for k in CLASSES + ["abstain"]},
                        "share_argmax_no_abstain": {k: round(g["argmax"][k] / n, 4) for k in CLASSES},
                        "defect_share_of_answered": round(sum(g["pred"][k] for k in CLASSES[1:]) / max(1, n - g["pred"]["abstain"]), 4),
                        "median_max_prob": round(float(np.median(g["conf"])), 4) if g["conf"] else None}
    allp = np.concatenate([p for _, _, _, p, _ in res])
    out = {"dataset": "CBD Coffee Bean Dataset, Mendeley Data 52877z55vr (2024), CC BY 4.0, https://data.mendeley.com/datasets/52877z55vr/1",
           "what_it_is": "OOD sanity check only: India, light box, size grades, NO per-bean defect labels. Not an accuracy measurement.",
           "model": str(model_path.relative_to(ROOT)), "abstain_below": thr, "photos": len(res), "beans": int(len(allp)),
           "overall_share_with_abstain": {k: round(sum(g["pred"][k] for g in by.values()) / len(allp), 4) for k in CLASSES + ["abstain"]},
           "overall_median_max_prob": round(float(np.median(allp.max(1))), 4),
           "per_grade": table}
    # photo-level fail-safe from the plan: "more than 15% of beans low-confidence -> not sure, take it to the cooperative"
    held = np.load(ROOT / "reports/beans_heldout_preds.npz"); rep0 = json.load(open(ROOT / "reports/beans_model.json"))
    hp = held[rep0["deploy_decision"]["deployed"]]; hy = held["y"]
    per_photo = [(Path(f).parent.name, p.max(1)) for f, _, _, p, _ in res if len(p)]
    import csv
    from common import CROPS_CSV
    photo_of = [r["photo"] for r in csv.DictReader(open(CROPS_CSV))]
    halves = {}
    for i, fd, c in zip(held["idx"], held["fold"], hp.max(1)): halves.setdefault((photo_of[i], int(fd)), []).append(c)
    halves = [np.array(v) for v in halves.values()]
    sweep = []
    for t in (thr, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        m = hp.max(1) >= t
        sweep.append({"abstain_below": t, "j4ck_heldout_coverage": round(float(m.mean()), 4),
                      "j4ck_heldout_accuracy_answered": round(float((hp[m].argmax(1) == hy[m]).mean()), 4) if m.any() else None,
                      "cbd_bean_abstain_share": round(float(np.mean(np.concatenate([c for _, c in per_photo]) < t)), 4),
                      "cbd_photos_where_15pct_rule_fires": round(float(np.mean([np.mean(c < t) > 0.15 for _, c in per_photo])), 4),
                      "j4ck_heldout_half_photos_where_15pct_rule_fires": round(float(np.mean([np.mean(c < t) > 0.15 for c in halves])), 4),
                      "j4ck_heldout_half_photos": len(halves)})
    out["photo_level_rule"] = {"rule": "photo answered 'not sure' when > 15% of its beans are below abstain_below (PLAN s.5)",
                               "at_deployed_threshold": sweep[0], "threshold_sweep": sweep,
                               "note": "the CBD columns are OOD behaviour only; CBD has no defect labels"}
    np.savez(ROOT / "reports/beans_ood_cbd_preds.npz", grade=np.array([Path(f).parent.name for f, _, _, p, _ in res for _ in range(len(p))]),
             photo=np.array([Path(f).stem for f, _, _, p, _ in res for _ in range(len(p))]), probs=allp)
    json.dump(out, open(ROOT / "reports/beans_ood_cbd.json", "w"), indent=1)
    rep_path = ROOT / "reports/beans_model.json"; rep = json.load(open(rep_path)); rep["cbd_ood"] = out
    json.dump(rep, open(rep_path, "w"), indent=1)
    print(f"{'grade':6s} {'photos':>6s} {'beans':>6s} " + " ".join(f"{k:>8s}" for k in CLASSES + ["abstain"]) + "  medconf")
    for gname, t in table.items():
        print(f"{gname:6s} {t['photos']:6d} {t['beans']:6d} " + " ".join(f"{t['share_with_abstain'][k]:8.3f}" for k in CLASSES + ["abstain"]) + f"  {t['median_max_prob']:.3f}")
    print("overall", out["overall_share_with_abstain"], "median max prob", out["overall_median_max_prob"])
    for row in out["photo_level_rule"]["threshold_sweep"]: print(row)

if __name__ == "__main__":
    main()
