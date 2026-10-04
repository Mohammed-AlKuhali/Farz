"""Extra evidence: LEAVE-ONE-PHOTO-OUT test for the defect photos (cross-photo generalisation).

The spatial folds keep train and test beans in the same photo (upper bound). Four of the five Farz classes are built
from 2-3 J4ckDev photos, so for each of those 10 photos we train on every OTHER photo (same recipe, seed 0) and test
on all beans of the held-out photo: does the model still name the right Farz class for a photo it never saw?
'good' (Normales) is a single photo, so it can never be left out - that gap is reported, not hidden.
Writes reports/beans_lopo.json and merges it into reports/beans_model.json under "leave_one_photo_out".
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch
import train_beans as tb
from common import ROOT, CLASSES, PHOTO_CLASS

def main():
    rows = tb.load_rows()
    out = {"what": "train on all other J4ckDev photos, test on every bean of the held-out photo; PyTorch, uncalibrated argmax, no abstention",
           "recipe": f"same as train_beans.py (epochs {tb.EPOCHS}, seed {tb.SEED})", "photos": {}}
    for photo, cls in sorted(PHOTO_CLASS.items()):
        if sum(1 for p, c in PHOTO_CLASS.items() if c == cls) < 2:
            out["photos"][photo] = {"class": cls, "skipped": "only photo of its class - cannot be left out"}; continue
        test = [r for r in rows if r["photo"] == photo]; train = [r for r in rows if r["photo"] != photo]
        m = tb.train(train, f"lopo-{photo}")
        p = torch.softmax(tb.logits_of(m, [r["img"] for r in test]), 1).numpy(); y = CLASSES.index(cls)
        dist = {c: round(float((p.argmax(1) == i).mean()), 4) for i, c in enumerate(CLASSES)}
        out["photos"][photo] = {"class": cls, "n": len(test), "recall_correct_class": dist[cls], "predicted_share": dist,
                                "median_max_prob_uncalibrated": round(float(np.median(p.max(1))), 4)}
        tb.log(f"LOPO {photo:16s} ({cls:8s}) n={len(test):3d} correct {dist[cls]:.3f}  {dist}")
    done = [v for v in out["photos"].values() if "n" in v]
    out["bean_weighted_accuracy"] = round(sum(v["n"] * v["recall_correct_class"] for v in done) / sum(v["n"] for v in done), 4)
    out["photo_mean_accuracy"] = round(float(np.mean([v["recall_correct_class"] for v in done])), 4)
    json.dump(out, open(ROOT / "reports/beans_lopo.json", "w"), indent=1)
    rep_path = ROOT / "reports/beans_model.json"; rep = json.load(open(rep_path)); rep["leave_one_photo_out"] = out
    json.dump(rep, open(rep_path, "w"), indent=1)
    print("bean-weighted", out["bean_weighted_accuracy"], "photo-mean", out["photo_mean_accuracy"])

if __name__ == "__main__":
    main()
