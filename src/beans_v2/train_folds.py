"""Leave-one-SOURCE-out (LOSO) and, inside J4ckDev, leave-one-PHOTO-out (LOPO) training of the v2 classifier.

  python train_folds.py loso [source ...]   -> for each held-out source: train on every other source, save
                                               CACHE/folds/loso_<source>.pt and the held-out logits
  python train_folds.py lopo [photo ...]    -> for each J4ckDev photo: train on all 11 other sources + the other 10
                                               J4ckDev photos, save CACHE/folds/lopo_<photo>.pt and that photo's logits
  python train_folds.py final               -> train on everything (deployed model weights)
  python train_folds.py licensed            -> train only on sources that state a licence (comparison)

Same recipe for every fold (train_common.train, fixed steps, no early stopping, nothing fitted on test data).
Folds whose outputs already exist are skipped, so the script can be re-run after an interruption.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
import train_common as tc

STEPS = 3000
OUT = tc.CACHE / "folds"
LICENSED = ["j4ckdev", "vicanadya16", "afiyah_deteksi", "afiyah_bijikopi", "loja_yolo", "lojano", "samruddh_grading"]


def run(name, train_idx, test_idx, meta, crops, seed=0):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"{name}.npz").exists():
        tc.log(f"skip {name} (exists)"); return
    tc.log(f"=== {name}: train {len(train_idx)} crops, test {len(test_idx)}")
    model, info = tc.train(meta, crops, train_idx, steps=STEPS, seed=seed, tag=name)
    torch.save(model.state_dict(), OUT / f"{name}.pt")
    lg = tc.logits_of(model, crops, test_idx) if len(test_idx) else np.zeros((0, len(tc.CLASSES)), np.float32)
    np.savez(OUT / f"{name}.npz", idx=np.asarray(test_idx), logits=lg)
    json.dump(info, open(OUT / f"{name}.json", "w"))
    tc.log(f"=== {name} done in {info['train_seconds']}s")


def main():
    mode = sys.argv[1]; args = sys.argv[2:]
    meta, crops = tc.load_cache()
    src = np.array([m["source"] for m in meta]); grp = np.array([m["group"] for m in meta])
    if mode == "loso":
        for s in (args or tc.SOURCES):
            run(f"loso_{s}", np.flatnonzero(src != s), np.flatnonzero(src == s), meta, crops)
    elif mode == "lopo":
        photos = sorted({g.split("/", 1)[1] for g in grp[src == "j4ckdev"]})
        for ph in (args or photos):
            test = np.flatnonzero(grp == f"j4ckdev/{ph}")
            run(f"lopo_{ph}", np.flatnonzero(grp != f"j4ckdev/{ph}"), test, meta, crops)
    elif mode == "halves":
        # v1's own spatial protocol inside J4ckDev (familiar set-up, unseen beans): fold k tests on the J4ckDev beans
        # with test_fold == k (data/crops/crops.csv); trains on all other sources + the other half of every J4ckDev photo
        import csv
        tf = [int(r["test_fold"]) for r in csv.DictReader(open(tc.ROOT / "data/crops/crops.csv"))]
        jidx = np.flatnonzero(src == "j4ckdev")  # cache rows 0..581 in crops.csv order
        assert len(jidx) == len(tf)
        for k in (args and [int(a) for a in args]) or (1, 2):
            test = jidx[np.array(tf) == k]
            run(f"half_{k}", np.setdiff1d(np.arange(len(meta)), test), test, meta, crops)
    elif mode == "final":
        run("final", np.arange(len(meta)), np.array([], int), meta, crops)
    elif mode == "licensed":
        keep = np.isin(src, LICENSED)
        run("licensed", np.flatnonzero(keep), np.flatnonzero(~keep), meta, crops)


if __name__ == "__main__":
    main()
