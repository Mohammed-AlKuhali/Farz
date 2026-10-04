"""LICENCE-STATED ("clean") v2 classifier: fold training for the honest evaluation of the shipped clean model.

The shipped clean model is trained ONLY on the 7 sources that state a licence (train_folds.LICENSED):
  CC0: afiyah_deteksi, afiyah_bijikopi, vicanadya16 | CC BY 4.0: loja_yolo | MIT: samruddh_grading |
  non-commercial: lojano (CC BY-NC 4.0), j4ckdev (CC BY-NC-SA 4.0)  => non-commercial, ShareAlike research prototype.
The 5 sources with no licence (mfu17, usk_coffee, daffa_defect, mindforge_doubleside, notplying_defects) are NEVER in any
training pool here; they are held-out TEST data only.

Its weights are the existing CACHE/folds/licensed.pt (train_folds.py licensed: same recipe train_common.train, 3,000 steps,
seed 0, training pool = exactly the 7 licensed sources; verified from licensed.json sampler_pool_sizes). This script trains
the fold models that evaluate it, all with the same recipe and seed:
  clean_loso_<group>  leave-one-source-out over the licensed sources; afiyah_deteksi + afiyah_bijikopi are ONE group
                      ("afiyah": same uploader, phone and backdrop; AUDIT_V2 finding L1), so they are always held out together
  clean_half_<k>      J4ckDev spatial half k held out (v1's protocol; used by the auditor's 1,200-tray J4ckDev test)
  clean_lopo_<photo>  one J4ckDev photo held out (11 folds)
Each fold saves CACHE/folds/<name>.pt, <name>.npz (held-out row indices + logits) and <name>.json.

Several workers may run at once: a fold is claimed by atomically creating <name>.lock, so workers never collide.
  python clean_folds.py worker          (run 2-3 of these in parallel)
"""
import csv, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
import train_common as tc
from train_folds import LICENSED, STEPS

OUT = tc.CACHE / "folds"
GROUPS = {"loja_yolo": ["loja_yolo"], "lojano": ["lojano"], "afiyah": ["afiyah_deteksi", "afiyah_bijikopi"],
          "j4ckdev": ["j4ckdev"], "samruddh_grading": ["samruddh_grading"], "vicanadya16": ["vicanadya16"]}
J4CK_PHOTOS = ["BrocadoLeve", "BrocadoSevero", "CerezaSeca", "Concha", "DXHongo", "Inmaduro", "MarronAVinagre", "Negros",
               "Normales", "PMordidoCortado", "Pergamino"]
# priority: loja_yolo first (local-calibration study), then the other LOSO folds (thresholds), halves, LOPO
ORDER = [f"clean_loso_{g}" for g in GROUPS] + ["clean_half_1", "clean_half_2"] + [f"clean_lopo_{p}" for p in J4CK_PHOTOS]


def split(name, meta):
    src = np.array([m["source"] for m in meta]); grp = np.array([m["group"] for m in meta])
    lic = np.isin(src, LICENSED)
    if name.startswith("clean_loso_"):
        held = np.isin(src, GROUPS[name[len("clean_loso_"):]])
    elif name.startswith("clean_half_"):
        k = int(name[-1])
        tf = np.array([int(r["test_fold"]) for r in csv.DictReader(open(tc.ROOT / "data/crops/crops.csv"))])
        jidx = np.flatnonzero(src == "j4ckdev"); assert len(jidx) == len(tf)
        held = np.zeros(len(meta), bool); held[jidx[tf == k]] = True
    elif name.startswith("clean_lopo_"):
        held = grp == f"j4ckdev/{name[len('clean_lopo_'):]}"
    else:
        raise ValueError(name)
    train_idx = np.flatnonzero(lic & ~held); test_idx = np.flatnonzero(held)
    assert not np.isin(src[train_idx], [s for s in tc.SOURCES if s not in LICENSED]).any()
    return train_idx, test_idx


def run(name, meta, crops):
    if (OUT / f"{name}.npz").exists(): return False
    try:
        fd = os.open(OUT / f"{name}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        return False
    tr, te = split(name, meta)
    tc.log(f"=== {name}: train {len(tr)} crops ({sorted(set(m['source'] for m in (meta[i] for i in tr[::50])))}), test {len(te)}")
    model, info = tc.train(meta, crops, tr, steps=STEPS, seed=0, tag=name)
    torch.save(model.state_dict(), OUT / f"{name}.pt")
    lg = tc.logits_of(model, crops, te)
    np.savez(OUT / f"{name}.npz", idx=te, logits=lg)
    info["train_sources"] = sorted({meta[i]["source"] for i in tr}); info["n_train"] = int(len(tr)); info["n_test"] = int(len(te))
    json.dump(info, open(OUT / f"{name}.json", "w"))
    tc.log(f"=== {name} done in {info['train_seconds']}s")
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, crops = tc.load_cache(in_memory=False)  # memmap: page cache shared between parallel workers
    names = sys.argv[2:] or ORDER
    for n in names: run(n, meta, crops)


if __name__ == "__main__":
    if sys.argv[1] == "worker": main()
