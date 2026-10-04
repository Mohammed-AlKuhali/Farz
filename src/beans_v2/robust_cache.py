"""Step 0 of the robust-signals study: run the FROZEN bean finder + v1 deployed model on every J4ckDev and CBD photo
once, and cache per-bean crops/features in the scratchpad (not in the repo).

Per photo : source, grade/class, pixel md5 of the decoded photo (dedupe), finder checks, colour gate (port), v1 verdict.
Per bean  : crop (128 px uint8), touching, area, v1 probs / raw logits / 576-d v1 features, 16 hand features,
            normalised-crop hand features, crop pixel md5.
Then (main process) 576-d frozen-ImageNet MobileNetV3-Small embeddings of raw and exposure-normalised crops.
"""
import glob, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
import robust_common as rc

_S = None
def _init():
    global _S
    _S = rc.v1_session(threads=1)

def _photo(args):
    src, f, label = args
    img = Image.open(f); img.load()
    md5 = rc.pixel_md5(np.asarray(img.convert("RGB")))
    r = rc.find_beans(img)
    cg = rc.colour_gate(r)
    crops = [b.crop for b in r.beans]
    probs, feat, logits = rc.v1_run(_S, crops)
    hand = np.stack([rc.hand_features(c) for c in crops]) if crops else np.zeros((0, len(rc.HAND_NAMES)), np.float32)
    ncrops = [rc.normalise_crop(c) for c in crops]
    hand_n = np.stack([rc.hand_features(c) for c in ncrops]) if crops else np.zeros((0, len(rc.HAND_NAMES)), np.float32)
    beans = dict(touching=np.array([b.touching for b in r.beans], bool), area=np.array([b.area for b in r.beans], np.float32),
                 side=np.array([max(b.x1 - b.x0, b.y1 - b.y0) for b in r.beans], np.float32),
                 cx=np.array([(b.x0 + b.x1) / 2 / r.work.shape[1] for b in r.beans], np.float32),
                 crop_md5=[rc.pixel_md5(c) for c in crops])
    return dict(src=src, file=rc.rel(f), label=label, md5=md5, checks=r.checks, cg=cg, median_area=r.median_area,
                work_hw=list(r.work.shape[:2]), crops=np.stack(crops) if crops else np.zeros((0, 128, 128, 3), np.uint8),
                ncrops=np.stack(ncrops) if ncrops else np.zeros((0, 128, 128, 3), np.uint8),
                probs=probs, feat=feat, logits=logits, hand=hand, hand_n=hand_n, **beans)

def main():
    t0 = time.time()
    jobs = [("j4ck", str(rc.J4CK_DIR / f"{stem}.jpg"), stem) for stem in sorted(rc.PHOTO_CLASS)]
    jobs += [("cbd", f, Path(f).parent.name) for f in sorted(glob.glob(str(rc.CBD_DIR / "**/*.jpg"), recursive=True))]
    with ProcessPoolExecutor(max_workers=16, initializer=_init) as ex:
        res = list(ex.map(_photo, jobs, chunksize=2))
    print(f"finder+v1 on {len(res)} photos in {time.time()-t0:.0f}s", flush=True)
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = rc.imagenet_extractor().to(dev)
    allc = np.concatenate([r["crops"] for r in res]); alln = np.concatenate([r["ncrops"] for r in res])
    emb = rc.embed(m, allc, device=dev); emb_n = rc.embed(m, alln, device=dev)
    print(f"embeddings {emb.shape} in {time.time()-t0:.0f}s", flush=True)
    # J4ckDev crops in data/crops must be identical to what the finder gives now (same frozen code)
    import csv
    rows = list(csv.DictReader(open(rc.CROPS_CSV)))
    disk = {(r["photo"], int(r["idx"])): rc.pixel_md5(np.asarray(Image.open(rc.ROOT / r["path"]).convert("RGB"))) for r in rows}
    same = sum(disk.get((r["label"], i)) == h for r in res if r["src"] == "j4ck" for i, h in enumerate(r["crop_md5"]))
    nb = np.array([len(r["crops"]) for r in res]); off = np.concatenate([[0], np.cumsum(nb)])
    rc.CACHE.mkdir(parents=True, exist_ok=True)
    np.save(rc.CACHE / "crops.npy", allc); np.save(rc.CACHE / "ncrops.npy", alln)
    np.savez(rc.CACHE / "beans.npz", photo_idx=np.repeat(np.arange(len(res)), nb), emb=emb, emb_n=emb_n,
             probs=np.concatenate([r["probs"] for r in res]), feat=np.concatenate([r["feat"] for r in res]),
             logits=np.concatenate([r["logits"] for r in res]), hand=np.concatenate([r["hand"] for r in res]),
             hand_n=np.concatenate([r["hand_n"] for r in res]), touching=np.concatenate([r["touching"] for r in res]),
             area=np.concatenate([r["area"] for r in res]), side=np.concatenate([r["side"] for r in res]),
             cx=np.concatenate([r["cx"] for r in res]), crop_md5=np.array(sum([r["crop_md5"] for r in res], [])), offsets=off)
    meta = [dict(src=r["src"], file=r["file"], label=r["label"], md5=r["md5"], checks=r["checks"], cg=r["cg"],
                 median_area=r["median_area"], work_hw=r["work_hw"], n=len(r["crops"])) for r in res]
    json.dump(dict(photos=meta, j4ck_crops_identical_to_data_crops=f"{same}/{len(rows)}"), open(rc.CACHE / "photos.json", "w"), default=float)
    print(f"cached {off[-1]} beans; data/crops identical {same}/{len(rows)}; {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
