"""(2) UNUSUAL-BEAN DETECTOR: relative, per photo, no labels.

Each bean is scored ONLY against the other beans of the same photo:
  hand : 16 colour/shape stats -> per-photo robust z per stat (median / 1.4826*MAD, MAD floored at 0.1 x the stat's
         global SD, clipped to +-8) -> root-mean-square over stats
  emb  : 576-d frozen ImageNet MobileNetV3-Small embedding, L2-normalised -> mean cosine distance to the k=5 nearest
         OTHER beans of the photo (leave-one-out kNN)
  both : mean of the two final z-scores
  v1f  : same kNN on the v1 model's own 576-d pooled features (already shipped, 0 extra bytes; but its backbone was
         fine-tuned on these J4ckDev crops, so synthetic J4ckDev trays are in-sample for it)
Final score = per-photo robust z of the raw score; a bean is FLAGGED when z > 3.5 (Iglewicz-Hoaglin modified-z cut-off,
fixed a priori, not tuned on any result below).

(a) synthetic trays (crop level; 300 trays per condition, 50 base beans + k=3 inserted beans, seed 0), each run on the
    raw crops and on exposure-normalised crops (robust_common.normalise_crop: per-crop background white balance +
    common 40 px effective resolution, applied to EVERY bean of the tray):
      A  base J4ckDev good (Normales)  + J4ckDev DEFECT beans (other photos)         -> defects from different photos
      B  base J4ckDev good             + CBD AAA beans (good, other camera/exposure)  -> CONTROL
      B2 base J4ckDev good             + J4ckDev good with a synthetic exposure shift -> CONTROL
      C  base = one real CBD AAA photo + J4ckDev DEFECT beans                        -> defects from different photos
      D  base = one real CBD AAA photo + J4ckDev good beans                          -> CONTROL
      E  base = one real CBD AAA photo + CBD 'Bits' beans from other photos (same light box, photo-level label only)
      F  base = one real CBD AAA photo + CBD AAA beans from OTHER AAA photos         -> CONTROL for E
(b) real CBD photos: share of beans flagged per grade (+ a variant with relative bean size area/photo-median added);
    real J4ckDev photos (each is ONE class) for the uniform-tray limitation.
(c) ONNX export of the frozen extractor: size, parity, CPU latency (Python ORT and onnxruntime-web wasm in Node).
Writes reports_v2/robust_unusual.json, robust_unusual.png.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import robust_common as rc

Z_CUT = 3.5
K = 5

def load():
    meta = json.load(open(rc.CACHE / "photos.json"))["photos"]
    z = dict(np.load(rc.CACHE / "beans.npz"))
    z["crops"] = np.load(rc.CACHE / "crops.npy", mmap_mode="r"); z["ncrops"] = np.load(rc.CACHE / "ncrops.npy", mmap_mode="r")
    return meta, z

def robust_z(x, floor=1e-9):
    med = np.median(x, 0); sc = np.maximum(1.4826 * np.median(np.abs(x - med), 0), floor)
    return (x - med) / sc

def score_hand(H, floors):
    zz = np.clip(robust_z(H, floors), -8, 8)
    return np.sqrt(np.nanmean(zz ** 2, 1))

def score_knn(E, k=K):
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    D = 1 - E @ E.T; np.fill_diagonal(D, np.inf)
    k = min(k, len(E) - 1)
    return np.sort(D, 1)[:, :k].mean(1)

def final_z(s):
    med = np.median(s); sc = max(1.4826 * np.median(np.abs(s - med)), 1e-6 * max(abs(med), 1e-9))
    return (s - med) / sc

def detect(H, E, floors, Fv=None):
    out = {"hand": final_z(score_hand(H, floors)), "emb": final_z(score_knn(E))}
    out["both"] = 0.5 * (out["hand"] + out["emb"])
    if Fv is not None: out["v1f"] = final_z(score_knn(Fv))
    return out

def auroc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg])); n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def exposure_shift(crop):
    """Synthetic 'same beans, different exposure': darker, warmer, slightly contrastier (gain + gamma), applied to the
    whole crop (bean AND sheet), as a different camera exposure / light would."""
    c = crop.astype(np.float32) / 255.0
    c = np.clip((c * np.array([0.95, 0.86, 0.72])) ** 1.15, 0, 1)
    return (c * 255).astype(np.uint8)

def main():
    t0 = time.time()
    meta, z = load()
    pi = z["photo_idx"]; src = np.array([m["src"] for m in meta]); lab = np.array([m["label"] for m in meta])
    bsrc, blab = src[pi], lab[pi]
    floors_raw = 0.1 * np.nanstd(z["hand"], 0); floors_n = 0.1 * np.nanstd(z["hand_n"], 0)
    # pools
    j_good = np.flatnonzero((bsrc == "j4ck") & (blab == "Normales"))
    j_def_by_photo = {p: np.flatnonzero((bsrc == "j4ck") & (blab == p)) for p in sorted(rc.PHOTO_CLASS) if p != "Normales"}
    aaa_photos = [i for i, m in enumerate(meta) if m["src"] == "cbd" and m["label"] == "AAA"]
    bits_photos = [i for i, m in enumerate(meta) if m["src"] == "cbd" and m["label"] == "Bits"]
    beans_of = {i: np.flatnonzero(pi == i) for i in range(len(meta))}
    # synthetic exposure-shifted copies of J4ckDev good beans (embeddings + hand stats computed here)
    ext = rc.imagenet_extractor()
    sh_raw = np.stack([exposure_shift(z["crops"][i]) for i in j_good]); sh_n = np.stack([rc.normalise_crop(c) for c in sh_raw])
    shift = dict(raw=dict(H=np.stack([rc.hand_features(c) for c in sh_raw]), E=rc.embed(ext, sh_raw)),
                 norm=dict(H=np.stack([rc.hand_features(c) for c in sh_n]), E=rc.embed(ext, sh_n)))
    v1s = rc.v1_session(threads=4)
    _, shift["raw"]["F"], _ = rc.v1_run(v1s, list(sh_raw)); _, shift["norm"]["F"], _ = rc.v1_run(v1s, list(sh_n))
    # v1 features of normalised crops (raw ones are cached)
    feat_n = rc.v1_run(v1s, list(z["ncrops"]))[1]
    print(f"prep {time.time()-t0:.0f}s", flush=True)

    def feats(idx, variant):
        if variant == "raw": return z["hand"][idx], z["emb"][idx], z["feat"][idx]
        return z["hand_n"][idx], z["emb_n"][idx], feat_n[idx]

    rng = np.random.default_rng(0)
    N_TRAYS, N_BASE, N_ODD = 300, 50, 3
    conds = ["A", "B", "B2", "C", "D", "E", "F"]
    results = {}
    for cond in conds:
        for variant in ["raw", "norm"]:
            floors = floors_raw if variant == "raw" else floors_n
            per = {m: dict(auc=[], rec=[], ffr=[], prec_num=0, prec_den=0) for m in ["hand", "emb", "both", "v1f"]}
            r2 = np.random.default_rng(1000 + conds.index(cond))
            for t in range(N_TRAYS):
                if cond in "AB" or cond == "B2":
                    base = r2.choice(j_good, N_BASE, replace=False)
                else:
                    bp = aaa_photos[r2.integers(len(aaa_photos))]; base = beans_of[bp]
                Hb, Eb, Fb = feats(base, variant)
                if cond in ("A", "C"):
                    ph = r2.choice(list(j_def_by_photo), N_ODD, replace=True)
                    odd = np.array([r2.choice(j_def_by_photo[p]) for p in ph]); Ho, Eo, Fo = feats(odd, variant)
                elif cond == "B":
                    op = r2.choice(aaa_photos, N_ODD); odd = np.array([r2.choice(beans_of[p]) for p in op]); Ho, Eo, Fo = feats(odd, variant)
                elif cond == "B2":
                    # shifted copies of good beans NOT in this tray's base
                    rest = np.setdiff1d(np.arange(len(j_good)), np.searchsorted(j_good, base))
                    o = r2.choice(rest, N_ODD, replace=False); S = shift[variant]; Ho, Eo, Fo = S["H"][o], S["E"][o], S["F"][o]
                elif cond == "D":
                    odd = r2.choice(j_good, N_ODD, replace=False); Ho, Eo, Fo = feats(odd, variant)
                elif cond == "E":
                    op = r2.choice(bits_photos, N_ODD); odd = np.array([r2.choice(beans_of[p]) for p in op]); Ho, Eo, Fo = feats(odd, variant)
                elif cond == "F":
                    others = [p for p in aaa_photos if p != bp]
                    op = r2.choice(others, N_ODD); odd = np.array([r2.choice(beans_of[p]) for p in op]); Ho, Eo, Fo = feats(odd, variant)
                H = np.concatenate([Hb, Ho]); E = np.concatenate([Eb, Eo]); F = np.concatenate([Fb, Fo])
                y = np.r_[np.zeros(len(Hb), bool), np.ones(len(Ho), bool)]
                sc = detect(H, E, floors, F)
                for m, s in sc.items():
                    P = per[m]; P["auc"].append(auroc(s[y], s[~y])); fl = s > Z_CUT
                    P["rec"].append(fl[y].mean()); P["ffr"].append(fl[~y].mean()); P["prec_num"] += fl[y].sum(); P["prec_den"] += fl.sum()
            results[f"{cond}/{variant}"] = {m: dict(mean_auroc=round(float(np.mean(P["auc"])), 4), recall_inserted=round(float(np.mean(P["rec"])), 4),
                                                    false_flag_rate_base=round(float(np.mean(P["ffr"])), 4),
                                                    precision=round(float(P["prec_num"] / max(P["prec_den"], 1)), 4)) for m, P in per.items()}
            print(cond, variant, {m: (v["mean_auroc"], v["recall_inserted"], v["false_flag_rate_base"]) for m, v in results[f"{cond}/{variant}"].items()}, flush=True)

    # (b) real photos
    real = {}
    rel_size = np.zeros(len(pi), np.float32)
    for i, m in enumerate(meta):
        b = beans_of[i]
        if len(b): rel_size[b] = np.log(z["area"][b] / np.median(z["area"][b]))
    per_photo = []
    for i, m in enumerate(meta):
        b = beans_of[i]
        if len(b) < 10: continue
        out = {}
        for variant in ["raw", "norm"]:
            H, E, F = feats(b, variant); floors = floors_raw if variant == "raw" else floors_n
            sc = detect(H, E, floors, F)
            Hs = np.concatenate([H, rel_size[b, None]], 1)
            sc["hand+size"] = final_z(score_hand(Hs, np.r_[floors, 0.05]))
            sc["both+size"] = 0.5 * (sc["hand+size"] + sc["emb"])
            for k2, s in sc.items(): out[f"{variant}/{k2}"] = s
        per_photo.append((i, out))
    keys = list(per_photo[0][1])
    for grp in rc.CBD_GRADES + ["J4CK:" + p for p in sorted(rc.PHOTO_CLASS)]:
        sel = [(i, o) for i, o in per_photo if (meta[i]["src"] == "cbd" and meta[i]["label"] == grp) or (grp.startswith("J4CK:") and meta[i]["src"] == "j4ck" and meta[i]["label"] == grp[5:])]
        if not sel: continue
        real[grp] = {"photos": len(sel), "beans": int(sum(len(o[keys[0]]) for _, o in sel))}
        for k2 in keys:
            fl = np.concatenate([o[k2] > Z_CUT for _, o in sel])
            real[grp][k2] = {"share_flagged": round(float(fl.mean()), 4), "mean_flags_per_photo": round(float(np.mean([np.sum(o[k2] > Z_CUT) for _, o in sel])), 3)}
    # Spearman between grade order (AAA..Bits) and flag share, per key
    from scipy.stats import spearmanr
    trend = {}
    for k2 in keys:
        ph_share = [(rc.CBD_GRADES.index(meta[i]["label"]), float(np.mean(o[k2] > Z_CUT))) for i, o in per_photo if meta[i]["src"] == "cbd"]
        a = np.array(ph_share); rho, pv = spearmanr(a[:, 0], a[:, 1]); trend[k2] = dict(spearman_rho_grade_order_vs_photo_flag_share=round(float(rho), 4), p=float(pv))
    # store per-bean z of the chosen variants for downstream (OOD gate / figures)
    np.savez(rc.CACHE / "unusual_scores.npz", photo=np.array([i for i, o in per_photo for _ in o[keys[0]]]),
             **{k2.replace("/", "__"): np.concatenate([o[k2] for _, o in per_photo]) for k2 in keys})
    res = {"what": "relative per-photo unusual-bean detector (no labels)", "z_cut": Z_CUT, "knn_k": K,
           "synthetic": {"n_trays": N_TRAYS, "base_beans": N_BASE, "inserted_beans": N_ODD, "results": results,
                         "conditions": {"A": "J4ck good + J4ck defects (other photos)", "B": "CONTROL J4ck good + CBD AAA (good, other camera)",
                                        "B2": "CONTROL J4ck good + J4ck good with synthetic exposure shift",
                                        "C": "real CBD AAA photo + J4ck defects", "D": "CONTROL real CBD AAA photo + J4ck good",
                                        "E": "real CBD AAA photo + CBD Bits beans (same light box)", "F": "CONTROL real CBD AAA photo + CBD AAA beans from other AAA photos"}},
           "real_photos": real, "cbd_grade_trend": trend}
    rc.dump(res, "robust_unusual.json")
    print(json.dumps(trend, indent=0))
    for g, v in real.items(): print(g, v["photos"], v["beans"], {k2: v[k2]["share_flagged"] for k2 in ["raw/both", "raw/emb", "raw/hand", "raw/hand+size", "raw/both+size", "raw/v1f"]})
    print(f"done {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
