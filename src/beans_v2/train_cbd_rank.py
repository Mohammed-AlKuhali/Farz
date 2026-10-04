"""Does the per-photo defect share follow the CBD grade folders? (CBD never used for training or fitting anything.)
Per photo: share of beans whose argmax is not 'good' (no abstention). v2 = models_v2 fp16 (deployed file) on the robust
cache crops (frozen bean finder); v1 = reports/beans_ood_cbd_preds.npz (v1's own saved probabilities, read-only).
Grade names are folder labels only: the CBD page does not define them, so the 'order' below is the folder order used by
robust_common.CBD_GRADES, not an official quality scale. Writes reports_v2/model_v2_cbd_rank.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import train_common as tc
from train_export import ort_probs

G = ["AAA", "AA", "A", "AB", "PB-I", "PB-II", "C", "Bulk", "Bits"]


def stats(lab, sh):
    out = {"median_photo_defect_share": {g: float(np.median(sh[lab == g])) for g in G}}
    for a, b in (("AAA", "Bits"), ("AAA", "C"), ("AAA", "PB-I"), ("AA", "Bits")):
        m = np.isin(lab, [a, b]); out[f"auroc_{b}_above_{a}"] = float(roc_auc_score(lab[m] == b, sh[m]))
    r = spearmanr([G.index(l) for l in lab], sh); out["spearman_vs_folder_order"] = {"rho": float(r.statistic), "p": float(r.pvalue), "n_photos": int(len(lab))}
    return out


def main():
    R = tc.SCRATCH / "robust_cache"
    pm = json.load(open(R / "photos.json"))["photos"]; z = np.load(R / "beans.npz"); pi = z["photo_idx"]
    crops = np.load(R / "crops.npy", mmap_mode="r")
    model = tc.ROOT / json.load(open(tc.REPORTS_V2 / "model_v2_export.json"))["recommended_file"]
    cb = np.flatnonzero(np.array([pm[i]["src"] for i in pi]) == "cbd")
    P = np.zeros((len(pi), 6), np.float32); P[cb] = ort_probs(model, crops[cb])
    ids = [i for i, m in enumerate(pm) if m["src"] == "cbd"]
    lab2 = np.array([pm[i]["label"] for i in ids]); sh2 = np.array([float((P[pi == i].argmax(1) != 0).mean()) for i in ids])
    v = np.load(tc.ROOT / "reports/beans_ood_cbd_preds.npz", allow_pickle=True)
    keys = sorted(set(zip(v["grade"], v["photo"])))
    lab1 = np.array([k[0] for k in keys]); sh1 = np.array([float((v["probs"][(v["grade"] == k[0]) & (v["photo"] == k[1])].argmax(1) != 0).mean()) for k in keys])
    res = {"protocol": __doc__.strip(), "v2": {"model": tc.rel(model), **stats(lab2, sh2)}, "v1": {"model": "models/farz_beans_fp32.onnx (saved probs)", **stats(lab1, sh1)}}
    json.dump(res, open(tc.REPORTS_V2 / "model_v2_cbd_rank.json", "w"), indent=1)
    tc.log("cbd rank:", {k: (res["v2"][k], res["v1"][k]) for k in ("auroc_Bits_above_AAA", "spearman_vs_folder_order")})


if __name__ == "__main__":
    main()
