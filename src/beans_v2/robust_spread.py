"""(2b, follow-up) PHOTO-LEVEL SPREAD: how mixed is the sample, as a whole?

The relative detector (robust_unusual.py) scores beans against their own photo, so a photo where MANY beans are
different (CBD 'Bits' photos are visibly mixed: black, brown, pale beans) raises the photo's own MAD and nothing stands
out. This script measures the complementary photo-level signal on CBD, where all photos share one light box, so absolute
spreads are comparable: per photo (touching blobs excluded),
  luma_iqr   IQR across beans of the bean median luminance (white-balanced)
  b_iqr      IQR across beans of Lab b* (yellow-blue)
  dark_rel   share of beans whose median luminance is > 25 levels below the photo median (relative 'dark' rule)
  emb_spread median over beans of the mean cosine distance to the 5 nearest other beans (frozen ImageNet embedding)
and reports per-grade medians, Spearman vs the grade order AAA..Bits, and AUROC 'AAA vs X' per grade.
No labels per bean exist; the grade is a photo-level label from the dataset folders.
Writes reports_v2/robust_spread.json and robust_spread.png.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import robust_common as rc
from robust_unusual import score_knn, auroc

def photo_stats(H, E):
    l = H[:, rc.HAND_NAMES.index("luma_med")]; b = H[:, rc.HAND_NAMES.index("lab_b")]
    q = lambda v: float(np.subtract(*np.percentile(v, [75, 25])))
    return dict(luma_iqr=q(l), b_iqr=q(b), dark_rel=float(np.mean(l < np.median(l) - 25)), emb_spread=float(np.median(score_knn(E))))

def main():
    meta = json.load(open(rc.CACHE / "photos.json"))["photos"]; z = np.load(rc.CACHE / "beans.npz"); pi = z["photo_idx"]
    rows = []
    for i, m in enumerate(meta):
        b = np.flatnonzero((pi == i) & ~z["touching"])
        if len(b) < 10: continue
        rows.append(dict(i=i, src=m["src"], label=m["label"], **photo_stats(z["hand"][b], z["emb"][b])))
    keys = ["luma_iqr", "b_iqr", "dark_rel", "emb_spread"]
    from scipy.stats import spearmanr
    cbd = [r for r in rows if r["src"] == "cbd"]
    res = {"what": __doc__.strip().split("\n")[0], "per_grade_median": {}, "spearman_vs_grade_order": {}, "auroc_AAA_vs_grade": {},
           "j4ck_photos": {r["label"]: {k: round(r[k], 4) for k in keys} for r in rows if r["src"] == "j4ck"}}
    for g in rc.CBD_GRADES:
        res["per_grade_median"][g] = {k: round(float(np.median([r[k] for r in cbd if r["label"] == g])), 4) for k in keys}
    for k in keys:
        rho, p = spearmanr([rc.CBD_GRADES.index(r["label"]) for r in cbd], [r[k] for r in cbd])
        res["spearman_vs_grade_order"][k] = dict(rho=round(float(rho), 4), p=float(p))
        aaa = np.array([r[k] for r in cbd if r["label"] == "AAA"])
        res["auroc_AAA_vs_grade"][k] = {g: round(float(auroc(np.array([r[k] for r in cbd if r["label"] == g]), aaa)), 4) for g in rc.CBD_GRADES[1:]}
    rc.dump(res, "robust_spread.json")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, k in zip(axes, keys):
        data = [[r[k] for r in cbd if r["label"] == g] for g in rc.CBD_GRADES]
        ax.boxplot(data, showfliers=False); ax.set_xticks(range(1, len(rc.CBD_GRADES) + 1)); ax.set_xticklabels(rc.CBD_GRADES, fontsize=8)
        for j, d in enumerate(data): ax.scatter(np.full(len(d), j + 1) + np.random.default_rng(j).uniform(-0.2, 0.2, len(d)), d, s=4, alpha=0.4, c="#2a7fb8")
        s = res["spearman_vs_grade_order"][k]; ax.set_title(f"{k}  (Spearman vs grade order rho={s['rho']:.2f})", fontsize=9)
    fig.suptitle("CBD photo-level spread per size grade (one light box, so absolute spreads are comparable within CBD only)", fontsize=10)
    fig.tight_layout(); fig.savefig(rc.REPORTS_V2 / "robust_spread.png", dpi=120); plt.close(fig)
    print(json.dumps({k: res[k] for k in ["per_grade_median", "spearman_vs_grade_order", "auroc_AAA_vs_grade"]}, indent=0))
    print(res["j4ck_photos"])

if __name__ == "__main__":
    main()
