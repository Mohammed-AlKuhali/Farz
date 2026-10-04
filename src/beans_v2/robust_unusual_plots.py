"""Figures for (2): reports_v2/robust_unusual.png (synthetic trays: defect vs control; CBD flag share per grade) and
reports_v2/robust_unusual_examples.png (most and least unusual beans in real CBD photos, raw/both score)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import robust_common as rc

def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = json.load(open(rc.REPORTS_V2 / "robust_unusual.json"))
    S = R["synthetic"]["results"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    pairs = [("A", "B", "J4ck good tray +3"), ("A", "B2", "J4ck good tray +3"), ("C", "D", "real CBD AAA photo +3"), ("E", "F", "real CBD AAA photo +3")]
    labels = {"A": "J4ck defects", "B": "CBD AAA good\n(CONTROL)", "B2": "same good beans,\nexposure shift (CONTROL)", "C": "J4ck defects",
              "D": "J4ck good\n(CONTROL)", "E": "CBD Bits", "F": "AAA from other\nphotos (CONTROL)"}
    for ax, metric, title in [(axes[0], "recall_inserted", "share of the 3 inserted beans flagged (z > 3.5)"),
                              (axes[1], "mean_auroc", "within-tray AUROC (inserted vs base)")]:
        conds = ["A", "B", "B2", "C", "D", "E", "F"]; x = np.arange(len(conds)); w = 0.2
        for j, (var, m, col) in enumerate([("raw", "both", "#2a7fb8"), ("norm", "both", "#9cc8e8"), ("raw", "v1f", "#7a7a7a"), ("norm", "v1f", "#c8c8c8")]):
            ax.bar(x + (j - 1.5) * w, [S[f"{c}/{var}"][m][metric] for c in conds], w, color=col, label=f"{m} score, {var} crops")
        ax.set_xticks(x); ax.set_xticklabels([f"{c}: {labels[c]}" for c in conds], fontsize=7, rotation=30, ha="right")
        for i, c in enumerate(conds):
            if c in ("B", "B2", "D", "F"): ax.axvspan(i - 0.5, i + 0.5, color="#f3d27a", alpha=0.3, lw=0)
        ax.set_title(title + "\n(shaded = CONTROLS: inserted beans are GOOD; high = photo-identity leak)", fontsize=9)
        ax.legend(fontsize=7); ax.set_ylim(0, 1.05)
    ax = axes[2]; real = R["real_photos"]; g = rc.CBD_GRADES; x = np.arange(len(g)); w = 0.27
    for j, (k, col) in enumerate([("raw/both", "#2a7fb8"), ("raw/hand+size", "#e08a20"), ("raw/v1f", "#7a7a7a")]):
        ax.bar(x + (j - 1) * w, [100 * real[q][k]["share_flagged"] for q in g], w, color=col, label=k)
    ax.set_xticks(x); ax.set_xticklabels(g); ax.set_ylabel("% of beans flagged unusual (z > 3.5)")
    ax.set_title("Real CBD photos (464): share flagged per size grade\n(no per-bean labels; relative score = heterogeneity within a photo)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rc.REPORTS_V2 / "robust_unusual.png", dpi=130); plt.close(fig)

    # examples: 3 AAA + 3 Bits photos, top-4 and median-2 beans by raw/both z
    meta = json.load(open(rc.CACHE / "photos.json"))["photos"]
    z = np.load(rc.CACHE / "beans.npz"); crops = np.load(rc.CACHE / "crops.npy", mmap_mode="r")
    sc = np.load(rc.CACHE / "unusual_scores.npz"); pi = z["photo_idx"]
    # unusual_scores rows follow photo order with >= 10 beans, same as cache order within each photo
    sel = []
    for grade in ["AAA", "Bits"]:
        ids = [i for i, m in enumerate(meta) if m["src"] == "cbd" and m["label"] == grade][:3]
        sel += [(grade, i) for i in ids]
    fig, axes = plt.subplots(len(sel), 7, figsize=(11, 1.75 * len(sel)))
    for r, (grade, i) in enumerate(sel):
        b = np.flatnonzero(pi == i); s = sc["raw__both"][sc["photo"] == i]
        order = np.argsort(-s); picks = list(order[:5]) + [order[len(order) // 2], order[-1]]
        for c, k in enumerate(picks):
            a = axes[r, c]; a.imshow(crops[b[k]]); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"z={s[k]:.1f}" + (" FLAG" if s[k] > 3.5 else ""), fontsize=7, color="#c03030" if s[k] > 3.5 else "black")
        axes[r, 0].set_ylabel(f"{grade}\n{Path(meta[i]['file']).name}", fontsize=7)
    fig.suptitle("Most unusual 5 beans, the median bean and the least unusual bean per photo (raw/both score)", fontsize=9)
    fig.tight_layout(); fig.savefig(rc.REPORTS_V2 / "robust_unusual_examples.png", dpi=80); plt.close(fig)

if __name__ == "__main__":
    main()
