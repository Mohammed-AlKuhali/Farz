"""Step 7: charts (static PNG for README / video), following the dataviz skill's reference palette and mark specs.
reports/beans_coverage.png  accuracy on answered beans vs coverage, per held-out fold + pooled, with the chosen threshold
reports/beans_confusion.png pooled held-out confusion matrix (row-normalised recall + counts), deployed format
Colours: categorical slots 1-3 (#2a78d6, #eb6834, #1baf7a) validated with the skill's validate_palette.js (light, all PASS;
aqua < 3:1 contrast -> relief = direct labels + legend). Sequential = the blue ramp. Text always in ink tokens.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from common import ROOT, CLASSES

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = ["#f0f5fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NAMES = {"good": "good", "dark": "dark\n(black/sour/fungus)", "insect": "insect\ndamage", "broken": "broken/\nimmature/shell", "unhulled": "unhulled\n(parchment/cherry)"}

def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-"); ax.set_axisbelow(True)

def curve(p, y):
    conf, ok = p.max(1), p.argmax(1) == y
    ts = np.round(np.arange(0.20, 1.0001, 0.01), 2)
    cov = np.array([(conf >= t).mean() for t in ts]); acc = np.array([ok[conf >= t].mean() if (conf >= t).any() else np.nan for t in ts])
    n = np.array([(conf >= t).sum() for t in ts])
    return ts, cov, acc, n

def main():
    rep = json.load(open(ROOT / "reports/beans_model.json"))
    d = np.load(ROOT / "reports/beans_heldout_preds.npz")
    fmt = rep["deploy_decision"]["deployed"]; p, y, fold = d[fmt], d["y"], d["fold"]
    chosen = rep["selective"]["chosen"]; thr = chosen["threshold"]

    # ---------------- coverage vs accuracy
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200); fig.patch.set_facecolor(SURF); style(ax)
    series = [("Fold 1 (right half held out)", fold == 1, S1), ("Fold 2 (left half held out)", fold == 2, S2), ("Both folds pooled", np.ones_like(fold, bool), S3)]
    for name, m, col in series:
        ts, cov, acc, n = curve(p[m], y[m]); keep = n >= 30
        ax.plot(100 * cov[keep], 100 * acc[keep], color=col, lw=2, solid_capstyle="round", solid_joinstyle="round", label=name, zorder=3)
        i = np.flatnonzero(keep)[0]  # end label at the full-coverage end
        ax.annotate(f"{100*acc[i]:.1f}%", (100 * cov[i], 100 * acc[i]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.axhline(90, color=AXIS, lw=1, zorder=2)
    ax.text(1.5, 90.4, "target: 90% correct on answered beans", fontsize=8.5, color=INK2, va="bottom")
    cx, cy = 100 * chosen["coverage"], 100 * chosen["accuracy_answered"]
    ax.scatter([cx], [cy], s=60, color=S3, edgecolor=SURF, linewidth=2, zorder=5)
    ax.annotate(f"abstain below {thr:.2f}:\nanswers {cx:.1f}% of beans,\n{cy:.1f}% of those correct", (cx, cy), xytext=(-150, -58), textcoords="offset points",
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlim(0, 104); lo = np.nanmin([100 * curve(p[m], y[m])[2][0] for _, m, _ in series])
    ax.set_ylim(max(50, np.floor(lo / 5) * 5 - 5), 101)
    ax.set_xlabel("Coverage: share of beans the model answers (%)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Accuracy on answered beans (%)", color=INK2, fontsize=9.5)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Farz bean classifier: answering fewer beans buys accuracy", x=0.06, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    ax.set_title(f"{fmt.upper()} ONNX fold models, J4ckDev beans held out spatially within each photo (n={len(y)}). "
                 "An UPPER BOUND:\ntrain and test beans share a photo, camera and light. Lines stop where fewer than 30 beans are answered.",
                 loc="left", fontsize=8.3, color=INK2)
    fig.tight_layout(); fig.savefig(ROOT / "reports/beans_coverage.png", facecolor=SURF); plt.close(fig)

    # ---------------- confusion matrix
    cm = np.array(rep["heldout_pooled"][fmt]["confusion"]); rec = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=200); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    ax.imshow(rec, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            dark = rec[i, j] > 0.55
            ax.text(j, i - 0.12, f"{100*rec[i,j]:.0f}%", ha="center", va="center", fontsize=11 if i == j else 9.5,
                    fontweight="bold" if i == j else "normal", color="#ffffff" if dark else INK)
            ax.text(j, i + 0.22, f"n={cm[i,j]}", ha="center", va="center", fontsize=7.5, color="#ffffff" if dark else INK2)
    for k in range(len(CLASSES) + 1):  # 2px surface gaps between cells
        ax.axhline(k - 0.5, color=SURF, lw=2); ax.axvline(k - 0.5, color=SURF, lw=2)
    ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=8.8, color=INK2)
    ax.set_yticks(range(len(CLASSES)), [f"{NAMES[c].replace(chr(10), ' ')}  (n={cm[i].sum()})" for i, c in enumerate(CLASSES)], fontsize=8.3, color=INK2)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xlabel("Predicted class", color=INK2, fontsize=9.5); ax.set_ylabel("True class (J4ckDev photo label)", color=INK2, fontsize=9.5)
    acc = rep["heldout_pooled"][fmt]["accuracy"]
    fig.suptitle("Where the bean classifier goes wrong", x=0.04, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    ax.set_title(f"Held-out beans, both spatial folds pooled, {fmt.upper()} ONNX, no abstention: {100*acc:.1f}% correct.\n"
                 "Cell = share of the row (recall). Upper bound: each class comes from one photo.", loc="left", fontsize=8.3, color=INK2)
    fig.tight_layout(); fig.savefig(ROOT / "reports/beans_confusion.png", facecolor=SURF, bbox_inches="tight", pad_inches=0.25); plt.close(fig)
    ood_chart(rep)
    print("wrote reports/beans_coverage.png, reports/beans_confusion.png, reports/beans_ood_cbd.png")

def ood_chart(rep):
    """100% stacked bars per CBD size grade: what the shipped model says about real photos from another setup."""
    if "cbd_ood" not in rep: return
    o = rep["cbd_ood"]; grades = list(o["per_grade"]); cats = CLASSES + ["abstain"]
    cols = {"good": S1, "dark": S2, "insect": S3, "broken": "#eda100", "unhulled": "#e87ba4", "abstain": AXIS}
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for gi, g in enumerate(grades):
        left = 0.0
        for c in cats:
            v = 100 * o["per_grade"][g]["share_with_abstain"][c]
            if v <= 0: continue
            ax.barh(gi, v - 0.4, left=left + 0.2, height=0.62, color=cols[c], linewidth=0)
            if v >= 7: ax.text(left + v / 2, gi, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=INK)
            left += v
    ax.set_yticks(range(len(grades)), [f"{g}  ({o['per_grade'][g]['beans']} beans)" for g in grades], fontsize=8.5, color=INK2)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.tick_params(length=0, colors=MUTED, labelsize=8.5)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_xlabel("Share of detected beans (%)", color=INK2, fontsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[c]) for c in cats]
    ax.legend(handles, cats, ncol=6, loc="upper left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.2)
    pr = o["photo_level_rule"]["at_deployed_threshold"]
    fig.suptitle("Out of distribution: the model calls AAA-grade beans from India 'broken'", x=0.03, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.03, 0.905, f"CBD dataset (India, light box, size grades, NO defect labels, so this is not an accuracy test): {o['photos']} photos, {o['beans']} beans.\n"
             f"Shipped {Path(o['model']).stem} model, abstain below {o['abstain_below']}. We expected AAA/AA/A to be mostly 'good'; measured 'good' for AAA: "
             f"{100*o['per_grade']['AAA']['share_with_abstain']['good']:.1f}%.\n"
             f"The photo-level 'not sure' rule (more than 15% of beans unsure) fires on {100*pr['cbd_photos_where_15pct_rule_fires']:.0f}% of these photos.",
             fontsize=8.3, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(ROOT / "reports/beans_ood_cbd.png", facecolor=SURF); plt.close(fig)

if __name__ == "__main__":
    main()
