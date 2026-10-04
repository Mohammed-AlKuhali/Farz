"""(1) COLOUR-ONLY BASELINE vs the v1 model, both under LEAVE-ONE-PHOTO-OUT on J4ckDev.

Question from the brief: "would a simpler tool do the same job?"  Two binary jobs:
  dark     vs not  (positives: Negros, MarronAVinagre, DXHongo)
  unhulled vs not  (positives: Pergamino = parchment, CerezaSeca = dried cherry)
Features: 16 per-bean colour/shape stats on the white-balanced 128 px crop (robust_common.hand_features).
Rules (no deep learning):
  R0  fixed, NOT fitted: "dark" if the bean's mean colour has HSV value < 0.30 (the app colour gate's own constant).
  R1  one feature, one threshold: feature, direction and threshold chosen on the TRAINING photos of each LOPO fold
      (max balanced accuracy over beans); applied to the held-out photo.
  R2  depth-2 decision tree on the 16 stats (class_weight balanced), fitted per LOPO fold.
  R3  logistic regression on the 16 stats (standardised, class_weight balanced), fitted per LOPO fold.
v1: reports/beans_lopo.json (same LOPO protocol, 5-class model retrained without the held-out photo, uncalibrated
argmax, no abstention); "v1 says dark" = argmax is dark. v1 could not leave Normales out (only 'good' photo), so the
head-to-head table uses the 10 photos v1 has; the rules are additionally scored on Normales.
Writes reports_v2/robust_colour.json and robust_colour.png.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import robust_common as rc

FEATS = rc.HAND_NAMES

def load_j4ck():
    meta = json.load(open(rc.CACHE / "photos.json"))["photos"]
    z = np.load(rc.CACHE / "beans.npz")
    pi = z["photo_idx"]; photos = np.array([meta[i]["label"] for i in pi]); src = np.array([meta[i]["src"] for i in pi])
    m = src == "j4ck"
    return dict(photo=photos[m], cls=np.array([rc.PHOTO_CLASS[p] for p in photos[m]]), X=z["hand"][m],
                touching=z["touching"][m]), z, meta, pi, src

def bal_acc(y, p):
    tpr = p[y].mean() if y.any() else np.nan; tnr = (~p[~y]).mean() if (~y).any() else np.nan
    return 0.5 * (tpr + tnr)

def fit_r1(X, y):
    best = (-1, None)
    for j in range(X.shape[1]):
        v = X[:, j]; qs = np.unique(np.percentile(v, np.linspace(1, 99, 197)))
        for t in qs:
            for sgn in (1, -1):
                p = (sgn * (v - t)) < 0  # sgn=1: positive if below t ; sgn=-1: positive if above t
                b = bal_acc(y, p)
                if b > best[0]: best = (b, (j, float(t), sgn))
    return best[1], best[0]

def apply_r1(rule, X):
    j, t, sgn = rule
    return (sgn * (X[:, j] - t)) < 0

def lopo(d, target, method):
    out = np.zeros(len(d["cls"]), bool); chosen = {}
    y = d["cls"] == target
    for ph in sorted(set(d["photo"])):
        te = d["photo"] == ph; tr = ~te
        Xtr, ytr, Xte = d["X"][tr], y[tr], d["X"][te]
        if method == "R0":
            out[te] = Xte[:, FEATS.index("val_mean")] < 0.30
            chosen[ph] = "val_mean < 0.30 (fixed)"
        elif method == "R1":
            rule, b = fit_r1(Xtr, ytr); out[te] = apply_r1(rule, Xte)
            chosen[ph] = f"{FEATS[rule[0]]} {'<' if rule[2] == 1 else '>'} {rule[1]:.3f} (train bal.acc {b:.3f})"
        elif method == "R2":
            from sklearn.tree import DecisionTreeClassifier, export_text
            t = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=0).fit(Xtr, ytr)
            out[te] = t.predict(Xte).astype(bool); chosen[ph] = export_text(t, feature_names=FEATS).replace("\n", " | ")
        elif method == "R3":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)).fit(Xtr, ytr)
            out[te] = m.predict(Xte).astype(bool); chosen[ph] = "logreg 16 stats"
    return out, chosen

def score(d, target, pred, photos_subset):
    y = d["cls"] == target
    m = np.isin(d["photo"], photos_subset)
    yy, pp = y[m], pred[m]
    per_photo = {ph: dict(cls=rc.PHOTO_CLASS[ph], positive=bool(rc.PHOTO_CLASS[ph] == target), n=int((d["photo"] == ph).sum()),
                          called_positive=round(float(pred[d["photo"] == ph].mean()), 4)) for ph in photos_subset}
    pos = [v["called_positive"] for v in per_photo.values() if v["positive"]]
    neg = [v["called_positive"] for v in per_photo.values() if not v["positive"]]
    return dict(n_beans=int(m.sum()), n_pos=int(yy.sum()), n_neg=int((~yy).sum()),
                sensitivity=round(float(pp[yy].mean()), 4), false_positive_rate=round(float(pp[~yy].mean()), 4),
                balanced_accuracy=round(float(bal_acc(yy, pp)), 4),
                photo_mean_sensitivity=round(float(np.mean(pos)), 4), photo_mean_fpr=round(float(np.mean(neg)), 4),
                photo_balanced_accuracy=round(float(0.5 * (np.mean(pos) + 1 - np.mean(neg))), 4), per_photo=per_photo)

def v1_lopo_pred(d, target):
    """Per-bean binary call implied by beans_lopo.json predicted_share: exact counts per photo (order within a photo
    is unknown but irrelevant for every metric here)."""
    lo = json.load(open(rc.ROOT / "reports/beans_lopo.json"))["photos"]
    out = np.zeros(len(d["cls"]), bool); have = []
    for ph, v in lo.items():
        if "n" not in v: continue
        idx = np.flatnonzero(d["photo"] == ph); assert len(idx) == v["n"], (ph, len(idx), v["n"])
        k = int(round(v["predicted_share"][target] * v["n"])); out[idx[:k]] = True; have.append(ph)
    return out, sorted(have)

def spatial(d):
    """v1's own (upper-bound) protocol: train on one half of every photo, test on the other half. Rule R1 is fitted
    on the training half; v1 numbers are the fp32 fold models' held-out predictions (reports/beans_heldout_preds.npz)."""
    import csv
    fold = np.array([int(r["test_fold"]) for r in csv.DictReader(open(rc.CROPS_CSV))])
    hp = np.load(rc.ROOT / "reports/beans_heldout_preds.npz")
    v1_arg = np.full(len(fold), -1); v1_arg[hp["idx"]] = hp["fp32"].argmax(1)
    out = {}
    for target in ["dark", "unhulled"]:
        y = d["cls"] == target; pr = np.zeros(len(y), bool); rules = {}
        for k in (1, 2):
            te = fold == k; rule, b = fit_r1(d["X"][~te], y[~te]); pr[te] = apply_r1(rule, d["X"][te])
            rules[k] = f"{FEATS[rule[0]]} {'<' if rule[2] == 1 else '>'} {rule[1]:.3f}"
        v1p = v1_arg == rc.CLASSES.index(target)
        out[target] = {"rule_R1": {"sensitivity": round(float(pr[y].mean()), 4), "false_positive_rate": round(float(pr[~y].mean()), 4),
                                   "balanced_accuracy": round(float(bal_acc(y, pr)), 4), "rules": rules},
                       "v1_fold_models": {"sensitivity": round(float(v1p[y].mean()), 4), "false_positive_rate": round(float(v1p[~y].mean()), 4),
                                          "balanced_accuracy": round(float(bal_acc(y, v1p)), 4)}}
    out["note"] = "UPPER BOUND for both: train and test beans share each photo's camera, light and sheet"
    return out

def main():
    d, z, meta, pi, src = load_j4ck()
    photos_all = sorted(set(d["photo"]))
    res = {"what": __doc__.strip().split("\n")[0], "protocol": "leave-one-photo-out on the 11 J4ckDev photos (582 crops); rules fitted only on the other photos",
           "features": FEATS, "tasks": {}}
    preds = {}
    for target in ["dark", "unhulled"]:
        v1p, v1_photos = v1_lopo_pred(d, target)
        T = {"positive_photos": [p for p in photos_all if rc.PHOTO_CLASS[p] == target],
             "v1_model": score(d, target, v1p, v1_photos)}
        methods = ["R0", "R1", "R2", "R3"] if target == "dark" else ["R1", "R2", "R3"]
        for meth in methods:
            p, chosen = lopo(d, target, meth); preds[(target, meth)] = p
            T[meth] = {"same_10_photos_as_v1": score(d, target, p, v1_photos), "all_11_photos": score(d, target, p, photos_all),
                       "good_photo_Normales_called_positive": round(float(p[d["photo"] == "Normales"].mean()), 4),
                       "rule_per_fold": chosen}
        res["tasks"][target] = T
    # final rules fitted on ALL J4ckDev (what the app would ship) + their behaviour on CBD (no labels: OOD only)
    cbd = src == "cbd"; grades = np.array([meta[i]["label"] for i in pi])
    final = {}
    for target in ["dark", "unhulled"]:
        y = d["cls"] == target; rule, b = fit_r1(d["X"], y)
        pc = apply_r1(rule, z["hand"][cbd])
        v1c = z["probs"][cbd].argmax(1) == rc.CLASSES.index(target)
        final[target] = {"R1_rule_all_j4ck": f"{FEATS[rule[0]]} {'<' if rule[2] == 1 else '>'} {rule[1]:.3f}", "train_bal_acc_in_sample": round(b, 4),
                         "cbd_share_called": {g: {"rule_R1": round(float(pc[grades[cbd] == g].mean()), 4),
                                                  "v1_argmax": round(float(v1c[grades[cbd] == g].mean()), 4)} for g in rc.CBD_GRADES}}
        if target == "dark":
            r0 = z["hand"][cbd][:, FEATS.index("val_mean")] < 0.30
            for g in rc.CBD_GRADES: final[target]["cbd_share_called"][g]["rule_R0"] = round(float(r0[grades[cbd] == g].mean()), 4)
    res["final_rules_and_cbd_behaviour"] = final
    res["within_photo_spatial_halves"] = spatial(d)
    res["cbd_note"] = "CBD has no per-bean labels: these shares are OOD behaviour, not accuracy."
    rc.dump(res, "robust_colour.json")
    plot(d, res, preds)
    for t, T in res["tasks"].items():
        print(t, "v1", {k: T["v1_model"][k] for k in ["sensitivity", "false_positive_rate", "balanced_accuracy", "photo_balanced_accuracy"]})
        for meth in [m for m in ["R0", "R1", "R2", "R3"] if m in T]:
            s = T[meth]["same_10_photos_as_v1"]
            print("  ", meth, {k: s[k] for k in ["sensitivity", "false_positive_rate", "balanced_accuracy", "photo_balanced_accuracy"]},
                  "Normales+", T[meth]["good_photo_Normales_called_positive"])
    print(json.dumps(final, indent=1))

def plot(d, res, preds):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    photos = sorted(set(d["photo"]), key=lambda p: (rc.PHOTO_CLASS[p], p))
    for ax, target, meth in [(axes[0], "dark", "R1"), (axes[1], "unhulled", "R1")]:
        T = res["tasks"][target]; x = np.arange(len(photos)); w = 0.38
        v1 = [T["v1_model"]["per_photo"].get(p, {}).get("called_positive", np.nan) for p in photos]
        r = [T[meth]["all_11_photos"]["per_photo"][p]["called_positive"] for p in photos]
        ax.bar(x - w / 2, v1, w, label="v1 model (LOPO)", color="#7a7a7a"); ax.bar(x + w / 2, r, w, label=f"colour rule {meth} (LOPO)", color="#2a7fb8")
        for i, p in enumerate(photos):
            if rc.PHOTO_CLASS[p] == target: ax.axvspan(i - 0.5, i + 0.5, color="#f3d27a", alpha=0.35, lw=0)
            if np.isnan(v1[i]): ax.text(i - w / 2, 0.02, "n/a", ha="center", fontsize=7, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([f"{p}\n({rc.PHOTO_CLASS[p]})" for p in photos], rotation=60, fontsize=7, ha="right")
        ax.set_ylim(0, 1.05); ax.set_ylabel(f"share of the photo's beans called '{target}'")
        ax.set_title(f"'{target}' vs not, leave-one-photo-out\n(shaded = true '{target}' photos; want high there, low elsewhere)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
    ax = axes[2]; X = d["X"]; i1, i2 = FEATS.index("luma_med"), FEATS.index("sat_mean")
    cols = {"good": "#3a9a3a", "dark": "#222222", "insect": "#c03030", "broken": "#e08a20", "unhulled": "#8a5ad0"}
    for c in rc.CLASSES:
        m = d["cls"] == c; ax.scatter(X[m, i1], X[m, i2], s=8, alpha=0.6, c=cols[c], label=c)
    ax.set_xlabel("bean median luminance (white-balanced crop, 0-255)"); ax.set_ylabel("bean mean saturation")
    ax.set_title("J4ckDev beans in two colour stats", fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rc.REPORTS_V2 / "robust_colour.png", dpi=130); plt.close(fig)

if __name__ == "__main__":
    main()
