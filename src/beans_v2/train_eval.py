"""Honest evaluation of the v2 classifier: leave-one-SOURCE-out (LOSO) and J4ckDev leave-one-PHOTO-out (LOPO).

Everything is computed from the fold outputs of train_folds.py (CACHE/folds/*.npz: held-out logits only).

Calibration and threshold are NEVER fitted on the source they are evaluated on:
  * weights: every held-out source has total weight 1, split equally between its good and defect rows
    ("source- and class-balanced"), so vicanadya16's 86k crops count as much as J4ckDev's 582.
  * temperature T_s for source s = argmin weighted NLL on the LOSO predictions of the OTHER 11 sources
    (NLL of an untyped-defect row = -log P(any defect)).
  * abstain threshold t_s for source s = picked on the OTHER 11 sources (rule below); the deployed T and t are fitted
    on all 12 sources' LOSO predictions (the final model has no held-out data of its own; same idea as v1).
  * threshold rule: confidence = max(P(good), 1 - P(good)); lowest t in 0.50..0.99 whose weighted answered good-vs-defect
    accuracy is >= 90% and stays >= 90% for every higher t that still answers >= 30 beans.
Writes reports_v2/model_v2_eval.json, model_v2_coverage.png, model_v2_recall.png.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.optimize import minimize_scalar
import train_common as tc

FOLDS = tc.CACHE / "folds"
TARGET = 0.90
J4CK_PHOTO_CLASS = {"Normales": "good", "Negros": "dark", "MarronAVinagre": "dark", "DXHongo": "dark", "BrocadoLeve": "insect",
                    "BrocadoSevero": "insect", "PMordidoCortado": "broken", "Inmaduro": "broken", "Concha": "broken",
                    "Pergamino": "unhulled", "CerezaSeca": "unhulled"}


def weights_for(src, y):
    w = np.zeros(len(y)); d = y != 0
    for s in np.unique(src):
        m = src == s; cells = [c for c in (m & d, m & ~d) if c.any()]
        for c in cells: w[c] = 1.0 / (len(cells) * c.sum())
    return w


def nll(L, y, w, T):
    p = tc.softmax(L, T); typed = y >= 0
    pr = np.where(typed, p[np.arange(len(y)), np.clip(y, 0, None)], p[:, 1:].sum(1))
    return float(-(w * np.log(np.clip(pr, 1e-12, 1))).sum() / w.sum())


def fit_T(L, y, w):
    r = minimize_scalar(lambda lt: nll(L, y, w, np.exp(lt)), bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def pick_threshold(p, y, w, target=TARGET, min_answered=30):
    curve = tc.coverage_curve(p, y, w, mode="binary")
    ok = [c for c in curve if c["answered"] >= min_answered and c["accuracy_answered"] is not None]
    for i, c in enumerate(ok):
        if all(d["accuracy_answered"] >= target for d in ok[i:]): return c, curve
    return None, curve


def pick_perclass(p, y, w, target=TARGET, min_answered=30):
    """Per-class thresholds: good call if P(good) >= tg, defect call if 1 - P(good) >= td. Each threshold is the lowest
    value in 0.50..0.99 whose weighted PRECISION of that call is >= target and stays >= target for every higher value
    answering >= min_answered beans. Both precisions >= target  =>  answered good-vs-defect accuracy >= target."""
    pg = p[:, 0]; truth_def = y != 0; out = {}
    for name, conf, right in (("good", pg, ~truth_def), ("defect", 1 - pg, truth_def)):
        rows = []
        for t in np.round(np.arange(0.50, 1.0001, 0.01), 2):
            m = conf >= t - 1e-12
            rows.append((float(t), int(m.sum()), float(np.average(right[m], weights=w[m])) if m.any() and w[m].sum() > 0 else None))
        ok = [r for r in rows if r[1] >= min_answered and r[2] is not None]
        pick = None
        for i, r in enumerate(ok):
            if all(q[2] >= target for q in ok[i:]): pick = r; break
        out[name] = {"threshold": pick[0] if pick else None, "answered": pick[1] if pick else 0, "precision": pick[2] if pick else None}
    return out


def at_perclass(p, y, tg, td):
    pg = p[:, 0]; truth_def = y != 0
    cg = pg >= (tg if tg is not None else 2); cd = (1 - pg) >= (td if td is not None else 2)
    ans = cg | cd; correct = (cg & ~truth_def) | (cd & truth_def)
    out = {"t_good": tg, "t_defect": td, "coverage": float(ans.mean()), "answered_binary_accuracy": float(correct[ans].mean()) if ans.any() else None,
           "good_beans": {"n": int((~truth_def).sum()), "called_good": float(cg[~truth_def].mean()) if (~truth_def).any() else None,
                          "called_defect": float(cd[~truth_def].mean()) if (~truth_def).any() else None},
           "defect_beans": {"n": int(truth_def.sum()), "called_good": float(cg[truth_def].mean()) if truth_def.any() else None,
                            "called_defect": float(cd[truth_def].mean()) if truth_def.any() else None}}
    return out


def pick_pair(p, y, w, target=TARGET, min_answered=30):
    """Two thresholds: good if P(good) >= tg, defect if 1 - P(good) >= td (tg >= 0.5 so a bean cannot be both).
    For every tg in 0.50..0.95 take the lowest td in 0.50..0.99 whose weighted answered good-vs-defect accuracy is >= target
    and stays >= target for every higher td answering >= min_answered beans; keep the (tg, td) with the largest weighted
    coverage. Needed because P(good) is capped well below P(defect) (label smoothing over 6 classes + T > 1 + summing 5
    defect classes), so one symmetric threshold almost never lets a bean be called good."""
    pg = p[:, 0]; truth_def = y != 0; best = None
    for tg in np.round(np.arange(0.50, 0.9501, 0.02), 2):
        cg = pg >= tg; rows = []
        for td in np.round(np.arange(0.50, 0.9901, 0.01), 2):
            cd = (1 - pg) >= td; ans = cg | cd; ok = (cg & ~truth_def) | (cd & truth_def)
            if ans.sum() < min_answered or w[ans].sum() == 0: continue
            rows.append((float(td), float(np.average(ok[ans], weights=w[ans])), float(w[ans].sum() / w.sum()), int(ans.sum())))
        for i, r in enumerate(rows):
            if all(q[1] >= target for q in rows[i:]):
                if best is None or r[2] > best["coverage"]: best = {"t_good": float(tg), "t_defect": r[0], "answered_accuracy": r[1], "coverage": r[2], "answered": r[3]}
                break
    return best


def at_threshold(p, y, t):
    """per-kind coverage and answered binary / typed accuracy at confidence threshold t (binary confidence)."""
    bp, bt, conf = tc.binary(p, y); ans = conf >= t - 1e-12
    out = {"threshold": t, "coverage": float(ans.mean()), "answered": int(ans.sum()),
           "answered_binary_accuracy": float((bp == bt)[ans].mean()) if ans.any() else None,
           "answered_typed_accuracy": float(tc.correct_typed(p, y)[ans].mean()) if ans.any() else None, "by_kind": {}}
    for i, k in enumerate(tc.KINDS):
        m = (y == i) if i < len(tc.CLASSES) else (y == -1)
        if not m.any(): continue
        a = m & ans
        out["by_kind"][k] = {"n": int(m.sum()), "coverage": float(a.sum() / m.sum()),
                             "answered_binary_accuracy": float((bp == bt)[a].mean()) if a.any() else None,
                             "answered_typed_accuracy": float(tc.correct_typed(p, y)[a].mean()) if a.any() else None,
                             "called_good": float((ans & m & ~bp).sum() / m.sum()), "called_defect": float((ans & m & bp).sum() / m.sum()),
                             "unsure": float((m & ~ans).sum() / m.sum())}
    return out


def load_folds(prefix):
    out = {}
    for f in sorted(FOLDS.glob(f"{prefix}_*.npz")):
        z = np.load(f); out[f.stem[len(prefix) + 1:]] = (z["idx"], z["logits"])
    return out


def main():
    meta = json.load(open(tc.CACHE / "meta.json"))
    src_all = np.array([m["source"] for m in meta]); y_all = tc.kind_targets(meta, range(len(meta)))
    loso = load_folds("loso")
    missing = [s for s in tc.SOURCES if s not in loso]
    srcs = [s for s in tc.SOURCES if s in loso]
    tc.log(f"eval: LOSO folds present {srcs}; missing {missing}")
    L = np.concatenate([loso[s][1] for s in srcs]); I = np.concatenate([loso[s][0] for s in srcs])
    S = src_all[I]; Y = y_all[I]; W = weights_for(S, Y)
    res = {"protocol": __doc__.strip(), "classes": tc.CLASSES, "kinds": tc.KINDS, "target_answered_accuracy": TARGET,
           "sources_evaluated": srcs, "sources_missing": missing, "per_source": {}}
    # pooled (deployed) calibration
    T_all = fit_T(L, Y, W); P_all = tc.softmax(L, T_all)
    chosen, curve = pick_threshold(P_all, Y, W)
    res["deployed"] = {"temperature": T_all, "threshold": chosen["threshold"] if chosen else None, "chosen_point_pooled": chosen,
                       "pooled_curve_binary": curve, "pooled_curve_top1": tc.coverage_curve(P_all, Y, W, mode="top1"),
                       "fitted_on": "all LOSO held-out predictions pooled, source- and class-balanced weights",
                       "ece_binary_pooled_weighted": tc.ece_binary(P_all, Y, W), "ece_binary_pooled_weighted_T1": tc.ece_binary(tc.softmax(L, 1.0), Y, W)}
    pcl = pick_perclass(P_all, Y, W)
    res["deployed"]["perclass_thresholds"] = {"good": pcl["good"]["threshold"], "defect": pcl["defect"]["threshold"], "detail": pcl,
                                              "rule": pick_perclass.__doc__.strip()}
    tc.log(f"deployed per-class thresholds {pcl}")
    pair = pick_pair(P_all, Y, W)
    res["deployed"]["pair_thresholds"] = {**pair, "rule": pick_pair.__doc__.strip(), "fitted_on": "all 12 LOSO held-out sets, source- and class-balanced weights"}
    tc.log(f"deployed pair thresholds {pair}")
    tc.log(f"deployed T={T_all:.3f}  threshold={res['deployed']['threshold']}  pooled coverage {chosen and chosen['coverage']:.3f}")
    # per held-out source, with T and t fitted on the other sources only
    curves = {}
    for s in srcs:
        o = S != s
        Ts = fit_T(L[o], Y[o], weights_for(S[o], Y[o])); ts_pt, _ = pick_threshold(tc.softmax(L[o], Ts), Y[o], weights_for(S[o], Y[o]))
        ts = ts_pt["threshold"] if ts_pt else 0.99
        Wo = weights_for(S[o], Y[o]); pcs = pick_perclass(tc.softmax(L[o], Ts), Y[o], Wo)
        prs = pick_pair(tc.softmax(L[o], Ts), Y[o], Wo)
        m = S == s; p = tc.softmax(L[m], Ts); y = Y[m]
        r = {"n": int(m.sum()), "T_fitted_on_other_sources": Ts, "threshold_fitted_on_other_sources": ts,
             "no_abstain": tc.summary(p, y), "ece_binary_uncalibrated_T1": tc.ece_binary(tc.softmax(L[m], 1.0), y),
             "at_own_held_out_threshold": at_threshold(p, y, ts),
             "perclass_thresholds_fitted_on_other_sources": {"good": pcs["good"]["threshold"], "defect": pcs["defect"]["threshold"]},
             "at_own_held_out_perclass": at_perclass(p, y, pcs["good"]["threshold"], pcs["defect"]["threshold"]),
             "pair_thresholds_fitted_on_other_sources": {"good": prs["t_good"], "defect": prs["t_defect"]},
             "at_own_held_out_pair": at_perclass(p, y, prs["t_good"], prs["t_defect"]),
             "at_deployed_threshold_T": at_threshold(tc.softmax(L[m], T_all), y, res["deployed"]["threshold"] or 0.99),
             "curve_binary": tc.coverage_curve(p, y, mode="binary"), "curve_top1": tc.coverage_curve(p, y, mode="top1")}
        res["per_source"][s] = r; curves[s] = (r["curve_binary"], ts)
        a = r["at_own_held_out_threshold"]; nb = r["no_abstain"]
        tc.log(f"  {s:22s} n={r['n']:6d} T={Ts:.2f} t={ts:.2f} | no-abstain bin-acc {nb['binary']['accuracy']:.3f} bal {nb['binary']['balanced_accuracy']:.3f} "
               f"macro {nb['macro_accuracy']:.3f} | at t: cov {a['coverage']:.3f} ans-bin-acc {a['answered_binary_accuracy']} | pair {r['pair_thresholds_fitted_on_other_sources']} cov {r['at_own_held_out_pair']['coverage']:.3f} acc {r['at_own_held_out_pair']['answered_binary_accuracy']}")
    # leave-one-source-out summary across sources (each source = one test)
    def mean_of(f): v = [f(res["per_source"][s]) for s in srcs]; v = [x for x in v if x is not None]; return float(np.mean(v)) if v else None
    res["loso_summary"] = {
        "mean_over_sources_no_abstain_binary_balanced_accuracy": mean_of(lambda r: r["no_abstain"]["binary"]["balanced_accuracy"]),
        "mean_over_sources_no_abstain_macro_accuracy": mean_of(lambda r: r["no_abstain"]["macro_accuracy"]),
        "mean_over_sources_coverage_at_own_threshold": mean_of(lambda r: r["at_own_held_out_threshold"]["coverage"]),
        "mean_over_sources_answered_binary_accuracy_at_own_threshold": mean_of(lambda r: r["at_own_held_out_threshold"]["answered_binary_accuracy"]),
        "sources_below_90_answered_binary_at_own_threshold": [s for s in srcs if (res["per_source"][s]["at_own_held_out_threshold"]["answered_binary_accuracy"] or 0) < 0.9],
    }
    # J4ckDev leave-one-photo-out (other 11 sources + the other 10 J4ckDev photos in training)
    lopo = load_folds("lopo")
    if lopo:
        Tj = res["per_source"].get("j4ckdev", {}).get("T_fitted_on_other_sources", T_all)
        tj = res["per_source"].get("j4ckdev", {}).get("threshold_fitted_on_other_sources", res["deployed"]["threshold"])
        ph_res = {}; allp, ally = [], []
        for ph, (idx, lg) in lopo.items():
            p = tc.softmax(lg, Tj); y = y_all[idx]; cls = J4CK_PHOTO_CLASS[ph]; pred = p.argmax(1)
            bp, bt, conf = tc.binary(p, y); ans = conf >= tj
            ph_res[ph] = {"class": cls, "n": int(len(y)), "recall_correct_class": float((pred == y).mean()),
                          "binary_correct_no_abstain": float((bp == bt).mean()),
                          "predicted_share_argmax": {c: float((pred == i).mean()) for i, c in enumerate(tc.CLASSES)},
                          "coverage_at_t": float(ans.mean()), "answered_binary_accuracy_at_t": float((bp == bt)[ans].mean()) if ans.any() else None,
                          "median_max_prob": float(np.median(p.max(1)))}
            allp.append(p); ally.append(y)
        P = np.concatenate(allp); Yj = np.concatenate(ally); pred = P.argmax(1)
        v1_photos = [k for k in ph_res if k != "Normales"]
        res["j4ck_lopo"] = {"protocol": "train on all 11 other sources + the other 10 J4ckDev photos; test on every bean of the held-out photo; "
                                        "T and threshold = those fitted for J4ckDev in LOSO (other sources only)",
                            "T": Tj, "threshold": tj, "photos": ph_res,
                            "bean_weighted_accuracy_all_11": float((pred == Yj).mean()),
                            "bean_weighted_accuracy_same_10_photos_as_v1": float(np.mean(np.concatenate([[ph_res[k]["recall_correct_class"]] * ph_res[k]["n"] for k in v1_photos]))),
                            "photo_mean_accuracy_same_10_photos_as_v1": float(np.mean([ph_res[k]["recall_correct_class"] for k in v1_photos])),
                            "v1_reference": {"bean_weighted_accuracy": 0.2739, "photo_mean_accuracy": 0.2975, "file": "reports/beans_lopo.json"},
                            "binary_bean_weighted_no_abstain": float(np.mean(np.concatenate([[ph_res[k]["binary_correct_no_abstain"]] * ph_res[k]["n"] for k in ph_res]))),
                            "summary_all_11": tc.summary(P, Yj)}
        tc.log(f"  J4ckDev LOPO: typed acc (10 v1 photos) {res['j4ck_lopo']['bean_weighted_accuracy_same_10_photos_as_v1']:.3f}  "
               f"all 11 {res['j4ck_lopo']['bean_weighted_accuracy_all_11']:.3f}  binary {res['j4ck_lopo']['binary_bean_weighted_no_abstain']:.3f}")
    # licence-clean comparison: a model trained ONLY on the 7 sources that state a licence, tested on the 5 that do not
    f = FOLDS / "licensed.npz"
    if f.exists():
        from train_folds import LICENSED
        z = np.load(f); idx, lg = z["idx"], z["logits"]
        o = np.isin(S, LICENSED)
        Tl = fit_T(L[o], Y[o], weights_for(S[o], Y[o])); tl_pt, _ = pick_threshold(tc.softmax(L[o], Tl), Y[o], weights_for(S[o], Y[o]))
        tl = tl_pt["threshold"] if tl_pt else 0.99
        lic = {"protocol": "train only on the 7 sources that state a licence (" + ", ".join(LICENSED) + "); test on every crop of the 5 "
                           "sources that state none; T and threshold fitted on the LOSO predictions of the 7 licensed sources only",
               "T": Tl, "threshold": tl, "per_source": {}}
        for s in sorted(set(src_all[idx])):
            m = src_all[idx] == s; p = tc.softmax(lg[m], Tl); y = y_all[idx][m]
            lic["per_source"][s] = {"no_abstain": tc.summary(p, y), "at_threshold": at_threshold(p, y, tl),
                                    "all_sources_model_LOSO_same_source": {"binary_balanced_accuracy": res["per_source"][s]["no_abstain"]["binary"]["balanced_accuracy"],
                                                                           "macro_accuracy": res["per_source"][s]["no_abstain"]["macro_accuracy"]} if s in res["per_source"] else None}
            tc.log(f"  licensed-only -> {s}: bal {lic['per_source'][s]['no_abstain']['binary']['balanced_accuracy']:.3f} macro {lic['per_source'][s]['no_abstain']['macro_accuracy']:.3f}")
        res["licensed_only_model"] = lic
    json.dump(res, open(tc.REPORTS_V2 / "model_v2_eval.json", "w"), indent=1)
    plot(res, curves)
    tc.log("wrote reports_v2/model_v2_eval.json")


def plot(res, curves):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    cmap = plt.get_cmap("tab20")
    for i, (s, (cv, ts)) in enumerate(curves.items()):
        xs = [c["coverage"] for c in cv if c["accuracy_answered"] is not None]; ys = [c["accuracy_answered"] for c in cv if c["accuracy_answered"] is not None]
        ax.plot(xs, ys, color=cmap(i), lw=1.6, label=s)
        pt = [c for c in cv if abs(c["threshold"] - ts) < 1e-9 and c["accuracy_answered"] is not None]
        if pt: ax.plot(pt[0]["coverage"], pt[0]["accuracy_answered"], "o", color=cmap(i), ms=6, mec="k")
    ax.axhline(TARGET, color="k", ls="--", lw=1)
    ax.set_xlabel("coverage (share of held-out beans answered)"); ax.set_ylabel("answered good-vs-defect accuracy")
    ax.set_title("Leave-one-source-out: coverage vs accuracy per held-out source\n(dot = threshold picked on the OTHER sources)", fontsize=10)
    ax.set_ylim(0.3, 1.01); ax.set_xlim(0, 1.01); ax.legend(fontsize=7, ncol=2, loc="lower left"); ax.grid(alpha=0.3)
    ax = axes[1]
    srcs = list(res["per_source"]); kinds = tc.KINDS
    M = np.full((len(srcs), len(kinds)), np.nan)
    for i, s in enumerate(srcs):
        for j, k in enumerate(kinds):
            v = res["per_source"][s]["no_abstain"]["per_kind_recall"].get(k)
            if v is not None: M[i, j] = v
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    for i in range(len(srcs)):
        for j in range(len(kinds)):
            if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(kinds))); ax.set_xticklabels([k.replace("_", "\n") for k in kinds], fontsize=8)
    ax.set_yticks(range(len(srcs))); ax.set_yticklabels(srcs, fontsize=8)
    ax.set_title("Held-out recall per label kind (argmax, no abstention)\n'defect_untyped' = called any defect class", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(tc.REPORTS_V2 / "model_v2_coverage.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
