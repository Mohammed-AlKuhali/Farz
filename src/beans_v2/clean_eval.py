"""Honest evaluation + temperature/threshold fit for the licence-stated ("clean") v2 model.

Protocol
  * Leave-one-source-out over the 7 licensed sources, with afiyah_deteksi + afiyah_bijikopi held out TOGETHER as one group
    ("afiyah"): 6 folds (clean_folds.py clean_loso_*). Every crop of the held-out group is scored by a model that never
    saw that group.
  * Weights: every held-out GROUP has total weight 1, split equally between its good and defect rows (so the afiyah twins
    count once, and vicanadya16's 86k crops count as much as J4ckDev's 582).
  * Deployed temperature T and the two thresholds (good if P(good) >= t_good, defect if P(defect) = 1 - P(good) >= t_defect)
    are fitted on ALL 6 groups' held-out predictions pooled (train_eval.fit_T and train_eval.pick_pair, unchanged: answered
    good-vs-defect accuracy >= 90% and stable for every higher t_defect; the pair with the largest coverage).
    Per-group rows use T and thresholds fitted on the OTHER 5 groups only.
  * The 5 sources that state no licence are TEST ONLY: scored by the shipped clean model (licensed.pt), never trained on.
  * J4ckDev leave-one-photo-out (clean_lopo_*): each photo scored by the model trained without it.
Writes reports_v2/clean_eval.json and CACHE/clean_emb_final_unlicensed.npz (embeddings for localcal.py).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import train_common as tc
import train_eval as te
import clean_common as cc
from clean_folds import GROUPS, J4CK_PHOTOS


def group_metrics(p, y, tg, td, ts=None):
    r = {"n": int(len(y)), "no_abstain": tc.summary(p, y), "at_pair": te.at_perclass(p, y, tg, td)}
    if ts is not None: r["at_symmetric"] = te.at_threshold(p, y, ts)
    # class-balanced coverage / answered accuracy at the pair
    pg = p[:, 0]; d = y != 0; cg = pg >= tg; cd = (1 - pg) >= td; ans = cg | cd; ok = (cg & ~d) | (cd & d)
    w = te.weights_for(np.zeros(len(y), int), y)
    r["at_pair"]["coverage_class_balanced"] = float(w[ans].sum() / w.sum()) if w.sum() else None
    r["at_pair"]["answered_accuracy_class_balanced"] = float(np.average(ok[ans], weights=w[ans])) if ans.any() and w[ans].sum() > 0 else None
    r["ece_binary"] = tc.ece_binary(p, y)
    return r


def main():
    meta = json.load(open(tc.CACHE / "meta.json"))
    src_all = np.array([m["source"] for m in meta]); y_all = tc.kind_targets(meta, range(len(meta)))
    grp_all = np.array([cc.GROUP_OF[s] for s in src_all])
    folds = {g: np.load(cc.FOLDS / f"clean_loso_{g}.npz") for g in GROUPS if (cc.FOLDS / f"clean_loso_{g}.npz").exists()}
    missing = [g for g in GROUPS if g not in folds]
    tc.log(f"clean_eval: LOSO groups present {list(folds)}; missing {missing}")
    L = np.concatenate([folds[g]["logits"] for g in folds]); I = np.concatenate([folds[g]["idx"] for g in folds])
    G = grp_all[I]; S = src_all[I]; Y = y_all[I]; W = te.weights_for(G, Y)
    for g in folds:  # integrity: every held-out row belongs to the group, and the group is complete
        assert set(src_all[folds[g]["idx"]]) == set(GROUPS[g]) and len(folds[g]["idx"]) == int(np.isin(src_all, GROUPS[g]).sum())
    res = {"protocol": __doc__.strip(), "licensed_sources": cc.LICENSED, "groups": GROUPS, "groups_present": list(folds), "groups_missing": missing,
           "unlicensed_test_only": cc.UNLICENSED, "final_weights": str(cc.FINAL_PT), "final_weights_train_info": json.load(open(cc.FOLDS / "licensed.json"))}
    T = te.fit_T(L, Y, W); P = tc.softmax(L, T)
    pair = te.pick_pair(P, Y, W); sym, _ = te.pick_threshold(P, Y, W)
    res["deployed"] = {"temperature": T, "pair_thresholds": {**pair, "rule": te.pick_pair.__doc__.strip()},
                       "symmetric_threshold": sym["threshold"] if sym else None, "symmetric_point": sym,
                       "ece_binary_pooled_weighted": tc.ece_binary(P, Y, W), "ece_binary_pooled_weighted_T1": tc.ece_binary(tc.softmax(L, 1.0), Y, W),
                       "fitted_on": f"clean LOSO held-out predictions of {list(folds)} pooled; group- and class-balanced weights (afiyah twins = one group)"}
    tc.log(f"clean deployed T={T:.4f} pair={pair} sym={sym and sym['threshold']}")
    # per held-out group: T and pair fitted on the OTHER groups only
    res["per_group"] = {}
    for g in folds:
        o = G != g; Wo = te.weights_for(G[o], Y[o])
        Tg = te.fit_T(L[o], Y[o], Wo); Po = tc.softmax(L[o], Tg)
        pg_ = te.pick_pair(Po, Y[o], Wo); sg, _ = te.pick_threshold(Po, Y[o], Wo)
        tg, td = (pg_["t_good"], pg_["t_defect"]) if pg_ else (2.0, 2.0)
        m = G == g
        row = {"T_fitted_on_other_groups": Tg, "pair_fitted_on_other_groups": [tg, td], "symmetric_fitted_on_other_groups": sg["threshold"] if sg else None,
               "all": group_metrics(tc.softmax(L[m], Tg), Y[m], tg, td, sg["threshold"] if sg else 0.99),
               "at_deployed_T_and_pair": group_metrics(tc.softmax(L[m], T), Y[m], pair["t_good"], pair["t_defect"])}
        if len(GROUPS[g]) > 1:
            row["per_source"] = {s: group_metrics(tc.softmax(L[m & (S == s)], Tg), Y[m & (S == s)], tg, td) for s in GROUPS[g]}
        res["per_group"][g] = row
        a = row["all"]; b = a["no_abstain"]["binary"]; ap = a["at_pair"]
        tc.log(f"  {g:18s} n={a['n']:6d} T={Tg:.2f} pair={tg:.2f}/{td:.2f} | bal {b['balanced_accuracy']:.3f} good-rec {b['good_recall']} | "
               f"cov {ap['coverage']:.3f} acc {ap['answered_binary_accuracy']} | balanced cov {ap['coverage_class_balanced']:.3f} acc {ap['answered_accuracy_class_balanced']}")
    with_good = [g for g in folds if (Y[G == g] == 0).any()]
    res["loso_summary"] = {
        "mean_balanced_accuracy_groups_with_good_beans": float(np.mean([res["per_group"][g]["all"]["no_abstain"]["binary"]["balanced_accuracy"] for g in with_good])) if with_good else None,
        "groups_with_good_beans": with_good,
        "groups_below_90_answered_at_own_pair": [g for g in folds if (res["per_group"][g]["all"]["at_pair"]["answered_binary_accuracy"] or 0) < 0.9]}
    # -------- the 5 unlicensed sources: TEST ONLY, shipped clean model
    model = cc.load_torch(cc.FINAL_PT)
    tidx = np.flatnonzero(np.isin(src_all, cc.UNLICENSED))
    crops = np.load(tc.CACHE / "crops.npy", mmap_mode="r")
    Lu, Eu = cc.embed_and_logits(model, crops[tidx])
    zl = np.load(cc.FOLDS / "licensed.npz")  # train_folds.py saved logits on exactly these rows when licensed.pt was trained
    o1 = np.argsort(zl["idx"]); o2 = np.argsort(tidx)
    res["integrity_final_logits_vs_saved"] = {"rows": int(len(tidx)), "same_rows": bool(np.array_equal(np.sort(zl["idx"]), np.sort(tidx))),
                                              "max_abs_logit_diff": float(np.abs(zl["logits"][o1] - Lu[o2]).max())}
    np.savez(tc.CACHE / "clean_emb_final_unlicensed.npz", idx=tidx, logits=Lu, embed=Eu.astype(np.float16))
    res["unlicensed_test"] = {}
    for s in cc.UNLICENSED:
        m = src_all[tidx] == s
        res["unlicensed_test"][s] = group_metrics(tc.softmax(Lu[m], T), y_all[tidx][m], pair["t_good"], pair["t_defect"], sym["threshold"] if sym else 0.99)
        b = res["unlicensed_test"][s]["no_abstain"]["binary"]; ap = res["unlicensed_test"][s]["at_pair"]
        tc.log(f"  TEST {s:22s} bal {b['balanced_accuracy']:.3f} | at pair cov {ap['coverage']:.3f} acc {ap['answered_binary_accuracy']}")
    # reference: the all-sources v2 model on the same 5 sources when each was held out (model_v2_eval.json)
    try:
        v2 = json.load(open(tc.REPORTS_V2 / "model_v2_eval.json"))
        res["reference_all_sources_v2_LOSO"] = {s: v2["per_source"][s]["no_abstain"]["binary"]["balanced_accuracy"] for s in cc.UNLICENSED}
        res["reference_licensed_model_old_fit"] = {s: v2["licensed_only_model"]["per_source"][s]["no_abstain"]["binary"]["balanced_accuracy"] for s in cc.UNLICENSED}
    except Exception as e:
        res["reference_all_sources_v2_LOSO"] = repr(e)
    # -------- J4ckDev leave-one-photo-out
    jl = {p: np.load(cc.FOLDS / f"clean_lopo_{p}.npz") for p in J4CK_PHOTOS if (cc.FOLDS / f"clean_lopo_{p}.npz").exists()}
    if jl:
        Tj = res["per_group"].get("j4ckdev", {}).get("T_fitted_on_other_groups", T)
        tgj, tdj = res["per_group"].get("j4ckdev", {}).get("pair_fitted_on_other_groups", [pair["t_good"], pair["t_defect"]])
        ph = {}; Pall, Yall = [], []
        for p, z in jl.items():
            pr = tc.softmax(z["logits"], Tj); y = y_all[z["idx"]]; c = cc.calls3(pr, tgj, tdj)
            truth_def = y != 0
            ph[p] = {"n": int(len(y)), "class": te.J4CK_PHOTO_CLASS[p], "typed_recall": float((pr.argmax(1) == y).mean()),
                     "good_vs_defect_correct_no_abstain": float(((pr[:, 0] < 0.5) == truth_def).mean()),
                     "coverage_at_pair": float((c != 2).mean()),
                     "answered_correct_at_pair": float(((c == 1) == truth_def)[c != 2].mean()) if (c != 2).any() else None}
            Pall.append(pr); Yall.append(y)
        Pc = np.concatenate(Pall); Yc = np.concatenate(Yall)
        same10 = [p for p in ph if p != "Normales"]
        res["j4ck_lopo"] = {"T": Tj, "pair": [tgj, tdj], "photos": ph, "photos_present": len(jl),
                            "typed_accuracy_bean_weighted_same_10_as_v1": float(sum(ph[p]["typed_recall"] * ph[p]["n"] for p in same10) / max(1, sum(ph[p]["n"] for p in same10))),
                            "summary_all": tc.summary(Pc, Yc)}
        tc.log(f"  J4ckDev LOPO ({len(jl)} photos): typed acc same-10 {res['j4ck_lopo']['typed_accuracy_bean_weighted_same_10_as_v1']:.3f}")
    cc.jdump(res, cc.EVAL_JSON)
    tc.log("wrote", cc.EVAL_JSON)


if __name__ == "__main__":
    main()
