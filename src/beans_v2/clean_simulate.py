"""End-to-end deploy simulation of the licence-stated ("clean") v2 model with the app's CURRENT photo rules.

Per photo: [retake gates] -> per bean: touching -> unsure; P(good) >= t_good -> good; 1 - P(good) >= t_defect -> defect;
else unsure -> src/beans/pipeline.decide (Python mirror of app/src/lib/rules.ts decide(): nothing answered / > 60% of
answered defective / > 15% unsure -> "not sure"; else band of the point estimate, clean < 5% <= some <= 20% < many).
Gates: every safety number is given with ALL retake gates ignored (stricter than the app), and also with a port of the
app's gate() order on the frozen Python bean finder's checks (+ colour gate where cached); the app's newer touching.ts
rules are not ported.

Sets
  cbd           464 real CBD photos (never used anywhere), shipped clean fp16 ONNX; per-grade shares; rank metrics
  stress_audit  777 = 111 AAA+AA photos x 7 software re-shoots, the AUDITOR's perturbations and crops (/tmp/audit_v2/ph_stress_*)
  stress_report 777 = the same photos x robust_ood.PERTURB (v2 report's re-shoots; cached bean-finder output)
  lojano_real / loja_yolo_real  real photos, each source's clean LOSO model (never saw the source)
  j4ck_lopo     11 real J4ckDev photos, each scored by its clean LOPO model
  j4ck_1200     the auditor's protocol (AUDIT_V2 (3d), /tmp/audit_v2/j4ck_trays_centres.py): 80-bean crop-level trays from
                the spatially held-out J4ckDev half, 200 per (fold, band) at 0 / 10 / 24 defects, rng seed 2026, clean_half_k
                models; v1 fold models re-run as an integrity check against the auditor's numbers
  trays_heldout 50-bean trays from each held-out group's own crops (auditor design: 0/2/10/15/30/45/100% defects, 200 each,
                seed 12345), clean LOSO logits; and the 5 unlicensed TEST sources with the shipped model
  app_demo      the app's 7 demo photos, shipped clean model (in-sample for the 5 synthetic J4ckDev trays)
Two threshold settings: "deployed" (T and pair fitted on all 6 clean LOSO groups) and "heldout" (T and pair fitted on the
other groups only; the honest one for a held-out group). Writes reports_v2/clean_deploy.json.
"""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import train_common as tc
import clean_common as cc

G9 = ["AAA", "AA", "A", "AB", "PB-I", "PB-II", "C", "Bulk", "Bits"]


def outcome(p, touching, tg, td, checks=None, cg=None):
    codes = cc.calls3(p, tg, td, touching)
    o_ng = cc.decide_codes(codes)[0]
    g = cc.app_gate(checks, cg) if checks is not None else ""
    return o_ng, ("retake:" + g if g else o_ng)


def rank_stats(lab, sh):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    out = {"median_photo_defect_share": {g: float(np.median(sh[lab == g])) for g in G9}}
    for a, b in (("AAA", "Bits"), ("AAA", "C"), ("AAA", "PB-I"), ("AA", "Bits"), ("AAA", "AA")):
        m = np.isin(lab, [a, b]); out[f"auroc_{b}_above_{a}"] = float(roc_auc_score(lab[m] == b, sh[m]))
    r = spearmanr([G9.index(l) for l in lab], sh); out["spearman_vs_folder_order"] = {"rho": float(r.statistic), "p": float(r.pvalue), "n_photos": int(len(lab))}
    return out


def main():
    T, tg, td, ev = cc.deployed()
    only = set(sys.argv[1:])
    sess = cc.ort_session(cc.ONNX16)
    meta = json.load(open(tc.CACHE / "meta.json"))
    src_all = np.array([m["source"] for m in meta]); y_all = tc.kind_targets(meta, range(len(meta)))
    touch_all = np.array([m["touching"] for m in meta], bool)
    pmeta_r = json.load(open(cc.ROBUST / "photos.json"))["photos"]
    def held(g):
        r = ev["per_group"][g]; return r["T_fitted_on_other_groups"], r["pair_fitted_on_other_groups"][0], r["pair_fitted_on_other_groups"][1]
    out_path = tc.REPORTS_V2 / "clean_deploy.json"
    res = json.load(open(out_path)) if (only and out_path.exists()) else {}
    res.update({"protocol": __doc__.strip(), "model_file": tc.rel(cc.ONNX16), "deployed_T": T, "deployed_pair": [tg, td]})
    res.setdefault("sets", {})

    # ------------------------------------------------ CBD
    if not only or "cbd" in only:
        b, pib, pm, tb = cc.cbd_rows()
        X = np.load(cc.ROBUST / "crops.npy", mmap_mode="r")[b]
        P, _ = cc.ort_run(sess, X)
        outs = {"gates_ignored": {}, "with_gates": {}}; rows = []
        for i, m in enumerate(pm):
            if m["src"] != "cbd": continue
            k = pib == i
            o_ng, o_g = outcome(P[k], tb[k], tg, td, m["checks"], m["cg"])
            outs["gates_ignored"].setdefault(m["label"], []).append(o_ng); outs["with_gates"].setdefault(m["label"], []).append(o_g)
            rows.append({"grade": m["label"], "file": Path(m["file"]).name, "gates_ignored": o_ng, "with_gates": o_g,
                         "argmax_defect_share": float((P[k].argmax(1) != 0).mean())})
        d = {}
        for kk, oo in outs.items():
            d[kk] = {"all": cc.tally(sum(oo.values(), [])), "AAA_many": sum(o == "many" for o in oo["AAA"]),
                     "per_grade": {g: cc.tally(oo[g]) for g in G9}}
        per_grade = {}
        for g in G9:
            ids = [i for i, m in enumerate(pm) if m["src"] == "cbd" and m["label"] == g]
            k = np.isin(pib, ids); c = cc.calls3(P[k], tg, td)
            per_grade[g] = {"photos": len(ids), "beans": int(k.sum()), "good": float((c == 0).mean()), "defect": float((c == 1).mean()),
                            "unsure": float((c == 2).mean()), "defect_share_of_answered": float((c == 1).sum() / max(1, (c != 2).sum())),
                            "argmax_good_no_abstain": float((P[k].argmax(1) == 0).mean())}
        lab = np.array([r["grade"] for r in rows]); sh = np.array([r["argmax_defect_share"] for r in rows])
        d["per_grade_bean_shares"] = per_grade; d["rank"] = rank_stats(lab, sh); d["photos"] = rows
        try:
            d["rank_all_sources_v2_reference"] = {k: v for k, v in json.load(open(tc.REPORTS_V2 / "model_v2_cbd_rank.json"))["v2"].items() if k != "model"}
        except Exception: pass
        res["sets"]["cbd"] = d
        tc.log("cbd:", d["gates_ignored"]["all"], "AAA many", d["gates_ignored"]["AAA_many"], d["with_gates"]["AAA_many"], "rank", d["rank"]["spearman_vs_folder_order"], d["rank"]["auroc_Bits_above_AAA"])

    # ------------------------------------------------ stress: auditor's 777 re-shoots (their crops)
    if not only or "stress" in only:
        rec = json.load(open(cc.AUDIT / "ph_stress.json")); C = np.load(cc.AUDIT / "ph_stress_crops.npy", mmap_mode="r"); Tch = np.load(cc.AUDIT / "ph_stress_touch.npy")
        P, _ = cc.ort_run(sess, C); off = 0; o1, o2, many = [], [], []
        for r in rec:
            n = r["n"]; sl = slice(off, off + n); off += n
            if "error" in r["checks"]: o1.append("error"); o2.append("error"); continue
            a, b_ = outcome(P[sl], Tch[sl], tg, td, r["checks"], None); o1.append(a); o2.append(b_)
            if a == "many": many.append(f"{Path(r['path']).parent.name}/{Path(r['path']).name}:{r['pert']}")
        assert off == len(C)
        res["sets"]["stress_audit"] = {"gates_ignored": cc.tally(o1), "with_finder_gates": cc.tally(o2), "confident_many_gates_ignored": many,
                                       "by_perturbation_gates_ignored": {p: cc.tally([o for o, r in zip(o1, rec) if r["pert"] == p]) for p in sorted({r["pert"] for r in rec})}}
        tc.log("stress_audit:", res["sets"]["stress_audit"]["gates_ignored"])
        rows = pickle.load(open(tc.CACHE / "photos_stress.pkl", "rb"))
        allc = np.concatenate([r["crops"] for r in rows if len(r["crops"])]); PP, _ = cc.ort_run(sess, allc); off = 0; o1, o2 = [], []
        for r in rows:
            n = len(r["crops"]); a, b_ = outcome(PP[off:off + n], r["touching"], tg, td, r["checks"], r["cg"]); off += n; o1.append(a); o2.append(b_)
        res["sets"]["stress_report"] = {"gates_ignored": cc.tally(o1), "with_gates": cc.tally(o2)}
        tc.log("stress_report:", res["sets"]["stress_report"]["gates_ignored"])
        del rows, allc

    # ------------------------------------------------ real held-out photos
    if not only or "real" in only:
        for name, s, fold in (("lojano_real", "lojano", "clean_loso_lojano"), ("loja_yolo_real", "loja_yolo", "clean_loso_loja_yolo")):
            rows = pickle.load(open(tc.CACHE / f"photos_{name}.pkl", "rb")); model = cc.load_torch(fold)
            Th, tgh, tdh = held(s)
            d = {"model": fold, "photos": len(rows)}
            Ls = [cc.embed_and_logits(model, r["crops"])[0] if len(r["crops"]) else np.zeros((0, 6)) for r in rows]
            for mode, (TT, a1, a2) in (("heldout", (Th, tgh, tdh)), ("deployed", (T, tg, td))):
                o1, o2, per = [], [], []
                for r, L in zip(rows, Ls):
                    p = tc.softmax(L, TT) if len(L) else np.zeros((0, 6))
                    a, b_ = outcome(p, r["touching"], a1, a2, r["checks"], r["cg"]); o1.append(a); o2.append(b_)
                    lab = "bueno" if "bueno" in r["path"] else ("defectuoso" if "defectuoso" in r["path"] else "mixed")
                    per.append({"file": r["path"].split("/")[-1], "label": lab, "gates_ignored": a, "with_gates": b_, "n": int(len(r["crops"]))})
                d[mode] = {"T": TT, "pair": [a1, a2], "gates_ignored": cc.tally(o1), "with_gates": cc.tally(o2), "photos": per}
                if s == "lojano":
                    d[mode]["bueno_gates_ignored"] = cc.tally([q["gates_ignored"] for q in per if q["label"] == "bueno"])
                    d[mode]["defectuoso_gates_ignored"] = cc.tally([q["gates_ignored"] for q in per if q["label"] == "defectuoso"])
            res["sets"][name] = d
            tc.log(name, {m: (d[m]["gates_ignored"]["outcomes"], d[m]["with_gates"]["given_band"]) for m in ("heldout", "deployed")})
        jm = {m["label"]: m for m in pmeta_r if m["src"] == "j4ck"}
        Th, tgh, tdh = held("j4ckdev"); d = {}
        for mode, (TT, a1, a2) in (("heldout", (Th, tgh, tdh)), ("deployed", (T, tg, td))):
            per = {}
            for f in sorted(cc.FOLDS.glob("clean_lopo_*.npz")):
                ph = f.stem[len("clean_lopo_"):]; z = np.load(f); p = tc.softmax(z["logits"], TT)
                a, b_ = outcome(p, touch_all[z["idx"]], a1, a2, jm[ph]["checks"], jm[ph]["cg"])
                per[ph] = {"truth": "clean (all good)" if ph == "Normales" else "100% defects", "gates_ignored": a, "with_gates": b_}
            d[mode] = {"T": TT, "pair": [a1, a2], "photos": per, "gates_ignored": cc.tally([v["gates_ignored"] for v in per.values()]),
                       "with_gates": cc.tally([v["with_gates"] for v in per.values()])}
        res["sets"]["j4ck_lopo"] = d
        tc.log("j4ck_lopo:", {m: d[m]["gates_ignored"]["outcomes"] for m in d})

    # ------------------------------------------------ auditor's 1,200-tray J4ckDev protocol
    if not only or "j1200" in only:
        import csv, collections
        rows = list(csv.DictReader(open(tc.ROOT / "data/crops/crops.csv")))
        jc = np.load(tc.CACHE / "crops.npy", mmap_mode="r")[:582]
        cls = np.array([r["cls"] for r in rows]); tf = np.array([int(r["test_fold"]) for r in rows])
        rng = np.random.default_rng(2026)
        DESIGN = {"clean": [0], "some": [10], "many": [24]}; NB = 80; NT = 200
        Th, tgh, tdh = held("j4ckdev")
        acc = {}
        for k in (1, 2):
            te_ = np.flatnonzero(tf == k)
            s1 = cc.ort_session(tc.ROOT / f"models/farz_fold{k}_fp32.onnx")
            p1 = s1.run(None, {"image": np.ascontiguousarray(np.asarray(jc[te_]).transpose(0, 3, 1, 2)).astype(np.float32) / 255})[0]
            c1 = np.where(p1.max(1) >= 0.59, np.where(p1.argmax(1) == 0, 0, 1), 2)
            z = np.load(cc.FOLDS / f"clean_half_{k}.npz"); assert np.array_equal(np.sort(z["idx"]), te_)
            L2 = z["logits"][np.argsort(z["idx"])]
            variants = {"v1_fold_abstain0.59 (integrity check vs AUDIT_V2)": c1,
                        f"clean_deployed_T{T:.3f}_{tg:.2f}/{td:.2f}": cc.calls3(tc.softmax(L2, T), tg, td),
                        f"clean_j4ck_heldout_T{Th:.3f}_{tgh:.2f}/{tdh:.2f}": cc.calls3(tc.softmax(L2, Th), tgh, tdh)}
            good = np.flatnonzero(cls[te_] == "good"); bad = np.flatnonzero(cls[te_] != "good")
            trays = []
            for tbn, nds in DESIGN.items():
                for nd in nds:
                    for _ in range(NT):
                        trays.append((tbn, np.r_[rng.choice(bad, nd, replace=False), rng.choice(good, NB - nd, replace=False)]))
            for nm, c in variants.items():
                a = acc.setdefault(nm, ([], []))
                for tbn, sel in trays:
                    a[0].append(tbn); a[1].append(cc.decide_codes(c[sel])[0])
        res["sets"]["j4ck_1200"] = {nm: cc.tray_score(t, o) for nm, (t, o) in acc.items()}
        res["sets"]["j4ck_1200"]["_auditor_reference"] = "AUDIT_V2 (3d): v1 793 bands, 736 correct, 57 wrong, 10 dangerous; all-sources v2 deployed 229 bands, 0 wrong, 0 dangerous"
        tc.log("j4ck_1200:", {nm: {k: v for k, v in s.items() if k != "by_truth"} for nm, s in res["sets"]["j4ck_1200"].items() if not nm.startswith("_")})

    # ------------------------------------------------ crop-level trays per held-out group + unlicensed test sources
    if not only or "trays" in only:
        rng = np.random.default_rng(12345)
        SH = [0.0, 0.02, 0.10, 0.15, 0.30, 0.45, 1.0]; NT = 200; NB = 50
        band = lambda r: "clean" if r < 0.05 else "some" if r <= 0.2 else "many"
        items = []
        for g in ev["per_group"]:
            z = np.load(cc.FOLDS / f"clean_loso_{g}.npz"); items.append((g, z["idx"], z["logits"], held(g)))
        zu = np.load(tc.CACHE / "clean_emb_final_unlicensed.npz")
        for s in cc.UNLICENSED:
            m = src_all[zu["idx"]] == s; items.append((f"TEST_{s}", zu["idx"][m], zu["logits"][m], (T, tg, td)))
        out = {}
        for name, idx, lg, (Th, a1, a2) in items:
            y = y_all[idx]; ok = ~touch_all[idx]; good = np.flatnonzero(ok & (y == 0)); bad = np.flatnonzero(ok & (y != 0))
            modes = [("heldout", (Th, a1, a2))] if name.startswith("TEST_") else [("heldout", (Th, a1, a2)), ("deployed", (T, tg, td))]
            for mode, (TT, b1, b2) in modes:
                codes = cc.calls3(tc.softmax(lg, TT), b1, b2); t_, o_ = [], []
                for sh in SH:
                    if sh < 1 and len(good) < NB: continue
                    if sh > 0 and len(bad) < NB: continue
                    nd = int(round(sh * NB))
                    for _ in range(NT):
                        sel = np.r_[rng.choice(bad, nd, replace=False) if nd else np.array([], int), rng.choice(good, NB - nd, replace=False) if NB - nd else np.array([], int)]
                        t_.append(band(nd / NB)); o_.append(cc.decide_codes(codes[sel])[0])
                out[f"{name}|{mode}"] = cc.tray_score(t_, o_) if t_ else {"trays": 0}
        tot = {}
        for k, v in out.items():
            if not v.get("trays"): continue
            key = ("TEST_unlicensed" if k.startswith("TEST_") else "licensed_heldout") + "|" + k.split("|")[1]
            t = tot.setdefault(key, {"trays": 0, "bands": 0, "correct": 0, "wrong": 0, "dangerous_clean_called_many": 0, "dangerous_many_called_clean": 0, "groups_with_bands": []})
            for f in ("trays", "bands", "correct", "wrong", "dangerous_clean_called_many", "dangerous_many_called_clean"): t[f] += v[f]
            if v["bands"]: t["groups_with_bands"].append(f"{k.split('|')[0]}: {v['bands']}")
        res["sets"]["trays_heldout"] = {"per_group": out, "totals": tot}
        tc.log("trays totals:", tot)

    # ------------------------------------------------ app demo photos (in-sample for J4ckDev trays)
    if not only or "demo" in only:
        rows = pickle.load(open(tc.CACHE / "photos_app_demo.pkl", "rb")); per = {}
        for r in rows:
            p, _ = cc.ort_run(sess, r["crops"]) if len(r["crops"]) else (np.zeros((0, 6)), None)
            a, b_ = outcome(p, r["touching"], tg, td, r["checks"], r["cg"])
            per[Path(r["path"]).name] = {"gates_ignored": a, "with_gates": b_, "n": int(len(r["crops"]))}
        res["sets"]["app_demo_in_sample"] = per
        tc.log("demo:", per)
    cc.jdump(res, out_path)
    tc.log("wrote", out_path)


if __name__ == "__main__":
    main()
