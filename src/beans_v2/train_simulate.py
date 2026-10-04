"""End-to-end simulation of candidate DEPLOY RULES for the v2 classifier.

Per photo: retake gates (app 23:05: too dark, blurred, colour "not green", bean count 20-160, too many touching)
  -> [optional photo-level OOD gate: hand-Mahalanobis, reports_v2/robust_gate_params.json, unchanged]
  -> per-bean call: touching -> unsure; else pGood >= t -> good; 1 - pGood >= t -> defect; else unsure
     [optional colour override: bean mean-colour HSV value < 0.30 (fixed, robust_colour R0) -> defect]
  -> app 23:05 rules: nothing answered / > 60% of answered are defects ("implausible") / > 15% unsure -> "not sure";
     otherwise band of the point estimate (clean < 5% <= some <= 20% < many).

Test sets (every bean is scored by a model that never saw its source / photo, with T and t fitted without it):
  cbd            464 real CBD photos (India light box; never used anywhere in training or fitting), FINAL model
  cbd_stress     111 CBD AAA+AA photos x 7 software re-shoots (robust_ood.PERTURB), FINAL model
  j4ck_lopo      11 real J4ckDev photos, each scored by its LOPO model
  lojano_real    36 real lojano tray photos (all beans one label), LOSO-lojano model
  notplying_real / loja_yolo_real  real photos of those sources, their LOSO model (dense piles / few beans)
  trays_<source> 50-bean trays drawn from the held-out source's own crops at 0 / 10 / 40 / 100 % defects
                 (no re-photographing: the crops are scored as they are; retake gates other than touching do not apply)
Writes reports_v2/model_v2_deploy.json and model_v2_deploy.png.
"""
import json, sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import train_common as tc

ROBUST = tc.SCRATCH / "robust_cache"
GATE = json.load(open(tc.REPORTS_V2 / "robust_gate_params.json"))["hand_mahalanobis_gate"]
R0_V = 0.30
VARIANTS = ["A_sym_t", "P_pair", "P_pair+ood_gate", "P_pair+dark_override", "P_pair+ood+dark"]
SWEEP = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
PERCLASS = {"t": None}
TRAY_N = 50; TRAYS_PER = 40; SHARES = [0.0, 0.10, 0.40, 1.0]


def maha_photo(H):
    if not len(H): return float("nan")
    mu, sd, P = np.array(GATE["mean"]), np.array(GATE["sd"]), np.array(GATE["precision"])
    Z = (np.nan_to_num(H, nan=0.0) - mu) / sd
    return float(np.median(np.sqrt(np.einsum("ij,jk,ik->i", Z, P, Z))))


def bean_calls(p, touching, t, hand=None, dark_override=False):
    """t = one threshold (symmetric) or (t_good, t_defect); a None threshold means that call is never made."""
    tg, td = (t, t) if not isinstance(t, (tuple, list)) else t
    tg = 2.0 if tg is None else tg; td = 2.0 if td is None else td
    pg = p[:, 0]
    c = np.where(pg >= tg, "good", np.where(1 - pg >= td, "defect", "unsure")).astype(object)
    if dark_override and hand is not None:
        c[hand[:, 7] < R0_V] = "defect"
    c[np.asarray(touching, bool)] = "unsure"
    return c


def decide(calls):
    import robust_common as rc
    n = len(calls); unsure = int((calls == "unsure").sum()); defects = int((calls == "defect").sum()); answered = n - unsure
    iv = rc.wilson(defects, answered); uf = unsure / n if n else 1.0
    if answered == 0: return "notsure:too_many_unsure", iv, uf
    if defects / answered > 0.6: return "notsure:implausible", iv, uf
    if uf > 0.15: return "notsure:too_many_unsure", iv, uf
    return rc.band_of_rate(iv["p"]), iv, uf


def verdict(p, touching, hand, thr, variant, checks=None, cg=None):
    """thr = {"sym": t, "pair": (t_good, t_defect)} (a bare number = symmetric t)"""
    import robust_common as rc
    if not isinstance(thr, dict): thr = {"sym": thr, "pair": thr}
    t = thr["sym"] if variant.startswith("A_") else thr["pair"]
    if checks is not None:
        g = rc.gate_reason(checks, cg)
        if g: return {"outcome": "retake:" + g}
    ood = maha_photo(hand)
    if "ood" in variant and ood > GATE["threshold"]:
        return {"outcome": "notsure:ood_gate", "ood": ood}
    calls = bean_calls(p, touching, t, hand, dark_override="dark" in variant)
    out, iv, uf = decide(calls)
    return {"outcome": out, "p_defect": iv["p"], "lo": iv["lo"], "hi": iv["hi"], "unsure_frac": uf, "ood": ood, "n": len(calls)}


def kind(o):
    return "retake" if o.startswith("retake") else ("notsure" if o.startswith("notsure") else "band")


def tally(outs):
    n = len(outs); c = {}
    for o in outs: c[o] = c.get(o, 0) + 1
    k = {"band": 0, "notsure": 0, "retake": 0}
    for o in outs: k[kind(o)] += 1
    return {"photos": n, "given_band": k["band"], "not_sure": k["notsure"], "retake": k["retake"],
            "pct_band": round(100 * k["band"] / n, 1) if n else None, "pct_not_sure": round(100 * k["notsure"] / n, 1) if n else None,
            "pct_retake": round(100 * k["retake"] / n, 1) if n else None, "outcomes": dict(sorted(c.items()))}


# ------------------------------------------------------------------------------------------------ models
def torch_model(name):
    import torch
    net = tc.Norm(tc.make_net(pretrained=False)); net.load_state_dict(torch.load(tc.CACHE / f"folds/{name}.pt", map_location="cpu"))
    return net.eval().to(tc.device())


def logits_np(model, crops):
    import torch
    if not len(crops): return np.zeros((0, 6), np.float32)
    out = []
    with torch.no_grad():
        for b in range(0, len(crops), 256):
            out.append(model(tc.to_float(np.stack(crops[b:b + 256]), tc.device())).float().cpu().numpy())
    return np.concatenate(out)


_W = {}
def _photo_job(args):
    """worker: find beans (+ optional perturbation) -> checks, colour gate, crops, touching, hand stats"""
    import robust_common as rc
    from PIL import Image
    path, pert = args
    im = Image.open(tc.ROOT / path); im = __import__("PIL.ImageOps", fromlist=["x"]).exif_transpose(im).convert("RGB")
    if pert:
        import robust_ood as ro
        im = ro.PERTURB[pert](im)
    r = rc.find_beans(im); cg = rc.colour_gate(r); crops = [b.crop for b in r.beans]
    hand = np.stack([rc.hand_features(c) for c in crops]) if crops else np.zeros((0, 16), np.float32)
    return dict(path=path, pert=pert, checks=r.checks, cg=cg, crops=np.stack(crops) if crops else np.zeros((0, 128, 128, 3), np.uint8),
                touching=np.array([b.touching for b in r.beans], bool), hand=hand)


def run_photos(jobs, workers=8, cache_name=None):
    """bean finder + colour gate + hand stats per photo (no model); cached in CACHE/photos_<name>.pkl"""
    import pickle
    f = tc.CACHE / f"photos_{cache_name}.pkl" if cache_name else None
    if f is not None and f.exists():
        rows = pickle.load(open(f, "rb"))
        if [(r["path"], r["pert"]) for r in rows] == list(jobs): return rows
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_photo_job, jobs, chunksize=2))
    if f is not None: pickle.dump(rows, open(f, "wb"))
    return rows


# ------------------------------------------------------------------------------------------------ main
def main():
    import robust_common as rc
    ev = json.load(open(tc.REPORTS_V2 / "model_v2_eval.json"))
    T_dep, t_dep = ev["deployed"]["temperature"], ev["deployed"]["threshold"]
    pr = ev["deployed"]["pair_thresholds"]
    THR_DEP = {"sym": t_dep, "pair": (pr["t_good"], pr["t_defect"])}
    def thr_of(s):
        r = ev["per_source"][s]
        return {"sym": r["threshold_fitted_on_other_sources"], "pair": (r["pair_thresholds_fitted_on_other_sources"]["good"], r["pair_thresholds_fitted_on_other_sources"]["defect"])}
    def set_pc(s=None): pass
    meta = json.load(open(tc.CACHE / "meta.json")); H_all = np.load(tc.CACHE / "hand.npy")
    src_all = np.array([m["source"] for m in meta]); y_all = tc.kind_targets(meta, range(len(meta)))
    touch_all = np.array([m["touching"] for m in meta], bool)
    res = {"protocol": __doc__.strip(), "variants": VARIANTS, "deployed_T": T_dep, "deployed_threshold": t_dep,
           "ood_gate": {"threshold": GATE["threshold"], "source": "reports_v2/robust_gate_params.json (unchanged)"},
           "dark_override": f"bean mean-colour HSV value < {R0_V} -> defect (fixed rule R0 of robust_colour, nothing fitted)", "sets": {}}
    res["deployed_pair"] = THR_DEP["pair"]
    only = set(sys.argv[1:])

    # ---------------- CBD (real photos) + per-grade bean shares, FINAL model (fp32 ONNX = deployed artifact)
    if not only or "cbd" in only:
        from train_export import ort_probs
        pmeta = json.load(open(ROBUST / "photos.json"))["photos"]; z = np.load(ROBUST / "beans.npz"); pi = z["photo_idx"]
        crops = np.load(ROBUST / "crops.npy", mmap_mode="r")
        exp = json.load(open(tc.REPORTS_V2 / "model_v2_export.json"))
        model_file = tc.ROOT / exp["recommended_file"]
        cbd_b = np.flatnonzero(np.array([pmeta[i]["src"] for i in pi]) == "cbd")
        P = np.zeros((len(pi), 6), np.float32); P[cbd_b] = ort_probs(model_file, crops[cbd_b])
        per_grade = {}; photo_rows = []
        for g in rc.CBD_GRADES:
            ph = [i for i, m in enumerate(pmeta) if m["src"] == "cbd" and m["label"] == g]
            b = np.flatnonzero(np.isin(pi, ph)); p = P[b]; pg = p[:, 0]
            call = np.where(pg >= THR_DEP["pair"][0], "good", np.where(1 - pg >= THR_DEP["pair"][1], "defect", "unsure"))
            am = p.argmax(1)
            per_grade[g] = {"photos": len(ph), "beans": int(len(b)),
                            "share_with_abstain": {"good": float((call == "good").mean()), "defect": float((call == "defect").mean()), "unsure": float((call == "unsure").mean())},
                            "defect_type_share_of_defect_calls_argmax": {c: float(((call == "defect") & (am == i)).sum() / max(1, (call == "defect").sum())) for i, c in enumerate(tc.CLASSES) if i},
                            "share_argmax_no_abstain": {c: float((am == i).mean()) for i, c in enumerate(tc.CLASSES)},
                            "defect_share_of_answered": float((call == "defect").sum() / max(1, (call != "unsure").sum())),
                            "median_binary_confidence": float(np.median(np.maximum(pg, 1 - pg)))}
        res["cbd_per_grade_bean_shares"] = {"model_file": exp["recommended_file"], "thresholds_good_defect": THR_DEP["pair"], "per_grade": per_grade,
                                            "v1_reference": "reports/beans_ood_cbd.json: v1 called 54.3% of all CBD beans broken and 0% of AAA beans good (with abstention)"}
        for v in VARIANTS:
            outs = {}
            for i, m in enumerate(pmeta):
                if m["src"] != "cbd": continue
                b = np.flatnonzero(pi == i)
                r = verdict(P[b], z["touching"][b], z["hand"][b], THR_DEP, v, m["checks"], m["cg"])
                outs.setdefault(m["label"], []).append(r["outcome"])
                if v == "P_pair": photo_rows.append(dict(grade=m["label"], file=Path(m["file"]).name, **{k: r.get(k) for k in ("outcome", "p_defect", "unsure_frac", "ood")}))
            allo = sum(outs.values(), [])
            res["sets"].setdefault("cbd", {})[v] = {"all": tally(allo), "per_grade": {g: tally(outs[g]) for g in rc.CBD_GRADES},
                                                    "AAA_confident_many": sum(o == "many" for o in outs["AAA"])}
        res["cbd_photo_rows_variant_P_pair"] = photo_rows
        sw = {}
        for tt in SWEEP:
            o = []; aaa = 0
            for i, m in enumerate(pmeta):
                if m["src"] != "cbd": continue
                b = np.flatnonzero(pi == i)
                q = verdict(P[b], z["touching"][b], z["hand"][b], {"sym": tt, "pair": (tt, tt)}, "A_sym_t", m["checks"], m["cg"])["outcome"]
                o.append(q); aaa += (m["label"] == "AAA" and q == "many")
            sw[str(tt)] = {**tally(o), "AAA_confident_many": aaa}
        res["sets"]["cbd"]["_threshold_sweep_rule_A_sensitivity_only"] = sw
        tc.log("cbd:", {v: (res["sets"]["cbd"][v]["all"]["given_band"], res["sets"]["cbd"][v]["AAA_confident_many"], res["sets"]["cbd"][v]["all"]["outcomes"]) for v in VARIANTS})

    # ---------------- CBD stress set (re-shot AAA + AA), final model
    if not only or "stress" in only:
        import robust_ood as ro
        from train_export import ort_probs
        pmeta = json.load(open(ROBUST / "photos.json"))["photos"]
        files = [m["file"] for m in pmeta if m["src"] == "cbd" and m["label"] in ("AAA", "AA")]
        jobs = [(f, p) for f in files for p in ro.PERTURB]
        rows = run_photos(jobs, cache_name="stress")
        model_file = tc.ROOT / json.load(open(tc.REPORTS_V2 / "model_v2_export.json"))["recommended_file"]
        allc = [r["crops"] for r in rows if len(r["crops"])]
        PP = ort_probs(model_file, np.concatenate(allc)); off = 0
        for r in rows:
            r["p"] = PP[off:off + len(r["crops"])]; off += len(r["crops"])
        for v in VARIANTS:
            outs = []; many = []
            for r in rows:
                p = r["p"]
                o = verdict(p, r["touching"], r["hand"], THR_DEP, v, r["checks"], r["cg"])["outcome"]
                outs.append(o)
                if o == "many": many.append(f"{Path(r['path']).parent.name}/{Path(r['path']).name}:{r['pert']}")
            res["sets"].setdefault("cbd_stress", {})[v] = {"all": tally(outs), "confident_many": many,
                                                           "by_perturbation": {pp: tally([o for o, r in zip(outs, rows) if r["pert"] == pp]) for pp in ro.PERTURB}}
        tc.log("stress:", {v: res["sets"]["cbd_stress"][v]["all"]["given_band"] for v in VARIANTS})

    # ---------------- the app's own demo photos (read-only), final model. The synthetic trays are pasted from J4ckDev
    # crops, which are in the final model's training set -> IN-SAMPLE, shown only to say what the demo would display.
    if not only or "demo" in only:
        from train_export import ort_probs
        files = sorted(str(p.relative_to(tc.ROOT)) for p in (tc.ROOT / "app/public/demo").glob("*.jpg"))
        rows = run_photos([(f, None) for f in files], cache_name="app_demo")
        model_file = tc.ROOT / json.load(open(tc.REPORTS_V2 / "model_v2_export.json"))["recommended_file"]
        for r in rows: r["p"] = ort_probs(model_file, r["crops"]) if len(r["crops"]) else np.zeros((0, 6), np.float32)
        for v in VARIANTS:
            per = {}
            for r in rows:
                p = r["p"]
                o = verdict(p, r["touching"], r["hand"], THR_DEP, v, r["checks"], r["cg"])
                per[Path(r["path"]).name] = {k: o.get(k) for k in ("outcome", "p_defect", "unsure_frac", "n")}
            res["sets"].setdefault("app_demo_in_sample", {})[v] = {"photos": per, "summary": tally([x["outcome"] for x in per.values()])}
        tc.log("demo:", {k: v["outcome"] for k, v in res["sets"]["app_demo_in_sample"]["P_pair"]["photos"].items()})

    # ---------------- J4ckDev-style HELD-OUT trays (familiar set-up, unseen beans): v1's spatial protocol. Fold k model
    # (train_folds.py halves) never saw the J4ckDev beans with test_fold == k. Each tray = 100 such beans pasted on a
    # 235 sheet with the app's demo-tray recipe (robust_ood.compose -> src/beans/make_demo.bean_rgba), JPEG q90, then the
    # FULL pipeline (bean finder, colour gate, retake gates, classifier, rules). Same seeds and recipe as robust_ood's 30
    # trays, plus 5 more seeds per cell. Thresholds: those fitted for J4ckDev in LOSO (other sources only).
    if (not only or "halves" in only) and (tc.CACHE / "folds/half_2.pt").exists():
        import csv, random, robust_ood as ro
        rows_csv = list(csv.DictReader(open(tc.ROOT / "data/crops/crops.csv")))
        fold = np.array([int(r["test_fold"]) for r in rows_csv])
        jcrops = np.load(tc.CACHE / "crops.npy", mmap_mode="r")[:len(rows_csv)]
        assert all(meta[i]["source"] == "j4ckdev" for i in range(len(rows_csv)))
        DEFN = {0: "clean", 12: "some", 30: "many"}
        thr = thr_of("j4ckdev"); Tj = ev["per_source"]["j4ckdev"]["T_fitted_on_other_sources"]
        trays = []
        for k in (1, 2):
            model = torch_model(f"half_{k}")
            pool = {c: [i for i, r in enumerate(rows_csv) if fold[i] == k and r["touching"] == "0" and r["cls"] == c] for c in ["good", "dark", "insect", "broken", "unhulled"]}
            for nd in DEFN:
                for t in range(10):
                    rng = random.Random(1000 * k + 10 * nd + t)
                    ng = 100 - nd; g = rng.sample(pool["good"], min(ng, len(pool["good"])))
                    g += [rng.choice(pool["good"]) for _ in range(ng - len(g))]
                    chosen = g + [rng.choice(pool[rng.choice(["dark", "insect", "broken", "unhulled"])]) for _ in range(nd)]
                    img = ro.compose([np.asarray(jcrops[i]) for i in chosen], rng, grid=10)
                    r = rc.find_beans(img); cgate = rc.colour_gate(r); cr = [b.crop for b in r.beans]
                    hand = np.stack([rc.hand_features(c) for c in cr]) if cr else np.zeros((0, 16), np.float32)
                    p = tc.softmax(logits_np(model, cr), Tj)
                    trays.append(dict(name=f"fold{k}_def{nd}_{t}", truth=DEFN[nd], p=p, touching=np.array([b.touching for b in r.beans], bool),
                                      hand=hand, checks=r.checks, cg=cgate))
        # LOCAL CALIBRATION (cross-fitted, sensitivity analysis, not the deployed default): T and (t_good, t_defect)
        # re-fitted on J4ckDev beans of the OTHER half, scored by the other half's model (never the tray's own beans).
        import train_eval as te
        loc = {}
        for k in (1, 2):
            zz = np.load(tc.CACHE / f"folds/half_{3 - k}.npz"); yk = tc.kind_targets(meta, zz["idx"]); wk = te.weights_for(np.array(["j4ckdev"] * len(yk)), yk)
            Tk = te.fit_T(zz["logits"], yk, wk); pk = te.pick_pair(tc.softmax(zz["logits"], Tk), yk, wk)
            loc[k] = {"T": Tk, "pair": (pk["t_good"], pk["t_defect"]) if pk else (2.0, 2.0), "fit": pk}
        res["j4ck_local_calibration"] = {str(k): v for k, v in loc.items()}
        for v in VARIANTS:
            outs = []
            for tr in trays:
                k = int(tr["name"][4]); pl = tc.softmax(np.log(np.clip(tr["p"], 1e-12, 1)) * Tj, loc[k]["T"])  # undo Tj, apply local T
                outs.append(verdict(pl, tr["touching"], tr["hand"], {"sym": loc[k]["pair"][1], "pair": loc[k]["pair"]}, v, tr["checks"], tr["cg"]))
            cell = {}
            for tb in ("clean", "some", "many"):
                o = [x["outcome"] for x, tr in zip(outs, trays) if tr["truth"] == tb]
                cell[tb] = {**tally(o), "truth_band": tb, "correct_band": sum(q == tb for q in o), "wrong_band": sum(kind(q) == "band" and q != tb for q in o),
                            "dangerous": sum((tb == "clean" and q == "many") or (tb == "many" and q == "clean") for q in o)}
            res["sets"].setdefault("j4ck_halves_trays_LOCAL_CALIBRATION", {})[v] = {"by_truth": cell, "all": tally([x["outcome"] for x in outs]),
                "trays": [{"name": tr["name"], "truth": tr["truth"], **{kk: x.get(kk) for kk in ("outcome", "p_defect", "unsure_frac", "ood")}} for x, tr in zip(outs, trays)]}
        sw = {}
        for tt in SWEEP:
            outs = [verdict(tr["p"], tr["touching"], tr["hand"], {"sym": tt, "pair": (tt, tt)}, "A_sym_t", tr["checks"], tr["cg"]) for tr in trays]
            sw[str(tt)] = {tb: {"correct_band": sum(x["outcome"] == tb for x, tr in zip(outs, trays) if tr["truth"] == tb),
                                "wrong_band": sum(kind(x["outcome"]) == "band" and x["outcome"] != tb for x, tr in zip(outs, trays) if tr["truth"] == tb),
                                "not_sure": sum(kind(x["outcome"]) == "notsure" for x, tr in zip(outs, trays) if tr["truth"] == tb)} for tb in ("clean", "some", "many")}
        res["j4ck_halves_threshold_sweep_rule_A_sensitivity_only"] = sw
        tc.log("halves trays LOCAL calib:", loc, {v: {tb: (c["correct_band"], c["wrong_band"], c["dangerous"], c["not_sure"]) for tb, c in res["sets"]["j4ck_halves_trays_LOCAL_CALIBRATION"][v]["by_truth"].items()} for v in VARIANTS})
        for v in VARIANTS:
            outs = [verdict(tr["p"], tr["touching"], tr["hand"], thr, v, tr["checks"], tr["cg"]) for tr in trays]
            cell = {}
            for tb in ("clean", "some", "many"):
                o = [x["outcome"] for x, tr in zip(outs, trays) if tr["truth"] == tb]
                cell[tb] = {**tally(o), "truth_band": tb, "correct_band": sum(q == tb for q in o), "wrong_band": sum(kind(q) == "band" and q != tb for q in o),
                            "dangerous": sum((tb == "clean" and q == "many") or (tb == "many" and q == "clean") for q in o)}
            res["sets"].setdefault("j4ck_halves_trays", {})[v] = {"by_truth": cell, "all": tally([x["outcome"] for x in outs]),
                                                                  "trays": [{"name": tr["name"], "truth": tr["truth"], **{kk: x.get(kk) for kk in ("outcome", "p_defect", "unsure_frac", "ood")}} for x, tr in zip(outs, trays)]}
        tc.log("halves trays:", {v: {tb: (c["correct_band"], c["wrong_band"], c["dangerous"], c["not_sure"]) for tb, c in res["sets"]["j4ck_halves_trays"][v]["by_truth"].items()} for v in VARIANTS})

    # ---------------- J4ckDev real photos, LOPO models
    if not only or "j4ck" in only:
        pmeta = json.load(open(ROBUST / "photos.json"))["photos"]
        jm = {m["label"]: m for m in pmeta if m["src"] == "j4ck"}
        Tj = ev["per_source"]["j4ckdev"]["T_fitted_on_other_sources"]; tj = ev["per_source"]["j4ckdev"]["threshold_fitted_on_other_sources"]; set_pc("j4ckdev")
        for v in VARIANTS:
            outs = {}
            for f in sorted((tc.CACHE / "folds").glob("lopo_*.npz")):
                ph = f.stem[5:]; zz = np.load(f); idx = zz["idx"]; p = tc.softmax(zz["logits"], Tj)
                r = verdict(p, touch_all[idx], H_all[idx], thr_of("j4ckdev"), v, jm[ph]["checks"], jm[ph]["cg"])
                outs[ph] = {**r, "truth": "clean (all good)" if ph == "Normales" else "100% defects"}
            res["sets"].setdefault("j4ck_lopo", {})[v] = {"photos": outs, "summary": tally([o["outcome"] for o in outs.values()])}
        tc.log("j4ck:", {k: v["outcome"] for k, v in res["sets"]["j4ck_lopo"]["P_pair"]["photos"].items()})

    # ---------------- real photos of held-out sources (lojano, notplying, loja_yolo), their LOSO model
    if not only or "real" in only:
        import csv
        by_src = {}
        for r in csv.DictReader(open(tc.ROOT / "data/crops_v2/crops_v2.csv")):
            if r["source"] in ("lojano", "notplying_defects", "loja_yolo"):
                by_src.setdefault(r["source"], {})[r["orig_path"]] = r["source_class"]
        # lojano: every raw tray photo (incl. the 8 brown-board ones the crop builder rejected)
        lojano_all = sorted(str(p.relative_to(tc.ROOT)) for p in (tc.ROOT / "data/raw2/lojano").rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                            and ("fotos en bruto" in str(p) or "completas" in str(p)))
        sets = {"lojano_real": ("lojano", lojano_all), "notplying_real": ("notplying_defects", sorted(by_src["notplying_defects"])),
                "loja_yolo_real": ("loja_yolo", sorted(by_src["loja_yolo"]))}
        for name, (s, files) in sets.items():
            rows = run_photos([(f, None) for f in files], cache_name=name)
            model = torch_model(f"loso_{s}")
            Ts = ev["per_source"][s]["T_fitted_on_other_sources"]; ts = ev["per_source"][s]["threshold_fitted_on_other_sources"]; set_pc(s)
            for r in rows: r["p"] = tc.softmax(logits_np(model, list(r["crops"])), Ts) if len(r["crops"]) else np.zeros((0, 6))
            for v in VARIANTS:
                outs = []; per = []
                for r in rows:
                    o = verdict(r["p"], r["touching"], r["hand"], thr_of(s), v, r["checks"], r["cg"])
                    lab = "bueno" if "bueno" in r["path"] else ("defectuoso" if "defectuoso" in r["path"] else "mixed")
                    outs.append(o["outcome"]); per.append({"file": r["path"].split("/")[-1], "label": lab, "outcome": o["outcome"], "p_defect": o.get("p_defect"), "n": int(r["checks"]["count"])})
                d = {"all": tally(outs), "photos": per}
                if name == "lojano_real":
                    d["bueno"] = tally([p["outcome"] for p in per if p["label"] == "bueno"]); d["defectuoso"] = tally([p["outcome"] for p in per if p["label"] == "defectuoso"])
                res["sets"].setdefault(name, {})[v] = d
            tc.log(name, res["sets"][name]["P_pair"]["all"])

    # ---------------- synthetic trays from each held-out source's own crops, LOSO logits
    if not only or "trays" in only:
        rng = np.random.default_rng(0)
        for s in tc.SOURCES:
            f = tc.CACHE / f"folds/loso_{s}.npz"
            if not f.exists(): continue
            zz = np.load(f); idx = zz["idx"]
            Ts = ev["per_source"][s]["T_fitted_on_other_sources"]; ts = ev["per_source"][s]["threshold_fitted_on_other_sources"]
            P = tc.softmax(zz["logits"], Ts); ok = ~touch_all[idx]; set_pc(s)
            good = np.flatnonzero(ok & (y_all[idx] == 0)); bad = np.flatnonzero(ok & (y_all[idx] != 0))
            truth_band = {0.0: "clean", 0.10: "some", 0.40: "many", 1.0: "many"}
            out = {}
            for v in VARIANTS:
                cell = {}; rng = np.random.default_rng(abs(hash(s)) % (1 << 31) if False else tc.SOURCES.index(s))  # same trays for every variant
                for sh in SHARES:
                    nd = int(round(sh * TRAY_N))
                    if (nd and not len(bad)) or (TRAY_N - nd and not len(good)): continue
                    outs = []
                    for _ in range(TRAYS_PER):
                        g = rng.choice(good, TRAY_N - nd, replace=len(good) < TRAY_N - nd) if TRAY_N - nd else np.array([], int)
                        b = rng.choice(bad, nd, replace=len(bad) < nd) if nd else np.array([], int)
                        sel = np.concatenate([g, b]).astype(int)
                        outs.append(verdict(P[sel], np.zeros(len(sel), bool), H_all[idx[sel]], thr_of(s), v)["outcome"])
                    tb = truth_band[sh]
                    t_ = tally(outs); t_["truth_band"] = tb
                    t_["correct_band"] = sum(o == tb for o in outs)
                    t_["wrong_band"] = sum(kind(o) == "band" and o != tb for o in outs)
                    t_["dangerous"] = sum((tb == "clean" and o == "many") or (tb == "many" and o == "clean") for o in outs)
                    cell[f"{int(sh*100)}pct"] = t_
                out[v] = cell
            sweep = {}
            for tt in SWEEP:
                cell = {}
                for sh in SHARES:
                    nd = int(round(sh * TRAY_N))
                    if (nd and not len(bad)) or (TRAY_N - nd and not len(good)): continue
                    r2 = np.random.default_rng(1); outs = []
                    for _ in range(TRAYS_PER):
                        g = r2.choice(good, TRAY_N - nd, replace=len(good) < TRAY_N - nd) if TRAY_N - nd else np.array([], int)
                        b = r2.choice(bad, nd, replace=len(bad) < nd) if nd else np.array([], int)
                        sel = np.concatenate([g, b]).astype(int)
                        outs.append(verdict(P[sel], np.zeros(len(sel), bool), H_all[idx[sel]], {"sym": tt, "pair": (tt, tt)}, "A_sym_t")["outcome"])
                    tb = truth_band[sh]
                    cell[f"{int(sh*100)}pct"] = {"truth_band": tb, "correct_band": sum(o == tb for o in outs), "wrong_band": sum(kind(o) == "band" and o != tb for o in outs),
                                                 "dangerous": sum((tb == "clean" and o == "many") or (tb == "many" and o == "clean") for o in outs),
                                                 "not_sure": sum(kind(o) == "notsure" for o in outs), "photos": len(outs)}
                sweep[str(tt)] = cell
            out["_threshold_sweep_rule_A_sensitivity_only"] = sweep
            res["sets"][f"trays_{s}"] = out
            tc.log(f"trays {s}:", {k: (c["correct_band"], c["wrong_band"], c["not_sure"]) for k, c in out["P_pair"].items()})

    # ---------------- colour rule vs v2 on held-out sources (dark vs not; defect vs good via the dark override)
    if not only or "colour" in only:
        cmp = {}
        for s in tc.SOURCES:
            f = tc.CACHE / f"folds/loso_{s}.npz"
            if not f.exists(): continue
            zz = np.load(f); idx = zz["idx"]; y = y_all[idx]; P = tc.softmax(zz["logits"], ev["per_source"][s]["T_fitted_on_other_sources"])
            ts = ev["per_source"][s]["threshold_fitted_on_other_sources"]; Hs = H_all[idx]
            d = {}
            dark = y == 1; typed = y >= 0
            if dark.any() and (typed & ~dark).any():
                for nm, pred in (("v2_argmax_dark", P.argmax(1) == 1), ("R0_val<0.30", Hs[:, 7] < R0_V), ("R1_luma_med<97.152", Hs[:, 0] < 97.152)):
                    sens = float(pred[dark].mean()); fpr = float(pred[typed & ~dark].mean())
                    d[nm] = {"sensitivity": sens, "false_positive_rate": fpr, "balanced_accuracy": 0.5 * (sens + 1 - fpr)}
            gd = y == 0; df = y != 0
            r0 = Hs[:, 7] < R0_V
            called_good = P[:, 0] >= thr_of(s)["pair"][0]
            d["dark_override"] = {"good_beans_with_R0": float(r0[gd].mean()) if gd.any() else None,
                                  "defect_beans_with_R0": float(r0[df].mean()) if df.any() else None,
                                  "defects_called_good_by_v2": int((df & called_good).sum()), "of_those_caught_by_R0": int((df & called_good & r0).sum()),
                                  "good_called_good_by_v2": int((gd & called_good).sum()), "of_those_flipped_to_defect_by_R0": int((gd & called_good & r0).sum())}
            cmp[s] = d
        res["colour_vs_model_heldout"] = cmp
        tc.log("colour:", {s: {k: round(v["balanced_accuracy"], 3) for k, v in d.items() if "balanced_accuracy" in v} for s, d in cmp.items()})

    prev = tc.REPORTS_V2 / "model_v2_deploy.json"
    if only and prev.exists():
        old = json.load(open(prev)); old["sets"].update(res["sets"])
        for k in res:
            if k != "sets": old[k] = res[k]
        res = old
    json.dump(res, open(prev, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tc.log("wrote reports_v2/model_v2_deploy.json")


def precompute():
    """run only the bean finder parts (no model) so the later simulation is fast"""
    import csv, robust_ood as ro
    pmeta = json.load(open(ROBUST / "photos.json"))["photos"]
    files = [m["file"] for m in pmeta if m["src"] == "cbd" and m["label"] in ("AAA", "AA")]
    run_photos([(f, p) for f in files for p in ro.PERTURB], cache_name="stress")
    by_src = {}
    for r in csv.DictReader(open(tc.ROOT / "data/crops_v2/crops_v2.csv")):
        if r["source"] in ("lojano", "notplying_defects", "loja_yolo"): by_src.setdefault(r["source"], {})[r["orig_path"]] = r["source_class"]
    lojano_all = sorted(str(p.relative_to(tc.ROOT)) for p in (tc.ROOT / "data/raw2/lojano").rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                        and ("fotos en bruto" in str(p) or "completas" in str(p)))
    for name, files in (("lojano_real", lojano_all), ("notplying_real", sorted(by_src["notplying_defects"])), ("loja_yolo_real", sorted(by_src["loja_yolo"]))):
        rows = run_photos([(f, None) for f in files], cache_name=name)
        tc.log(f"precomputed {name}: {len(rows)} photos")


if __name__ == "__main__":
    if sys.argv[1:] == ["precompute"]: precompute()
    else: main()
