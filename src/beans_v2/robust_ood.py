"""(3) PHOTO-LEVEL OUT-OF-DISTRIBUTION GATE: "this sample does not look like what the model learned from -> not sure,
take the sample to the cooperative", decided BEFORE any band is shown.

Candidate photo scores (higher = less like the J4ckDev training photos). Per bean, then the photo's MEDIAN bean:
  S1 v1_lowconf  share of beans with v1 max-prob < 0.59 (the app's existing '>15% unsure' rule is S1 > 0.15)
  S2 v1_energy   -logsumexp(raw v1 logits)
  S3 emb_knn     mean cosine distance to the 5 nearest J4ckDev beans, frozen ImageNet MobileNetV3-Small 576-d
  S4 v1f_knn     same, in the v1 model's own 576-d pooled features
  S5 hand_maha   Mahalanobis distance of the 16 colour/shape stats to the J4ckDev bean cloud (Ledoit-Wolf)
J4ckDev photos are scored LEAVE-ONE-PHOTO-OUT: the bank / covariance excludes the photo's own beans, and S1/S2/S4 use
v1-recipe models retrained without that photo (robust_lopo_v1.py). CBD and other photos use the full bank and the
deployed v1 model. Verdicts use a Python port of the app's rules (gate -> per-bean call -> '>15% unsure' -> Wilson band).
Calibration (as specified): pick the threshold from J4ckDev LOPO photos (must pass) and CBD AAA photos only (none may get a
confident 'many'); every other CBD grade, the app's demo trays and perturbed J4ckDev photos are held out.
Writes reports_v2/robust_ood.json and robust_ood.png.
"""
import json, sys, glob, io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image, ImageFilter
import robust_common as rc

ABSTAIN = json.load(open(rc.V1_LABELS))["abstain_below"]
KB = 5
SCORES = ["v1_lowconf", "v1_energy", "emb_knn", "v1f_knn", "hand_maha", "hand_maha_colour", "hand_maha_shape", "v1f_unfam_share"]
COLOUR_COLS = list(range(11)); SHAPE_COLS = list(range(11, 16))  # robust_common.HAND_NAMES order
TB = {"value": None}  # bean-level familiarity threshold, set from the held-out halves before anything else is scored
TBQ = {}  # sensitivity sweep: percentile of held-out-half distances -> threshold (95 is the pre-chosen value)

def unit(E): return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)

def knn_to_bank(E, B, k=KB):
    D = 1 - unit(E) @ unit(B).T
    return np.sort(D, 1)[:, :k].mean(1)

class Maha:
    def __init__(s, X, cols=None):
        from sklearn.covariance import LedoitWolf
        s.cols = cols if cols is not None else list(range(X.shape[1])); X = X[:, s.cols]
        s.bank = X if cols is None else None
        X = X[~np.isnan(X).any(1)]; s.mu = X.mean(0); s.sd = X.std(0) + 1e-6
        s.P = np.linalg.inv(LedoitWolf().fit((X - s.mu) / s.sd).covariance_)
    def __call__(s, X):
        Z = (np.nan_to_num(X[:, s.cols], nan=0.0) - s.mu) / s.sd
        return np.sqrt(np.einsum("ij,jk,ik->i", Z, s.P, Z))

def energy(logits):
    m = logits.max(1, keepdims=True); return -(m[:, 0] + np.log(np.exp(logits - m).sum(1)))

def photo_scores(probs, logits, emb, feat, hand, bankE, bankF, maha):
    d = knn_to_bank(feat, bankF)
    if not isinstance(maha, dict): maha = maha_set(maha)
    return dict(hand_maha_colour=float(np.median(maha["colour"](hand))), hand_maha_shape=float(np.median(maha["shape"](hand))),v1f_unfam_share=float(np.mean(d > TB["value"])), _unfam=(d > TB["value"]).tolist(), _d=d.tolist(),v1_lowconf=float(np.mean(probs.max(1) < ABSTAIN)), v1_energy=float(np.median(energy(logits))),
                emb_knn=float(np.median(knn_to_bank(emb, bankE))), v1f_knn=float(np.median(knn_to_bank(feat, bankF))),
                hand_maha=float(np.median(maha["all"](hand))))

def maha_set(m):
    """from a full Maha (fitted on some bank) build the three variants on the same bank"""
    return {"all": m, "colour": Maha(m.bank, COLOUR_COLS), "shape": Maha(m.bank, SHAPE_COLS)}

# ---------------------------------------------------------------------------- extra photos (demo trays, perturbations)
_S = None; _X = None
def _init():
    global _S, _X
    _S = rc.v1_session(threads=1); _X = rc.imagenet_extractor()

PERTURB = {
    "darker_x0.6": lambda im: Image.eval(im, lambda v: int(v * 0.6)),
    "warm_cast": lambda im: Image.merge("RGB", [c.point(lambda v, g=g: min(255, int(v * g))) for c, g in zip(im.split(), (1.0, 0.85, 0.65))]),
    "half_res": lambda im: im.resize((im.width // 3, im.height // 3), Image.BILINEAR),
    "blur_r3": lambda im: im.filter(ImageFilter.GaussianBlur(3)),
    "brighter_x1.3": lambda im: Image.eval(im, lambda v: min(255, int(v * 1.3))),
    "cool_cast": lambda im: Image.merge("RGB", [c.point(lambda v, g=g: min(255, int(v * g))) for c, g in zip(im.split(), (0.75, 0.9, 1.0))]),
    "jpeg_q15": lambda im: Image.open(io.BytesIO((lambda b: (im.save(b, "JPEG", quality=15), b.getvalue())[1])(io.BytesIO()))).convert("RGB"),
}

def _extra(args):
    name, path, pert = args
    im = Image.open(path).convert("RGB")
    if pert: im = PERTURB[pert](im)
    r = rc.find_beans(im); cg = rc.colour_gate(r); crops = [b.crop for b in r.beans]
    probs, feat, logits = rc.v1_run(_S, crops)
    hand = np.stack([rc.hand_features(c) for c in crops]) if crops else np.zeros((0, 16), np.float32)
    emb = rc.embed(_X, crops) if crops else np.zeros((0, 576), np.float32)
    return dict(name=name, pert=pert, checks=r.checks, cg=cg, touching=np.array([b.touching for b in r.beans]), probs=probs,
                logits=logits, feat=feat, hand=hand, emb=emb)

def integrated(probs, touching, checks, cg, sc):
    """S7: the app's own rule with one change - a bean whose v1 features are unfamiliar (kNN distance > TB) is 'unsure',
    exactly like a low-confidence or touching bean; the existing '>15% unsure -> not sure' rule then decides."""
    p = np.array(probs, np.float32).copy(); p[np.array(sc["_unfam"], bool)] = 0.0
    return rc.v1_verdict(p, touching, checks, cg, ABSTAIN)["band"]

def compose(crops, rng, grid=8):
    """Same layout recipe as src/beans/make_demo.make_tray (sheet 235 + noise, random rotation, ~60 px beans, grid
    with jitter, JPEG q90 round trip), for any list of crops."""
    from make_demo import bean_rgba, CELL, MARGIN, BEAN_PX
    rng.shuffle(crops); side = 2 * MARGIN + grid * CELL
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    canvas = Image.fromarray(np.clip(rc.SHEET_TARGET + nprng.normal(0, 1.2, (side, side, 3)), 0, 255).astype(np.uint8), "RGB")
    for i, c in enumerate(crops):
        b = bean_rgba(np.asarray(c)); s = BEAN_PX * rng.uniform(0.88, 1.12) / max(b.size)
        b = b.resize((max(8, round(b.width * s)), max(8, round(b.height * s))), Image.BICUBIC).rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)
        gy, gx = divmod(i, grid)
        cx = MARGIN + gx * CELL + CELL // 2 + rng.randint(-10, 10); cy = MARGIN + gy * CELL + CELL // 2 + rng.randint(-10, 10)
        canvas.paste(b, (cx - b.width // 2, cy - b.height // 2), b)
    buf = io.BytesIO(); canvas.save(buf, "JPEG", quality=90); buf.seek(0)
    return Image.open(buf).convert("RGB")

def integ_all(probs, touching, checks, cg, sc):
    out = {"band_int": integrated(probs, touching, checks, cg, sc), "band_int_sweep": {}}
    d = np.array(sc["_d"])
    for q, t in TBQ.items():
        p = np.array(probs, np.float32).copy(); p[d > t] = 0.0
        out["band_int_sweep"][str(q)] = rc.v1_verdict(p, touching, checks, cg, ABSTAIN)["band"]
    return out

# ----------------------------------------------------------------------------
def main():
    if "--analyse-only" in sys.argv:
        rows = json.load(open(rc.CACHE / "ood_rows.json")); TB["value"] = json.load(open(rc.REPORTS_V2 / "robust_ood.json"))["integrated_rule_S7"]["bean_familiarity_threshold"]
        for r in rows: r.pop("_unfam", None); r.pop("_d", None)
        res = analyse(rows, True); rc.dump(res, "robust_ood.json"); plot(rows, res); return
    meta = json.load(open(rc.CACHE / "photos.json"))["photos"]; z = np.load(rc.CACHE / "beans.npz"); pi = z["photo_idx"]
    jidx = [i for i, m in enumerate(meta) if m["src"] == "j4ck"]
    jb = np.isin(pi, jidx)  # J4ckDev beans (cache order == crops.csv order)
    bankE, bankF, bankH = z["emb"][jb], z["feat"][jb], z["hand"][jb]
    jphoto = np.array([meta[i]["label"] for i in pi[jb]])
    lopo = np.load(rc.CACHE / "lopo_v1.npz") if (rc.CACHE / "lopo_v1.npz").exists() else None
    maha_all = Maha(bankH)
    rows = []
    # bean-level familiarity threshold S6/S7: 95th percentile of v1-feature kNN distance of HELD-OUT half beans to the
    # fold model's own training half (fixed here, before any CBD / LOPO / demo photo is scored)
    import csv
    fold = np.array([int(r["test_fold"]) for r in csv.DictReader(open(rc.CROPS_CSV))])
    crops = np.load(rc.CACHE / "crops.npy", mmap_mode="r"); jbeans = np.flatnonzero(jb)
    jcrops = [crops[i] for i in jbeans]; jtouch = z["touching"][jb]
    fold_out = {k: rc.v1_run(rc.v1_session(rc.ROOT / f"models/farz_fold{k}_fp32.onnx", threads=8), jcrops) for k in (1, 2)}
    dh = np.concatenate([knn_to_bank(fold_out[k][1][fold == k], fold_out[k][1][fold != k]) for k in (1, 2)])
    TB["value"] = float(np.percentile(dh, 95))
    for q in (90, 95, 99, 99.5): TBQ[q] = float(np.percentile(dh, q))
    for i, m in enumerate(meta):
        b = np.flatnonzero(pi == i)
        if m["src"] == "j4ck":
            ph = m["label"]; keep = jphoto != ph; mh = Maha(bankH[keep])
            if lopo is not None:
                f = list(lopo["order"]).index(ph); own = lopo["photos"] == ph
                pr, lg, ft = lopo["probs"][f][own], lopo["logits"][f][own], lopo["feat"][f][own]; bF = lopo["feat"][f][~own]
            else:
                pr, lg, ft, bF = z["probs"][b], z["logits"][b], z["feat"][b], bankF[keep]
            sc = photo_scores(pr, lg, z["emb"][b], ft, z["hand"][b], bankE[keep], bF, mh)
            v = rc.v1_verdict(pr, z["touching"][b], m["checks"], m["cg"], ABSTAIN)
            v_in = rc.v1_verdict(z["probs"][b], z["touching"][b], m["checks"], m["cg"], ABSTAIN)
            rows.append(dict(set="j4ck_lopo", name=ph, grade=rc.PHOTO_CLASS[ph], band=v["band"], band_v0=v.get("band_v0", v["band"]), about=bool(v.get("about", False)), p_def=float(v.get("p", float("nan"))), unsure_frac=float(v.get("unsure_frac", float("nan"))), why=v.get("why", ""), band_insample=v_in["band"],
                             **integ_all(pr, z["touching"][b], m["checks"], m["cg"], sc), **sc))
        else:
            sc = photo_scores(z["probs"][b], z["logits"][b], z["emb"][b], z["feat"][b], z["hand"][b], bankE, bankF, maha_all)
            v = rc.v1_verdict(z["probs"][b], z["touching"][b], m["checks"], m["cg"], ABSTAIN)
            rows.append(dict(set="cbd", name=Path(m["file"]).name, grade=m["label"], band=v["band"], band_v0=v.get("band_v0", v["band"]), about=bool(v.get("about", False)), p_def=float(v.get("p", float("nan"))), unsure_frac=float(v.get("unsure_frac", float("nan"))), why=v.get("why", ""),
                             **integ_all(z["probs"][b], z["touching"][b], m["checks"], m["cg"], sc), **sc))
    # J4ckDev spatial halves (v1's own held-out protocol): fold-k model scores the half it never saw; bank = training half
    for k in (1, 2):
        pr, ft, lg = fold_out[k]
        test = fold == k; mh = Maha(bankH[~test])
        for ph in sorted(rc.PHOTO_CLASS):
            sel = test & (jphoto == ph)
            if sel.sum() < 3: continue
            sc = photo_scores(pr[sel], lg[sel], bankE[sel], ft[sel], bankH[sel], bankE[~test], ft[~test], mh)
            mi = [i for i in jidx if meta[i]["label"] == ph][0]
            ck = dict(meta[mi]["checks"]); ck["count"] = int(sel.sum())
            v = rc.v1_verdict(pr[sel], jtouch[sel], ck, meta[mi]["cg"], ABSTAIN)
            rows.append(dict(set="j4ck_halves", name=f"{ph}#half{k}", grade=rc.PHOTO_CLASS[ph], band=v["band"], band_v0=v.get("band_v0", v["band"]), about=bool(v.get("about", False)), p_def=float(v.get("p", float("nan"))), unsure_frac=float(v.get("unsure_frac", float("nan"))), why=v.get("why", ""),
                             **integ_all(pr[sel], jtouch[sel], ck, meta[mi]["cg"], sc), **sc))
    # J4ckDev-style HELD-OUT synthetic trays: 100 beans pasted on a 235 sheet with the app's own demo-tray recipe
    # (src/beans/make_demo.bean_rgba), beans drawn ONLY from fold-k's test half, scored by fold-k model + training-half banks.
    # A fold's test half has < 100 non-touching good beans, so good beans are drawn without replacement first and the
    # remainder is filled with replacement (a repeated bean is pasted with a new rotation/size).
    import random
    from make_demo import bean_rgba, CELL, MARGIN, BEAN_PX
    rows_csv = list(csv.DictReader(open(rc.CROPS_CSV))); X = rc.imagenet_extractor()
    DEFN = {0: "clean", 12: "some", 30: "many"}  # true defect share 0 / 12 / 30 % of 100 -> Wilson band clean / some / many
    for k in (1, 2):
        sess = rc.v1_session(rc.ROOT / f"models/farz_fold{k}_fp32.onnx", threads=8); tr = fold != k; mh = Maha(bankH[tr])
        pool = {c: [i for i, r in enumerate(rows_csv) if fold[i] == k and r["touching"] == "0" and r["cls"] == c] for c in rc.CLASSES}
        for nd in DEFN:
            for t in range(5):
                rng = random.Random(1000 * k + 10 * nd + t)
                ng = 100 - nd; g = rng.sample(pool["good"], min(ng, len(pool["good"])))
                g += [rng.choice(pool["good"]) for _ in range(ng - len(g))]
                chosen = g + [rng.choice(pool[rng.choice(rc.CLASSES[1:])]) for _ in range(nd)]
                img = compose([jcrops[i] for i in chosen], rng, grid=10)
                r = rc.find_beans(img); cg = rc.colour_gate(r); cr = [b.crop for b in r.beans]
                pr, ft, lg = rc.v1_run(sess, cr); emb = rc.embed(X, cr); hd = np.stack([rc.hand_features(c) for c in cr])
                sc = photo_scores(pr, lg, emb, ft, hd, bankE[tr], fold_out[k][1][tr], mh)
                tch = np.array([b.touching for b in r.beans])
                v = rc.v1_verdict(pr, tch, r.checks, cg, ABSTAIN)
                rows.append(dict(set="j4ck_trays", name=f"fold{k}_def{nd}_{t}", grade=DEFN[nd], band=v["band"], band_v0=v.get("band_v0", v["band"]), about=bool(v.get("about", False)), p_def=float(v.get("p", float("nan"))), unsure_frac=float(v.get("unsure_frac", float("nan"))), why=v.get("why", ""),
                                 **integ_all(pr, tch, r.checks, cg, sc), n_found=len(cr), **sc))
    # demo trays + perturbed J4ckDev photos (held out from calibration)
    jobs = [(Path(p).name, p, None) for p in sorted(glob.glob(str(rc.ROOT / "app/public/demo/*.jpg")))]
    jobs += [(stem, str(rc.J4CK_DIR / f"{stem}.jpg"), pert) for stem in sorted(rc.PHOTO_CLASS) for pert in PERTURB]
    # STRESS SET (held out from every calibration): top-grade CBD photos (AAA, AA) re-shot in software - another
    # exposure, colour cast, resolution, blur or compression - i.e. "a different phone / light" for the same beans
    jobs += [(f"cbdP|{m['label']}|{Path(m['file']).name}", str(rc.ROOT / m["file"]), pert) for m in meta
             if m["src"] == "cbd" and m["label"] in ("AAA", "AA") for pert in PERTURB]
    with ProcessPoolExecutor(max_workers=12, initializer=_init) as ex:
        extra = list(ex.map(_extra, jobs))
    for e in extra:
        cbdp = e["name"].startswith("cbdP|")
        if not len(e["probs"]):
            rows.append(dict(set="cbd_perturbed" if cbdp else ("demo" if e["pert"] is None else "j4ck_perturbed"),
                             name=e["name"], grade=e["name"].split("|")[1] if cbdp else (e["pert"] or ""), pert=e["pert"], band="retake:count",
                             band_v0="retake:count", about=False, band_int="retake:count", band_int_sweep={str(q): "retake:count" for q in TBQ}, **{k: float("nan") for k in SCORES})); continue
        if cbdp:
            sc = photo_scores(e["probs"], e["logits"], e["emb"], e["feat"], e["hand"], bankE, bankF, maha_all); st = "cbd_perturbed"
        elif e["pert"] is None:
            sc = photo_scores(e["probs"], e["logits"], e["emb"], e["feat"], e["hand"], bankE, bankF, maha_all); st = "demo"
        else:  # perturbed J4ckDev photo: score against the bank WITHOUT that photo (S3, S5); S1/S2/S4 use the deployed model (in-sample)
            keep = jphoto != e["name"]
            sc = photo_scores(e["probs"], e["logits"], e["emb"], e["feat"], e["hand"], bankE[keep], bankF[keep], Maha(bankH[keep])); st = "j4ck_perturbed"
        v = rc.v1_verdict(e["probs"], e["touching"], e["checks"], e["cg"], ABSTAIN)
        rows.append(dict(set=st, name=e["name"], grade=e["name"].split("|")[1] if cbdp else (e["pert"] or ""), pert=e["pert"], band=v["band"], band_v0=v.get("band_v0", v["band"]), about=bool(v.get("about", False)), p_def=float(v.get("p", float("nan"))), unsure_frac=float(v.get("unsure_frac", float("nan"))), why=v.get("why", ""),
                         **integ_all(e["probs"], e["touching"], e["checks"], e["cg"], sc), **sc))
    json.dump(rows, open(rc.CACHE / "ood_rows.json", "w"))
    for r in rows: r.pop("_unfam", None); r.pop("_d", None)
    res = analyse(rows, lopo is not None)
    rc.dump(res, "robust_ood.json")
    plot(rows, res)

def analyse(rows, have_lopo):
    from robust_unusual import auroc
    sets = {k: [r for r in rows if r["set"] == k] for k in ["j4ck_halves", "j4ck_trays", "j4ck_lopo", "cbd", "demo", "j4ck_perturbed", "cbd_perturbed"]}
    CP = sets["cbd_perturbed"]
    C = sets["cbd"]; AAA = [r for r in C if r["grade"] == "AAA"]
    conf = lambda r: r["band"] in ("clean", "some", "many")
    res = {"what": __doc__.strip().split("\n")[0], "lopo_v1_models_used": have_lopo, "abstain_below": ABSTAIN,
           "sets": {k: len(v) for k, v in sets.items()}, "baseline_v1_without_ood_gate": {}, "scores": {}}
    for g in rc.CBD_GRADES:
        R = [r for r in C if r["grade"] == g]
        res["baseline_v1_without_ood_gate"][g] = {b: sum(r["band"] == b for r in R) for b in ["clean", "some", "many", "unsure"]} | {"photos": len(R)}
    res["baseline_v1_without_ood_gate"]["cbd_all_confident_band_share"] = round(float(np.mean([conf(r) for r in C])), 4)
    res["rules_version"] = rc.RULES_VERSION
    res["baseline_app_rules_2240_for_comparison"] = {
        "note": "same photos through the 22:40 app rules (band only if the whole 95% interval is in one band; no 'implausible' rule)",
        "cbd_confident": f"{sum(r['band_v0'] in ('clean', 'some', 'many') for r in C)}/{len(C)}",
        "cbd_AAA_many": f"{sum(r['band_v0'] == 'many' for r in AAA)}/{len(AAA)}",
        "j4ck_trays_correct": f"{sum(r['band_v0'] == r['grade'] for r in sets['j4ck_trays'])}/{len(sets['j4ck_trays'])}",
        "j4ck_trays_wrong_confident": sum(r['band_v0'] in ('clean', 'some', 'many') and r['band_v0'] != r['grade'] for r in sets['j4ck_trays'])}
    res["baseline_v1_without_ood_gate"]["cbd_unsure_reason_note"] = "current rules: 'implausible' (> 60% defects) was added at 23:05 and its 0.6 was tuned on CBD-like photos (rules.ts comment), so CBD is partly in-sample for it"
    res["baseline_v1_without_ood_gate"]["cbd_about_share_of_confident"] = round(float(np.mean([r['about'] for r in C if conf(r)])), 4) if any(conf(r) for r in C) else None
    for k in ["j4ck_halves", "j4ck_lopo", "demo"]:
        res["baseline_v1_without_ood_gate"][k] = {r["name"]: r["band"] for r in sets[k]}
    res["baseline_v1_without_ood_gate"]["j4ck_lopo_in_sample_deployed_model"] = {r["name"]: r["band_insample"] for r in sets["j4ck_lopo"]}
    for s in SCORES:
        out = {"per_photo": {k: {r["name"]: round(r[s], 4) for r in sets[k]} for k in ["j4ck_halves", "j4ck_lopo", "demo"]},
               "auroc_vs_cbd": {k: round(float(auroc(np.array([r[s] for r in C]), np.array([r[s] for r in sets[k]]))), 4) for k in ["j4ck_halves", "j4ck_trays", "j4ck_lopo"]},
               "cbd_aaa_many_min": float(min([r[s] for r in AAA if r["band_v0"] == "many"], default=np.nan)),
               "cbd_aaa_many_min_note": "AAA photos v1 calls 'many' under the 22:40 rules (no 'implausible' rule): the 27 photos where v1's own evidence says 'many'", "calibrations": {}}
        calibs = {"fixed_app_rule_0.15": None} if s in ("v1_lowconf", "v1f_unfam_share") else {}
        calibs.update({"on_j4ck_halves": "j4ck_halves", "on_j4ck_lopo": "j4ck_lopo", "zero_AAA_many": "AAA"})
        for cname, cset in calibs.items():
            if cset is None: tau, how = 0.15, "fixed 15% share (the app's existing unsure-share cut-off), not fitted"
            elif cset == "AAA":
                tau = float(np.nextafter(out["cbd_aaa_many_min"], -np.inf)); how = "largest threshold that refuses EVERY CBD AAA photo whose v1 evidence says 'many' (22:40 rules; calibrated on AAA only)"
            else:
                jmax = float(np.nanmax([r[s] for r in sets[cset]])); hi = out["cbd_aaa_many_min"]
                if np.isnan(hi) or hi > jmax: tau, how = (jmax + 0.5 * (hi - jmax)) if not np.isnan(hi) else jmax, "midpoint: max J4ckDev score (must pass) .. min CBD-AAA 'many' score (must be refused)"
                else: tau, how = jmax, "NO GAP: threshold = max J4ckDev score so every calibration photo passes; AAA 'many' photos above it are refused, the rest are not"
            gate = lambda r: bool(r[s] > tau) if not np.isnan(r[s]) else False
            o = {"threshold": float(tau), "rule": how,
                 "false_refusal": {k: f"{sum(gate(r) for r in sets[k])}/{len(sets[k])}" for k in ["j4ck_halves", "j4ck_trays", "j4ck_lopo"]},
                 "false_refusal_names": {k: [r["name"] for r in sets[k] if gate(r)] for k in ["j4ck_halves", "j4ck_lopo"]},
                 "j4ck_trays_confident_after_gate": f"{sum((not gate(r)) and conf(r) for r in sets['j4ck_trays'])}/{len(sets['j4ck_trays'])}",
                 "j4ck_trays_confident_and_correct_after_gate": f"{sum((not gate(r)) and r['band'] == r['grade'] for r in sets['j4ck_trays'])}/{len(sets['j4ck_trays'])}",
                 "demo_trays_refused": {r["name"]: gate(r) for r in sets["demo"]},
                 "j4ck_perturbed_refused": {p: f"{sum(gate(r) for r in sets['j4ck_perturbed'] if r['grade'] == p)}/{sum(1 for r in sets['j4ck_perturbed'] if r['grade'] == p)}" for p in PERTURB},
                 "cbd_refused_share": round(float(np.mean([gate(r) for r in C])), 4),
                 "cbd_aaa_confident_many_after_gate": f"{sum((not gate(r)) and r['band'] == 'many' for r in AAA)}/{len(AAA)}",
                 "cbd_false_confidence_any_band_all_grades": round(float(np.mean([(not gate(r)) and conf(r) for r in C])), 4),
                 "cbd_per_grade": {g: {"photos": sum(r["grade"] == g for r in C), "refused": sum(gate(r) for r in C if r["grade"] == g),
                                       "confident_band_after_gate": sum((not gate(r)) and conf(r) for r in C if r["grade"] == g)} for g in rc.CBD_GRADES}}
            out["calibrations"][cname] = o
            print(f"{s:11s} {cname:20s} tau={tau:.4f} FR halves {o['false_refusal']['j4ck_halves']} trays {o['false_refusal']['j4ck_trays']} (conf {o['j4ck_trays_confident_after_gate']}, correct {o['j4ck_trays_confident_and_correct_after_gate']}) lopo {o['false_refusal']['j4ck_lopo']} "
                  f"CBD refused {o['cbd_refused_share']} AAA-many {o['cbd_aaa_confident_many_after_gate']} CBD conf {o['cbd_false_confidence_any_band_all_grades']} "
                  f"demo {sum(o['demo_trays_refused'].values())}/{len(o['demo_trays_refused'])} pert {o['j4ck_perturbed_refused']}")
        print("   AUROC", out["auroc_vs_cbd"], "lopo", out["per_photo"]["j4ck_lopo"])
        res["scores"][s] = out
    # Gate comparison on ONE definition: the verdict the farmer would get AFTER the gate (refused -> "unsure").
    # Photo-score gates refuse the whole photo; S7 changes bean calls inside decide() (its verdict is band_int).
    T = lambda sc, c: res["scores"][sc]["calibrations"][c]["threshold"]
    ref = {"hand_maha": lambda x: x["hand_maha"] > T("hand_maha", "zero_AAA_many"),
           "energy": lambda x: x["v1_energy"] > T("v1_energy", "zero_AAA_many"),
           "emb_knn": lambda x: x["emb_knn"] > T("emb_knn", "zero_AAA_many")}
    AFTER = {"none (app rules 23:05)": lambda x: x["band"],
             "none (app rules 22:40, for reference)": lambda x: x["band_v0"],
             "hand_maha@zeroAAA": lambda x: "unsure" if ref["hand_maha"](x) else x["band"],
             "energy@zeroAAA": lambda x: "unsure" if ref["energy"](x) else x["band"],
             "emb_knn@zeroAAA": lambda x: "unsure" if ref["emb_knn"](x) else x["band"],
             "hand_maha OR emb_knn @zeroAAA": lambda x: "unsure" if (ref["hand_maha"](x) or ref["emb_knn"](x)) else x["band"],
             "S7 (95th pct)": lambda x: x["band_int"],
             "S7 (99th pct)": lambda x: x["band_int_sweep"]["99"],
             "hand_maha@zeroAAA + S7 (95th)": lambda x: "unsure" if ref["hand_maha"](x) else x["band_int"]}
    isconf = lambda b: b in ("clean", "some", "many")
    held = [r for r in C if r["grade"] != "AAA"]; TR = sets["j4ck_trays"]
    # margin of the 23:05 'implausible' rule: defect share among answered beans, for photos that pass the 15% unsure rule
    def margin(R):
        # photos the 15% unsure rule would NOT stop: only the implausible rule stands between them and a band
        v = np.array([r["p_def"] for r in R if not r["band"].startswith("retake") and np.isfinite(r["unsure_frac"]) and r["unsure_frac"] <= 0.15], float)
        v = v[np.isfinite(v)]
        return {"photos_with_unsure_share_le_15pct": int(len(v)), "of_photos": len(R), "min_defect_share": float(v.min()) if len(v) else None,
                "share_below_0.6": round(float(np.mean(v <= 0.6)), 4) if len(v) else None,
                "share_in_0.5_to_0.6": round(float(np.mean((v > 0.5) & (v <= 0.6))), 4) if len(v) else None,
                "percentiles_5_25_50": [round(float(q), 4) for q in np.percentile(v, [5, 25, 50])] if len(v) else None}
    res["implausible_rule_margin"] = {"cbd": margin(C), "stress_cbd_AAA_AA_perturbed": margin(CP),
                                      "why_counts_cbd": {w: sum(r.get("why") == w for r in C) for w in ["implausible", "too_many_unsure", ""]},
                                      "why_counts_stress": {w: sum(r.get("why") == w for r in CP) for w in ["implausible", "too_many_unsure", ""]},
                                      "stress_confident_photos": [dict(name=r["name"], pert=r["pert"], band=r["band"], p_def=r["p_def"], unsure_frac=r["unsure_frac"]) for r in CP if isconf(r["band"])]}
    print("implausible margin", json.dumps(res["implausible_rule_margin"]))
    res["gate_comparison"] = {}
    for name, after in AFTER.items():
        lost = lambda R: f"{sum(isconf(r['band']) and not isconf(after(r)) for r in R)}/{sum(isconf(r['band']) for r in R)}"
        res["gate_comparison"][name] = {
            "j4ck_trays_correct": f"{sum(after(r) == r['grade'] for r in TR)}/{len(TR)}",
            "j4ck_trays_wrong_confident": sum(isconf(after(r)) and after(r) != r["grade"] for r in TR),
            "j4ck_halves_confident_lost": lost(sets["j4ck_halves"]), "j4ck_lopo_confident_lost": lost(sets["j4ck_lopo"]),
            "j4ck_perturbed_confident": f"{sum(isconf(after(r)) for r in sets['j4ck_perturbed'])}/{len(sets['j4ck_perturbed'])}",
            "app_demo": {r["name"]: after(r) for r in sets["demo"]},
            "cbd_all_confident": f"{sum(isconf(after(r)) for r in C)}/{len(C)}",
            "cbd_AAA_many": f"{sum(after(r) == 'many' for r in AAA)}/{len(AAA)}",
            "cbd_heldout_grades_confident": f"{sum(isconf(after(r)) for r in held)}/{len(held)}",
            "STRESS_cbd_AAA_AA_perturbed_confident": f"{sum(isconf(after(r)) for r in CP)}/{len(CP)}",
            "STRESS_some_or_many": f"{sum(after(r) in ('some', 'many') for r in CP)}/{len(CP)}",
            "STRESS_by_perturbation": {pp: f"{sum(isconf(after(r)) for r in CP if r['pert'] == pp)}/{sum(1 for r in CP if r['pert'] == pp)}" for pp in PERTURB},
            "STRESS_bands": {b: sum(after(r) == b for r in CP) for b in ["clean", "some", "many"]}}
        print(f"{name:40s}", {k: v for k, v in res["gate_comparison"][name].items() if k not in ("app_demo", "STRESS_by_perturbation")})
    # S7 integrated rule: unfamiliar beans become 'unsure' inside the app's own decide(); no extra photo threshold
    bands = ["clean", "some", "many", "unsure"]
    integ = {"bean_familiarity_threshold": TB["value"], "how": "95th percentile of held-out-half beans' v1-feature 5-NN cosine distance to their fold model's training half",
             "per_set": {}}
    for k in ["j4ck_halves", "j4ck_trays", "j4ck_lopo", "demo", "j4ck_perturbed"]:
        integ["per_set"][k] = {r["name"] + (f"/{r['grade']}" if k == "j4ck_perturbed" else ""): {"before": r["band"], "after": r["band_int"]} for r in sets[k]}
    integ["cbd_per_grade"] = {g: {"before": {b: sum(r["band"] == b for r in C if r["grade"] == g) for b in bands},
                                  "after": {b: sum(r["band_int"] == b for r in C if r["grade"] == g) for b in bands}} for g in rc.CBD_GRADES}
    integ["cbd_confident_share_before"] = round(float(np.mean([conf(r) for r in C])), 4)
    integ["cbd_confident_share_after"] = round(float(np.mean([r["band_int"] in ("clean", "some", "many") for r in C])), 4)
    integ["cbd_aaa_confident_many_after"] = f"{sum(r['band_int'] == 'many' for r in AAA)}/{len(AAA)}"
    T = sets["j4ck_trays"]
    integ["j4ck_trays"] = {"true_band": {b: sum(r["grade"] == b for r in T) for b in ["clean", "some", "many"]},
                           "before": {"confident": sum(conf(r) for r in T), "correct": sum(r["band"] == r["grade"] for r in T),
                                      "wrong_confident": sum(conf(r) and r["band"] != r["grade"] for r in T)},
                           "after": {"confident": sum(r["band_int"] in ("clean", "some", "many") for r in T), "correct": sum(r["band_int"] == r["grade"] for r in T),
                                     "wrong_confident": sum(r["band_int"] in ("clean", "some", "many") and r["band_int"] != r["grade"] for r in T)}}
    for k in ["j4ck_halves", "j4ck_trays", "j4ck_lopo", "demo"]:
        R = sets[k]
        integ[f"{k}_newly_refused"] = f"{sum(r['band'] != 'unsure' and r['band_int'] == 'unsure' for r in R)}/{sum(r['band'] in ('clean', 'some', 'many') for r in R)} confident photos turned 'not sure'"
    sweep = {}
    for q in (sorted(rows[0]["band_int_sweep"]) if rows and "band_int_sweep" in rows[0] else []):
        bi = lambda r: r["band_int_sweep"][q]
        sweep[q] = {"cbd_all_confident": f"{sum(bi(r) in ('clean', 'some', 'many') for r in C)}/{len(C)}",
                    "cbd_AAA_confident_many": f"{sum(bi(r) == 'many' for r in AAA)}/{len(AAA)}",
                    "j4ck_trays_confident_correct": f"{sum(bi(r) == r['grade'] for r in T)}/{len(T)}",
                    "j4ck_trays_confident_wrong": sum(bi(r) in ('clean', 'some', 'many') and bi(r) != r["grade"] for r in T),
                    "j4ck_lopo_confident": f"{sum(bi(r) in ('clean', 'some', 'many') for r in sets['j4ck_lopo'])}/{len(sets['j4ck_lopo'])}",
                    "j4ck_perturbed_confident": f"{sum(bi(r) in ('clean', 'some', 'many') for r in sets['j4ck_perturbed'])}/{len(sets['j4ck_perturbed'])}",
                    "STRESS_cbd_AAA_AA_perturbed_confident": f"{sum(bi(r) in ('clean', 'some', 'many') for r in CP)}/{len(CP)}"}
    integ["sensitivity_sweep_percentile_of_heldout_half_distances"] = sweep
    integ["sweep_note"] = "95 was fixed before looking at CBD; the other percentiles are shown for sensitivity only"
    print("S7 sweep", json.dumps(sweep, indent=0))
    res["integrated_rule_S7"] = integ
    print(json.dumps({k: v for k, v in integ.items() if k != "per_set"}, indent=0))
    print(json.dumps(integ["per_set"]["j4ck_halves"]), json.dumps(integ["per_set"]["j4ck_lopo"]), json.dumps(integ["per_set"]["demo"]))
    return res

def plot(rows, res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(20, 9)); axes = axes.ravel()
    order = ["j4ck_halves", "j4ck_trays", "j4ck_lopo", "demo", "j4ck_perturbed"] + rc.CBD_GRADES
    names = ["J4ck\nhalves", "J4ck\nheld-out\ntrays", "J4ck\nLOPO", "app\ndemo", "J4ck\nperturbed"] + rc.CBD_GRADES
    for ax, s in zip(axes, SCORES):
        for j, grp in enumerate(order):
            R = [r for r in rows if (r["set"] == grp) or (r["set"] == "cbd" and r["grade"] == grp)]
            if not len(R): continue
            v = np.array([r[s] for r in R], float)
            col = "#3a9a3a" if grp.startswith("j4ck") or grp == "demo" else "#2a7fb8"
            many = np.array([r["band"] == "many" for r in R])
            xj = j + np.random.default_rng(j).uniform(-0.25, 0.25, len(v))
            ax.scatter(xj[~many], v[~many], s=7, c=col, alpha=0.6); ax.scatter(xj[many], v[many], s=14, c="#c03030", marker="x")
        cal = res["scores"][s]["calibrations"]
        ax.axhline(cal["on_j4ck_halves"]["threshold"], color="k", ls="--", lw=1, label="all J4ck halves pass")
        ax.axhline(cal["zero_AAA_many"]["threshold"], color="#c03030", ls=":", lw=1.2, label="zero AAA 'many'")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(names, fontsize=7)
        a = res["scores"][s]["auroc_vs_cbd"]
        ax.set_title(f"{s}  (AUROC vs CBD: halves {a['j4ck_halves']:.2f}, trays {a['j4ck_trays']:.2f}, LOPO {a['j4ck_lopo']:.2f})", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Photo-level OOD scores. Green = J4ckDev-style, blue = CBD (India light box); red x = v1 would say 'many defects'. Refused = above the line.", fontsize=11)
    fig.tight_layout(); fig.savefig(rc.REPORTS_V2 / "robust_ood.png", dpi=110); plt.close(fig)
    # gate comparison
    G = res["gate_comparison"]; names = list(G)
    frac = lambda t: (lambda a, b: a / b if b else 0.0)(*map(int, t.split("/")))
    series = [("cbd_all_confident", "#2a7fb8", "CBD photos (464) given a band (want 0)"),
              ("STRESS_cbd_AAA_AA_perturbed_confident", "#c03030", "STRESS: AAA/AA photos re-shot (other exposure/cast/res.) given a band (want 0)"),
              ("j4ck_trays_correct", "#3a9a3a", "held-out J4ckDev-style trays (30): correct band (want 1)")]
    fig, ax = plt.subplots(figsize=(14, 5.8)); x = np.arange(len(names)); w = 0.26
    for k, (key, col, lab) in enumerate(series):
        ax.bar(x + (k - 1) * w, [frac(G[n][key]) for n in names], w, color=col, label=lab)
    for i2, n in enumerate(names):
        wr = G[n]["j4ck_trays_wrong_confident"]
        if wr: ax.text(i2 + w, frac(G[n]["j4ck_trays_correct"]) + 0.02, f"{wr} wrong", ha="center", fontsize=7, color="#c03030")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8); ax.set_ylim(0, 1.08); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("OOD gate options under the app's current rules: confident answers on out-of-distribution photos vs. correct answers kept", fontsize=10)
    fig.tight_layout(); fig.savefig(rc.REPORTS_V2 / "robust_ood_gates.png", dpi=120); plt.close(fig)

if __name__ == "__main__":
    main()
