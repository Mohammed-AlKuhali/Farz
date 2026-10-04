"""Local-calibration study (pre-registered in reports_v2/localcal.md BEFORE any result; that block is hashed below).

Frozen base model embeddings (the clean model's penultimate feature, L2-normalised = the ONNX "embed" output) + a local
head built from K grader-labelled beans (good / defect) from a few calibration photos of ONE source; tested on the beans
of DIFFERENT photos of that source, on 500 trays per draw, and on all 464 CBD photos (safety elsewhere).

  primary  : loja_yolo  (base clean_loso_loja_yolo: loja_yolo never in training)
  secondary: lojano (base clean_loso_lojano), mindforge_doubleside and usk_coffee (base = shipped clean model; test-only)
  heads    : PROTOTYPES (pre-registered primary) and logistic regression (reported alternative); baseline = base model
             alone with the clean deployed thresholds
Writes reports_v2/localcal.json and CACHE/localcal_tables.md (markdown tables pasted into localcal.md).
"""
import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import train_common as tc
import clean_common as cc

KS = [20, 50, 100]; DRAWS = 5
SOURCES = {"loja_yolo": "clean_loso_loja_yolo", "lojano": "clean_loso_lojano",
           "mindforge_doubleside": "FINAL", "usk_coffee": "FINAL"}
SI = {s: i for i, s in enumerate(SOURCES)}
TRAY_N = 50; TRAY_MIX = [(0, "clean", 167), (6, "some", 167), (15, "many", 166)]
PREREG_SHA_AT_0214 = "d97a6694d2f253e877b831b2758602d42135235c1e552f730f5795794b34837c"


# ------------------------------------------------------------------------------------------------ calibration draw
def draw_calibration(groups, ybin, K, rng):
    """groups: photo id per (non-touching) bean; ybin: 0 good / 1 defect. Pre-registered procedure. -> (cal idx, test mask, n photos)"""
    photos = np.unique(groups); order = rng.permutation(len(photos)); photos = photos[order]
    has_good = {p for p in np.unique(groups[ybin == 0])}
    chosen = []; ng = nd = 0
    for p in photos:
        if ng >= K // 2: break
        if p in has_good:
            chosen.append(p); m = groups == p; ng += int((ybin[m] == 0).sum()); nd += int((ybin[m] == 1).sum())
    if nd < K // 2:
        for p in photos:
            if nd >= K // 2: break
            if p in chosen: continue
            chosen.append(p); m = groups == p; ng += int((ybin[m] == 0).sum()); nd += int((ybin[m] == 1).sum())
    if ng < K // 2 or nd < K // 2: return None
    pool = np.isin(groups, chosen)
    cg = rng.choice(np.flatnonzero(pool & (ybin == 0)), K // 2, replace=False)
    cd = rng.choice(np.flatnonzero(pool & (ybin == 1)), K // 2, replace=False)
    return np.concatenate([cg, cd]), ~pool, len(chosen)


# ------------------------------------------------------------------------------------------------ heads
def l2n(v): return v / max(np.linalg.norm(v), 1e-12)


def proto_fit(E, y):
    """-> (mu_good, mu_defect, m_hi, loo_scores). y 0/1."""
    Eg, Ed = E[y == 0].astype(np.float64), E[y == 1].astype(np.float64)
    sg, sd = Eg.sum(0), Ed.sum(0)
    mu_g, mu_d = l2n(sg), l2n(sd)
    loo = np.empty(len(y))
    for i in range(len(y)):
        if y[i] == 0: g, d = l2n(sg - E[i]), mu_d
        else: g, d = mu_g, l2n(sd - E[i])
        loo[i] = E[i] @ d - E[i] @ g
    wrong = (loo >= 0).astype(int) != y
    m_hi = max(float(np.abs(loo[wrong]).max()) if wrong.any() else 0.0, float(np.median(np.abs(loo))))
    return mu_g, mu_d, m_hi, loo


def proto_score(E, mu_g, mu_d): return E.astype(np.float64) @ mu_d - E.astype(np.float64) @ mu_g


def logit_fit(E, y):
    from sklearn.linear_model import LogisticRegression
    mk = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf = mk().fit(E.astype(np.float64), y)
    loo = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        loo[i] = mk().fit(E[m].astype(np.float64), y[m]).predict_proba(E[i:i + 1].astype(np.float64))[0, 1] - 0.5
    wrong = (loo >= 0).astype(int) != y
    m_hi = max(float(np.abs(loo[wrong]).max()) if wrong.any() else 0.0, float(np.median(np.abs(loo))))
    return clf, m_hi, loo


def answer(score, m_hi, base_def):
    """score >= 0 -> local says defect. Answer when local == base binary call, or |score| >= m_hi. -> codes 0/1/2, how"""
    local = (score >= 0).astype(int)
    agree = local == base_def; high = np.abs(score) >= m_hi
    ans = agree | high
    return np.where(ans, local, 2), {"agree": int(agree.sum()), "high_margin_only": int((high & ~agree).sum())}


# ------------------------------------------------------------------------------------------------ metrics
def bean_metrics(codes, ybin):
    ans = codes != 2; ok = codes == ybin
    w = np.where(ybin == 0, 0.5 / max(1, (ybin == 0).sum()), 0.5 / max(1, (ybin == 1).sum()))
    return {"n": int(len(ybin)), "n_good": int((ybin == 0).sum()), "n_defect": int((ybin == 1).sum()),
            "coverage_balanced": float(w[ans].sum() / w.sum()), "answered_accuracy_balanced": float(np.average(ok[ans], weights=w[ans])) if ans.any() else None,
            "coverage_plain": float(ans.mean()), "answered_accuracy_plain": float(ok[ans].mean()) if ans.any() else None,
            "good_beans_called_good_defect_unsure": [float((codes[ybin == 0] == k).mean()) for k in (0, 1, 2)] if (ybin == 0).any() else None,
            "defect_beans_called_defect_good_unsure": [float((codes[ybin == 1] == k).mean()) for k in (1, 0, 2)] if (ybin == 1).any() else None}


def trays(codes_by_head, ybin, seed):
    """500 trays from test beans; the SAME trays for every head. -> {head: tray_score}"""
    rng = np.random.default_rng(seed)
    g, d = np.flatnonzero(ybin == 0), np.flatnonzero(ybin == 1)
    sels, truths, repl = [], [], False
    for nd, tb, nt in TRAY_MIX:
        for _ in range(nt):
            rg = len(g) < TRAY_N - nd; rd = len(d) < nd; repl |= rg or rd
            sels.append(np.concatenate([rng.choice(g, TRAY_N - nd, replace=rg), rng.choice(d, nd, replace=rd) if nd else np.array([], int)]).astype(int))
            truths.append(tb)
    out = {}
    for h, c in codes_by_head.items():
        outs = [cc.decide_codes(c[s])[0] for s in sels]
        out[h] = cc.tray_score(truths, outs)
    out["_sampled_with_replacement"] = bool(repl); out["_test_pool_good_defect"] = [int(len(g)), int(len(d))]
    return out


def cbd_eval(codes, pi, pmeta):
    """per-photo band on all CBD photos, gates ignored (stricter) and with the app-order gate port."""
    import robust_common as rc
    res = {"gates_ignored": {}, "with_gates": {}}
    for i, m in enumerate(pmeta):
        if m["src"] != "cbd": continue
        b = pi == i
        o = cc.decide_codes(codes[b])[0]
        res["gates_ignored"].setdefault(m["label"], []).append(o)
        g = cc.app_gate(m["checks"], m.get("cg"))
        res["with_gates"].setdefault(m["label"], []).append("retake:" + g if g else o)
    out = {}
    for k, d in res.items():
        allo = sum(d.values(), [])
        out[k] = {"all": cc.tally(allo), "AAA_many": sum(o == "many" for o in d["AAA"]), "AAA": cc.tally(d["AAA"]),
                  "bands_by_grade": {gr: cc.tally(d[gr])["given_band"] for gr in rc.CBD_GRADES}}
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    T, tg, td, ev = cc.deployed()
    prereg = open(tc.REPORTS_V2 / "localcal.md").read().split("\n## Results")[0]
    meta = json.load(open(tc.CACHE / "meta.json"))
    src_all = np.array([m["source"] for m in meta]); y_all = tc.kind_targets(meta, range(len(meta)))
    grp_all = np.array([m["group"] for m in meta]); touch_all = np.array([m["touching"] for m in meta], bool)
    crops = np.load(tc.CACHE / "crops.npy", mmap_mode="r")
    cb, cpi, pmeta, ctouch = cc.cbd_rows()
    ccrops = np.ascontiguousarray(np.load(cc.ROBUST / "crops.npy", mmap_mode="r")[cb])
    res = {"protocol": __doc__.strip(), "prereg_file_sha256_at_02_14": PREREG_SHA_AT_0214,
           "prereg_block_sha256_now": hashlib.sha256(prereg.encode()).hexdigest(),
           "base_T": T, "baseline_pair": [tg, td], "Ks": KS, "draws": DRAWS, "tray_design": {"beans": TRAY_N, "mix": TRAY_MIX}, "sources": {}}
    base_cache = {}
    for s, base in SOURCES.items():
        if base not in base_cache:
            model = cc.load_torch(cc.FINAL_PT if base == "FINAL" else base)
            Lc, Ec = cc.embed_and_logits(model, ccrops)
            base_cache[base] = {"model": model, "cbd": (tc.softmax(Lc, T), Ec)}
        model = base_cache[base]["model"]
        rows = np.flatnonzero((src_all == s) & ~touch_all)
        L, E = cc.embed_and_logits(model, crops[rows]); P = tc.softmax(L, T)
        ybin = (y_all[rows] != 0).astype(int); groups = grp_all[rows]
        base_def = (P[:, 0] < 0.5).astype(int)
        Pc, Ec = base_cache[base]["cbd"]; cbase_def = (Pc[:, 0] < 0.5).astype(int)
        cbase_codes = cc.calls3(Pc, tg, td, ctouch)
        sres = {"base_model": base, "n_beans_non_touching": int(len(rows)), "n_touching_excluded": int(((src_all == s) & touch_all).sum()),
                "n_photos": int(len(np.unique(groups))), "n_good": int((ybin == 0).sum()), "n_defect": int((ybin == 1).sum()),
                "base_alone_all_beans": bean_metrics(cc.calls3(P, tg, td), ybin),
                "base_argmax_all_beans": bean_metrics(base_def, ybin),
                "cbd_base_alone": cbd_eval(cbase_codes, cpi, pmeta), "runs": {}}
        tc.log(f"{s}: {len(rows)} beans, base alone {sres['base_alone_all_beans']['coverage_balanced']:.3f} cov / {sres['base_alone_all_beans']['answered_accuracy_balanced']}")
        for K in KS:
            runs = []
            for d in range(DRAWS):
                seed = 10000 * SI[s] + 100 * K + d
                dr = draw_calibration(groups, ybin, K, np.random.default_rng(seed))
                if dr is None:
                    runs.append({"draw": d, "skipped": "source cannot supply K/2 of each class"}); continue
                cal, test, nph = dr
                Ecal, ycal = E[cal], ybin[cal]
                mu_g, mu_d, m_hi, loo = proto_fit(Ecal, ycal)
                sp = proto_score(E, mu_g, mu_d); codes_p, how_p = answer(sp[test], m_hi, base_def[test])
                clf, m_hi_l, loo_l = logit_fit(Ecal, ycal)
                sl = clf.predict_proba(E.astype(np.float64))[:, 1] - 0.5; codes_l, how_l = answer(sl[test], m_hi_l, base_def[test])
                codes_b = cc.calls3(P[test], tg, td)
                yt = ybin[test]
                r = {"draw": d, "seed": seed, "calibration_photos": nph, "calibration_photo_ids": sorted(set(groups[cal].tolist())),
                     "n_test": int(test.sum()), "n_test_photos": int(len(np.unique(groups[test]))),
                     "proto": {"m_hi": m_hi, "loo_errors": int(((loo >= 0).astype(int) != ycal).sum()), "how": how_p, "beans": bean_metrics(codes_p, yt),
                               "local_head_alone_no_abstain": bean_metrics((sp[test] >= 0).astype(int), yt)},
                     "logistic": {"m_hi": m_hi_l, "loo_errors": int(((loo_l >= 0).astype(int) != ycal).sum()), "how": how_l, "beans": bean_metrics(codes_l, yt)},
                     "baseline": {"beans": bean_metrics(codes_b, yt)}}
                r["trays"] = trays({"proto": codes_p, "logistic": codes_l, "baseline": codes_b}, yt, seed + 50000)
                # CBD with THIS source's base model and THIS calibration (gates ignored = stricter)
                cs = proto_score(Ec, mu_g, mu_d); ccodes, _ = answer(cs, m_hi, cbase_def); ccodes = np.where(ctouch, 2, ccodes)
                cl = clf.predict_proba(Ec.astype(np.float64))[:, 1] - 0.5; lcodes, _ = answer(cl, m_hi_l, cbase_def); lcodes = np.where(ctouch, 2, lcodes)
                r["cbd"] = {"proto": cbd_eval(ccodes, cpi, pmeta), "logistic": cbd_eval(lcodes, cpi, pmeta)}
                r["calibration_beans_rows"] = [int(rows[i]) for i in cal]
                r["calibration_labels"] = ycal.tolist()
                r["proto_params"] = {"mu_good": mu_g.astype(np.float32).tolist(), "mu_defect": mu_d.astype(np.float32).tolist()} if (s == "loja_yolo" and d == 0) else None
                runs.append(r)
                pb = r["proto"]["beans"]; tp = r["trays"]["proto"]
                tc.log(f"  {s} K={K} d={d}: photos {nph} | proto cov {pb['coverage_balanced']:.3f} acc {pb['answered_accuracy_balanced']} "
                       f"| trays bands {tp['bands']} wrong {tp['wrong']} dang {tp['dangerous_clean_called_many']} | CBD AAA many {r['cbd']['proto']['gates_ignored']['AAA_many']} "
                       f"| logit cov {r['logistic']['beans']['coverage_balanced']:.3f} acc {r['logistic']['beans']['answered_accuracy_balanced']}")
            sres["runs"][str(K)] = runs
        res["sources"][s] = sres
        cc.jdump(res, tc.REPORTS_V2 / "localcal.json")  # save early (network/compute drops)
    res["verdict"] = verdict(res)
    res["seconds"] = round(time.time() - t0)
    cc.jdump(res, tc.REPORTS_V2 / "localcal.json")
    tables(res)
    tc.log("localcal verdict:", res["verdict"]["verdict"])


def summarise(runs, head):
    ok = [r for r in runs if "skipped" not in r]
    if not ok: return None
    acc = [r[head]["beans"]["answered_accuracy_balanced"] or 0 for r in ok]; cov = [r[head]["beans"]["coverage_balanced"] for r in ok]
    accp = [r[head]["beans"]["answered_accuracy_plain"] or 0 for r in ok]; covp = [r[head]["beans"]["coverage_plain"] for r in ok]
    tr = [r["trays"][head] for r in ok]
    out = {"draws": len(ok), "mean_acc_bal": float(np.mean(acc)), "min_acc_bal": float(np.min(acc)), "mean_cov_bal": float(np.mean(cov)), "min_cov_bal": float(np.min(cov)),
           "mean_acc_plain": float(np.mean(accp)), "mean_cov_plain": float(np.mean(covp)),
           "trays": int(sum(t["trays"] for t in tr)), "bands": int(sum(t["bands"] for t in tr)), "correct": int(sum(t["correct"] for t in tr)),
           "wrong": int(sum(t["wrong"] for t in tr)), "dangerous": int(sum(t["dangerous_clean_called_many"] for t in tr)),
           "many_called_clean": int(sum(t["dangerous_many_called_clean"] for t in tr)),
           "photos_per_draw": [r["calibration_photos"] for r in ok]}
    if head in ("proto", "logistic"):
        out["cbd_AAA_many_per_draw_gates_ignored"] = [r["cbd"][head]["gates_ignored"]["AAA_many"] for r in ok]
        out["cbd_bands_per_draw_gates_ignored"] = [r["cbd"][head]["gates_ignored"]["all"]["given_band"] for r in ok]
    return out


def verdict(res):
    s = res["sources"].get("loja_yolo")
    v = {"rule": "pre-registered (reports_v2/localcal.md)", "per_K": {}}
    go_k = None; go_k_log = None
    for K in KS:
        runs = s["runs"][str(K)]
        p = summarise(runs, "proto"); l = summarise(runs, "logistic")
        if p is None: continue
        c1 = p["mean_acc_bal"] >= 0.90 and p["mean_cov_bal"] >= 0.50; c2 = p["dangerous"] == 0; c3 = all(x == 0 for x in p["cbd_AAA_many_per_draw_gates_ignored"]) and p["draws"] == DRAWS
        l1 = l["mean_acc_bal"] >= 0.92 and l["mean_cov_bal"] >= 0.55; l2 = l["dangerous"] == 0; l3 = all(x == 0 for x in l["cbd_AAA_many_per_draw_gates_ignored"]) and l["draws"] == DRAWS
        v["per_K"][str(K)] = {"proto": p, "proto_criteria_1_2_3": [c1, c2, c3], "logistic": l, "logistic_exception_criteria": [l1, l2, l3]}
        if go_k is None and c1 and c2 and c3: go_k = K
        if go_k_log is None and l1 and l2 and l3: go_k_log = K
    if go_k is not None: v["verdict"] = f"GO (prototypes, K = {go_k})"; v["K"] = go_k; v["head"] = "proto"
    elif go_k_log is not None: v["verdict"] = f"GO (logistic, pre-registered exception, K = {go_k_log})"; v["K"] = go_k_log; v["head"] = "logistic"
    else: v["verdict"] = "NO-GO"; v["K"] = None; v["head"] = None
    return v


def tables(res):
    lines = []
    for s, sr in res["sources"].items():
        lines.append(f"\n### {s} (base `{sr['base_model']}`; {sr['n_beans_non_touching']} non-touching beans in {sr['n_photos']} photos/groups: {sr['n_good']} good, {sr['n_defect']} defect)\n")
        b = sr["base_alone_all_beans"]
        acc = "n/a" if b["answered_accuracy_balanced"] is None else f"{100*b['answered_accuracy_balanced']:.1f}%"
        lines.append(f"Base model alone (thresholds {res['baseline_pair'][0]:.2f}/{res['baseline_pair'][1]:.2f}), all beans of the source: balanced coverage {100*b['coverage_balanced']:.1f}%, balanced answered accuracy {acc}.\n")
        lines.append("| K | head | calib. photos per draw | mean balanced coverage (min) | mean balanced answered acc (min) | plain coverage / acc | trays: bands / wrong / dangerous (clean->many) / many->clean (of 2,500) | CBD AAA 'many' per draw (gates ignored) | CBD photos banded per draw |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for K in KS:
            for head in ("proto", "logistic", "baseline"):
                m = summarise(sr["runs"][str(K)], head)
                if m is None: lines.append(f"| {K} | {head} | skipped | | | | | | |"); continue
                cb = m.get("cbd_AAA_many_per_draw_gates_ignored", ["-"]); cbb = m.get("cbd_bands_per_draw_gates_ignored", ["-"])
                lines.append(f"| {K} | {head} | {m['photos_per_draw']} | {100*m['mean_cov_bal']:.1f}% ({100*m['min_cov_bal']:.1f}%) | {100*m['mean_acc_bal']:.1f}% ({100*m['min_acc_bal']:.1f}%) | "
                             f"{100*m['mean_cov_plain']:.1f}% / {100*m['mean_acc_plain']:.1f}% | {m['bands']} / {m['wrong']} / {m['dangerous']} / {m['many_called_clean']} | {cb} | {cbb} |")
    (tc.CACHE / "localcal_tables.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
