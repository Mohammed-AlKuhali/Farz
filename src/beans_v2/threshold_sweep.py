"""Pre-registered "sound" threshold sweep (reports_v2/threshold_sweep.md, pre-registration block fixed 04:53 BST).

t_good in {0.50 .. 0.75}, t_defect fixed at the shipped 0.91, shipped T. Trays:
  A  6 licensed groups, each scored by its clean LOSO fold (afiyah twins together); 50- and 100-bean trays
  B  5 unlicensed test-only sources, shipped model (licensed.pt logits)
  C  the auditor's 1,200 J4ckDev trays (clean_half_1/2; identical draws to clean_simulate.py "j1200")
Photo rule = src/beans/pipeline.decide (the Python mirror of app/src/lib/rules.ts decide()).
Usage: FARZ_SCRATCH=<scratch> python threshold_sweep.py   -> reports_v2/threshold_sweep.json and the md Results section.
"""
import csv, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import train_common as tc
import clean_common as cc

CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
T_DEFECT = 0.91
SHARES = [0.0, 0.02, 0.10, 0.15, 0.30, 0.45, 1.0]
SIZES = [50, 100]
NT = 200
MD = tc.REPORTS_V2 / "threshold_sweep.md"
OUT = tc.REPORTS_V2 / "threshold_sweep.json"


def band_of(r):
    return "clean" if r < 0.05 else "some" if r <= 0.2 else "many"


def prereg_block():
    txt = MD.read_text()
    return txt.split("\n\n## Results")[0].rstrip("\n")


def build_trays():
    """-> list of units: {name, kind, trays: [(size, truth, idx array into that unit's prob rows)], logits}"""
    meta = json.load(open(tc.CACHE / "meta.json"))
    src_all = np.array([m["source"] for m in meta]); y_all = tc.kind_targets(meta, range(len(meta)))
    touch_all = np.array([m["touching"] for m in meta], bool)
    _, _, _, ev = cc.deployed()
    rng = np.random.default_rng(20261004)
    units = []
    srcs = [(g, "licensed_heldout", np.load(cc.FOLDS / f"clean_loso_{g}.npz")) for g in ev["per_group"]]
    zu = np.load(tc.CACHE / "clean_emb_final_unlicensed.npz")
    for s in cc.UNLICENSED:
        m = src_all[zu["idx"]] == s
        srcs.append((s, "unlicensed_test_only", {"idx": zu["idx"][m], "logits": zu["logits"][m]}))
    for name, kind, z in srcs:
        idx, lg = z["idx"], z["logits"]
        y = y_all[idx]; ok = ~touch_all[idx]
        good = np.flatnonzero(ok & (y == 0)); bad = np.flatnonzero(ok & (y != 0))
        trays = []
        for nb in SIZES:
            for sh in SHARES:
                nd = int(round(sh * nb))
                if nb - nd > len(good) or nd > len(bad): continue
                for _ in range(NT):
                    sel = np.r_[rng.choice(bad, nd, replace=False) if nd else np.array([], int),
                                rng.choice(good, nb - nd, replace=False) if nb - nd else np.array([], int)]
                    trays.append((nb, band_of(nd / nb), sel))
        units.append({"name": name, "kind": kind, "logits": lg, "trays": trays, "n_good": int(len(good)), "n_defect": int(len(bad)),
                      "model": f"clean_loso_{name}" if kind == "licensed_heldout" else "shipped (licensed.pt)"})
    # C: auditor's 1,200 J4ckDev trays, identical draws to clean_simulate.py
    rows = list(csv.DictReader(open(tc.ROOT / "data/crops/crops.csv")))
    cls = np.array([r["cls"] for r in rows]); tf = np.array([int(r["test_fold"]) for r in rows])
    rng2 = np.random.default_rng(2026)
    DESIGN = {"clean": [0], "some": [10], "many": [24]}; NB = 80
    L_all, trays = [], []; off = 0
    for k in (1, 2):
        te_ = np.flatnonzero(tf == k)
        z = np.load(cc.FOLDS / f"clean_half_{k}.npz"); assert np.array_equal(np.sort(z["idx"]), te_)
        L2 = z["logits"][np.argsort(z["idx"])]
        good = np.flatnonzero(cls[te_] == "good"); bad = np.flatnonzero(cls[te_] != "good")
        for tbn, nds in DESIGN.items():
            for nd in nds:
                for _ in range(200):
                    sel = np.r_[rng2.choice(bad, nd, replace=False), rng2.choice(good, NB - nd, replace=False)]
                    trays.append((NB, tbn, sel + off))
        L_all.append(L2); off += len(L2)
    units.append({"name": "j4ckdev_1200_auditor", "kind": "auditor_j4ckdev_1200", "logits": np.concatenate(L_all), "trays": trays,
                  "model": "clean_half_1 / clean_half_2", "n_good": None, "n_defect": None})
    return units


def score(units, T, tg):
    res = {}
    for u in units:
        codes = cc.calls3(tc.softmax(u["logits"], T), tg, T_DEFECT)
        per_size = {}
        tot = {"trays": 0, "bands": 0, "wrong": 0, "dangerous": 0, "clean_called_many": 0, "many_called_clean": 0,
               "notsure_implausible": 0, "notsure_too_many_unsure": 0}
        by_truth = {}
        for nb, truth, sel in u["trays"]:
            o = cc.decide_codes(codes[sel])[0]
            d = per_size.setdefault(str(nb), {"trays": 0, "bands": 0, "wrong": 0, "dangerous": 0})
            for t in (tot, d): t["trays"] += 1
            bt = by_truth.setdefault(truth, {}); bt[o] = bt.get(o, 0) + 1
            if o.startswith("notsure"):
                tot["notsure_implausible" if "implausible" in o else "notsure_too_many_unsure"] += 1
                continue
            for t in (tot, d):
                t["bands"] += 1
                if o != truth: t["wrong"] += 1
                if (truth == "clean" and o == "many") or (truth == "many" and o == "clean"): t["dangerous"] += 1
            if truth == "clean" and o == "many": tot["clean_called_many"] += 1
            if truth == "many" and o == "clean": tot["many_called_clean"] += 1
        tot["wrong_share_of_bands"] = round(tot["wrong"] / tot["bands"], 4) if tot["bands"] else None
        tot["per_size"] = per_size; tot["by_truth"] = by_truth
        res[u["name"]] = tot
    return res


def verdict(per_t):
    rows = {}
    chosen = None
    for tg in CANDIDATES:
        r = per_t[f"{tg:.2f}"]
        dang = sum(v["dangerous"] for v in r.values())
        fails_b = [k for k, v in r.items() if v["bands"] >= 10 and v["wrong"] / v["bands"] > 0.05]
        ok = dang == 0 and not fails_b
        rows[f"{tg:.2f}"] = {"dangerous_total": dang, "units_failing_b": fails_b, "qualifies": ok,
                             "bands_total": sum(v["bands"] for v in r.values()), "wrong_total": sum(v["wrong"] for v in r.values())}
        if ok and chosen is None: chosen = tg
    return rows, chosen


def main():
    block = prereg_block()
    h = hashlib.sha256(block.encode()).hexdigest()
    t0 = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    T, tg0, td0, _ = cc.deployed()
    assert abs(td0 - T_DEFECT) < 1e-9 and abs(tg0 - 0.50) < 1e-9
    units = build_trays()
    per_t = {f"{tg:.2f}": score(units, T, tg) for tg in CANDIDATES}
    rows, chosen = verdict(per_t)
    fallback = chosen is None
    if fallback: chosen = 0.75
    out = {"prereg_block_sha256": h, "run_started": t0, "run_finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "temperature": T, "t_defect": T_DEFECT, "candidates": CANDIDATES,
           "units": {u["name"]: {"kind": u["kind"], "model": u["model"], "trays": len(u["trays"]), "n_good_beans": u["n_good"],
                                 "n_defect_beans": u["n_defect"]} for u in units},
           "verdict_rows": rows, "chosen_t_good": chosen, "no_candidate_qualified_fallback_0_75": fallback,
           "integrity_note": "t_good 0.50 on the J4ckDev 1,200 set should reproduce clean_deploy.json 241 bands / 0 wrong / 0 dangerous",
           "per_t_good": per_t}
    cc.jdump(out, OUT)
    tc.log("threshold sweep: chosen", chosen, "fallback" if fallback else "", {k: (v["dangerous_total"], v["units_failing_b"]) for k, v in rows.items()})


if __name__ == "__main__":
    main()
