"""Calibration demo set for the app (run ONLY when reports_v2/localcal.json says GO).

  models_v2/localcal_demo/calibration/   the labelled crops of a loja_yolo calibration draw (2-3 photos), PNG 128 px
  models_v2/localcal_demo/test/          3 loja_yolo TEST photos (not in that draw's calibration photos), unchanged JPEG files
  models_v2/localcal_demo/manifest.json  labels, source photo of every crop, the per-bean truth of the test photos, the
                                         head computed with the SHIPPED clean fp16 ONNX (prototypes + m_hi), and what the
                                         Python pipeline (frozen bean finder + shipped ONNX + local head + rules mirror)
                                         gives on each test photo
  models_v2/localcal_demo/CREDITS.md     CC BY 4.0 attribution
Caveat written into the manifest: the shipped clean model was TRAINED on loja_yolo (a licensed source), so a demo with the
shipped model on loja photos is IN-SAMPLE; the measured evidence is from the loja-held-out base model (localcal.json).
"""
import csv, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
import train_common as tc
import clean_common as cc
import localcal as lc

OUT = tc.MODELS_V2 / "localcal_demo"
CREDIT = ("Coffee bean photos and labels: \"Deteccion de defectos del grano\" (v14), Roboflow Universe project tesis-kmw54, "
          "https://universe.roboflow.com/tesis-kmw54/deteccion-de-defectos-del-grano , mirrored on Kaggle as "
          "cristianyagg/defectos-grano-de-cafe-de-la-provincia-de-loja (Loja province, Ecuador). Licence: CC BY 4.0 "
          "(https://creativecommons.org/licenses/by/4.0/), as stated in the dataset's README.dataset.txt. ")


def main(n_photos_max=3):
    v = json.load(open(tc.REPORTS_V2 / "localcal.json"))
    if not v["verdict"]["verdict"].startswith("GO"):
        print("verdict is", v["verdict"]["verdict"], "- no demo set exported"); return
    K = v["verdict"]["K"]; T, tg, td, _ = cc.deployed()
    meta = json.load(open(tc.CACHE / "meta.json"))
    src_all = np.array([m["source"] for m in meta]); y_all = tc.kind_targets(meta, range(len(meta)))
    grp_all = np.array([m["group"] for m in meta]); touch_all = np.array([m["touching"] for m in meta], bool)
    rows = np.flatnonzero((src_all == "loja_yolo") & ~touch_all)
    ybin = (y_all[rows] != 0).astype(int); groups = grp_all[rows]
    # pick the first draw (d = 0..4, pre-registered seeds) at the GO K whose calibration uses <= n_photos_max photos
    pick = None
    for d in range(lc.DRAWS):
        seed = 10000 * lc.SI["loja_yolo"] + 100 * K + d
        dr = lc.draw_calibration(groups, ybin, K, np.random.default_rng(seed))
        if dr and dr[2] <= n_photos_max: pick = (d, seed, dr); break
    if pick is None:
        d, seed = 0, 10000 * lc.SI["loja_yolo"] + 100 * K
        pick = (d, seed, lc.draw_calibration(groups, ybin, K, np.random.default_rng(seed)))
    d, seed, (cal, test, nph) = pick
    if OUT.exists(): shutil.rmtree(OUT)
    (OUT / "calibration").mkdir(parents=True); (OUT / "test").mkdir()
    crops = np.load(tc.CACHE / "crops.npy", mmap_mode="r")
    sess = cc.ort_session(cc.ONNX16)
    P, E = cc.ort_run(sess, crops[rows[cal]])
    mu_g, mu_d, m_hi, loo = lc.proto_fit(E, ybin[cal])
    cal_items = []
    for j, i in enumerate(cal):
        r = meta[rows[i]]; lab = "good" if ybin[i] == 0 else "defect"
        fn = f"{j:03d}_{lab}_{Path(r['group']).name}.png"
        shutil.copy(tc.ROOT / r["path"], OUT / "calibration" / fn)
        cal_items.append({"file": f"calibration/{fn}", "label": lab, "source_label": r["source_class"], "source_photo": Path(r["group"]).name})
    # test photos: from this draw's test photos, the 3 with the most labelled non-touching beans that include good AND defect beans
    tg_ids = np.unique(groups[test]); cand = []
    for g in tg_ids:
        m = groups == g; cand.append((int((ybin[m] == 0).sum() > 0 and (ybin[m] == 1).sum() > 0), int(m.sum()), g))
    cand.sort(reverse=True); chosen = [c[2] for c in cand[:3]]
    by_photo = {}
    for r in csv.DictReader(open(tc.ROOT / "data/crops_v2/crops_v2.csv")):
        if r["source"] == "loja_yolo" and r["group"] in chosen: by_photo.setdefault(r["group"], {"orig": r["orig_path"], "beans": []})["beans"].append(
            {"farz_class": r["farz_class"], "source_class": r["source_class"], "bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"], r["bbox_y1"]], "keep": r["keep"]})
    import robust_common as rc
    test_items = []
    for g in chosen:
        src = tc.ROOT / by_photo[g]["orig"]; dst = OUT / "test" / src.name; shutil.copy(src, dst)
        im = Image.open(src).convert("RGB"); res = rc.find_beans(im); cg = rc.colour_gate(res)
        cr = np.stack([b.crop for b in res.beans]) if res.beans else np.zeros((0, 128, 128, 3), np.uint8)
        p, e = cc.ort_run(sess, cr)
        s = lc.proto_score(e, mu_g, mu_d); codes, how = lc.answer(s, m_hi, (p[:, 0] < 0.5).astype(int))
        codes = np.where([b.touching for b in res.beans], 2, codes) if len(cr) else codes
        o, det = cc.decide_codes(codes) if len(cr) else ("notsure:too_many_unsure", {})
        gate = cc.app_gate(res.checks, cg)
        test_items.append({"file": f"test/{src.name}", "group": g, "image_size": list(im.size), "labelled_beans_in_dataset": by_photo[g]["beans"],
                           "finder_beans": int(len(cr)), "calls_good_defect_unsure": [int((codes == k).sum()) for k in (0, 1, 2)],
                           "rules_outcome_gates_ignored": o, "app_gate_port": gate or "pass",
                           "base_alone_calls_good_defect_unsure": [int((cc.calls3(p, tg, td) == k).sum()) for k in (0, 1, 2)]})
    man = {"what": "Farz local-calibration demo set (loja_yolo, CC BY 4.0)", "credit": CREDIT, "K": K, "draw": d, "seed": seed, "calibration_photos": nph,
           "calibration": cal_items, "test": test_items,
           "head_from_shipped_model": {"model": tc.rel(cc.ONNX16), "mu_good": mu_g.astype(np.float32).tolist(), "mu_defect": mu_d.astype(np.float32).tolist(),
                                       "m_hi": m_hi, "loo_errors": int(((loo >= 0).astype(int) != ybin[cal]).sum())},
           "IN_SAMPLE_WARNING": "The shipped clean model was trained on loja_yolo (all its photos, incl. these). A demo with it on these photos is in-sample "
                                "and flatters the method. The measured, held-out evidence is reports_v2/localcal.json (base model trained without loja_yolo)."}
    cc.jdump(man, OUT / "manifest.json")
    (OUT / "CREDITS.md").write_text("# Credits\n\n" + CREDIT + "\n\nFiles in `calibration/` are 128x128 crops cut from the dataset's photos by Farz's bean finder "
                                    "(changed: cropped, white-balanced to a grey sheet, resized). Files in `test/` are the dataset's own JPEG files, unchanged. "
                                    "Labels come from the dataset's YOLO polygons (mapped to good / defect by Farz).\n")
    tc.log("demo set exported", OUT, "K", K, "draw", d, "photos", nph, [t["rules_outcome_gates_ignored"] for t in test_items])


if __name__ == "__main__":
    main()
