"""Farz v2 data hunter: build per-bean crops from every external dataset (FROZEN crop contract).

  python src/beans_v2/data_build_crops.py --sources mfu17,usk_coffee   # crop sources (per-source CSV)
  python src/beans_v2/data_build_crops.py --merge                       # dedupe across sources + J4ckDev

Outputs
  data/crops_v2/<source>/<farz_class>/<source>_<item>_<k>.png  128x128 RGB (bean finder contract)
  data/crops_v2/<source>/crops.csv                            one row per crop (before cross-source dedupe)
  data/crops_v2/crops_v2.csv                                  merged; keep==0 rows are duplicates (files parked in _dropped/)
  data/crops_v2/summary_v2.json                               counts per source x class, dedupe stats

Dedupe (before any split): (1) pixel MD5 of the decoded, EXIF-rotated source image; any source image identical to a
J4ckDev photo or to an image already kept from an earlier source (order: J4ckDev, then SOURCE_ORDER) is dropped.
(2) pixel MD5 of the 128 px crop, same rule. (3) exact 64-bit dHash collisions of crops ACROSS sources (and against
the v1 J4ckDev crops) are counted and reported, not dropped (dHash collides on look-alike plain beans).
(4) near-duplicates (same bean re-shot / re-encoded): 32x32 grey thumbnail correlation >= 0.998 between crops of
different groups -> later one dropped (all vicanadya16 excluded from this check).
"""
import argparse, csv, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from data_common import ROOT, load_rgb, pixel_md5, dhash64, crops_tray, crop_single, crop_precrop, iou, FARZ_CLASSES, crop_mask_chroma, crop_box, sheet_wb_factor
from data_sources import ITEMS, SOURCES, SCA_MAP, LOJA_MAP, NOTPLYING_MAP

OUT = os.path.join(ROOT, "data", "crops_v2")
SOURCE_ORDER = ["mfu17", "loja_yolo", "lojano", "usk_coffee", "samruddh_grading", "afiyah_deteksi", "afiyah_bijikopi", "daffa_defect", "mindforge_doubleside",
                "notplying_defects", "vicanadya16"]
FIELDS = ["crop_path", "source", "source_class", "farz_class", "group", "split_hint", "orig_path", "mode",
          "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "n_found", "touching", "flags", "src_pixel_md5",
          "crop_pixel_md5", "dhash", "licence"]


def _save(crop, source, cls, name):
    d = os.path.join(OUT, source, cls); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + ".png")
    Image.fromarray(crop).save(p, compress_level=1)
    return os.path.relpath(p, ROOT)


def _row(item, crop, cls, sc, name, mode, bbox, n_found, touching, flags, src_md5):
    h = __import__("hashlib").md5(); h.update(crop.tobytes())
    return dict(crop_path=_save(crop, item["source"], cls, name), source=item["source"], source_class=sc,
                farz_class=cls, group=item["group"], split_hint=item.get("split_hint", ""),
                orig_path=os.path.relpath(item["path"], ROOT), mode=mode,
                bbox_x0=round(bbox[0], 1), bbox_y0=round(bbox[1], 1), bbox_x1=round(bbox[2], 1), bbox_y1=round(bbox[3], 1),
                n_found=n_found, touching=int(touching), flags=flags, src_pixel_md5=src_md5,
                crop_pixel_md5=h.hexdigest(), dhash=format(dhash64(crop), "016x"),
                licence=SOURCES[item["source"]]["licence"])


TRAY_MIN_SHEET_LUMA = 120.0   # the finder assumes beans darker than a light sheet; dark boards break it
AREA_REL = (0.4, 2.5)          # label cleaning only: area vs the photo's median bean area
EDGE_PX = 2                    # label cleaning only: drop blobs hugging the frame (corner vignette artefacts)


def tray_keep(r, b):
    h, w = r.work.shape[:2]
    rel = b.area / r.median_area if r.median_area else 0.0
    if not (AREA_REL[0] <= rel <= AREA_REL[1]): return False, rel
    if b.x0 <= EDGE_PX or b.y0 <= EDGE_PX or b.x1 >= w - EDGE_PX or b.y1 >= h - EDGE_PX: return False, rel
    return True, rel


def process(args):
    idx, item = args
    src = item["source"]; mode = SOURCES[src]["mode"]
    try:
        im = load_rgb(item["path"])
    except Exception as e:
        return [], {"item": item["path"], "error": repr(e)}
    md5 = pixel_md5(im)
    stem = f"{src}_{idx:06d}"
    rows = []
    if mode == "single":
        crop, info = crop_single(im)
        if crop is None:
            return [], {"item": item["path"], "error": info.get("flags")}
        rows.append(_row(item, crop, item["farz_class"], item["source_class"], stem, mode, info["bbox"],
                         info["n_found"], False, info["flags"], md5))
    elif mode == "mask_chroma":
        crop, info = crop_mask_chroma(im)
        if crop is None:
            return [], {"item": item["path"], "error": info.get("flags")}
        rows.append(_row(item, crop, item["farz_class"], item["source_class"], stem, mode, info["bbox"], 1, False,
                         info["flags"], md5))
    elif mode == "box":
        W, H = im.size; wbf = sheet_wb_factor(im)
        for k, (name, bx) in enumerate(item["polys"]):
            box = (bx[0] * W, bx[1] * H, bx[2] * W, bx[3] * H)
            crop, info = crop_box(im, box, wbf)
            rows.append(_row(item, crop, NOTPLYING_MAP[name], name, f"{stem}_{k:03d}", mode, box, len(item["polys"]),
                             False, info["flags"], md5))
        return rows, {"item": item["path"], "n_polys": len(item["polys"])}
    elif mode == "precrop":
        crop, info = crop_precrop(im)
        rows.append(_row(item, crop, item["farz_class"], item["source_class"], stem, mode, info["bbox"], 1, False,
                         info["flags"], md5))
    elif mode == "tray_single_class":
        r, boxes = crops_tray(im)
        if r.checks["sheet_luma"] < TRAY_MIN_SHEET_LUMA:
            return [], {"item": item["path"], "n_beans": len(r.beans), "rejected_photo": f"sheet_luma={r.checks['sheet_luma']:.0f}"}
        dropped = 0
        for k, (b, box) in enumerate(zip(r.beans, boxes)):
            ok, rel = tray_keep(r, b)
            if not ok: dropped += 1; continue
            rows.append(_row(item, b.crop, item["farz_class"], item["source_class"], f"{stem}_{k:03d}", mode, box,
                             len(r.beans), b.touching, f"area_rel={rel:.2f}", md5))
        return rows, {"item": item["path"], "n_beans": len(r.beans), "kept": len(rows), "dropped_qc": dropped}
    elif mode == "tray_polygons":
        W, H = im.size
        r, boxes = crops_tray(im)
        polys = [(name, (bx[0] * W, bx[1] * H, bx[2] * W, bx[3] * H)) for name, bx in item["polys"]]
        used = set(); matched = 0
        if r.checks["sheet_luma"] < TRAY_MIN_SHEET_LUMA:
            return [], {"item": item["path"], "n_beans": len(r.beans), "n_polys": len(polys), "matched": 0,
                        "rejected_photo": f"sheet_luma={r.checks['sheet_luma']:.0f}"}
        for k, (b, box) in enumerate(zip(r.beans, boxes)):
            best, bi = 0.0, -1
            for j, (name, pb) in enumerate(polys):
                if j in used: continue
                v = iou(box, pb)
                if v > best: best, bi = v, j
            if best < 0.5: continue
            used.add(bi); matched += 1
            name = polys[bi][0]
            rows.append(_row(item, b.crop, LOJA_MAP[name], name, f"{stem}_{k:03d}", mode, box, len(r.beans),
                             b.touching, f"iou={best:.2f}", md5))
        return rows, {"item": item["path"], "n_beans": len(r.beans), "n_polys": len(polys), "matched": matched}
    return rows, {"item": item["path"], "ok": True}


def build(source, workers):
    items = list(ITEMS[source]())
    t = time.time()
    rows, logs = [], []
    with Pool(workers) as pool:
        for rr, lg in pool.imap(process, list(enumerate(items)), chunksize=8):
            rows.extend(rr); logs.append(lg)
    os.makedirs(os.path.join(OUT, source), exist_ok=True)
    with open(os.path.join(OUT, source, "crops.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    errs = [l for l in logs if "error" in l]
    stats = dict(source=source, items=len(items), crops=len(rows), errors=len(errs), error_examples=errs[:5],
                 seconds=round(time.time() - t, 1))
    if SOURCES[source]["mode"] == "tray_polygons":
        stats["polygons_total"] = sum(l.get("n_polys", 0) for l in logs)
        stats["beans_found_total"] = sum(l.get("n_beans", 0) for l in logs)
        stats["matched_iou_ge_0.5"] = sum(l.get("matched", 0) for l in logs)
    if SOURCES[source]["mode"].startswith("tray"):
        stats["photos_rejected_dark_sheet"] = [os.path.basename(l["item"]) + " " + l["rejected_photo"] for l in logs if "rejected_photo" in l]
    if SOURCES[source]["mode"] == "tray_single_class":
        stats["beans_found_total"] = sum(l.get("n_beans", 0) for l in logs)
        stats["dropped_by_qc"] = sum(l.get("dropped_qc", 0) for l in logs)
        stats["per_photo"] = {os.path.basename(l["item"]): [l.get("n_beans"), l.get("kept")] for l in logs}
    json.dump(stats, open(os.path.join(OUT, source, "build_log.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_photo"}))


NEAR_THR = 0.998


def _thumb(p):
    a = np.asarray(Image.open(os.path.join(ROOT, p)).convert("L").resize((32, 32), Image.BILINEAR), np.float32).ravel()
    a -= a.mean(); n = np.linalg.norm(a)
    return a / n if n > 0 else a


DROPPED = os.path.join(OUT, "_dropped")


def _park(rel):
    """Move a dropped crop out of the class folders into data/crops_v2/_dropped/ (restored at the next merge,
    so --merge is repeatable)."""
    src = os.path.join(ROOT, rel)
    if os.path.exists(src):
        dst = os.path.join(DROPPED, os.path.relpath(src, OUT)); os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)


def _restore_parked():
    for f in glob.glob(os.path.join(DROPPED, "**", "*.png"), recursive=True):
        dst = os.path.join(OUT, os.path.relpath(f, DROPPED)); os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(f, dst)


def near_dup_pass(all_rows):
    idx = [k for k, r in enumerate(all_rows) if int(r["keep"]) and r["source"] != "vicanadya16"]
    V = np.stack([_thumb(all_rows[k]["crop_path"]) for k in idx]).astype(np.float32)
    grp = [all_rows[k]["group"] for k in idx]
    dropped = set(); stats = {}
    for i0 in range(0, len(idx), 2048):
        S = V[i0:i0 + 2048] @ V.T
        for a, j in np.argwhere(S >= NEAR_THR):
            i = i0 + a
            if j <= i or grp[i] == grp[j] or i in dropped or j in dropped: continue
            dropped.add(j)
            r = all_rows[idx[j]]; r["keep"] = 0; r["dup_of"] = "near:" + all_rows[idx[i]]["crop_path"] + f" r={S[a, j]:.4f}"
            key = r["source"] if r["source"] == all_rows[idx[i]]["source"] else all_rows[idx[i]]["source"] + "+" + r["source"]
            stats[key] = stats.get(key, 0) + 1
            _park(r["crop_path"])
    return stats


def merge():
    _restore_parked()
    # J4ckDev reference hashes
    j4_src = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "green", "ImageDataset", "*.jpg"))):
        j4_src[pixel_md5(load_rgb(f))] = os.path.basename(f)
    j4_crop_md5, j4_dh = {}, {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "crops", "*", "*.png"))):
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        h = __import__("hashlib").md5(); h.update(a.tobytes())
        j4_crop_md5[h.hexdigest()] = f; j4_dh.setdefault(format(dhash64(a), "016x"), []).append(f)
    seen_src, seen_crop = dict((k, "j4ckdev") for k in j4_src), dict((k, "j4ckdev") for k in j4_crop_md5)
    all_rows, dup_stats = [], {}
    for s in SOURCE_ORDER:
        p = os.path.join(OUT, s, "crops.csv")
        if not os.path.exists(p): continue
        rows = list(csv.DictReader(open(p)))
        st = dict(rows_in=len(rows), dup_source_image_within=0, dup_source_image_cross=0,
                  dup_crop_within=0, dup_crop_cross=0)
        own_src = set(); own_crop = set()
        src_first_owner = {}
        for r in rows:
            m = r["src_pixel_md5"]; c = r["crop_pixel_md5"]
            r["keep"] = 1; r["dup_of"] = ""
            # one source image yields several crops (trays): only a DIFFERENT file with the same pixels is a dup
            if m in seen_src:
                r["keep"] = 0; r["dup_of"] = "src:" + seen_src[m]; st["dup_source_image_cross"] += 1
            elif m in src_first_owner and src_first_owner[m] != r["orig_path"]:
                r["keep"] = 0; r["dup_of"] = "src:" + src_first_owner[m]; st["dup_source_image_within"] += 1
            elif c in seen_crop:
                r["keep"] = 0; r["dup_of"] = "crop:" + seen_crop[c]; st["dup_crop_cross"] += 1
            elif c in own_crop:
                r["keep"] = 0; r["dup_of"] = "crop:" + s; st["dup_crop_within"] += 1
            src_first_owner.setdefault(m, r["orig_path"])
            if r["keep"]:
                own_crop.add(c); own_src.add(m)
            elif r["crop_path"]:
                _park(r["crop_path"])
        for m in own_src: seen_src.setdefault(m, s)
        for c in own_crop: seen_crop.setdefault(c, s)
        st["rows_kept"] = sum(int(r["keep"]) for r in rows)
        dup_stats[s] = st
        all_rows.extend(rows)
    # (4) near-duplicates: 32x32 grey thumbnails, mean-centred + L2-normalised, dot >= NEAR_THR, different group
    #     -> the later row (SOURCE_ORDER, then file order) is dropped. vicanadya16 is not checked (86k tight crops).
    near = near_dup_pass(all_rows)
    # near-duplicate report: identical dHash across different sources / vs J4ckDev crops
    by_dh = {}
    for r in all_rows:
        if int(r["keep"]): by_dh.setdefault(r["dhash"], set()).add(r["source"])
    cross_pairs = {}
    for dh, ss in by_dh.items():
        if len(ss) > 1:
            k = "+".join(sorted(ss)); cross_pairs[k] = cross_pairs.get(k, 0) + 1
    within = {}
    seen_dh = {}
    for r in all_rows:
        if not int(r["keep"]): continue
        k = (r["source"], r["dhash"])
        if k in seen_dh and seen_dh[k] != r["orig_path"]:
            within[r["source"]] = within.get(r["source"], 0) + 1
        seen_dh.setdefault(k, r["orig_path"])
    vs_j4 = {}
    for r in all_rows:
        if int(r["keep"]) and r["dhash"] in j4_dh:
            vs_j4[r["source"]] = vs_j4.get(r["source"], 0) + 1
    with open(os.path.join(OUT, "crops_v2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS + ["keep", "dup_of"]); w.writeheader(); w.writerows(all_rows)
    counts = {}
    for r in all_rows:
        if int(r["keep"]):
            counts.setdefault(r["source"], {c: 0 for c in FARZ_CLASSES})[r["farz_class"]] += 1
    groups = {}
    for r in all_rows:
        if int(r["keep"]): groups.setdefault(r["source"], set()).add(r["group"])
    summary = dict(counts_kept=counts, totals_by_class={c: sum(v[c] for v in counts.values()) for c in FARZ_CLASSES},
                   total_kept=sum(sum(v.values()) for v in counts.values()),
                   distinct_groups_kept={k: len(v) for k, v in groups.items()},
                   dedupe=dup_stats, near_dup_dropped=near, j4ckdev_photos_hashed=len(j4_src), j4ckdev_crops_hashed=len(j4_crop_md5),
                   near_dup_identical_dhash_cross_source_buckets=cross_pairs,
                   identical_dhash_vs_j4ckdev_v1_crops=vs_j4,
                   identical_dhash_within_source_different_files=within)
    json.dump(summary, open(os.path.join(OUT, "summary_v2.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    for s in [x for x in a.sources.split(",") if x]:
        build(s, a.workers)
    if a.merge:
        merge()
