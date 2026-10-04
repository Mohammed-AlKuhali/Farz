"""Measure the result rules (Python mirror of app/src/lib/rules.ts, in src/beans/pipeline.py) BEFORE and AFTER the
3 Oct 2026 changes, on:
  * all 464 real CBD photos (India; no defect labels) — twice: as stored (no EXIF rotation, like the older Python
    reports) and with ImageOps.exif_transpose (what the app does: the browser applies the EXIF orientation);
  * the 22 J4ckDev held-out half-photos (out-of-fold predictions in reports/beans_heldout_preds.npz);
  * every demo tray in app/public/demo (deployed model, as the app runs it).

BEFORE = the shipped rule until 3 Oct 23:00: > 15% unsure -> not sure; else a band only if the WHOLE 95% Wilson
         interval sits inside one band, otherwise not sure (c07).
AFTER  = nothing answered / > 60% defects among answered ("implausible") / > 15% unsure -> not sure (c07);
         otherwise the band of the POINT estimate, marked "about" when the interval crosses a band edge.

The quality gates (dark, blur, count 20-160, touching) are applied first in both, as in the app. The colour gate
exists only in JS (colourgate.ts); all 464 CBD photos pass it (app/reports/colourgate_all_photos.json).

usage: .venv/bin/python app/scripts/measure_rules.py   -> app/reports/rules_measure.json
"""
import collections, csv, glob, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/beans"))
import numpy as np
from PIL import Image, ImageOps
import pipeline
from common import CBD_DIR, CLASSES, CROPS_CSV

OUT = ROOT / "app/reports/rules_measure.json"


def decide_before(calls):
    counts = {k: 0 for k in CLASSES + ["unsure"]}
    for c in calls: counts[c] += 1
    total = len(calls); defects = sum(counts[k] for k in CLASSES[1:]); answered = total - counts["unsure"]
    p, lo, hi = pipeline.wilson(defects, answered)
    if answered == 0 or counts["unsure"] / total > pipeline.MAX_UNSURE_FRAC: return {"band": "unsure", "reason": "too_many_unsure"}
    a, b = pipeline.band_of_rate(lo), pipeline.band_of_rate(hi)
    return {"band": a, "reason": ""} if a == b else {"band": "unsure", "reason": "interval_spans_bands"}


def gate(checks):
    if checks["too_dark"]: return "dark"
    if checks["too_blurry"]: return "blur"
    if not (pipeline_min <= checks["count"] <= pipeline_max): return "count"
    if checks["too_many_touching"]: return "spread"
    return None


from beanfinder import MIN_BEANS as pipeline_min, MAX_BEANS as pipeline_max  # noqa: E402

_S = None; _T = None
def _init(model_path, thr):
    global _S, _T
    _S = pipeline.session(model_path, threads=1); _T = thr


def _run(args):
    f, exif = args
    im = Image.open(f)
    orient = im.getexif().get(274)
    if exif: im = ImageOps.exif_transpose(im)
    r, p, pred = pipeline.run_photo(_S, im, _T)
    calls = pipeline.app_calls(r, pred)
    g = gate(r.checks)
    before = {"band": "retake", "reason": g} if g else decide_before(calls)
    after = {"band": "retake", "reason": g} if g else {k: v for k, v in pipeline.decide(calls).items() if k in ("band", "about", "reason", "p", "lo", "hi")}
    counts = collections.Counter(calls)
    return {"file": str(Path(f).relative_to(ROOT)) if str(f).startswith(str(ROOT)) else Path(f).name, "grade": Path(f).parent.name, "exif_orientation": orient,
            "beans": r.checks["count"], "counts": {k: counts.get(k, 0) for k in CLASSES + ["unsure"]},
            "before": before, "after": after}


def rate(r):
    c = r["counts"]; ans = r["beans"] - c["unsure"]
    return round(sum(c[k] for k in CLASSES[1:]) / ans, 4) if ans else None


def summarise(rows):
    s = {"photos": len(rows)}
    for when in ("before", "after"):
        bands = collections.Counter(r[when]["band"] for r in rows)
        reasons = collections.Counter(r[when]["reason"] for r in rows if r[when]["band"] == "unsure")
        s[when] = {"outcomes": dict(bands), "not_sure_reasons": dict(reasons),
                   "given_a_band": sum(r[when]["band"] in ("clean", "some", "many") for r in rows)}
        if when == "after": s[when]["about"] = sum(bool(r[when].get("about")) for r in rows)
    return s


def main():
    t0 = time.time()
    model_path, labels = pipeline.deployed(); thr = labels["abstain_below"]
    files = sorted(glob.glob(str(CBD_DIR / "**/*.jpg"), recursive=True))
    out = {"what": __doc__.strip().splitlines()[0], "model": str(model_path.relative_to(ROOT)), "abstain_below": thr,
           "rules": {"max_unsure_frac": pipeline.MAX_UNSURE_FRAC, "implausible_defect_frac": pipeline.IMPLAUSIBLE_DEFECT_FRAC,
                     "bands": "clean <5%, some 5-20%, many >20% (our rule of thumb, not an official grade)"}}
    with ProcessPoolExecutor(initializer=_init, initargs=(str(model_path), thr)) as ex:
        for key, exif in (("cbd_exif_applied_like_app", True), ("cbd_as_stored_no_exif", False)):
            rows = list(ex.map(_run, [(f, exif) for f in files], chunksize=4))
            per_grade = {}
            for g in sorted({r["grade"] for r in rows}):
                gr = [r for r in rows if r["grade"] == g]
                per_grade[g] = summarise(gr)
                per_grade[g]["confident_many_before"] = sum(r["before"]["band"] == "many" for r in gr)
                per_grade[g]["many_after"] = sum(r["after"]["band"] == "many" for r in gr)
            out[key] = {"all": summarise(rows), "per_grade": per_grade,
                        "confident_many_before": sum(r["before"]["band"] == "many" for r in rows),
                        "many_after": sum(r["after"]["band"] == "many" for r in rows),
                        "min_defect_rate_of_before_many": min([rate(r) for r in rows if r["before"]["band"] == "many"], default=None),
                        "max_defect_rate_of_after_band": max([rate(r) for r in rows if r["after"]["band"] in ("clean", "some", "many")], default=None),
                        "photos": rows}
            print(key, json.dumps(out[key]["all"]), "many before", out[key]["confident_many_before"], "after", out[key]["many_after"], f"{time.time()-t0:.0f}s", flush=True)
        # demo trays (deployed model, exactly the files the app ships; EXIF applied like the app)
        demo = sorted(glob.glob(str(ROOT / "app/public/demo/*.jpg")))
        trays = list(ex.map(_run, [(f, True) for f in demo]))
        out["demo_trays"] = trays
        for r in trays: print("demo", Path(r["file"]).name, r["counts"], "before", r["before"], "after", r["after"])
        # synthetic adversarial inputs from the app-QA run (/tmp/farz_verify, copied to the scratchpad), if present
        adv_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
        if adv_dir and adv_dir.exists():
            adv = sorted(glob.glob(str(adv_dir / "*.jpg")))
            arows = list(ex.map(_run, [(f, True) for f in adv]))
            for r in arows: r["file"] = Path(r["file"]).name if not r["file"].startswith("data") else r["file"]
            out["synthetic_adversarial_python_no_colour_gate"] = {"note": "SYNTHETIC images drawn by the app-QA agent (skin, legumes, leaves...). Python has no colour gate, so photos the app refuses with c06 still appear here; see the Node run for the full app pipeline.", "photos": arows}
            for r in arows: print("adv", r["file"], r["beans"], "before", r["before"]["band"], "after", r["after"]["band"], r["after"].get("reason"))
    # J4ckDev held-out half-photos: out-of-fold predictions (each bean scored by the fold model that never saw it)
    held = np.load(ROOT / "reports/beans_heldout_preds.npz")
    rep = json.load(open(ROOT / "reports/beans_model.json"))
    probs = held[rep["deploy_decision"]["deployed"]]
    rows = list(csv.DictReader(open(CROPS_CSV)))
    halves = collections.defaultdict(list)
    for i, fd, pr in zip(held["idx"], held["fold"], probs):
        r = rows[int(i)]
        c = "unsure" if r["touching"] == "1" or pr.max() < thr else CLASSES[int(pr.argmax())]
        halves[(r["photo"], int(fd), r["cls"])].append(c)
    hrows = []
    for (ph, fd, cls), calls in sorted(halves.items()):
        counts = collections.Counter(calls)
        a = pipeline.decide(calls)
        hrows.append({"photo": ph, "test_fold": fd, "true_class": cls, "beans": len(calls),
                      "counts": {k: counts.get(k, 0) for k in CLASSES + ["unsure"]},
                      "before": decide_before(calls), "after": {k: a[k] for k in ("band", "about", "reason", "p", "lo", "hi")},
                      "note": "half-photo has < 20 beans: in the app the count gate (c05) would fire first" if len(calls) < 20 else ""})
    out["j4ck_heldout_half_photos_out_of_fold"] = {"all": summarise(hrows), "photos": hrows,
        "note": "Each J4ckDev photo holds ONE class, so every defect half-photo is ~100% defects by construction."}
    for r in hrows: print("half", r["photo"], r["test_fold"], r["beans"], r["counts"], r["before"]["band"], "->", r["after"]["band"], r["after"]["reason"])
    out["seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT, out["seconds"], "s")


if __name__ == "__main__":
    main()
