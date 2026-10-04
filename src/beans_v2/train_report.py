"""Assemble reports_v2/model_v2.json (headline numbers, each with its protocol) from model_v2_eval.json,
model_v2_export.json and model_v2_deploy.json, and print the markdown tables used in model_v2.md."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import train_common as tc

R = tc.REPORTS_V2


def f3(x): return "–" if x is None else f"{x:.3f}"
def pc(x): return "–" if x is None else f"{100*x:.1f}%"


def main():
    ev = json.load(open(R / "model_v2_eval.json")); ex = json.load(open(R / "model_v2_export.json")); dp = json.load(open(R / "model_v2_deploy.json"))
    out = {"what": "Farz v2 per-bean classifier: headline numbers. Every number names its protocol; details in model_v2_eval.json, "
                   "model_v2_export.json, model_v2_deploy.json.",
           "classes": tc.CLASSES, "deployed": {"temperature": ev["deployed"]["temperature"], "abstain_threshold": ev["deployed"]["threshold"],
                                               "pooled_LOSO_point": ev["deployed"]["chosen_point_pooled"], "fitted_on": ev["deployed"]["fitted_on"],
                                               "model_file": ex["recommended_file"], "format": ex["recommended_format"]},
           "loso": {}, "loso_summary": ev["loso_summary"], "j4ck_lopo": {k: v for k, v in ev.get("j4ck_lopo", {}).items() if k != "summary_all_11"},
           "export": {k: ex[k] for k in ("fp32", "fp16") if k in ex}, "int8": ex.get("int8"),
           "cbd_bean_shares": dp.get("cbd_per_grade_bean_shares"), "deploy_simulation": {}}
    for s, r in ev["per_source"].items():
        a = r["at_own_held_out_threshold"]; nb = r["no_abstain"]
        out["loso"][s] = {"n": r["n"], "T": r["T_fitted_on_other_sources"], "t": r["threshold_fitted_on_other_sources"],
                          "macro_accuracy_no_abstain": nb["macro_accuracy"], "per_kind_recall_no_abstain": nb["per_kind_recall"],
                          "binary_no_abstain": nb["binary"], "ece_binary": nb["ece_binary"], "ece_binary_T1": r["ece_binary_uncalibrated_T1"], "ece_top1": nb["ece_top1"],
                          "coverage_at_t": a["coverage"], "answered_binary_accuracy_at_t": a["answered_binary_accuracy"],
                          "answered_typed_accuracy_at_t": a["answered_typed_accuracy"], "by_kind_at_t": a["by_kind"]}
    for name, d in dp["sets"].items():
        out["deploy_simulation"][name] = {v: (d[v].get("all") or d[v].get("summary") or {k: {kk: c[kk] for kk in ("photos", "correct_band", "wrong_band", "dangerous", "not_sure", "truth_band")} for k, c in d[v].items() if not k.startswith("_")})
                                          for v in d if not v.startswith("_")}
        if name.startswith("trays_") and "_threshold_sweep_rule_A_sensitivity_only" in d:
            out["deploy_simulation"][name]["_threshold_sweep_rule_A_sensitivity_only"] = d["_threshold_sweep_rule_A_sensitivity_only"]
        if name == "cbd":
            for v in d:
                if v.startswith("_"): out["deploy_simulation"][name][v] = d[v]; continue
                out["deploy_simulation"][name][v] = {**d[v]["all"], "AAA_confident_many": d[v]["AAA_confident_many"]}
    out["colour_vs_model_heldout"] = dp.get("colour_vs_model_heldout")
    if "licensed_only_model" in ev:
        out["licensed_only_model"] = {"protocol": ev["licensed_only_model"]["protocol"], "per_source": {
            s: {"binary_balanced_accuracy": r["no_abstain"]["binary"]["balanced_accuracy"], "macro_accuracy": r["no_abstain"]["macro_accuracy"],
                "coverage_at_t": r["at_threshold"]["coverage"], "answered_binary_accuracy": r["at_threshold"]["answered_binary_accuracy"],
                "all_sources_LOSO": r["all_sources_model_LOSO_same_source"]} for s, r in ev["licensed_only_model"]["per_source"].items()}}
    # held-out totals for the recommended rule (computed, not typed)
    V = "P_pair"; tot = {"band": 0, "not_sure": 0, "retake": 0, "n": 0, "wrong_band": 0}
    for nm in ("cbd", "cbd_stress", "j4ck_lopo", "lojano_real", "notplying_real", "loja_yolo_real"):
        t = dp["sets"][nm][V].get("all") or dp["sets"][nm][V].get("summary")
        tot["band"] += t["given_band"]; tot["not_sure"] += t["not_sure"]; tot["retake"] += t["retake"]; tot["n"] += t["photos"]
    real_n = sum((dp["sets"][nm][V].get("all") or dp["sets"][nm][V].get("summary"))["photos"] for nm in ("cbd", "j4ck_lopo", "lojano_real", "notplying_real", "loja_yolo_real"))
    real_band = sum((dp["sets"][nm][V].get("all") or dp["sets"][nm][V].get("summary"))["given_band"] for nm in ("cbd", "j4ck_lopo", "lojano_real", "notplying_real", "loja_yolo_real"))
    for c in dp["sets"]["j4ck_halves_trays"][V]["by_truth"].values():
        tot["band"] += c["given_band"]; tot["not_sure"] += c["not_sure"]; tot["retake"] += c["retake"]; tot["n"] += c["photos"]; tot["wrong_band"] += c["wrong_band"]
    for sname in tc.SOURCES:
        for c in dp["sets"].get(f"trays_{sname}", {}).get(V, {}).values():
            tot["band"] += c["correct_band"] + c["wrong_band"]; tot["not_sure"] += c["not_sure"]; tot["n"] += c["photos"]; tot["wrong_band"] += c["wrong_band"]
    out["deploy_rule_recommended"] = {
        "name": "P_pair",
        "steps": ["app retake gates (23:05) unchanged", "per bean (models_v2/farz_beans_v2_fp16.onnx): touching -> unsure; pGood=probs[0]; pGood >= %.2f -> good; else 1-pGood >= %.2f -> defect (type = argmax probs[1:], 'possible' only); else unsure" % tuple(dp["deployed_pair"]),
                  "app photo rules (23:05) unchanged: implausible > 60%% -> not sure; unsure > 15%% -> not sure; band of point estimate", "no OOD gate, no colour override, unusual-bean detector not in the verdict (each measured to add nothing)"],
        "held_out_totals_photos_and_trays": {**tot, "pct_band": round(100 * tot["band"] / tot["n"], 1), "pct_not_sure": round(100 * tot["not_sure"] / tot["n"], 1), "pct_retake": round(100 * tot["retake"] / tot["n"], 1)},
        "real_held_out_photos": {"n": real_n, "given_band": real_band},
        "cbd_AAA_confident_many": dp["sets"]["cbd"][V]["AAA_confident_many"],
        "app_demo_in_sample": dp["sets"]["app_demo_in_sample"][V]["summary"],
        "freeze_recommendation": "Keep v1 + 23:05 rules in the app for the freeze; v2+P_pair is equally safe on CBD (0/464), closes the 1/777 stress leak, but gives a band on ~1% of held-out photos/trays, 0 of the real held-out photos, 0/60 J4ckDev-style trays and 0/7 demo photos; with local calibration v2 is worse than v1 in the J4ckDev set-up. Swap only if a 'not sure'-only app is acceptable.",
    }
    out["cbd_rank"] = json.load(open(R / "model_v2_cbd_rank.json"))
    json.dump(out, open(R / "model_v2.json", "w"), indent=1)
    print("DEPLOY TOTALS", out["deploy_rule_recommended"]["held_out_totals_photos_and_trays"], out["deploy_rule_recommended"]["real_held_out_photos"])

    # ---------------- markdown tables
    print("\n### LOSO table\n")
    print("| Held-out source | n | T | t | macro acc (typed, no abstain) | good-vs-defect bal. acc (no abstain) | good recall | defect recall | ECE binary (T_s) | ECE binary (T=1) | coverage at t | answered good-vs-defect acc | answered typed acc | good: answered / called good | defect: answered / called defect |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s, r in out["loso"].items():
        b = r["binary_no_abstain"]; bk = r["by_kind_at_t"]
        g = bk.get("good"); dkinds = [k for k in bk if k != "good"]
        nd = sum(bk[k]["n"] for k in dkinds)
        d_cov = sum(bk[k]["coverage"] * bk[k]["n"] for k in dkinds) / nd if nd else None
        d_def = sum(bk[k]["called_defect"] * bk[k]["n"] for k in dkinds) / nd if nd else None
        gs = f"{pc(g['coverage'])} / {pc(g['called_good'])}" if g else "–"
        ds = f"{pc(d_cov)} / {pc(d_def)}" if nd else "–"
        print(f"| {s} | {r['n']:,} | {r['T']:.2f} | {r['t']:.2f} | {f3(r['macro_accuracy_no_abstain'])} | {f3(b['balanced_accuracy'])} | {f3(b['good_recall'])} | {f3(b['defect_recall'])} | {f3(r['ece_binary'])} | {f3(r['ece_binary_T1'])} | {pc(r['coverage_at_t'])} | {pc(r['answered_binary_accuracy_at_t'])} | {pc(r['answered_typed_accuracy_at_t'])} | {gs} | {ds} |")
    print("\n### pair thresholds (held-out)\n")
    print("| Held-out source | t_good / t_defect (fitted on the other 11) | coverage | answered good-vs-defect acc | good beans: called good / called defect / unsure | defect beans: called defect / called good / unsure |")
    print("|---|---|---|---|---|---|")
    for s_, r in ev["per_source"].items():
        a = r["at_own_held_out_pair"]; g = a["good_beans"]; d = a["defect_beans"]
        gs = f"{pc(g['called_good'])} / {pc(g['called_defect'])} / {pc(1 - g['called_good'] - g['called_defect'])}" if g["n"] else "–"
        ds = f"{pc(d['called_defect'])} / {pc(d['called_good'])} / {pc(1 - d['called_good'] - d['called_defect'])}" if d["n"] else "–"
        print(f"| {s_} | {a['t_good']:.2f} / {a['t_defect']:.2f} | {pc(a['coverage'])} | {pc(a['answered_binary_accuracy'])} | {gs} | {ds} |")
    print("\n### per-kind recall (argmax, no abstain)\n")
    print("| Held-out source | " + " | ".join(tc.KINDS) + " |"); print("|---|" + "---|" * len(tc.KINDS))
    for s, r in out["loso"].items():
        print(f"| {s} | " + " | ".join(f3(r["per_kind_recall_no_abstain"].get(k)) for k in tc.KINDS) + " |")
    if "j4ck_lopo" in ev:
        print("\n### J4ckDev LOPO\n")
        print("| Photo | class | n | v1 LOPO recall | v2 LOPO recall (typed) | v2 good-vs-defect correct | coverage at t | answered good-vs-defect acc | v2 argmax shares |")
        print("|---|---|---|---|---|---|---|---|---|")
        v1 = json.load(open(tc.ROOT / "reports/beans_lopo.json"))["photos"]
        for ph, r in ev["j4ck_lopo"]["photos"].items():
            v1r = v1.get(ph, {}).get("recall_correct_class")
            sh = ", ".join(f"{k} {v:.2f}" for k, v in r["predicted_share_argmax"].items() if v >= 0.05)
            print(f"| {ph} | {r['class']} | {r['n']} | {f3(v1r) if v1r is not None else 'n/a (only good photo)'} | {f3(r['recall_correct_class'])} | {f3(r['binary_correct_no_abstain'])} | {pc(r['coverage_at_t'])} | {pc(r['answered_binary_accuracy_at_t'])} | {sh} |")
    print("\n### CBD bean shares\n")
    cb = dp.get("cbd_per_grade_bean_shares", {}).get("per_grade", {})
    print("| Grade | photos | beans | v2 good | v2 defect | v2 unsure | v2 defect share of answered | v2 argmax good (no abstain) | v1 good (with abstain) | v1 broken (with abstain) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    v1c = json.load(open(tc.ROOT / "reports/beans_ood_cbd.json"))["per_grade"]
    for g, r in cb.items():
        sw = r["share_with_abstain"]; v = v1c.get(g, {}).get("share_with_abstain", {})
        print(f"| {g} | {r['photos']} | {r['beans']:,} | {pc(sw['good'])} | {pc(sw['defect'])} | {pc(sw['unsure'])} | {pc(r['defect_share_of_answered'])} | {pc(r['share_argmax_no_abstain']['good'])} | {pc(v.get('good'))} | {pc(v.get('broken'))} |")
    print("\n### deploy simulation (photos)\n")
    for name in ("cbd", "cbd_stress", "j4ck_lopo", "lojano_real", "notplying_real", "loja_yolo_real", "app_demo_in_sample", "j4ck_halves_trays", "j4ck_halves_trays_LOCAL_CALIBRATION"):
        if name not in dp["sets"]: continue
        print(f"\n{name}:")
        print("| Rule | photos | given a band | not sure | retake | outcomes |"); print("|---|---|---|---|---|---|")
        for v, d in dp["sets"][name].items():
            if v.startswith("_"): continue
            t = d.get("all") or d.get("summary")
            extra = f" (AAA many = {d['AAA_confident_many']})" if "AAA_confident_many" in d else ""
            extra += f" (confident many: {len(d['confident_many'])})" if "confident_many" in d else ""
            print(f"| {v} | {t['photos']} | {t['given_band']} ({t['pct_band']}%) | {t['not_sure']} ({t['pct_not_sure']}%) | {t['retake']} ({t['pct_retake']}%) | {t['outcomes']}{extra} |")
        if name == "cbd" and "_threshold_sweep_rule_A_sensitivity_only" in dp["sets"]["cbd"]:
            for tt, c in dp["sets"]["cbd"]["_threshold_sweep_rule_A_sensitivity_only"].items(): print(f"  sweep t={tt}: band {c['given_band']} AAA many {c['AAA_confident_many']} {c['outcomes']}")
        if name == "lojano_real":
            for v, d in dp["sets"][name].items(): print(f"  {v}: bueno {d['bueno']['outcomes']} | defectuoso {d['defectuoso']['outcomes']}")
        if name in ("j4ck_lopo", "app_demo_in_sample"):
            for v, d in dp["sets"][name].items():
                if v.startswith("_"): continue
                print(f"  {v}: " + "; ".join(f"{k}: {x['outcome']}" + (f" ({100*x['p_defect']:.0f}% def)" if x.get('p_defect') is not None else "") for k, x in d["photos"].items()))
    for nm in ("j4ck_halves_trays", "j4ck_halves_trays_LOCAL_CALIBRATION"):
        if nm in dp["sets"]:
            print(f"\n{nm} by truth:")
            for v, d in dp["sets"][nm].items(): print(f"  {v}: " + "; ".join(f"{tb}: correct {c['correct_band']} wrong {c['wrong_band']} (dangerous {c['dangerous']}) not sure {c['not_sure']} retake {c['retake']}" for tb, c in d["by_truth"].items()))
    if "j4ck_halves_threshold_sweep_rule_A_sensitivity_only" in dp: print("  halves sweep:", dp["j4ck_halves_threshold_sweep_rule_A_sensitivity_only"])
    if "j4ck_local_calibration" in dp: print("  local calibration:", dp["j4ck_local_calibration"])
    print("\n### trays\n")
    print("| Held-out source | truth | " + " | ".join(f"{v}: correct / wrong (dangerous) / not sure" for v in dp["variants"]) + " |")
    print("|---|---|" + "---|" * len(dp["variants"]))
    for s in tc.SOURCES:
        t = dp["sets"].get(f"trays_{s}")
        if not t: continue
        for sh in t[dp["variants"][0]]:
            if sh.startswith("_"): continue
            cells = []
            for v in dp["variants"]:
                c = t[v][sh]; cells.append(f"{c['correct_band']} / {c['wrong_band']} ({c['dangerous']}) / {c['not_sure']}")
            print(f"| {s} | {sh} -> {t[dp['variants'][0]][sh]['truth_band']} | " + " | ".join(cells) + " |")
    # totals over trays
    print()
    for v in dp["variants"]:
        tot = {"photos": 0, "correct_band": 0, "wrong_band": 0, "dangerous": 0, "not_sure": 0}
        for s in tc.SOURCES:
            for c in dp["sets"].get(f"trays_{s}", {}).get(v, {}).values():
                for k in tot: tot[k] += c[k]
        print(f"trays total {v}: {tot}")
    for tt in ("0.6", "0.7", "0.8", "0.85", "0.9", "0.95"):
        tot = {"photos": 0, "correct_band": 0, "wrong_band": 0, "dangerous": 0, "not_sure": 0}
        for s in tc.SOURCES:
            for c in dp["sets"].get(f"trays_{s}", {}).get("_threshold_sweep_rule_A_sensitivity_only", {}).get(tt, {}).values():
                for k in tot: tot[k] += c[k]
        print(f"sweep t={tt}: {tot}")
    print("\n### colour vs model\n")
    for s, d in (dp.get("colour_vs_model_heldout") or {}).items():
        print(s, {k: (round(v["balanced_accuracy"], 3), round(v["sensitivity"], 3), round(v["false_positive_rate"], 3)) for k, v in d.items() if "balanced_accuracy" in v}, d["dark_override"])


if __name__ == "__main__":
    main()
