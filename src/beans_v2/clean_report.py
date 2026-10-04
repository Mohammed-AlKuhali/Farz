"""Collect the headline numbers of the licence-stated ("clean") v2 model into reports_v2/clean_model.json.
Every value is copied from clean_eval.json / clean_export.json / clean_deploy.json (nothing computed here except sums)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import train_common as tc

R = tc.REPORTS_V2


def main():
    ev = json.load(open(R / "clean_eval.json")); ex = json.load(open(R / "clean_export.json")); dp = json.load(open(R / "clean_deploy.json"))["sets"]
    d = ev["deployed"]
    out = {"model": {"fp16": ex["fp16"]["file"], "fp16_bytes": ex["fp16"]["bytes"], "fp16_sha256": ex["fp16"]["sha256"], "fp32": ex["fp32"]["file"],
                     "fp32_bytes": ex["fp32"]["bytes"], "labels": "models_v2/farz_beans_v2_clean_labels.json", "outputs": ["probs [N,6]", "embed [N,1024] L2-normalised"],
                     "training_sources": ev["licensed_sources"], "licence_of_weights": "non-commercial, ShareAlike (CC BY-NC-SA 4.0 via J4ckDev; CC BY-NC 4.0 via lojano)",
                     "unlicensed_sources_used_for": "held-out test only"},
           "temperature": d["temperature"], "pair_thresholds": {"t_good": d["pair_thresholds"]["t_good"], "t_defect": d["pair_thresholds"]["t_defect"],
                                                               "pooled_coverage": d["pair_thresholds"]["coverage"], "pooled_answered_accuracy": d["pair_thresholds"]["answered_accuracy"]},
           "fp16_vs_fp32": {k: ex["fp16"][k] for k in ("top_class_agreement_vs_fp32", "call3_agreement_vs_fp32_at_pair", "max_abs_prob_diff_vs_fp32", "embed_cosine_vs_fp32_min_mean")},
           "wasm_parity": {k: {kk: v.get(kk) for kk in ("n", "max_abs_prob_diff_vs_python_ort", "max_abs_embed_diff_vs_python_ort", "argmax_agreement", "call3_agreement_at_pair")} for k, v in ex["wasm_parity"].items()},
           "loso_licensed": {g: {"n": r["all"]["n"], "balanced_accuracy_no_abstain": r["all"]["no_abstain"]["binary"]["balanced_accuracy"],
                                 "pair_from_other_groups": r["pair_fitted_on_other_groups"], "coverage": r["all"]["at_pair"]["coverage"],
                                 "answered_accuracy": r["all"]["at_pair"]["answered_binary_accuracy"]} for g, r in ev["per_group"].items()},
           "loso_mean_balanced_accuracy_groups_with_good_beans": ev["loso_summary"]["mean_balanced_accuracy_groups_with_good_beans"],
           "unlicensed_test_only": {s: {"n": r["n"], "balanced_accuracy_no_abstain": r["no_abstain"]["binary"]["balanced_accuracy"], "coverage": r["at_pair"]["coverage"],
                                        "answered_accuracy": r["at_pair"]["answered_binary_accuracy"]} for s, r in ev["unlicensed_test"].items()},
           "cbd": {"photos_banded_gates_ignored": dp["cbd"]["gates_ignored"]["all"]["given_band"], "AAA_many_gates_ignored": dp["cbd"]["gates_ignored"]["AAA_many"],
                   "AAA_many_with_gates": dp["cbd"]["with_gates"]["AAA_many"], "spearman": dp["cbd"]["rank"]["spearman_vs_folder_order"]["rho"],
                   "auroc_Bits_above_AAA": dp["cbd"]["rank"]["auroc_Bits_above_AAA"], "auroc_AA_above_AAA": dp["cbd"]["rank"]["auroc_AA_above_AAA"]},
           "stress_777_auditor_banded": dp["stress_audit"]["gates_ignored"]["given_band"], "stress_777_report_banded": dp["stress_report"]["gates_ignored"]["given_band"],
           "j4ck_1200": {k: {kk: v[kk] for kk in ("trays", "bands", "correct", "wrong", "dangerous_clean_called_many")} for k, v in dp["j4ck_1200"].items() if not k.startswith("_")},
           "trays_heldout_totals": dp["trays_heldout"]["totals"]}
    for nm in ("lojano_real", "loja_yolo_real"):
        if nm in dp: out[nm] = {m: {"gates_ignored": dp[nm][m]["gates_ignored"]["outcomes"], "with_gates_banded": dp[nm][m]["with_gates"]["given_band"]} for m in ("heldout", "deployed")}
    if "j4ck_lopo" in dp: out["j4ck_lopo_real"] = {m: dp["j4ck_lopo"][m]["gates_ignored"]["outcomes"] for m in ("heldout", "deployed")}
    if "j4ck_lopo" in ev: out["j4ck_lopo_typed_accuracy_same_10_as_v1"] = ev["j4ck_lopo"]["typed_accuracy_bean_weighted_same_10_as_v1"]
    json.dump(out, open(R / "clean_model.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
