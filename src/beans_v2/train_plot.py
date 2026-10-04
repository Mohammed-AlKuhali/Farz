"""Chart for reports_v2/model_v2.md from model_v2_deploy.json:
left  = outcome of every simulated test set under the recommended rule P_pair (and the local-calibration sensitivity run)
right = threshold sweep (rule A, sensitivity only) on the 1,560 held-out-source trays and the 464 CBD photos."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import train_common as tc


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = json.load(open(tc.REPORTS_V2 / "model_v2_deploy.json")); S = d["sets"]; V = "P_pair"
    rows = []  # (label, correct band, wrong band, dangerous, not sure, retake, n)

    def add_photos(label, t, correct=None):
        rows.append((label, correct if correct is not None else 0, (t["given_band"] - (correct or 0)) if correct is not None else t["given_band"], 0, t["not_sure"], t["retake"], t["photos"], correct is not None))

    add_photos("CBD 464 real photos (final model)", S["cbd"][V]["all"])
    add_photos("CBD stress 777 re-shoots (final)", S["cbd_stress"][V]["all"])
    add_photos("J4ckDev 11 real photos (LOPO)", S["j4ck_lopo"][V]["summary"])
    add_photos("lojano 36 real photos (LOSO)", S["lojano_real"][V]["all"])
    add_photos("notplying 204 real photos (LOSO)", S["notplying_real"][V]["all"])
    add_photos("loja_yolo 315 real photos (LOSO)", S["loja_yolo_real"][V]["all"])
    add_photos("app demo 7 photos (IN-SAMPLE)", S["app_demo_in_sample"][V]["summary"])
    for nm, lab in (("j4ck_halves_trays", "J4ckDev-style 60 trays, unseen beans"), ("j4ck_halves_trays_LOCAL_CALIBRATION", "same 60 trays, LOCAL calibration (sensitivity)")):
        c = S[nm][V]["by_truth"]
        rows.append((lab, sum(x["correct_band"] for x in c.values()), sum(x["wrong_band"] - x["dangerous"] for x in c.values()), sum(x["dangerous"] for x in c.values()),
                     sum(x["not_sure"] for x in c.values()), sum(x["retake"] for x in c.values()), sum(x["photos"] for x in c.values()), True))
    tot = [0, 0, 0, 0, 0]
    for s in tc.SOURCES:
        for c in S.get(f"trays_{s}", {}).get(V, {}).values():
            tot[0] += c["correct_band"]; tot[1] += c["wrong_band"] - c["dangerous"]; tot[2] += c["dangerous"]; tot[3] += c["not_sure"]; tot[4] += c["photos"]
    rows.append(("1,560 trays from 12 held-out sources (LOSO)", tot[0], tot[1], tot[2], tot[3], 0, tot[4], True))

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]; labels = [r[0] for r in rows]; n = np.array([r[6] for r in rows], float); left = np.zeros(len(rows))
    parts = [("band, truth known: correct", 1, "#2e7d32"), ("band, wrong (adjacent) / truth unknown", 2, "#f9a825"), ("band, dangerous clean<->many", 3, "#c62828"),
             ("not sure", 4, "#9e9e9e"), ("retake", 5, "#d0d0d0")]
    for nm, j, col in parts:
        vals = 100 * np.array([r[j] for r in rows]) / n
        ax.barh(labels, vals, left=left, color=col, label=nm); left += vals
    for i, r in enumerate(rows):
        ax.text(101, i, f"band {r[1] + r[2] + r[3]}/{r[6]}", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 118); ax.set_xlabel("% of photos / trays"); ax.tick_params(axis="y", labelsize=8)
    ax.set_title(f"Recommended rule P_pair (v2, t_good {d['deployed_pair'][0]}, t_defect {d['deployed_pair'][1]}, 23:05 app rules)\n"
                 "every photo/tray scored by a model that never saw it (except the in-sample demo row)", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.09), ncol=3)
    ax = axes[1]
    ts = ["0.6", "0.7", "0.8", "0.85", "0.9", "0.95"]
    agg = {t: [0, 0, 0, 0] for t in ts}
    for s in tc.SOURCES:
        sw = S.get(f"trays_{s}", {}).get("_threshold_sweep_rule_A_sensitivity_only", {})
        for t in ts:
            for c in sw.get(t, {}).values():
                agg[t][0] += c["correct_band"]; agg[t][1] += c["wrong_band"] - c["dangerous"]; agg[t][2] += c["dangerous"]; agg[t][3] += c["photos"]
    x = np.arange(len(ts))
    ax.plot(x, [agg[t][0] for t in ts], "o-", color="#2e7d32", label="held-out trays: correct band")
    ax.plot(x, [agg[t][1] for t in ts], "o-", color="#f9a825", label="held-out trays: wrong band (adjacent)")
    ax.plot(x, [agg[t][2] for t in ts], "o-", color="#c62828", label="held-out trays: dangerous (clean<->many)")
    cs = S["cbd"].get("_threshold_sweep_rule_A_sensitivity_only", {})
    ax.plot(x, [cs.get(t, {}).get("given_band", np.nan) for t in ts], "s--", color="#1565c0", label="CBD photos given a band (of 464)")
    ax.set_xticks(x); ax.set_xticklabels(ts); ax.set_xlabel("one symmetric bean threshold t (rule A)\nsensitivity only - NOT used to choose anything")
    ax.set_ylabel("count (of 1,560 trays / 464 photos)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"Threshold sweep (held-out sources)\nt picked for >= 90% answered accuracy = {d['deployed_threshold']}", fontsize=10)
    fig.tight_layout(); fig.savefig(tc.REPORTS_V2 / "model_v2_deploy.png", dpi=120); plt.close(fig)
    tc.log("wrote reports_v2/model_v2_deploy.png")


if __name__ == "__main__":
    main()
