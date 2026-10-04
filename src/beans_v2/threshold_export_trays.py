"""Export the threshold sweep's held-out trays (threshold_sweep.py build_trays(), same seeds) as JSON for the app's own
TypeScript rules (app/tests/threshold.app.test.ts): per unit the P(good) of every bean a tray uses (shipped T) and the
trays as index lists + truth band. Lets the shipped callBean() + decide() re-measure the sweep with the app's code.
Usage: FARZ_SCRATCH=<scratch> python threshold_export_trays.py <out.json>
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import train_common as tc
import clean_common as cc
from threshold_sweep import build_trays


def main(out):
    T, _, _, _ = cc.deployed()
    units = build_trays()
    res = {"temperature": T, "units": []}
    for u in units:
        used = np.unique(np.concatenate([sel for _, _, sel in u["trays"]]))
        remap = {int(k): i for i, k in enumerate(used)}
        pg = tc.softmax(u["logits"], T)[used, 0]
        res["units"].append({"name": u["name"], "kind": u["kind"], "pgood": [round(float(x), 6) for x in pg],
                             "trays": [{"size": int(nb), "truth": tr, "beans": [remap[int(k)] for k in sel]} for nb, tr, sel in u["trays"]]})
    json.dump(res, open(out, "w"))
    tc.log("exported", sum(len(u["trays"]) for u in res["units"]), "trays to", out)


if __name__ == "__main__":
    main(sys.argv[1])
