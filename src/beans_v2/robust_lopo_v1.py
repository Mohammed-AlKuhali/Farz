"""Leave-one-photo-out v1 models, re-run ONLY to get per-bean outputs for the OOD-gate study (3).

reports/beans_lopo.json keeps only per-photo shares. To score photo-level OOD signals of the v1 model (max-prob,
energy, feature-space kNN) on a J4ckDev photo the model has NOT seen, we retrain the exact v1 recipe
(src/beans/train_beans.py: train(), seed 0, 40 epochs) on the other 10 photos for each of the 11 photos, including
Normales (the v1 LOPO skipped it because it is the only 'good' photo; here the model simply never sees 'good', which is
exactly the situation of a gate facing a new kind of photo).
The v1 training module is executed from its source with its log redirected to the scratchpad, so no v1 file is
written. Outputs (scratchpad): robust_cache/lopo_v1.npz with probs (T = deployed 0.825), raw logits, 576-d features of
the held-out photo's beans AND of the training photos' beans (the in-fold bank) per fold.
"""
import sys, time, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import robust_common as rc

def load_tb():
    src = (rc.ROOT / "src/beans/train_beans.py").read_text()
    src = src.replace('LOG = open(ROOT / "logs/beans_train.log", "a")', f'LOG = open({str(rc.SCRATCH / "robust_lopo_v1_train.log")!r}, "a")')
    assert "robust_lopo_v1_train.log" in src
    mod = types.ModuleType("tb_v2"); mod.__file__ = str(rc.ROOT / "src/beans/train_beans.py")
    sys.path.insert(0, str(rc.ROOT / "src/beans"))
    exec(compile(src, "train_beans.py(v2-exec)", "exec"), mod.__dict__)
    return mod

def main():
    import torch
    tb = load_tb(); T = 0.8248995244503021
    rows = tb.load_rows(); photos = np.array([r["photo"] for r in rows])
    out = {}
    t0 = time.time()
    for ph in sorted(rc.PHOTO_CLASS):
        tr = [r for r in rows if r["photo"] != ph]
        m = tb.train(tr, f"v2-lopo-{ph}")
        net = m.net
        feats = []
        hook = net.avgpool.register_forward_hook(lambda mod, i, o: feats.append(o.flatten(1).detach().float().cpu()))
        lg = tb.logits_of(m, [r["img"] for r in rows]).numpy(); hook.remove()
        f = torch.cat(feats).numpy()
        p = np.exp(lg / T - (lg / T).max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        out[ph] = dict(logits=lg, probs=p, feat=f)
        print(f"{ph}: done {time.time()-t0:.0f}s", flush=True)
    np.savez(rc.CACHE / "lopo_v1.npz", photos=photos, order=np.array(sorted(rc.PHOTO_CLASS)),
             logits=np.stack([out[p]["logits"] for p in sorted(rc.PHOTO_CLASS)]),
             probs=np.stack([out[p]["probs"] for p in sorted(rc.PHOTO_CLASS)]),
             feat=np.stack([out[p]["feat"] for p in sorted(rc.PHOTO_CLASS)]))
    # sanity: recompute v1 LOPO recall on the 10 photos and compare to reports/beans_lopo.json
    import json
    lo = json.load(open(rc.ROOT / "reports/beans_lopo.json"))["photos"]
    for ph in sorted(rc.PHOTO_CLASS):
        if "n" not in lo[ph]: continue
        pr = out[ph]["probs"][photos == ph].argmax(1)
        print(ph, "recall now", round(float((pr == rc.CLASSES.index(rc.PHOTO_CLASS[ph])).mean()), 4), "v1 report", lo[ph]["recall_correct_class"])

if __name__ == "__main__":
    main()
