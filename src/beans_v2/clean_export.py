"""Export the licence-stated ("clean") v2 model with TWO outputs, fp32 and fp16, and check them.

  input  "image" float32 [N,3,128,128] RGB in [0,1] (crop contract of src/beans/beanfinder.py; same as v1/v2)
  output "probs" float32 [N,6] = softmax(logits / T) (good, dark, insect, broken, unhulled, other_defect); ImageNet
         normalisation and T (clean_eval.json, fitted on clean LOSO held-out predictions) are inside the graph
  output "embed" float32 [N,1024] = the penultimate feature (Hardswish output feeding the last Linear), L2-normalised.
         Computed as u = h / max|h|, embed = u / ||u|| (identical maths to h / ||h||) so that the fp16 graph cannot
         overflow when squaring (|u| <= 1, sum of 1,024 squares <= 1,024).
FP16 = onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True). Kept only if its top-class agreement with
fp32 is >= 99% on ALL CBD bean crops (never used for training, T, thresholds or anything else).
Also writes the inputs/expected outputs for the onnxruntime-web wasm parity check (clean_wasm_parity.mjs) and runs it.
Writes models_v2/farz_beans_v2_clean_{fp32,fp16}.onnx, models_v2/farz_beans_v2_clean_labels.json, reports_v2/clean_export.json.
"""
import hashlib, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
import train_common as tc
import clean_common as cc


class Export(torch.nn.Module):
    def __init__(s, norm, T):
        super().__init__(); s.m = norm; s.register_buffer("t", torch.tensor(float(T)))
    def forward(s, image):
        lg, h = cc.penultimate(s.m, image)
        u = h / h.abs().amax(1, keepdim=True).clamp_min(1e-6)
        e = u / torch.sqrt((u * u).sum(1, keepdim=True)).clamp_min(1e-6)
        return torch.softmax(lg / s.t, dim=1), e


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    T, tg, td, ev = cc.deployed()
    net = cc.load_torch(cc.FINAL_PT, dev=torch.device("cpu"))
    exp = Export(net, T).eval()
    cc.ONNX32.parent.mkdir(exist_ok=True)
    torch.onnx.export(exp, torch.rand(1, 3, 128, 128), str(cc.ONNX32), input_names=["image"], output_names=["probs", "embed"],
                      dynamic_axes={"image": {0: "n"}, "probs": {0: "n"}, "embed": {0: "n"}}, opset_version=17, dynamo=False)
    import onnx
    from onnxconverter_common import float16
    onnx.checker.check_model(onnx.load(str(cc.ONNX32)))
    m16 = float16.convert_float_to_float16(onnx.load(str(cc.ONNX32)), keep_io_types=True)
    onnx.save(m16, str(cc.ONNX16)); onnx.checker.check_model(onnx.load(str(cc.ONNX16)))
    b, _, _, _ = cc.cbd_rows()
    X = np.ascontiguousarray(np.load(cc.ROBUST / "crops.npy", mmap_mode="r")[b])
    s32, s16 = cc.ort_session(cc.ONNX32), cc.ort_session(cc.ONNX16)
    P32, E32 = cc.ort_run(s32, X); P16, E16 = cc.ort_run(s16, X)
    with torch.no_grad():
        xt = torch.from_numpy(X[:2048].astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
        pt, et = exp(xt); pt, et = pt.numpy(), et.numpy()
    # embeddings must equal the PyTorch helper used by the calibration study (F.normalize of h)
    _, Eh = cc.embed_and_logits(net, X[:2048])
    c3 = lambda p: cc.calls3(p, tg, td)
    cos = (E16 * E32).sum(1) / (np.linalg.norm(E16, axis=1) * np.linalg.norm(E32, axis=1))
    res = {"protocol": __doc__.strip(), "temperature": T, "pair_thresholds": [tg, td], "agreement_set": f"all {len(X)} CBD bean crops (robust cache)",
           "final_weights": str(cc.FINAL_PT), "final_weights_sha256": sha(cc.FINAL_PT),
           "fp32": {"file": tc.rel(cc.ONNX32), "bytes": cc.ONNX32.stat().st_size, "sha256": sha(cc.ONNX32),
                    "max_abs_prob_diff_vs_pytorch_first_2048": float(np.abs(pt - P32[:2048]).max()),
                    "max_abs_embed_diff_vs_pytorch_first_2048": float(np.abs(et - E32[:2048]).max()),
                    "max_abs_embed_diff_onnx_vs_calibration_helper_first_2048": float(np.abs(Eh - E32[:2048]).max()),
                    "embed_norm_min_max": [float(np.linalg.norm(E32, axis=1).min()), float(np.linalg.norm(E32, axis=1).max())]},
           "fp16": {"file": tc.rel(cc.ONNX16), "bytes": cc.ONNX16.stat().st_size, "sha256": sha(cc.ONNX16),
                    "how": "onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)",
                    "top_class_agreement_vs_fp32": float((P16.argmax(1) == P32.argmax(1)).mean()),
                    "call3_agreement_vs_fp32_at_pair": float((c3(P16) == c3(P32)).mean()),
                    "max_abs_prob_diff_vs_fp32": float(np.abs(P16 - P32).max()),
                    "embed_cosine_vs_fp32_min_mean": [float(cos.min()), float(cos.mean())],
                    "max_abs_embed_diff_vs_fp32": float(np.abs(E16 - E32).max()),
                    "any_nan_or_inf": bool(~np.isfinite(P16).all() or ~np.isfinite(E16).all())}}
    res["fp16"]["kept"] = res["fp16"]["top_class_agreement_vs_fp32"] >= 0.99 and not res["fp16"]["any_nan_or_inf"]
    # latency, laptop CPU, 1 thread, batch 50
    for k, f in (("fp32", cc.ONNX32), ("fp16", cc.ONNX16)):
        s = cc.ort_session(f, threads=1); x = np.random.default_rng(0).random((50, 3, 128, 128), dtype=np.float32)
        for _ in range(3): s.run(None, {"image": x})
        ts = []
        for _ in range(15):
            t0 = time.perf_counter(); s.run(None, {"image": x}); ts.append((time.perf_counter() - t0) * 1000)
        res[k]["latency_ms_batch50_1thread_laptop_cpu"] = float(np.median(ts))
    tc.log("clean export:", {k: {kk: v for kk, v in res[k].items() if kk not in ("sha256", "how", "file")} for k in ("fp32", "fp16")})
    # wasm parity inputs: 300 random CBD crops (seed 1), Python ORT outputs for both files
    wd = tc.CACHE / "clean_wasm"; wd.mkdir(exist_ok=True)
    sel = np.sort(np.random.default_rng(1).choice(len(X), 300, replace=False))
    xs = np.ascontiguousarray(X[sel].astype(np.float32).transpose(0, 3, 1, 2) / 255.0); xs.tofile(wd / "x.bin")
    for tag, s in (("fp32", s32), ("fp16", s16)):
        p, e = s.run(["probs", "embed"], {"image": xs}); p.astype(np.float32).tofile(wd / f"py_{tag}_probs.bin"); e.astype(np.float32).tofile(wd / f"py_{tag}_embed.bin")
    json.dump({"n": 300, "d": int(E32.shape[1]), "t_good": tg, "t_defect": td}, open(wd / "meta.json", "w"))
    res["wasm_parity"] = {}
    for tag, f in (("fp32", cc.ONNX32), ("fp16", cc.ONNX16)):
        try:
            out = subprocess.run(["node", str(Path(__file__).parent / "clean_wasm_parity.mjs"), str(wd), str(f), tag], capture_output=True, text=True, timeout=600)
            res["wasm_parity"][tag] = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {"error": out.stderr[-800:]}
        except Exception as e:
            res["wasm_parity"][tag] = {"error": repr(e)}
        tc.log("wasm", tag, res["wasm_parity"][tag])
    labels = {"classes": tc.CLASSES, "defect_classes": tc.CLASSES[1:], "good_at_or_above": tg, "defect_at_or_above": td,
              "temperature_baked_in": T, "input": 128, "outputs": {"probs": "[N,6] softmax(logits/T)", "embed": f"[N,{E32.shape[1]}] L2-normalised penultimate feature (for on-device calibration)"},
              "model_file": tc.rel(cc.ONNX16 if res["fp16"]["kept"] else cc.ONNX32),
              "decision": "pGood = probs[0]; good if pGood >= good_at_or_above; else defect if 1 - pGood >= defect_at_or_above "
                          "(type = argmax of probs[1:], shown only as 'possible'); otherwise unsure. Touching blobs are unsure.",
              "training_data": "ONLY the 7 sources that state a licence: afiyah_deteksi, afiyah_bijikopi, vicanadya16 (CC0), loja_yolo (CC BY 4.0), "
                               "samruddh_grading (MIT), lojano (CC BY-NC 4.0), J4ckDev (CC BY-NC-SA 4.0). The 5 sources with no licence were used only as "
                               "held-out test data. Licence of these weights: NON-COMMERCIAL research prototype, ShareAlike (CC BY-NC-SA 4.0 from J4ckDev).",
              "thresholds_fitted_on": "clean leave-one-source-out predictions over the licensed sources, afiyah twins held out together (reports_v2/clean_eval.json)"}
    json.dump(labels, open(cc.LABELS, "w"), indent=1)
    res["labels_json"] = labels
    cc.jdump(res, tc.REPORTS_V2 / "clean_export.json")
    tc.log("wrote reports_v2/clean_export.json")


if __name__ == "__main__":
    main()
