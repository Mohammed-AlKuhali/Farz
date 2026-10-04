"""Export the deployed v2 model and try smaller formats.

  1. models_v2/farz_beans_v2_fp32.onnx : final weights (train_folds.py final) + ImageNet normalisation + temperature
     (from train_eval.py: fitted on the pooled LOSO held-out predictions) + softmax. input "image" [N,3,128,128] RGB
     float in [0,1], output "probs" [N,6] (good, dark, insect, broken, unhulled, other_defect).
  2. FP16: onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True).
  3. INT8 static QDQ (onnxruntime.quantization.quantize_static), calibrated on 1,200 TRAINING crops (100 per source,
     seed 0), with HardSwish / HardSigmoid / Mul nodes and the squeeze-excitation convs excluded (nodes_to_exclude);
     variants also keep the stem conv and the classifier in fp32.
  A smaller file is kept ONLY if its argmax agreement with fp32 is >= 99% on held-out crops: all CBD bean crops
  (never used for training, temperature, threshold or calibration). Agreement of the deployed 3-way call
  (good / defect / unsure at the abstain threshold) is reported too.
Writes models_v2/*.onnx, models_v2/farz_beans_v2_labels.json, reports_v2/model_v2_export.json.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
import train_common as tc

ROBUST = tc.SCRATCH / "robust_cache"


def cbd_crops():
    meta = json.load(open(ROBUST / "photos.json"))["photos"]; z = np.load(ROBUST / "beans.npz")
    pi = z["photo_idx"]; src = np.array([meta[i]["src"] for i in pi])
    crops = np.load(ROBUST / "crops.npy", mmap_mode="r")
    return np.ascontiguousarray(crops[np.flatnonzero(src == "cbd")])


def ort_probs(path, crops, bs=256):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = []
    for b in range(0, len(crops), bs):
        x = np.ascontiguousarray(crops[b:b + bs]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        out.append(s.run(["probs"], {"image": x})[0])
    return np.concatenate(out)


def call3(p, t):
    tg, td = (t, t) if not isinstance(t, tuple) else t
    pg = p[:, 0]
    return np.where(pg >= tg, 0, np.where(1 - pg >= td, 1, 2))


def latency_ms(path, n=200):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((50, 3, 128, 128), dtype=np.float32)
    for _ in range(5): s.run(None, {"image": x})
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); s.run(None, {"image": x}); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def main():
    tc.MODELS_V2.mkdir(exist_ok=True)
    ev = json.load(open(tc.REPORTS_V2 / "model_v2_eval.json"))
    T, thr = ev["deployed"]["temperature"], ev["deployed"]["threshold"]
    pr = ev["deployed"]["pair_thresholds"]; tg, td = pr["t_good"], pr["t_defect"]
    net = tc.Norm(tc.make_net(pretrained=False)); net.load_state_dict(torch.load(tc.CACHE / "folds/final.pt", map_location="cpu")); net.eval()
    dep = tc.Deploy(net, T).eval()
    fp32 = tc.MODELS_V2 / "farz_beans_v2_fp32.onnx"
    torch.onnx.export(dep, torch.rand(1, 3, 128, 128), str(fp32), input_names=["image"], output_names=["probs"],
                      dynamic_axes={"image": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
    X = cbd_crops()
    p32 = ort_probs(fp32, X)
    with torch.no_grad():
        pt = torch.cat([dep(torch.from_numpy(X[b:b + 256].astype(np.float32).transpose(0, 3, 1, 2) / 255.0)) for b in range(0, 2048, 256)]).numpy()
    res = {"protocol": __doc__.strip(), "temperature": T, "abstain_threshold_symmetric": thr, "pair_thresholds": [tg, td], "held_out_agreement_set": f"{len(X)} CBD bean crops",
           "fp32": {"file": tc.rel(fp32), "bytes": fp32.stat().st_size, "max_abs_diff_vs_pytorch_first_2048": float(np.abs(pt - p32[:2048]).max()),
                    "latency_ms_batch50_1thread_cpu": latency_ms(fp32)}}
    tc.log(f"fp32 {fp32.stat().st_size} B, torch diff {res['fp32']['max_abs_diff_vs_pytorch_first_2048']:.2e}")

    def compare(path, name, extra=None):
        p = ort_probs(path, X)
        r = {"file": tc.rel(path), "bytes": path.stat().st_size, "argmax_agreement": float((p.argmax(1) == p32.argmax(1)).mean()),
             "call3_agreement_at_symmetric_threshold": float((call3(p, thr) == call3(p32, thr)).mean()),
             "call3_agreement_at_pair_thresholds": float((call3(p, (tg, td)) == call3(p32, (tg, td))).mean()),
             "max_abs_prob_diff": float(np.abs(p - p32).max()), "latency_ms_batch50_1thread_cpu": latency_ms(path), **(extra or {})}
        r["kept"] = r["argmax_agreement"] >= 0.99
        tc.log(f"{name}: {r['bytes']} B argmax agree {r['argmax_agreement']:.4f} call agree {r['call3_agreement_at_pair_thresholds']:.4f} kept={r['kept']}")
        return r

    # FP16
    import onnx
    from onnxconverter_common import float16
    m16 = float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True)
    f16 = tc.MODELS_V2 / "farz_beans_v2_fp16.onnx"; onnx.save(m16, str(f16))
    res["fp16"] = compare(f16, "fp16", {"how": "onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)"})

    # INT8 static, excluded nodes
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = tc.CACHE / "final_pre.onnx"; quant_pre_process(str(fp32), str(pre))
    g = onnx.load(str(pre)).graph
    ops = {}
    for n in g.node: ops.setdefault(n.op_type, []).append(n.name)
    se_convs = [n.name for n in g.node if n.op_type == "Conv" and ("fc1" in n.name or "fc2" in n.name)]
    convs = [n.name for n in g.node if n.op_type == "Conv"]
    base_ex = ops.get("HardSwish", []) + ops.get("HardSigmoid", []) + ops.get("Mul", []) + se_convs
    meta, crops = tc.load_cache(in_memory=False)
    src = np.array([m["source"] for m in meta]); rng = np.random.default_rng(0)
    cal_idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(src == s), 100, replace=False) for s in tc.SOURCES]))
    C = np.ascontiguousarray(crops[cal_idx]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0

    class Calib(CalibrationDataReader):
        def __init__(s): s.it = iter([{"image": C[i:i + 16]} for i in range(0, len(C), 16)])
        def get_next(s): return next(s.it, None)

    variants = {"int8_ex_hswish_hsig_mul_se": base_ex,
                "int8_ex_hswish_hsig_mul_se_stem_head": base_ex + convs[:1] + ops.get("Gemm", []),
                "int8_ex_hswish_hsig_mul_se_first4convs_head": base_ex + convs[:4] + ops.get("Gemm", [])}
    res["int8"] = {"op_counts": {k: len(v) for k, v in ops.items()}, "se_convs_excluded": len(se_convs), "calibration": f"{len(C)} training crops, 100 per source, seed 0, MinMax", "variants": {}}
    for name, ex in variants.items():
        out = tc.MODELS_V2 / f"farz_beans_v2_{name}.onnx"
        try:
            quantize_static(str(pre), str(out), Calib(), quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, nodes_to_exclude=sorted(set(ex)))
            res["int8"]["variants"][name] = compare(out, name, {"nodes_excluded": len(set(ex))})
            if not res["int8"]["variants"][name]["kept"]: out.unlink()
        except Exception as e:
            res["int8"]["variants"][name] = {"error": repr(e)[:300]}; tc.log(name, "failed", e)
    pre.unlink(missing_ok=True)
    if not res["fp16"]["kept"]: f16.unlink()
    kept = [("fp16", res["fp16"])] if res["fp16"]["kept"] else []
    kept += [(k, v) for k, v in res["int8"]["variants"].items() if v.get("kept")]
    best = min(kept, key=lambda kv: kv[1]["bytes"]) if kept else ("fp32", res["fp32"])
    res["recommended_file"] = best[1]["file"]; res["recommended_format"] = best[0]
    labels = {"classes": tc.CLASSES, "defect_classes": tc.CLASSES[1:], "good_at_or_above": tg, "defect_at_or_above": td,
              "abstain_below": round(thr, 2), "input": 128,
              "decision": "pGood = probs[0]; good if pGood >= good_at_or_above; else defect if 1 - pGood >= defect_at_or_above "
                          "(type = argmax of probs[1:], shown only as 'possible'); otherwise unsure. Touching blobs are unsure. "
                          "abstain_below = the single symmetric threshold (rule A, reported for comparison only).",
              "temperature_baked_in": T, "model_file": best[1]["file"],
              "note": "Farz v2 per-bean classifier. input 'image' float32 [N,3,128,128] RGB in [0,1] (crop contract of "
                      "src/beans/beanfinder.py); output 'probs' [N,6] = softmax(logits/T). Temperature and abstain threshold "
                      "were fitted on leave-one-source-out held-out predictions (reports_v2/model_v2_eval.json). Trained on 12 "
                      "sources incl. 5 that state no licence and 2 that are non-commercial (see reports_v2/model_v2.md)."}
    json.dump(labels, open(tc.MODELS_V2 / "farz_beans_v2_labels.json", "w"), indent=1)
    res["labels_json"] = labels
    json.dump(res, open(tc.REPORTS_V2 / "model_v2_export.json", "w"), indent=1)
    tc.log("export done; recommended", best[0], best[1]["file"])


if __name__ == "__main__":
    main()
