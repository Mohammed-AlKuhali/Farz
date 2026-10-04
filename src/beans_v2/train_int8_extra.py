"""Extra INT8 static attempts (after the three in train_export.py all failed the 99% agreement bar): Percentile and
Entropy calibration with the same excluded nodes, and a variant that quantises only the 1x1 (pointwise) convs.
Same held-out agreement set (all CBD crops) and the same 99% bar. Appends to reports_v2/model_v2_export.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, onnx
import train_common as tc
from train_export import cbd_crops, ort_probs, call3

def main():
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static, CalibrationMethod
    from onnxruntime.quantization.shape_inference import quant_pre_process
    ex = json.load(open(tc.REPORTS_V2 / "model_v2_export.json"))
    fp32 = tc.ROOT / ex["fp32"]["file"]; X = cbd_crops(); p32 = ort_probs(fp32, X)
    tg, td = ex["pair_thresholds"]
    pre = tc.CACHE / "final_pre2.onnx"; quant_pre_process(str(fp32), str(pre))
    g = onnx.load(str(pre)).graph
    init = {i.name: i for i in g.initializer}
    ops = {}
    for n in g.node: ops.setdefault(n.op_type, []).append(n.name)
    se = [n.name for n in g.node if n.op_type == "Conv" and ("fc1" in n.name or "fc2" in n.name)]
    base_ex = ops.get("HardSwish", []) + ops.get("HardSigmoid", []) + ops.get("Mul", []) + se
    pw = [n.name for n in g.node if n.op_type == "Conv" and n.name not in se and n.input[1] in init and list(init[n.input[1]].dims[2:]) == [1, 1]]
    meta, crops = tc.load_cache(in_memory=False); src = np.array([m["source"] for m in meta]); rng = np.random.default_rng(0)
    cal = np.sort(np.concatenate([rng.choice(np.flatnonzero(src == s), 100, replace=False) for s in tc.SOURCES]))
    C = np.ascontiguousarray(crops[cal]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    class Calib(CalibrationDataReader):
        def __init__(s): s.it = iter([{"image": C[i:i + 16]} for i in range(0, len(C), 16)])
        def get_next(s): return next(s.it, None)
    tries = {"int8_ex_hswish_hsig_mul_se_percentile": dict(nodes_to_exclude=base_ex, calibrate_method=CalibrationMethod.Percentile),
             "int8_ex_hswish_hsig_mul_se_entropy": dict(nodes_to_exclude=base_ex, calibrate_method=CalibrationMethod.Entropy),
             "int8_pointwise_convs_only_minmax": dict(nodes_to_quantize=pw)}
    out = {}
    for name, kw in tries.items():
        f = tc.MODELS_V2 / f"farz_beans_v2_{name}.onnx"
        try:
            quantize_static(str(pre), str(f), Calib(), quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, **kw)
            p = ort_probs(f, X)
            r = {"bytes": f.stat().st_size, "argmax_agreement": float((p.argmax(1) == p32.argmax(1)).mean()),
                 "call3_agreement_at_pair_thresholds": float((call3(p, (tg, td)) == call3(p32, (tg, td))).mean()), "max_abs_prob_diff": float(np.abs(p - p32).max())}
            r["kept"] = r["argmax_agreement"] >= 0.99
            if not r["kept"]: f.unlink()
        except Exception as e:
            r = {"error": repr(e)[:300]}
        out[name] = r; tc.log(name, r)
    pre.unlink(missing_ok=True)
    ex["int8"]["extra_attempts"] = {"pointwise_convs": len(pw), "variants": out}
    json.dump(ex, open(tc.REPORTS_V2 / "model_v2_export.json", "w"), indent=1)

if __name__ == "__main__":
    main()
