"""Hand-off artefacts for the two OOD gates measured in robust_ood.py (nothing is written to app/ or models/).

S7 (bean familiarity inside the app's own unsure rule) needs, at run time, the v1 model's 576-d pooled features.
  -> writes <out>/farz_beans_fp32_with_feat.onnx: the SHIPPED v1 graph, same weights, plus one extra output 'feat'
     (Identity of '/m/net/Flatten_output_0'); checks that 'probs' is bit-identical to the shipped file on 300 crops.
  -> writes <out>/robust_s7_bank_fp16.bin: L2-normalised deployed-model features of the 582 J4ckDev training crops,
     float16, row-major [582, 576]; checks that fp16 vs fp32 bank changes no CBD/J4ckDev S7 verdict.
Hand-Mahalanobis gate (pure arithmetic on 16 per-bean colour/shape stats): mean, sd, precision matrix, threshold.
Writes reports_v2/robust_gate_params.json (parameters + file hashes; the binary files stay in <out>, default scratchpad).
Usage: python robust_export_gate.py [out_dir]
"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import robust_common as rc
import robust_ood as ro

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import onnx, onnxruntime as ort
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else rc.SCRATCH
    meta = json.load(open(rc.CACHE / "photos.json"))["photos"]; z = np.load(rc.CACHE / "beans.npz"); pi = z["photo_idx"]
    crops = np.load(rc.CACHE / "crops.npy", mmap_mode="r")
    jidx = [i for i, m in enumerate(meta) if m["src"] == "j4ck"]; jb = np.isin(pi, jidx)
    R = json.load(open(rc.REPORTS_V2 / "robust_ood.json"))
    # ---- v1 + feat output
    m = onnx.load(str(rc.V1_MODEL))
    m.graph.node.append(onnx.helper.make_node("Identity", ["/m/net/Flatten_output_0"], ["feat"], name="feat_out"))
    m.graph.output.append(onnx.helper.make_tensor_value_info("feat", onnx.TensorProto.FLOAT, ["n", 576]))
    onnx.checker.check_model(m)
    f = out / "farz_beans_fp32_with_feat.onnx"; onnx.save(m, str(f))
    rng = np.random.default_rng(0); idx = np.sort(rng.choice(len(crops), 300, replace=False))
    x = np.stack([crops[i] for i in idx]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    a = ort.InferenceSession(str(rc.V1_MODEL), so, providers=["CPUExecutionProvider"]).run(["probs"], {"image": x})[0]
    b, feat = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"]).run(["probs", "feat"], {"image": x})
    # ---- bank
    bank = ro.unit(z["feat"][jb]).astype(np.float32); bank16 = bank.astype(np.float16)
    bf = out / "robust_s7_bank_fp16.bin"; bank16.tofile(bf)
    tb = R["integrated_rule_S7"]["bean_familiarity_threshold"]
    def unfam(F, B): return np.sort(1 - ro.unit(F) @ B.T, 1)[:, :ro.KB].mean(1) > tb
    changed = 0; total = 0; vchg = 0
    for i, mm in enumerate(meta):
        bb = np.flatnonzero(pi == i)
        if not len(bb): continue
        u32, u16 = unfam(z["feat"][bb], bank), unfam(z["feat"][bb], bank16.astype(np.float32))
        changed += int((u32 != u16).sum()); total += len(bb)
        if mm["src"] == "cbd":
            v = []
            for u in (u32, u16):
                p = z["probs"][bb].copy(); p[u] = 0
                v.append(rc.v1_verdict(p, z["touching"][bb], mm["checks"], mm["cg"], ro.ABSTAIN)["band"])
            vchg += v[0] != v[1]
    # ---- hand Mahalanobis (fitted on all 582 J4ckDev beans, the bank used for every CBD score)
    M = ro.Maha(z["hand"][jb])
    hm = R["scores"]["hand_maha"]["calibrations"]["zero_AAA_many"]
    res = {"what": __doc__.strip().split("\n")[0],
           "S7_bean_familiarity": {
               "rule": "for each bean: d = mean of the 5 smallest (1 - cosine) between its L2-normalised 'feat' and the bank rows; "
                       "if d > threshold the bean is 'unsure' (same as max-prob < 0.59 or touching); then the app's existing rule "
                       "'> 15% unsure -> not sure, take the sample to the cooperative' decides",
               "threshold": tb, "k": ro.KB, "threshold_source": R["integrated_rule_S7"]["how"],
               "model_with_feat": {"file": f.name, "bytes": f.stat().st_size, "sha256": sha(f),
                                   "probs_max_abs_diff_vs_shipped": float(np.abs(a - b).max()), "n_crops": len(idx),
                                   "shipped_model": rc.rel(rc.V1_MODEL), "shipped_bytes": rc.V1_MODEL.stat().st_size},
               "bank": {"file": bf.name, "bytes": bf.stat().st_size, "sha256": sha(bf), "shape": list(bank16.shape), "dtype": "float16, row-major, rows L2-normalised"},
               "fp16_bank_vs_fp32": {"beans": total, "unfamiliar_flag_changes": changed, "cbd_verdict_changes": int(vchg)}},
           "hand_mahalanobis_gate": {
               "rule": "photo score = median over beans of sqrt(z' P z), z = (stats - mean) / sd, stats = the 16 per-bean values of "
                       "robust_common.hand_features on the 128 px crop; photo refused ('not sure') when score > threshold",
               "feature_names": rc.HAND_NAMES, "mean": M.mu.tolist(), "sd": M.sd.tolist(), "precision": M.P.tolist(),
               "threshold": hm["threshold"], "threshold_rule": hm["rule"],
               "warning": "thin margin: J4ckDev real half-photos reach 4.53 (excl. CerezaSeca), CBD AAA minimum is 4.91; CBD C and Bits photos score LOWER than J4ckDev halves (their dark beans resemble J4ckDev defect beans) and pass"},
           "files_written_to": "scratchpad (not shipped); regenerate with: python src/beans_v2/robust_export_gate.py <out_dir>"}
    rc.dump(res, "robust_gate_params.json")
    print(json.dumps({k: v for k, v in res["S7_bean_familiarity"].items() if k != "rule"}, indent=1))

if __name__ == "__main__":
    main()
