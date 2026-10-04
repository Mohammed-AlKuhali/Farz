"""(2c) Browser cost of the unusual-bean detector's frozen feature extractor.

Exports torchvision MobileNetV3-Small IMAGENET1K_V1 (features + global average pool, ImageNet normalisation inside the
graph; input 'image' float32 [N,3,128,128] RGB in [0,1] = the bean-finder crop contract; output 'emb' [N,576]) to ONNX,
plus an fp16-weights copy if onnxconverter-common is available. Measures file size, parity vs PyTorch, and median
latency for one 50-bean photo (batch 50) on: onnxruntime Python CPU 1 thread, and onnxruntime-web wasm (Node, 1 thread,
the package version the app ships). Laptop numbers, NOT a phone.
Model files go to the scratchpad (this task does not own models/). Writes reports_v2/robust_export.json.
Usage: python robust_export.py [out_dir]
"""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import robust_common as rc

def decision_parity(out_dir):
    """Does fp16 change which beans the relative detector flags? 40 CBD photos (first 5 of each of 8 grades), 'emb' and
    'both' scores, z > 3.5, ONNX fp16 embeddings vs the cached PyTorch fp32 embeddings."""
    import onnxruntime as ort
    from robust_unusual import detect, Z_CUT
    f16 = out_dir / "robust_feat_mnv3s_imagenet_fp16.onnx"
    if not f16.exists(): return {"skipped": "no fp16 file"}
    meta = json.load(open(rc.CACHE / "photos.json"))["photos"]; z = np.load(rc.CACHE / "beans.npz"); pi = z["photo_idx"]
    crops = np.load(rc.CACHE / "crops.npy", mmap_mode="r"); floors = 0.1 * np.nanstd(z["hand"], 0)
    s = ort.InferenceSession(str(f16), providers=["CPUExecutionProvider"])
    ids = [i for g in rc.CBD_GRADES[:8] for i in [j for j, m in enumerate(meta) if m["src"] == "cbd" and m["label"] == g][:5]]
    agree = {"emb": [0, 0], "both": [0, 0]}; changed = {"emb": 0, "both": 0}; flagged = {"emb": 0, "both": 0}
    for i in ids:
        b = np.flatnonzero(pi == i)
        x = np.stack([crops[k] for k in b]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        e16 = s.run(None, {"image": x})[0]
        a32 = detect(z["hand"][b], z["emb"][b], floors); a16 = detect(z["hand"][b], e16, floors)
        for m in agree:
            f32, f16_ = a32[m] > Z_CUT, a16[m] > Z_CUT
            agree[m][0] += int((f32 == f16_).sum()); agree[m][1] += len(b); changed[m] += int((f32 != f16_).sum()); flagged[m] += int(f32.sum())
    return {"photos": len(ids), "beans": agree["emb"][1], **{m: {"flag_agreement": round(agree[m][0] / agree[m][1], 5), "beans_changed": changed[m],
            "beans_flagged_fp32": flagged[m]} for m in agree}}

def main():
    import torch, onnxruntime as ort
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else rc.SCRATCH
    out_dir.mkdir(parents=True, exist_ok=True)
    m = rc.imagenet_extractor().eval()
    f32 = out_dir / "robust_feat_mnv3s_imagenet_fp32.onnx"
    torch.onnx.export(m, torch.zeros(1, 3, 128, 128), str(f32), input_names=["image"], output_names=["emb"],
                      dynamic_axes={"image": {0: "n"}, "emb": {0: "n"}}, opset_version=17, dynamo=False)
    crops = np.load(rc.CACHE / "crops.npy", mmap_mode="r")
    rng = np.random.default_rng(0); idx = np.sort(rng.choice(len(crops), 200, replace=False))
    x = np.stack([crops[i] for i in idx]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    with torch.no_grad(): ref = m(torch.from_numpy(x)).numpy()
    res = {"what": __doc__.strip().split("\n")[0], "params": int(sum(p.numel() for p in m.parameters())), "models": {}}
    files = [("fp32", f32)]
    try:
        import onnx
        from onnxconverter_common import float16
        f16 = out_dir / "robust_feat_mnv3s_imagenet_fp16.onnx"
        mm = float16.convert_float_to_float16(onnx.load(str(f32)), keep_io_types=True); onnx.save(mm, str(f16)); files.append(("fp16", f16))
    except Exception as e:
        res["fp16_skipped"] = repr(e)
    for name, f in files:
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        s = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
        out = s.run(None, {"image": x})[0]
        cos = (out * ref).sum(1) / (np.linalg.norm(out, axis=1) * np.linalg.norm(ref, axis=1))
        xb = x[:50]; s.run(None, {"image": xb}); ts = []
        for _ in range(15):
            t = time.perf_counter(); s.run(None, {"image": xb}); ts.append(time.perf_counter() - t)
        res["models"][name] = {"file": f.name, "bytes": f.stat().st_size, "max_abs_diff_vs_torch": float(np.abs(out - ref).max()),
                               "min_cosine_vs_torch": float(cos.min()), "median_ms_batch50_python_ort_cpu_1thread": round(1000 * float(np.median(ts)), 1)}
        # wasm in Node (same onnxruntime-web the app ships)
        js = rc.SCRATCH / "robust_wasm_bench.mjs"
        js.write_text(f"""
import {{ createRequire }} from "node:module"; import {{ readFileSync }} from "node:fs";
const require = createRequire({json.dumps(str(rc.ROOT / "app/package.json"))});
const ort = require("onnxruntime-web"); ort.env.wasm.numThreads = 1;
const s = await ort.InferenceSession.create(readFileSync({json.dumps(str(f))}), {{ executionProviders: ["wasm"] }});
const x = new Float32Array(readFileSync({json.dumps(str(rc.SCRATCH / "robust_x50.bin"))}).buffer.slice(0));
const t = new ort.Tensor("float32", x, [50, 3, 128, 128]);
let o = await s.run({{ image: t }}); const ts = [];
for (let i = 0; i < 10; i++) {{ const a = performance.now(); o = await s.run({{ image: t }}); ts.push(performance.now() - a); }}
ts.sort((a, b) => a - b);
console.log(JSON.stringify({{ version: ort.env.versions?.web, node: process.version, median_ms: ts[5], first: Array.from(o.emb.data.slice(0, 576)) }}));
""")
        xb.astype(np.float32).tofile(rc.SCRATCH / "robust_x50.bin")
        try:
            p = subprocess.run(["node", str(js)], capture_output=True, text=True, timeout=300)
            w = json.loads(p.stdout.strip().splitlines()[-1])
            res["models"][name]["wasm"] = {"runtime": f"onnxruntime-web {w['version']} wasm, numThreads=1, Node {w['node']}",
                                           "median_ms_batch50": round(w["median_ms"], 1),
                                           "max_abs_diff_vs_python_bean0": float(np.abs(np.array(w["first"]) - out[0]).max())}
        except Exception as e:
            res["models"][name]["wasm"] = {"error": repr(e), "stderr": p.stderr[-500:] if 'p' in dir() else ""}
        print(name, res["models"][name], flush=True)
    res["decision_parity_fp16_vs_torch"] = decision_parity(out_dir)
    res["machine"] = "MacBook M5 Pro (laptop CPU), NOT a phone"
    res["note"] = "model files written to the scratchpad, not shipped; the v1 classifier (6.1 MB fp32) is a separate download"
    rc.dump(res, "robust_export.json")

if __name__ == "__main__":
    main()
