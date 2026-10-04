/**
 * Classifier input contract (src/beans/beanfinder.py docstring + farz_beans_labels.json):
 * crop = CROP x CROP RGB uint8 -> float32 in [0,1], NCHW. Normalisation (ImageNet mean/std) and the
 * temperature are inside the ONNX graph; output 'probs' [N,6] already sums to 1; output 'embed' [N,1024] is
 * L2-normalised (used only by the cooperative calibration pilot, src/lib/localcal.ts).
 */
import { CROP } from "./beanfinder";
import { V2_THRESHOLDS, type Thresholds } from "./rules";

export function cropsToTensorData(crops: Uint8Array[]): Float32Array {
  const n = crops.length;
  const plane = CROP * CROP;
  const out = new Float32Array(n * 3 * plane);
  for (let b = 0; b < n; b++) {
    const c = crops[b];
    const base = b * 3 * plane;
    for (let i = 0; i < plane; i++) {
      out[base + i] = c[i * 3] / 255;
      out[base + plane + i] = c[i * 3 + 1] / 255;
      out[base + 2 * plane + i] = c[i * 3 + 2] / 255;
    }
  }
  return out;
}

export interface Labels {
  classes: string[];
  /** good if P(good) >= this */
  good_at_or_above: number;
  /** defect if 1 - P(good) >= this */
  defect_at_or_above: number;
  input: number;
  model_sha256?: string;
  note?: string;
}

export function labelsThresholds(l: Partial<Labels>): Thresholds {
  const g = Number(l.good_at_or_above), d = Number(l.defect_at_or_above);
  return { good: Number.isFinite(g) ? g : V2_THRESHOLDS.good, defect: Number.isFinite(d) ? d : V2_THRESHOLDS.defect };
}
