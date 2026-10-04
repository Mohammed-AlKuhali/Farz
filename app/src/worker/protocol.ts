import type { Checks } from "../lib/beanfinder";
import type { ColourGate } from "../lib/colourgate";
import type { Thresholds } from "../lib/rules";

export type ToWorker =
  | { type: "init"; base: string }
  | {
      type: "analyze"; id: number; width: number; height: number; rgba: ArrayBuffer; skipModelIfGateFails?: boolean;
      /** lower bean minimum for the grader's calibration photos (rules.gate minBeans) */
      minBeans?: number;
    }
  /** classify ready-made 128x128 RGB crops (the bundled calibration demo set) */
  | { type: "embed"; id: number; n: number; crops: ArrayBuffer };

export interface WorkerBean {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  touching: boolean;
  probs: number[] | null;
}

export type FromWorker =
  | {
      type: "ready"; stub: boolean; classes: string[]; thresholds: Thresholds; modelBytes: number;
      /** sha256 of the model file (from the labels json, checked against the bytes when crypto.subtle exists) */
      modelSha: string; embedDim: number; error?: string;
    }
  | {
      type: "result";
      id: number;
      workW: number;
      workH: number;
      scale: number;
      checks: Checks;
      colour: ColourGate;
      beans: WorkerBean[];
      /** L2-normalised `embed` output, beans x embedDim, packed (null for the stub or when not classified) */
      embeds: Float32Array | null;
      embedDim: number;
      classified: boolean;
      stub: boolean;
      ms: { finder: number; model: number; total: number };
    }
  | { type: "embedded"; id: number; probs: number[][]; embeds: Float32Array | null; embedDim: number }
  | { type: "error"; id?: number; message: string };
