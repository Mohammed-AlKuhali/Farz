# Farz local calibration: JS-portable algorithm spec (prototype head, v1)

Written 02:36 BST Sun 4 Oct 2026, after the pre-registered study in `reports_v2/localcal.md` returned **GO (prototypes, K = 20)** on loja_yolo. That verdict covers **one source**. Read "Scope and safety" before wiring any of this into the default app path. Every number here comes from `reports_v2/localcal.json` (study) or `reports_v2/clean_export.json` (model file). Python reference implementation: `src/beans_v2/localcal.py` (`proto_fit`, `proto_score`, `answer`).

## 1. Inputs

| What | Where it comes from | Shape / type |
|---|---|---|
| Model | `models_v2/farz_beans_v2_clean_fp16.onnx` (sha256 `c40c10f915f7a3ab78f1aead50d3cd027113b1a53d031eea21ffec331c5bc87a`, 3,082,642 B) | input `image` float32 [N,3,128,128] RGB in [0,1] (the existing crop contract, unchanged) |
| `probs` | ONNX output 1 | float32 [N,6]: good, dark, insect, broken, unhulled, other_defect. Temperature T = 2.0459 is baked in |
| `embed` | ONNX output 2 | float32 [N,1024], already L2-normalised (norm 1 within 2e-7 in fp32; fp16 vs fp32 cosine >= 0.99988 on 23,447 CBD crops; onnxruntime-web wasm vs Python max abs diff 1.5e-3, min cosine 0.999995 on 300 crops) |
| `touching` | the bean finder's existing per-bean flag | boolean |

Labels used by calibration are two-way only: `good` or `defect` (any defect type).

## 2. Calibration (once per cooperative set-up: same sheet, same light, same phone)

1. The grader photographs a few handfuls of the cooperative's own coffee on the usual sheet. The app finds the beans; the grader marks each bean `good` or `defect`. **Touching beans are never used.**
2. Take **n = 10 good and 10 defect** beans (K = 20, the measured setting; the study also measured K = 50 and 100 with similar results on loja_yolo). If the grader marks more, draw n of each uniformly at random. Refuse to calibrate with fewer than 10 of either class.
3. In float64 (plain JS numbers):
   - `Sg = Σ embed_i` over the good beans, `Sd = Σ embed_i` over the defect beans (component-wise, 1,024 values each).
   - `muG = Sg / ||Sg||`, `muD = Sd / ||Sd||` (Euclidean norm).
4. Leave-one-out margins, for every calibration bean i:
   - if i is good: `g = (Sg - e_i) / ||Sg - e_i||`, `d = muD`; if i is defect: `g = muG`, `d = (Sd - e_i) / ||Sd - e_i||`
   - `s_i = dot(e_i, d) - dot(e_i, g)`; the LOO call is `defect` if `s_i >= 0`, else `good`; `wrong_i` = LOO call != label.
5. `mHi = max( max{|s_i| : wrong_i} (0 if no wrong_i), median{|s_i|} )`. Median of an even count = mean of the two middle values (numpy convention).
6. Store (IndexedDB, one record per cooperative set-up):

```json
{
  "format": "farz-localcal-1",
  "model_sha256": "c40c10f915f7a3ab78f1aead50d3cd027113b1a53d031eea21ffec331c5bc87a",
  "embed_dim": 1024,
  "n_good": 10, "n_defect": 10,
  "mu_good":   "<base64 of Float32Array(1024)>",
  "mu_defect": "<base64 of Float32Array(1024)>",
  "m_hi": 0.6348,
  "loo_errors": 0,
  "created": "2026-10-04T02:40:00Z",
  "labelled_embeddings": "<optional: base64 Float32Array(20*1024) + labels, so the head can be rebuilt; 80 KB>"
}
```
A stored head is **invalid if `model_sha256` differs** from the loaded model file: re-calibrate.

## 3. Per-bean decision (replaces `callBean` while a local head is active)

```
if touching:                       call = "unsure"
s     = dot(embed, muD) - dot(embed, muG)        // float64
local = s >= 0 ? "defect" : "good"
base  = probs[0] >= 0.5 ? "good" : "defect"      // the base model's own binary call (T already in the graph)
if local == base:                  call = local          // "agree"
else if |s| >= mHi:                call = local          // "high local margin"
else:                              call = "unsure"
// for display only: if call == "defect", possible type = argmax(probs[1..5]) (types are NOT reliable; never show as fact)
```
The photo-level rules are **unchanged**: `decide()` in `app/src/lib/rules.ts` (nothing answered / > 60% of answered defective / > 15% unsure -> "not sure"; else the band of the point estimate, clean < 5% <= some <= 20% < many). Only the defect total matters for the band. If `rules.ts` keeps the 5-class `CLASSES`, pass a defect as any defect class name (the Python study passed `"dark"`). `other_defect` from the 6-class model must count as a defect.

Without a local head the app uses the clean model's own two thresholds (`models_v2/farz_beans_v2_clean_labels.json`: good if P(good) >= 0.50, defect if 1 - P(good) >= 0.91).

## 4. What was measured (loja_yolo held out from the base model; 5 calibration draws per K; test = beans of different photos)

| K | calibration photos per draw | balanced coverage, mean (min) | balanced answered accuracy, mean (min) | trays with a band / wrong / dangerous (of 2,500) | CBD AAA called "many" (5 draws, gates ignored) |
|---|---|---|---|---|---|
| 20 | 4, 3, 2, 4, 6 | 78.3% (70.8%) | 96.3% (95.5%) | 131 / 17 / 0 | 0, 0, 0, 0, 0 |
| 50 | 11, 12, 8, 12, 9 | 80.8% (78.8%) | 96.6% (95.9%) | 133 / 8 / 0 | 0, 0, 0, 0, 0 |
| 100 | 26, 19, 18, 22, 19 | 77.2% (72.9%) | 96.5% (95.4%) | 93 / 20 / 0 | 0, 0, 0, 0, 0 |
| base model alone (no head) | – | 65.7% | 97.4% | 0 / 0 / 0 | – |

Trays: 50 beans each from the test photos, 167 at 0%, 167 at 12%, 166 at 30% defects. All 17 wrong bands at K = 20 were one band off (12 "some" trays called "many", 5 "clean" trays called "some").

## 5. Scope and safety (read this before shipping)

- **Validated on one source only.** The GO bar was met on loja_yolo, where the base model held out from loja_yolo was already accurate (97.4% balanced answered accuracy at 65.7% coverage, which meets the accuracy bar on its own). The head raises coverage by about 13 points and keeps accuracy at about 96%.
- **The same procedure failed on the secondary sources** (`localcal.md`, results):
  - lojano (base held out from lojano): 84.9% answered accuracy at K = 20, and **11 clean trays called "many"** in 2,500 (10 of them from one draw); with lojano heads, **2 top-grade CBD photos (one AA, one A) were called "many"** (no AAA). The logistic-regression alternative, which the spec does not use, called 2 CBD AAA photos "many": one with a lojano head and one with a Mindforge head.
  - Mindforge: 78.7–80.3% accuracy; **3 clean trays called "many"** (1 at K = 20, 2 at K = 50).
  - USK: 63–67% accuracy; calibration does not rescue it.
- **Most trays still end as "not sure".** Even on loja_yolo, the head answers about 78% of beans, so the 15% unsure rule still fires on most trays: 131 of 2,500 trays got a band at K = 20.
- **Measured on crops, not on new photos.** The test trays are drawn from the bean finder's crops of real photos. Nothing was re-photographed, and nothing ran on a phone.
- **The shipped model saw loja_yolo.** The study's base model never did. Any in-app demo on loja_yolo photos with the shipped model is in-sample, so it looks better than a real new farm would.

Recommendation: ship local calibration only as a clearly labelled **pilot / demo mode**, off by default, with a mandatory acceptance check on labelled photos that were not used for calibration before it is used on a farmer's coffee.

Exploratory only (post-hoc, NOT part of the pre-registered rule, NOT validated): a guard that refuses a head whose LOO error rate exceeds 10% would have kept 13 of 15 loja_yolo draws and rejected every Mindforge and USK draw. It would still have kept one lojano draw that called 1 clean tray "many", so it is not sufficient on its own.

## 6. Demo set (`models_v2/localcal_demo/`, CC BY 4.0, credits in `CREDITS.md`)

- `calibration/`: the 20 labelled crops (10 good, 10 defect) of loja_yolo draw 1 at K = 20: photos IMG_4758, IMG_5488 and IMG_5749.
- `test/`: 3 loja_yolo test photos, not used in that draw's calibration: IMG_4804, IMG_4807, IMG_4809. They are the original JPEGs, unchanged.
- `manifest.json`: labels and the dataset's per-bean truth for each test photo. It also holds the head computed with the shipped fp16 model (m_hi 0.6348, 0 LOO errors) and what the Python pipeline gives.
- **The app will not accept these photos as they are.** Each test photo has 10–11 beans, so the app's gate answers "use about 100 beans" (count < 20). This is true of every loja_yolo photo: 1,311 crops over 315 photos. A demo needs a demo-only bypass of the count gate, or a composite tray. That is the app owner's call; nothing in `app/` was changed.
- Gates ignored, the Python pipeline (shipped model, in-sample) gives:

  | Photo | Truth (good / defect) | Result |
  |---|---|---|
  | IMG_4804 | 5 / 5 | **many** |
  | IMG_4807 | 2 / 8 | not sure (implausible) |
  | IMG_4809 | 2 / 8 | not sure (implausible) |

## 7. The spec runs as written in the app's runtime (measured 02:42 BST)
`node src/beans_v2/localcal_spec_check.mjs` implements §2 in plain JS. It runs onnxruntime-web 1.30.0 (wasm, 1 thread, Node 26, the app's own package, read only) on the 20 demo calibration crops. Against the Python values in `manifest.json`:
- m_hi: 0.634825 (JS) vs 0.634810 (Python), an absolute difference of 1.5e-5;
- LOO errors: 0 in both;
- cosine of mu_good, JS vs Python: 0.9999997; mu_defect: 0.9999991.

Not run: the app's TypeScript bean finder on the demo photos, and any phone.
