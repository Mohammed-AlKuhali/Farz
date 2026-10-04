# NOTICE: licences and attribution

Farz is a hackathon prototype (Hack-Nation 2026, World Bank challenge). Different parts of this repository
carry different licences. In short: **code = MIT; the trained bean classifier = CC BY-NC-SA 4.0, a
non-commercial, ShareAlike research prototype; images derived from J4ckDev = CC BY-NC-SA 4.0; the CBD and
loja_yolo images = CC BY 4.0.** Licences are recorded as each source states them; this is not legal advice.

## What is licensed how

| Part | Files | Licence |
|---|---|---|
| Source code | `app/src`, `app/scripts`, `app/tests/*.ts`, `app/e2e`, `src/`, configs | MIT, see `LICENSE` (owner: Mohammed) |
| App icons | `app/public/icons/*` (drawn as SVG for Farz, rasterised by `app/scripts/make-icons.mjs`) | MIT, like the code |
| **Trained bean classifier** (the only model file in this repository) | `app/public/models/farz_beans.onnx` (3,082,642 bytes, sha256 `c40c10f915f7a3ab78f1aead50d3cd027113b1a53d031eea21ffec331c5bc87a`) + `farz_beans_labels.json` | **CC BY-NC-SA 4.0**, see `MODEL_LICENSE.md` |
| Synthetic demo trays | `app/public/demo/tray_synthetic_*.jpg` | CC BY-NC-SA 4.0 (made from J4ckDev bean crops) |
| Synthetic "touching" trays | `app/tests/touching/*.jpg` | CC BY-NC-SA 4.0 (made from J4ckDev bean crops) |
| Synthetic dark-lot test trays | `app/tests/darklot/*.jpg` (`tray_synthetic_00.jpg` with the beans recoloured) | CC BY-NC-SA 4.0 (made from J4ckDev bean crops) |
| J4ckDev test fixtures | `tests/fixtures/crops/*`, `tests/fixtures/photos/j4ck_*.png`, `app/tests/fixtures/classifier_v2/crops/*`, `app/tests/fixtures/classifier_v2/photos/j4ck_*.png` | CC BY-NC-SA 4.0 (J4ckDev crops / downscaled photos) |
| CBD images | `app/public/demo/cbd_*.jpg` (unmodified copies), `tests/fixtures/photos/cbd_*.png` and `app/tests/fixtures/classifier_v2/photos/cbd_*.png` (downscaled), the bean crops in `reports_v2/robust_unusual_examples.png` | CC BY 4.0 (CBD Coffee Bean Dataset) |
| Calibration demo | `app/public/calib-demo/calibration/*.png` (crops: cut out, white-balanced, resized), `app/public/calib-demo/test/*.jpg` (photos from the dataset, resized to 2400 × 1600 px); credit in `app/public/calib-demo/CREDITS.md` | CC BY 4.0 (loja_yolo) |
| App screenshots | `app/screens/*.png` | CC BY-NC-SA 4.0. They show the synthetic J4ckDev trays (CC BY-NC-SA 4.0), the CBD photo `cbd_aaa_2.jpg` (CC BY 4.0) and loja_yolo calibration crops and photos (CC BY 4.0); credit those datasets as below |
| ONNX Runtime Web | `app/public/ort/*` + npm `onnxruntime-web` | MIT, Microsoft Corporation, see `third_party/onnxruntime-web.LICENSE.txt` |
| Fonts (bundled at build time from npm) | `@fontsource/noto-naskh-arabic`, `@fontsource/reem-kufi` | SIL Open Font License 1.1, see `third_party/fonts/` |
| Voice clips | `app/public/audio/*.mp3` + `pack.json` | see "Voice clips" below |
| Measured reports | `reports/`, `reports_v2/`, `app/reports/` | our own measurements (MIT, like the code). Numbers and charts only; the one exception is the CBD crops noted above. Results on the evaluation-only datasets below are numbers, not images |

## Training data of the shipped model (only datasets that state a licence)

The classifier was trained on 92,690 bean crops (sampler log `reports_v2/clean_eval.json` → `final_weights_train_info`) from these datasets and nothing else:

| Dataset | Crops used in training | Licence (as stated by the source) | Redistributed here? | Source |
|---|---|---|---|---|
| afiyah_bijikopi (single beans; same uploader and phone as afiyah_deteksi) | 1,170 | **CC0 1.0** | No | https://www.kaggle.com/datasets/afiyahmusrah/bijikopideteksi |
| afiyah_deteksi (single beans; same uploader and phone as afiyah_bijikopi) | 985 | **CC0 1.0** | No | https://www.kaggle.com/datasets/afiyahmusrah/deteksi-biji-kopi |
| J4ckDev Green Coffee Beans | 582 | **CC BY-NC-SA 4.0** | Only derived crops, composites and downscaled photos (listed above) | https://github.com/J4ckDev/GreenCoffeeBeansDataset |
| loja_yolo: "Deteccion de defectos del grano" (Loja, Ecuador; Roboflow export mirrored on Kaggle) | 1,311 | **CC BY 4.0 (README.dataset.txt in the download; the Kaggle page says Apache 2.0)** | Calibration demo crops and resized photos in `app/public/calib-demo/` (listed above) | https://www.kaggle.com/datasets/cristianyagg/defectos-grano-de-cafe-de-la-provincia-de-loja |
| lojano (Loja, Ecuador; good vs defective trays) | 1,182 | **CC BY-NC 4.0** | No | https://www.kaggle.com/datasets/patopucho/lojano-arabica-coffee |
| samruddh_grading: Coffee Bean Grading Dataset (Coorg, India; grades A and D used) | 1,337 | **MIT (dataset card)** | No | https://huggingface.co/datasets/SamruddhK/coffee-bean-grading-dataset |
| vicanadya16: coffee defect, 16 classes (tight crops, no sound beans) | 86,123 | **CC0 1.0** | No | https://www.kaggle.com/datasets/vicanadya/coffee-defect-16-classes |

The two afiyah sets come from one uploader, one phone model and one backdrop; they are always held out together
in our tests. CC0 sets from anonymous uploaders: provenance not verified. Exported only after
`scripts/publish_clean.sh` checked that the shipped file's sha256 matches `reports_v2/clean_model.json` and that
both training records (`clean_model.json → model.training_sources`, `clean_eval.json → final_weights_train_info`)
list only these licence-stated datasets.

**Pretrained starting point:** the network was initialised from torchvision's ImageNet-pretrained
MobileNetV3-Small weights (`IMAGENET1K_V1`). **Licence not verified** for those weights (the torchvision code is
BSD-3-Clause; that is the code's licence, not the weights').

## Evaluation-only datasets (no licence stated)

| Dataset | How Farz uses it | Licence | Source |
|---|---|---|---|
| mfu17: coffee green bean with 17 defects (Thailand) | **used for evaluation only, not redistributed, not in the shipped weights** | none stated (all rights reserved by default) | https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original |
| usk_coffee: Kaggle mirror of USK-Coffee (Indonesia) | **used for evaluation only, not redistributed, not in the shipped weights** | none stated (all rights reserved by default) | https://www.kaggle.com/datasets/mfaisalriftiarrasyid/duardata |
| daffa_defect: Coffee Beans Defect Classification | **used for evaluation only, not redistributed, not in the shipped weights** | none stated (all rights reserved by default) | https://github.com/daffakurnia11/Coffee-Beans-Defect-Classification |
| mindforge_doubleside: Green Coffee Bean Double Side | **used for evaluation only, not redistributed, not in the shipped weights** | none stated (all rights reserved by default) | https://github.com/Mindforge-inc/GreenCoffeeBeanDoubleSide |
| notplying_defects: Dataset Coffee Bean Defects | **used for evaluation only, not redistributed, not in the shipped weights** | none stated (all rights reserved by default) | https://github.com/Notplying/DatasetCoffeeBeanDefects |

No image, crop or file from these five datasets is in this repository. Our reports contain only numbers measured
on them (accuracy, coverage, tray simulations). USK-Coffee's authors distribute it through a form at
https://coffee.comvislab-usk.org/ ; we used a third-party Kaggle mirror.

## Other datasets

| Dataset | How Farz uses it | Licence | Redistributed here? | Source |
|---|---|---|---|---|
| **CBD Coffee Bean Dataset** (Mendeley 52877z55vr) | Bean-counting test (no AI), photo-rule safety test, out-of-distribution check; never used for training or for fitting thresholds | **CC BY 4.0** | 2 demo photos, 1 downscaled test photo (stored in two test folders), bean crops in one chart (listed above) | https://data.mendeley.com/datasets/52877z55vr/1 |
| BRACOL (Mendeley yy2k5y8mxg) | Leaf-disease pipeline smoke test only (`reports/metrics_mnv3s_smoke.json`); not in the app | CC BY 4.0 | No | https://data.mendeley.com/datasets/yy2k5y8mxg |
| JMuBEN (Mendeley t2r6rszp5c) | Leaf smoke test only; not in the app | CC BY 4.0 | No | https://data.mendeley.com/datasets/t2r6rszp5c |
| Uganda coffee leaf (Mendeley k36wnd6knb) | Leaf smoke test only (held-out country); not in the app | CC BY 4.0 | No | https://data.mendeley.com/datasets/k36wnd6knb |

Downloaded during research but not used for any shipped file or reported result, and not redistributed:
JMuBEN2, RoCoLe, DECAFIA / CoffeeLeaf-CO (see `docs/DATA_CARD.md`). Other bean datasets that were checked and
not used are listed in `docs/DATA_SOURCES_V2.md` §6.

**Attribution** for CC BY / CC BY-NC / CC BY-NC-SA material: dataset name, authors as listed on the source page
above, licence as listed, and the changes made ("cropped per bean, white-balanced to a grey sheet, resized,
composited onto a grey background" for derived images). Credits are also shown inside the app
(`app/public/demo/manifest.json`, `app/public/calib-demo/manifest.json`).

## Voice clips

The 15 clips in `app/public/audio/` are recordings made for Farz by the project owner; `app/public/audio/pack.json`
describes the voice (the app's About screen shows the same text). They are **not** covered by the MIT code licence
in `LICENSE`; ask the project owner before reusing them outside this app. No synthetic or text-to-speech voice is
included: `scripts/publish_clean.sh` checked every clip against the text-to-speech placeholders used during
development (byte copies and same-waveform copies) before export.

## Runtime npm dependencies (licence read from each installed package)

| Package | Version | Licence |
|---|---|---|
| @fontsource/noto-naskh-arabic | 5.3.0 | OFL-1.1 |
| @fontsource/reem-kufi | 5.3.0 | OFL-1.1 |
| onnxruntime-web | 1.30.0 | MIT |
| react | 19.3.0 | MIT |
| react-dom | 19.3.0 | MIT |
| workbox-window | 7.4.1 | MIT (a dev dependency, bundled into the built app by vite-plugin-pwa) |

Dev dependencies are listed in `app/package.json` / `app/package-lock.json` with their own licences.

## Not published

Some docs cite files that are deliberately **not** in this repository: `docs/sources/*` (the challenge brief,
the event deck and terms, the full UNDP report text, research notes), internal planning and production notes,
internal review, regression and audit reports (for example `reports/REGRESSION_*.md`),
`logs/`, `data/` (third-party datasets: download them from the links above), `models_v2/` and the fold weights
(`*.pt`), and the superseded v1 model file `models/farz_beans_fp32.onnx` (J4ckDev only; its measured results are
in `reports/`). Quotes from those sources in the docs are short and attributed; the UNDP report is public at the
link given in `README.md`.
