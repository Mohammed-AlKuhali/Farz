# Model licence

**The trained bean classifier is licensed CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/).
It is a **non-commercial, ShareAlike research prototype**.

Files: `app/public/models/farz_beans.onnx` (3,082,642 bytes, sha256 `c40c10f915f7a3ab78f1aead50d3cd027113b1a53d031eea21ffec331c5bc87a`) and
`app/public/models/farz_beans_labels.json`. The same licence covers every image in this repository derived from
the J4ckDev dataset (see `NOTICE.md`).

## What it was trained on

Only datasets that state a licence (92,690 bean crops (sampler log `reports_v2/clean_eval.json` → `final_weights_train_info`)):

| Dataset | Licence (as stated by the source) | Source |
|---|---|---|
| afiyah_bijikopi (single beans; same uploader and phone as afiyah_deteksi) | CC0 1.0 | https://www.kaggle.com/datasets/afiyahmusrah/bijikopideteksi |
| afiyah_deteksi (single beans; same uploader and phone as afiyah_bijikopi) | CC0 1.0 | https://www.kaggle.com/datasets/afiyahmusrah/deteksi-biji-kopi |
| J4ckDev Green Coffee Beans | CC BY-NC-SA 4.0 | https://github.com/J4ckDev/GreenCoffeeBeansDataset |
| loja_yolo: "Deteccion de defectos del grano" (Loja, Ecuador; Roboflow export mirrored on Kaggle) | CC BY 4.0 (README.dataset.txt in the download; the Kaggle page says Apache 2.0) | https://www.kaggle.com/datasets/cristianyagg/defectos-grano-de-cafe-de-la-provincia-de-loja |
| lojano (Loja, Ecuador; good vs defective trays) | CC BY-NC 4.0 | https://www.kaggle.com/datasets/patopucho/lojano-arabica-coffee |
| samruddh_grading: Coffee Bean Grading Dataset (Coorg, India; grades A and D used) | MIT (dataset card) | https://huggingface.co/datasets/SamruddhK/coffee-bean-grading-dataset |
| vicanadya16: coffee defect, 16 classes (tight crops, no sound beans) | CC0 1.0 | https://www.kaggle.com/datasets/vicanadya/coffee-defect-16-classes |

Starting point: torchvision's ImageNet-pretrained MobileNetV3-Small weights (`IMAGENET1K_V1`), **licence not
verified**.

Five datasets that state **no licence** (mfu17, the usk_coffee Kaggle mirror, daffa_defect, mindforge_doubleside,
notplying_defects) were **used for evaluation only**. They are not in these weights and are not redistributed.

## Why CC BY-NC-SA 4.0

- **ShareAlike**: J4ckDev is CC BY-NC-SA 4.0. A model trained on it is treated as an adapted work, so the weights are released under the same licence.
- **NonCommercial**: J4ckDev (CC BY-NC-SA 4.0) and lojano (CC BY-NC 4.0) forbid commercial use.
- The other sources (afiyah_bijikopi: CC0 1.0, afiyah_deteksi: CC0 1.0, loja_yolo: CC BY 4.0, samruddh_grading: MIT, vicanadya16: CC0 1.0) allow this. CC BY 4.0 and MIT require attribution, which is given here and in `NOTICE.md`.

Terms, in short:
- **Attribution**: credit this project and the datasets listed above.
- **NonCommercial**: no commercial use. Farz is a **prototype only**.
- **ShareAlike**: adaptations must use the same licence.

The MIT licence in `LICENSE` covers the source code only, not these weights or the dataset-derived images. A
commercial or field deployment needs a model retrained on data licensed for that use (for example, labelled bean
photos collected with a cooperative, with consent). This file records licences as the sources state them; it is
not legal advice.
