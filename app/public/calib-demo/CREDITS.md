# Credits

Coffee bean photos and labels: "Deteccion de defectos del grano" (v14), Roboflow Universe project tesis-kmw54, https://universe.roboflow.com/tesis-kmw54/deteccion-de-defectos-del-grano , mirrored on Kaggle as cristianyagg/defectos-grano-de-cafe-de-la-provincia-de-loja (Loja province, Ecuador). Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), as stated in the dataset's README.dataset.txt. 

Files in `calibration/` are 128x128 crops cut from the dataset's photos by Farz's bean finder (changed: cropped, white-balanced to a grey sheet, resized). Files in `test/` are the dataset's own photos, changed: resized from 6000x4000 to 2400x1600 pixels (re-encoded as JPEG). Labels come from the dataset's YOLO polygons (mapped to good / defect by Farz). In the app these dataset labels stand in for a cooperative grader; no grader labelled these beans.
