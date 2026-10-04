# Farz v2: green coffee bean defect datasets (data hunter report)

*Data hunter, Sat 3 Oct 2026, ~22:40 to ~23:50 BST (machine clock); crops rebuilt from scratch 23:38-23:48. Owner paths: `data/raw2/`, `data/crops_v2/`, `src/beans_v2/data_*.py`, this file. Nothing in `app/`, `models/`, `reports/` or `src/beans/` was touched; `src/beans/beanfinder.py` is imported read-only.*

**Why this exists.** v1 learned from J4ckDev only: 11 photos, one photo per class. Leave-one-photo-out accuracy was 27.4%, below the 38.2% scored by always guessing the commonest type (148 of 387 beans are "dark"; `docs/EVIDENCE.md` M7, M19), and on CBD it called 54% of beans broken. A classifier cannot learn "defect" rather than "which photo" from one photo per class. This report lists every labelled green-bean defect dataset found that downloads without an account. It records each licence as the source states it, downloads the usable ones, removes duplicates, and cuts every bean into the same 128 px crop the app uses.

**Rules followed.** No account was created, no form submitted, nothing pushed. Kaggle data came through the public API (`/api/v1/datasets/download/{owner}/{slug}`; search, metadata and file listing also worked without login). Hugging Face and GitHub data came from public `resolve`, `raw` and `codeload` URLs. Each licence string is copied from the dataset page, the API metadata, or a licence/README file in the download, and the location is named. Every count below is measured, and the script that measured it is named. Claims marked "stated by source" are the uploader's words and were not checked.

## 1. Headline

**117,561 kept crops from 11 sources** (v1 had 582 crops from 1 source). Every count is a kept crop after exact and near-duplicate removal. Measured by `src/beans_v2/data_build_crops.py --merge` and printed by `src/beans_v2/data_report.py`. "Distinct photos/groups" = distinct values of the `group` column (photo for trays, bean for Mindforge's two faces, file for single-bean sets). **Split on `group`, never on rows.**

| Source | good | dark | insect | broken | unhulled | other_defect | Total kept | Distinct photos/groups | Source images in | Crops built | Dropped as duplicates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mfu17 | 0 | 303 | 111 | 250 | 161 | 137 | 962 | 962 | 979 | 979 | 17 |
| loja_yolo | 146 | 472 | 135 | 367 | 191 | 0 | 1,311 | 315 | 419 | 1,325 | 14 |
| lojano | 598 | 0 | 0 | 0 | 0 | 584 | 1,182 | 28 | 36 | 1,182 | 0 |
| usk_coffee | 5,970 | 0 | 0 | 0 | 0 | 1,999 | 7,969 | 7,969 | 8,000 | 8,000 | 31 |
| samruddh_grading | 467 | 0 | 0 | 0 | 0 | 870 | 1,337 | 1,337 | 1,375 | 1,370 | 33 |
| afiyah_deteksi | 658 | 0 | 0 | 0 | 0 | 327 | 985 | 985 | 986 | 986 | 1 |
| afiyah_bijikopi | 579 | 0 | 0 | 0 | 0 | 591 | 1,170 | 1,170 | 1,200 | 1,200 | 30 |
| daffa_defect | 500 | 0 | 0 | 0 | 0 | 500 | 1,000 | 1,000 | 1,000 | 1,000 | 0 |
| mindforge_doubleside | 1,447 | 1,026 | 1,039 | 2,357 | 0 | 1,503 | 7,372 | 5,066 | 11,412 | 7,373 | 1 |
| notplying_defects | 0 | 1,363 | 1,362 | 5,142 | 257 | 26 | 8,150 | 204 | 204 | 8,152 | 2 |
| vicanadya16 | 0 | 22,316 | 16,138 | 24,749 | 20,014 | 2,906 | 86,123 | 924 | 86,129 | 86,129 | 6 |
| **Total** | **10,365** | **25,480** | **18,785** | **32,865** | **20,623** | **9,443** | **117,561** | | | | |

| Subset (kept crops) | good | dark | insect | broken | unhulled | other_defect | Total | Groups |
|---|---|---|---|---|---|---|---|---|
| All 11 sources | 10,365 | 25,480 | 18,785 | 32,865 | 20,623 | 9,443 | 117,561 | 19,960 |
| Without vicanadya16 (86k low-res crops, no good class) | 10,365 | 3,164 | 2,647 | 8,116 | 609 | 6,537 | 31,438 | 19,036 |
| Only sources that state a licence (CC0, CC BY, CC BY-NC, MIT: vicanadya16, afiyah x2, loja_yolo, lojano, samruddh) | 2,448 | 22,788 | 16,273 | 25,116 | 20,205 | 5,278 | 92,108 | 4,759 |
| Same, excluding non-commercial lojano (CC0, CC BY, MIT) | 1,850 | 22,788 | 16,273 | 25,116 | 20,205 | 4,694 | 90,926 | 4,731 |

Sources per class: good 8; dark 5; insect 5; broken 5; unhulled 4 (loja_yolo, mfu17, notplying, vicanadya16); other_defect 10.

**Biggest risk for the trainer: the same confound that sank v1, now at source level.** The typed defect classes (dark, insect, broken, unhulled) come mostly from vicanadya16 and notplying_defects, and **neither has a single good bean**. Good beans come mostly from USK and other plain single-bean sets that have **no typed defects**. A model can score well by learning "crowded low-res crop = defect, clean single bean = good". Only two sources have good beans and typed defects in the same photo style: loja_yolo (146 good) and mindforge_doubleside (1,447 good). Six more pair good with untyped defects in one style: lojano, afiyah x2, daffa, samruddh and USK. Recommended: (1) cap each source per epoch, (2) report leave-one-source-out results, (3) check good-vs-defect inside loja_yolo and Mindforge on their own before trusting any pooled number.

## 2. Datasets used (downloaded, cropped, in `data/crops_v2/`)

| # | Name in `data/raw2/` | URL | Licence (verbatim, where found) | Download size (bytes, `stat`) | Classes (count as downloaded) | Single bean or tray | Background | Camera (EXIF on a 60-image sample, or stated) | Country | Green? |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `mfu17` | https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original | Kaggle licence field **"Unknown"** (metadata API: `unknown`). The page asks users to cite https://doi.org/10.1016/j.atech.2024.100680 | 77,149,430 zip | 17 SCA-style defects, 979 images: Broken 62, Cut 66, Dry Cherry 54, Fade 35, Floater 48, Full Black 41, Full Sour 75, Fungus Damage 75, Husk 53, Immature 78, Parchment 54, Partial Black 65, Partial Sour 50, Severe Insect 57, Shell 57, Slight Insect 55, Withered 54. **No "good" class** | Single bean, 500x500 | Light grey/white | Canon EOS M50 (14/60 have EXIF) | Thailand (page: Mae Fah Luang University; paper title "...defect classification in Thai Arabica green coffee beans", Smart Agric. Technol. 9 (2024) 100680; article licence not checked, page returned 403) | Yes, Arabica green |
| 2 | `usk_coffee` | https://www.kaggle.com/datasets/mfaisalriftiarrasyid/duardata (mirror of USK-Coffee, authors' page https://coffee.comvislab-usk.org/) | Mirror: **"Unknown"**. The authors' page states no licence. Its official download sits behind a Google Form, which we did not use | 66,187,699 zip | premium 2,000; longberry 2,000; peaberry 2,000; defect 2,000 (train/val/test folders) | Single bean, 256x256 | White | "digital camera" (authors' page); no EXIF | Indonesia, Banda Aceh (KNT Coffee), stated by the authors' page | Yes, Arabica green |
| 3 | `vicanadya16` | https://www.kaggle.com/datasets/vicanadya/coffee-defect-16-classes | Kaggle **"CC0: Public Domain"** (metadata API: `CC0-1.0`). No description, no provenance | 157,006,914 zip | 16 SCA defect classes, 86,129 pre-cut crops from 924 source photos (file names `DSCFnnnn_JPG_k`). **No "good" class** | Tight crops (long side p10 35 px, median 47 px, p90 92 px; 3,000-crop sample) cut from crowded trays; neighbouring beans visible | Other beans | None in crops; `DSCF` names follow Fujifilm's convention (not verified) | Unknown | Yes, green |
| 4 | `loja_yolo` | https://www.kaggle.com/datasets/cristianyagg/defectos-grano-de-cafe-de-la-provincia-de-loja (a Roboflow export of https://universe.roboflow.com/tesis-kmw54/deteccion-de-defectos-del-grano v14) | Kaggle **"Apache 2.0"**; `README.dataset.txt` in the download: **"License: CC BY 4.0"**. Treated as CC BY 4.0 | 186,674,815 zip | 9 classes, 3,050 YOLO polygons on 419 labelled photos (72 more photos have no labels): cortado 450, negro_parcial 373, cereza_seca 369, por_hongo 341, agrio_parcial 310, broca_leve_severa 306, grano_negro 303, concha 302, normal 296 | Tray, mean 7.3 labelled beans per labelled photo (3,050/419), polygon per bean | Light grey paper | No EXIF; 6000x4000 | Ecuador, Loja province (title) | Yes |
| 5 | `lojano` | https://www.kaggle.com/datasets/patopucho/lojano-arabica-coffee | Kaggle **"Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"** | 717,955,744 zip | good vs defective: 10+10 raw tray photos (50 beans each) and 8+8 "generalisation" photos; also 500+500 and 200+200 pre-cut 256 px crops (not used, because they are cut from the same photos) | Tray, all beans in a photo share one label | White sheet (main); white board or brown wood (generalisation) | Canon EOS R50 | Ecuador, Loja, Coffea arabica (stated by source) | Yes |
| 6 | `afiyah_deteksi` | https://www.kaggle.com/datasets/afiyahmusrah/deteksi-biji-kopi | Kaggle **"CC0: Public Domain"** (`CC0-1.0`). No description | 1,351,349,522 zip | bijibagus (good) 659; bijirusak (damaged) 327 | Single bean, 3056x3056 (595) or 4096x1840 (390) | White paper | samsung Galaxy A05s | Not stated (Indonesian labels) | Yes |
| 7 | `afiyah_bijikopi` | https://www.kaggle.com/datasets/afiyahmusrah/bijikopideteksi | Kaggle **"CC0: Public Domain"** (`CC0-1.0`). No description | 6,252,517,523 zip | bijikopibagus 600; bijikopirusak 600 (Kaggle file listing) | Single bean, 6112x6112 | White paper | samsung Galaxy A05s | Not stated (Indonesian labels) | Yes |
| 8 | `samruddh_grading` | https://huggingface.co/datasets/SamruddhK/coffee-bean-grading-dataset | HF card **"license: mit"** | Subset downloaded: all 875 Grade D + 500 random Grade A (seed 0) = 1,375 files. Full set 13.4 GB (API sizes: CGA 3,857,124,898; CGB 3,246,389,315; CGC 2,340,963,046; CGD 1,993,253,048) | Grade A "Premium" 1,000; B "Good" 1,000; C "Standard" 1,002; D "Defective" 875 (card). Used: A -> good, D -> other_defect. B and C are not used: grades are not defect types | Single bean (a few Grade D images hold 2 pieces) | White backdrop, strong phone shadows | Galaxy S23, Pixel 7, Xiaomi, iPhone 15 Pro Max | India, Coorg/Kodagu, Karnataka. **The card contradicts itself on species**: "Arabica" in the abstract, "Coffea robusta" in the table | Yes ("pre-roast") |
| 9 | `daffa_defect` | https://github.com/daffakurnia11/Coffee-Beans-Defect-Classification (folder `Source/`) | **No licence**: no LICENSE file, GitHub API `license: null`, no README | 1,000 JPGs (folder listing) | Defected 500; Undefected 500 | Single bean, 3456x3456 | White | Canon EOS 60D, Xiaomi | Not stated | Yes |
| 10 | `mindforge_doubleside` | https://github.com/Mindforge-inc/GreenCoffeeBeanDoubleSide | **No licence**: no LICENSE file, GitHub API `license: null` | 217,264,827 zip | 12 classes, 11,824 images = 5,912 beans x 2 faces. Normal 2,290; Broken 1,578; Inspect(insect)-damaged 1,577; Immature 1,183; Fissure 1,146; Fungus-damaged 1,094; Shell 887; Over-dried 802; Sour 465; Black 256; Floater 128; Foreign matter 6 (faces labelled `U-1`/`D-1`; 412 faces marked `-0` are not used) | Single bean, close-up, 640x480 | Black PCB window inside a white jig (**not a plain sheet**) | No EXIF | Not stated. The README is in Korean. It matches the 12-class (11 defects + normal) dual-sided set described in Agriculture 16:1796 (2026); that link is **not verified** | Yes |
| 11 | `notplying_defects` | https://github.com/Notplying/DatasetCoffeeBeanDefects | **No licence**: no LICENSE file, GitHub API `license: null`; description "Dataset Cacat Biji Kopi" | 840,177,047 zip | 204 photos (Grade1 133, Grade2 34, Grade3 37), 8,152 YOLO boxes over the 16 SCA classes (`notes.json`): Broken/Chipped/Cut 4,219; Severe Insect 946; Full Black 902; Shell 477; Immature 448; Slight Insect 416; Hull/Husk 240; Fungus 139; Full Sour 137; Partial Sour 111; Partial Black 74; Withered 19; Dried Cherry 11; Floater 7; Parchment 6. **Only defects are boxed** | Dense pile of hundreds of beans (by eye on one photo); only defects are boxed | White paper | samsung Galaxy S24 | Not stated (Indonesian description) | Yes |

Licence warning for the trainer. Sources 1, 2, 9, 10 and 11 state no licence. Absent a licence the default is "all rights reserved". Every row of `crops_v2.csv` carries its licence string so that a model can be trained on the CC0 / CC BY / MIT subset only, if that is the call. Source 5 is CC BY-NC 4.0, like J4ckDev. Source 4 is CC BY 4.0. Sources 3, 6 and 7 are CC0. Source 8 is MIT.

## 3. Mapping to Farz v2 classes

`src/beans_v2/data_sources.py` holds the mapping. `source_class` is kept on every row, so a trainer can re-map or drop classes.

| Farz v2 | Source labels mapped here |
|---|---|
| good | USK premium, longberry, peaberry (shape classes, not defects); Loja `normal`; lojano `bueno`; afiyah `bijibagus`/`bijikopibagus`; Samruddh Grade A; daffa `Undefected`; Mindforge `Normal` |
| dark | full/partial black, full/partial sour, fungus (MFU17, vicanadya, Notplying); Loja `grano_negro`, `negro_parcial`, `agrio_parcial`, `por_hongo`; Mindforge `Black`, `Sour`, `Fungus-damaged` |
| insect | slight/severe insect (all SCA sets); Loja `broca_leve_severa`; Mindforge `Inspect-damaged` |
| broken | broken, cut, shell, immature (all SCA sets); Loja `cortado`, `concha`; Mindforge `Broken`, `Shell`, `Immature` |
| unhulled | parchment, dry cherry / cherry pod, hull/husk (SCA sets); Loja `cereza_seca` |
| other_defect | defect of unknown type: USK `defect`, lojano `defectuoso`, afiyah `bijirusak`/`bijikopirusak`, Samruddh Grade D, daffa `Defected`; plus floater, withered, fade, foreign matter (SCA sets), Mindforge `Fissure`, `Over-dried`, `Floater`, `Foreign matter` |

Choices a trainer may want to undo: (a) peaberry and longberry are mapped to good, because the SCA does not count them as defects; (b) foreign matter is not a bean, but it stays in other_defect with its `source_class`; (c) hull/husk goes to unhulled (pieces of dried pod); (d) withered and fade go to other_defect, not broken.

## 4. How crops were made (`src/beans_v2/data_build_crops.py`)

All paths end in a 128x128 RGB PNG on the 235-grey sheet (`SHEET_TARGET`) with the contract's `PAD` = 0.18. The mode is recorded per row in `crops_v2.csv` (`mode`, `flags`).

| Mode | Sources | What happens | Contract fidelity |
|---|---|---|---|
| `tray_single_class` | lojano | `beanfinder.find_beans(photo)` exactly as the app runs it; every bean takes the photo's label. Label cleaning: photos whose sheet luma is below 120 are rejected (the finder assumes beans darker than a light sheet; the 8 brown-board photos measured 82-87); blobs with area outside 0.4-2.5x the photo's median, or within 2 px of the frame, are dropped | **Exact** |
| `tray_polygons` | loja_yolo | `find_beans(photo)`; each found bean takes the label of the annotation it overlaps at box IoU >= 0.5; unmatched beans are dropped | **Exact** |
| `single` | mfu17, usk_coffee, afiyah x2, samruddh, daffa | Photos <= 700 px: pad on every side with the photo's own border colour (30% of the long side) so the bean cannot touch the frame, then `find_beans`, then keep the largest bean. Larger photos: locate the bean with OpenCV (chroma first, then flat-field difference; **locating only**), cut a window 3x the bean, pad, run `find_beans` on the original pixels. The QC rejects a window if the finder's bean is not 0.4-2x the located size, or touches the window edge | **Exact** finder crop on a cut-out of the photo |
| `precrop` | vicanadya16 | Tight crop pasted centred on a 235-grey square with `PAD`, resized. **No white balance is possible** (no sheet in view); neighbouring beans stay visible | Geometry only |
| `box` | notplying_defects | White balance from the photo's border band (as the contract does), then a square box crop padded by `PAD`, taken at original resolution; outside the photo = 235 | WB and geometry; neighbours visible (dense pile) |
| `mask_chroma` | mindforge_doubleside | The finder cannot run on a black plate, so: white balance from the bright, low-colour jig pixels; bean mask = Otsu on chroma, largest blob, holes filled; the bean is pasted onto 235 grey with `PAD`. Masks are rejected when solidity < 0.93 (plate edges stuck to the bean), extent > 0.88 (rectangular patches), or the mask touches the frame (bean cut off; on dark beans the chroma mask also grabbed brown jig patches at the frame edge). Rejection differs by class (dark Black beans have little colour and are lost most often; see the build log), so class balance shifts | Synthetic sheet (mask composite) |

## 5. Deduplication (before any split)

Rules, in this order: J4ckDev first, then `SOURCE_ORDER` in `data_build_crops.py`.
1. **Source-image pixel MD5** of the decoded, EXIF-rotated RGB array. Any image equal to a J4ckDev photo, or to a different file already kept, is dropped with all its crops.
2. **Crop pixel MD5** (128 px PNG), same rule.
3. **Near-duplicates dropped**: 32x32 grey thumbnails (mean-centred, L2-normalised); two crops from different groups with dot product >= 0.998 are treated as the same bean (re-shot or re-encoded), and the later one is dropped. vicanadya16 (86k tight crops) is not checked. How the threshold was chosen: all 30 USK pairs at >= 0.998 were inspected by eye. Every one was the same bean in the same pose (train/premium 837-891 duplicated as 947-999), and pixel MD5 missed them because they are separate shots. A further 17 pairs were sampled from other sources at >= 0.998. 14 were the same bean. 3 were the same rectangular plate patch, not a bean; Mindforge mask failures like these are now removed by the extent rule. At 0.985, sampled pairs were different look-alike beans, so that threshold is not used.
4. **Report only**: exact 64-bit dHash collisions of crops across sources, against the 582 v1 J4ckDev crops, and within a source across different files. dHash also collides on plain look-alike beans, so these counts are an upper bound on near-duplicates, not proof.

Measured result (`summary_v2.json`):

| Source | Source images equal to another file (pixel MD5) | Crops equal (pixel MD5) | Near-duplicates dropped (r >= 0.998) |
|---|---|---|---|
| mfu17 | 13 | 0 | 4 |
| loja_yolo | 0 | 0 | 14 |
| lojano | 0 | 0 | 0 |
| usk_coffee | 1 | 0 | 30 |
| samruddh_grading | 32 | 0 | 1 |
| afiyah_deteksi | 0 | 1 | 0 |
| afiyah_bijikopi | 27 | 0 | 3 |
| daffa_defect | 0 | 0 | 0 |
| mindforge_doubleside | 0 | 0 | 1 |
| notplying_defects | 0 | 1 | 1 |
| vicanadya16 | 5 | 1 | not checked |

- **Label conflicts among exact duplicates: 5.** mfu17 has 1 image in both Fade and Partial Sour (`Fade_08` = `Partial Sour_06`). vicanadya16 has 4 beans boxed twice with different labels (e.g. floater and immature, partial sour and slight insect). The first copy is kept, so its label is ambiguous; `dup_of` lists them. All 54 near-duplicate pairs carried the same label.
- **Cross-dataset exact duplicates: 0.** That covers every pair of sources and all 11 J4ckDev photos (582 J4ckDev crops hashed). None of the v2 images is a copy of J4ckDev.
- **Cross-dataset near-duplicates at r >= 0.998: 0.** Every near-duplicate dropped was inside one source.
- afiyah_deteksi and afiyah_bijikopi come from the same uploader and phone (Galaxy A05s) but share no identical image. Their image sizes differ (3056 or 4096x1840 vs 6112).
- The within-source near-duplicates matter for splitting. USK's own folders put the same bean in train/premium 837-891 and again as 947-999. loja_yolo has the same bean in different photos (4 of its 14 pairs were checked by eye). Pixel MD5 cannot catch either.
- Exact dHash collisions are counted in `summary_v2.json`, but they are dominated by plain look-alike beans (e.g. 88 buckets are shared by usk_coffee and vicanadya16, two sets with different image sizes and capture setups). They are not evidence of copying.

Per source class, kept crops:

- **mfu17**: Immature -> broken 66; Cut -> broken 65; Broken -> broken 62; Shell -> broken 57; Full Sour -> dark 75; Fungus Damange -> dark 75; Partial Black -> dark 63; Partial Sour -> dark 49; Full Black -> dark 41; Severe Insect Damange -> insect 56; Slight Insect Damage -> insect 55; Withered -> other_defect 54; Floater -> other_defect 48; Fade -> other_defect 35; Dry Cherry -> unhulled 54; Parchment -> unhulled 54; Husk -> unhulled 53
- **loja_yolo**: cortado -> broken 253; concha -> broken 114; grano_negro -> dark 135; negro_parcial -> dark 133; agrio_parcial -> dark 104; por_hongo -> dark 100; normal -> good 146; broca_leve_severa -> insect 135; cereza_seca -> unhulled 191
- **lojano**: bueno -> good 598; defectuoso -> other_defect 584
- **usk_coffee**: longberry -> good 2,000; peaberry -> good 1,999; premium -> good 1,971; defect -> other_defect 1,999
- **samruddh_grading**: CGA -> good 467; CGD -> other_defect 870
- **afiyah_deteksi**: bijibagus -> good 658; bijirusak -> other_defect 327
- **afiyah_bijikopi**: bijikopibagus -> good 579; bijikopirusak -> other_defect 591
- **daffa_defect**: Undefected -> good 500; Defected -> other_defect 500
- **mindforge_doubleside**: Broken -> broken 928; Immature -> broken 850; Shell -> broken 579; Fungus-damaged -> dark 634; Sour -> dark 305; Black -> dark 87; Normal -> good 1,447; Inspect-damaged -> insect 1,039; Over-dried -> other_defect 724; Fissure -> other_defect 664; Floater -> other_defect 112; Foreign matter -> other_defect 3
- **notplying_defects**: Broken/Chipped/Cut -> broken 4,217; Shell -> broken 477; Immature/Unripe -> broken 448; Full Black -> dark 902; Fungus Damaged -> dark 139; Full Sour -> dark 137; Partial Sour -> dark 111; Partial Black -> dark 74; Severe Insect Damage -> insect 946; Slight Insect Damage -> insect 416; Withered -> other_defect 19; Floater -> other_defect 7; Hull/Husk -> unhulled 240; Dried Cherry/Pod -> unhulled 11; Parchment/Pergamino -> unhulled 6
- **vicanadya16**: broken -> broken 10,960; immature -> broken 7,850; shell -> broken 5,939; full black -> dark 8,589; partial sour -> dark 6,283; full sour -> dark 3,585; partial black -> dark 2,792; fungus -> dark 1,067; slight insect -> insect 14,237; severe insect -> insect 1,901; withered -> other_defect 1,497; floater -> other_defect 1,190; foreign material -> other_defect 219; parchment -> unhulled 17,368; hull -> unhulled 2,364; cherry pod -> unhulled 282

## 6. Candidates checked and not used

| Candidate | Why not used |
|---|---|
| Roboflow Universe sets (e.g. nur-muhammad-himawan/arabica-green-coffee-bean-defect; robusta-coffee-bean-defects) | Export requires a Roboflow account (rule: no accounts). Loja is used through its Kaggle mirror |
| USK-Coffee official site (coffee.comvislab-usk.org) | Download behind a Google Form (no form submissions); the Kaggle mirror was used instead |
| Kaggle mfaisalriftiarrasyid/usk-coffee-rmbg, berglingmurphy/coffeebean-usk, HF duvdar/usk-coffee, Kaggle ardiyanto24/coffee-bean-classification-dataset | Further USK-Coffee copies (same 4 classes; same 8,000-file layout or a subset). Not downloaded |
| Kaggle bangsul12/data-fiks | Its page says it is built from the MFU17 images plus a Roboflow "good" set (SAM polygons). Derivative, licence "unknown" |
| Kaggle harishs34/coffee-bean-grading-dataset (11.4 GB, "unknown") | Same `CGA/CGA1.jpg` naming and the same Galaxy S23 EXIF as SamruddhK, so probably a mirror (not verified by hash). The MIT-licensed HF original was used |
| Kaggle zaafirrahman/arabica-beans (2.5 GB, "unknown") | Lot grades mutu1-5 per tray photo, no per-bean labels. Possible later use as an out-of-distribution lot test |
| Kaggle danielmejiar/green-coffee-beans (4.4 GB, "unknown") | 600 `IMG_*.jpg` with no labels |
| Kaggle vinsentoothpaste/coffee-beans-defect-detection (4.3 GB, MIT) | Mixed Label Studio/Roboflow exports with no class-name file |
| Kaggle ilhamau/indonesian-coffee-beans; gpiosenka and sot2542 coffee-bean sets; GitHub dont-text-me/RoastedCoffeeDefectDataset | Roasted |
| Kaggle abe16s/ethiopian-coffee-beans; adityaahfazh (arabica/liberika/robusta); Zenodo 21441551 (14 varieties + adulteration); Zenodo 11061522 (Liberica varieties); Zenodo 18917349 (BaLiCoM, no files) | Variety or origin labels, not defects |
| Zenodo 14271151; Kaggle andressac, cienciacafeto, harisyunanda | Coffee cherries, not green beans |
| HF rasyidf/coffee-beans and SriPrasanna/coffee-beans | Labels "0"-"3" with no stated meaning |
| HF everycoffee/autotrain-data-coffee-beans | No licence; Roboflow x3-augmented 224 px webcam frames; origin unknown. Small; skipped for time |
| HF sayidj/Asalan-Gayo-Coffee; GitHub CoffeeDataSet/CoffeeDataSet-Coffee-beans-data-set- | No labels |
| GitHub lonlonago/coffee-bean-defect-classification-dataset-17-types-of-defects | "Sponsor to obtain the full data files" (paid) |
| GitHub AlexHuc, Mary0612 | Code only; they use the MFU17 Kaggle set |
| Yunnan YOLOv10 paper (PMC13276431) | "Data will be made available on request." |
| CBD (Mendeley 52877z55vr) | Already in v1; size grades, no per-bean defect labels |

Not searched exhaustively: Papers-with-Code was not searched separately; a web search naming figshare returned no figshare-hosted green-bean defect set; a Mendeley web search returned only CBD and the leaf sets already in v1. No Brazilian, Colombian, Ethiopian or Vietnamese green-bean defect image set was found that downloads without an account. The Ethiopian GitHub repo (samuelbirhanuabebe251) holds a model and 3 photos, not a dataset.

## 7. What this data still does not cover (for the model card)

- **Still no Yemeni beans.** Countries stated by a source: Thailand (mfu17), Indonesia (USK), Ecuador (loja_yolo, lojano), India (samruddh). No country is stated for afiyah x2 or Notplying (Indonesian-language labels), daffa, Mindforge (Korean README) or vicanadya. Yemen's natural-process beans are not represented.
- **"good" comes from 8 sources, but 3 of them state no licence** (daffa and Mindforge have none; USK's mirror says "Unknown"). Those three hold 7,917 of the 10,365 good crops.
- **Insect, broken and dark now have thousands of examples from 5 sources each, and unhulled from 4.** That makes a leave-one-SOURCE-out test possible, which is the honest replacement for v1's leave-one-photo-out.
- **Resolution mismatch** (median bean long side before the 128 px resize, from bbox in `crops_v2.csv` and v1 `crops.csv`). v1 J4ckDev: 64 px in the 1200-px work image. lojano: about 34 px and loja_yolo about 106 px at work scale (170 and 530 px in the 6000-px originals). Single-bean sources are far larger: USK 157, mfu17 220, Mindforge 268, samruddh 388, afiyah 406 and 791, daffa 704 px. Box crops: notplying 78 px, vicanadya16 47 px. Keep v1's resolution-degrade augmentation, or downscale single-bean crops to the app's scale.
- **Context mismatch.** vicanadya and Notplying crops show neighbouring beans, and Mindforge crops are mask composites. All three should be tested as separate domains, not just pooled.
- **Not in J4ckDev's style.** Several sets use a phone with harsh shadows (Samruddh) or a dark plate (Mindforge). That variety is the point, but per-source accuracy must be reported.

## 8. Observations for the bean-finder owner (measured on these data; `beanfinder.py` was not changed)

- **Frame-edge blobs pass.** The finder drops components with `x0 <= 0` / `y0 <= 0`. Blobs that start at x0 = 1 or y0 = 1 are kept. In the 20 main lojano photos (white sheet, 50 beans each going by the 500+500 pre-cut crops), the finder returned 1,040 blobs. Area outliers it returned included blue corner-vignette regions at y0 = 1 or x0 = 1, with area 3-46x the median bean (checked by eye on IMG_0246). The label-cleaning rules here (area 0.4-2.5x median, 2 px edge margin) kept 982 of the 1,040.
- **Dark boards break it.** On the 8 lojano photos on a brown wooden board (sheet luma 82-87), the finder returned 38-72 blobs per photo for 25 beans. It segmented shadows below the beans, because the beans there are lighter than the board.
- **Fewer blobs than beans on loja_yolo.** 3,050 annotated beans gave 1,857 finder blobs, and only 1,325 matched an annotation at IoU >= 0.5. Whether the shortfall is merged touching beans or missed beans was not measured.

## 9. Files

- `data/raw2/<source>/` raw downloads (zips kept next to them).
- `data/crops_v2/<source>/<farz_class>/*.png`: kept crops. `data/crops_v2/<source>/crops.csv`: per-source rows before cross-source dedupe. `data/crops_v2/<source>/build_log.json`: items, errors, QC drops.
- `data/crops_v2/_dropped/`: crops removed as duplicates. They are parked, not deleted, so `--merge` can be re-run.
- `data/crops_v2/crops_v2.csv`: all rows. Columns: crop_path, source, source_class, farz_class, group (split on this: photo or bean id), split_hint (the source's own split), orig_path, mode, bbox, n_found, touching, flags, src_pixel_md5, crop_pixel_md5, dhash, licence, keep, dup_of. **Use rows with keep == 1.**
- `data/crops_v2/contact_sheet_v2.jpg`: 16 random kept crops per source, one row per source in alphabetical order. Use it to eyeball each domain.
- `data/crops_v2/summary_v2.json`: counts and dedupe statistics. `src/beans_v2/data_report.py` prints the tables in this file from it.
- Re-run: `python src/beans_v2/data_build_crops.py --sources <names> --workers 8`, then `--merge`, then `python src/beans_v2/data_report.py`.
