# Storage Cleanup Plan

This file records the recommended cleanup for the local research folder and Google Drive project folder. It is intentionally conservative: do not delete raw data or credentials without a backup.

## Local Folder

Local folder inspected:

```text
C:\Users\ruoch\Desktop\CU\Research\H&E ML
```

Recommended final layout:

```text
H&E ML/
|-- README_LOCAL.md
|-- 01_raw_and_annotations/
|   |-- H&E quant-Rama/
|   `-- QuPath Project/
|-- 02_tile_dataset/
|   `-- spatial_tiles_dataset/
|-- 03_model_outputs/
|   `-- pipeline_outputs/
|-- 04_manuscript_and_reports/
|   |-- Chvasta et al Reya Lab Bioinformatics_Draft_V6_RCH.docx
|   `-- H&E_ML_Report export, if downloaded
|-- 05_legacy_scripts/
|   |-- he_ml_pipeline.py
|   |-- threshold_tune.py
|   |-- debug_other.py
|   `-- requirements.txt
`-- 99_private_do_not_share/
    `-- reya-lab.pem
```

Files already preserved in GitHub:

- `he_ml_pipeline.py`
- `threshold_tune.py`
- `debug_other.py`
- `requirements.txt`
- QuPath Groovy scripts
- lightweight result CSVs
- README figures and example tiles

Files to keep locally but not in GitHub:

- raw whole-slide images
- full tile dataset
- QuPath project database
- model checkpoints
- manuscript drafts
- private key files

## Google Drive Folder

Drive folder inspected:

<https://drive.google.com/drive/folders/1bzPsdvUn9KUEjALNeJGkmOVVipVVXvzz>

Recommended final Drive layout:

```text
Reya H&E ML Project/
|-- 00_README_PROJECT_INDEX
|-- 01_reports/
|   |-- H&E_ML_Report
|   `-- Chvasta et al Reya Lab Bioinformatics_Draft_V6_RCH.docx
|-- 02_notebooks_archive/
|   |-- H&E_Image_Analysis_Nov6.ipynb
|   |-- H&E_Image_Analysis_Nov10.ipynb
|   |-- H&E_Image_Analysis_Dec30.ipynb
|   |-- H&E_Image_Analysis_Jan30.ipynb
|   |-- H&E_Image_Analysis_Feb1.ipynb
|   |-- H&E_Image_Analysis_Feb6.ipynb
|   `-- HE_Pipeline_Evaluation.ipynb
|-- 03_qupath/
|   |-- QuPath Project/
|   `-- H&E quant-Rama/
|-- 04_tiles/
|   `-- spatial_tiles_dataset/
|-- 05_model_outputs/
|   `-- pipeline outputs and selected checkpoints, if approved
`-- 99_archive_do_not_publish/
```

## Cleanup Recommendation

Do not delete anything yet. First reorganize into the folder layout above. After confirming the GitHub repo, Drive folder, and local backup all contain the right materials, old duplicate notebooks can be moved into `02_notebooks_archive`.

## Why Not Put Everything On GitHub?

GitHub should show the clean, reproducible, public story. Drive/local storage should hold raw data, full outputs, large checkpoints, and private research materials.
