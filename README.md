# H&E Pancreas Pathology Tile Classifier

Deep learning workflow for classifying annotated pancreatic H&E histology tiles from Reya Lab whole-slide images. The project packages QuPath tile export scripts, a PyTorch training pipeline, leave-one-slide-out evaluation outputs, threshold tuning, and lightweight public figures for reproducible review.

![Workflow](figures/pipeline_overview.png)

## Highlights

- **Task:** classify H&E image tiles into `ADM`, `PanIN_LG`, `PanIN_HG`, and `Other`.
- **Model:** WideResNet-50-2 trained with per-fold Macenko stain normalization, slide-stratified sampling, MixUp augmentation, focal loss, test-time augmentation, and spatial consensus.
- **Evaluation:** leave-one-slide-out validation across six slides (`R4-22`, `R4-23`, `R4-25`, `R4-27`, `R4-28`, `R4-29`).
- **Post-processing:** per-class probability threshold tuning to improve tissue-class macro F1.
- **Data policy:** raw whole-slide images, large checkpoints, and private credentials are excluded from git.

## Example Tiles

![Example tile mosaic](figures/example_tile_mosaic.png)

## Results Snapshot

Threshold tuning improved tissue-class macro F1 on every held-out slide in the saved evaluation outputs.

![Threshold tuning summary](figures/threshold_tuning_summary.png)

Additional figures:

- [All-class confusion matrix](figures/confusion_matrix_all.png)
- [Tissue-only confusion matrix](figures/confusion_matrix_tissue.png)
- [Original vs tuned confusion matrices](figures/confusion_matrix_tuned_vs_original.png)
- [Tile class distribution](figures/tile_class_distribution.png)

The exported tile distribution used for development is summarized in [results/tile_class_counts.csv](results/tile_class_counts.csv). Full per-tile prediction CSVs are in [results/](results/).

## Repository Layout

```text
.
├── src/                  # Training and threshold tuning pipelines
├── scripts/              # Debug and public figure generation utilities
├── qupath_scripts/       # QuPath scripts used to export annotated tiles
├── figures/              # README and process figures
├── results/              # Lightweight evaluation CSVs and summaries
├── examples/tiles/       # Small representative tile samples
├── data/                 # Data access notes only; raw data is excluded
├── models/               # Model checkpoint notes only; checkpoints are excluded
└── docs/                 # Methods and reproducibility notes
```

## Quick Start

Create an environment:

```bash
conda create -n he-pathology python=3.10
conda activate he-pathology
pip install -r requirements.txt
```

Run leave-one-slide-out training and evaluation:

```bash
python src/he_ml_pipeline.py \
  --data_dir /path/to/spatial_tiles_dataset \
  --output_dir pipeline_outputs
```

Run threshold tuning after prediction CSVs are generated:

```bash
python src/threshold_tune.py --output_dir pipeline_outputs
```

Regenerate public figures:

```bash
python scripts/make_public_figures.py
```

## Data And Model Access

This repository intentionally does not include raw whole-slide images, full tile datasets, or `.pth` checkpoint files. The detailed project folder is maintained on Google Drive:

<https://drive.google.com/drive/folders/1bzPsdvUn9KUEjALNeJGkmOVVipVVXvzz>

See [docs/data.md](docs/data.md) and [models/README.md](models/README.md) for expected local paths and artifact handling.

## Methods

The main method is documented in [docs/methods.md](docs/methods.md). In brief:

1. Annotate histologic regions in QuPath.
2. Export coordinate-aware tiles into class folders.
3. Exclude PDAC from final leave-one-slide-out modeling because the available PDAC tiles are not evenly represented across slides.
4. Train a single-stage four-class classifier.
5. Save per-tile prediction probabilities, apply spatial consensus, and tune per-class thresholds.

## Citation

If this repository is useful, cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

Code is released under the MIT License. Data and images remain subject to the Reya Lab/project data-use terms and are not automatically covered by the code license.
