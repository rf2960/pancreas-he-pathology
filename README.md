# Pancreas H&E Pathology AI

[![Validate repository](https://github.com/rf2960/H-E/actions/workflows/validate.yml/badge.svg)](https://github.com/rf2960/H-E/actions/workflows/validate.yml)

Computer vision pipeline for classifying pancreatic H&E histology tiles from Reya Lab whole-slide images. This repository presents the full public research workflow: QuPath annotation export, tile-level data engineering, PyTorch modeling, leave-one-slide-out validation, post-hoc threshold tuning, reproducibility docs, and publication-style figures.

![Workflow](figures/pipeline_overview.png)

## Why This Project Matters

Pathology ML projects are difficult because the data are large, class-imbalanced, slide-correlated, and sensitive. This repo demonstrates an end-to-end DS/ML workflow that handles those realities instead of only showing a notebook:

- **Computer vision:** H&E tile classification with an ImageNet-pretrained WideResNet-50-2 backbone.
- **Medical image preprocessing:** QuPath annotation export, coordinate-aware tile parsing, and per-fold Macenko stain normalization.
- **Imbalanced learning:** focal loss, class weighting, balanced slide-stratified sampling, and controlled `Other` downsampling.
- **Robust validation:** leave-one-slide-out evaluation to reduce slide leakage.
- **Inference engineering:** 8-pass test-time augmentation, confidence-aware spatial consensus, and per-class threshold tuning.
- **ML reporting:** model card, reproducibility notes, result CSVs, metric summaries, and artifact/data governance.

## Results At A Glance

The saved evaluation outputs cover **39,944 tile predictions** across **6 held-out slides**, including **4,210 tissue lesion tiles**.

| Metric | Value |
|---|---:|
| Mean original tissue macro F1 | 0.413 |
| Mean tuned tissue macro F1 | 0.573 |
| Mean threshold-tuning gain | +0.160 |
| Held-out slides | 6 |

Threshold tuning improved tissue-class macro F1 on every held-out slide in the saved outputs.

![Threshold tuning summary](figures/threshold_tuning_summary.png)

![Per-slide tuning gain](figures/per_slide_f1_delta.png)

## Model Architecture

![Model architecture](figures/model_architecture.png)

## Example Tiles

Representative public examples are selected with a content-aware sampler so README examples avoid blank background tiles.

![Example tile mosaic](figures/example_tile_mosaic.png)

## Additional Figures

- [Tuned class metric heatmap](figures/class_f1_heatmap.png)
- [Learned threshold heatmap](figures/threshold_heatmap.png)
- [All-class confusion matrix](figures/confusion_matrix_all.png)
- [Tissue-only confusion matrix](figures/confusion_matrix_tissue.png)
- [Original vs tuned confusion matrices](figures/confusion_matrix_tuned_vs_original.png)
- [Tile class distribution](figures/tile_class_distribution.png)

## Repository Layout

```text
.
|-- src/                  # Training and threshold tuning pipelines
|-- scripts/              # Figure generation, metrics, and maintenance utilities
|-- qupath_scripts/       # QuPath scripts used to export annotated tiles
|-- figures/              # README and process figures
|-- results/              # Lightweight evaluation CSVs and metric summaries
|-- examples/tiles/       # Small representative tile samples
|-- data/                 # Data access notes only; raw data is excluded
|-- models/               # Model checkpoint notes only; checkpoints are excluded
|-- docs/                 # Methods, model card, experiment report, reproducibility
`-- tests/                # Repository integrity tests for CI
```

## Quick Start

Create an environment:

```bash
conda env create -f environment.yml
conda activate he-pathology
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

Regenerate public figures and metric summaries:

```bash
python scripts/make_public_figures.py
python scripts/summarize_results.py
```

Run repository checks:

```bash
python -m pytest
```

## Data And Model Access

This repository intentionally does not include raw whole-slide images, full tile datasets, private credentials, or `.pth` checkpoint files. The detailed project folder is maintained on Google Drive:

<https://drive.google.com/drive/folders/1bzPsdvUn9KUEjALNeJGkmOVVipVVXvzz>

See [docs/data.md](docs/data.md), [models/README.md](models/README.md), and [docs/model_card.md](docs/model_card.md) for expected local paths, artifact handling, and model limitations.

## Documentation

- [Methods](docs/methods.md)
- [Experiment report](docs/experiment_report.md)
- [Model card](docs/model_card.md)
- [Reproducibility](docs/reproducibility.md)
- [Resume positioning](docs/resume_positioning.md)
- [Project inventory](docs/project_inventory.md)

## Citation

If this repository is useful, cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

Code is released under the MIT License. Data and images remain subject to the Reya Lab/project data-use terms and are not automatically covered by the code license.
