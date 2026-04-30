# Methods

## Overview

This workflow classifies annotated pancreatic H&E tiles into four public-model classes:

- `ADM`
- `PanIN_LG`
- `PanIN_HG`
- `Other`

PDAC tiles are present in the development dataset but are excluded from the final leave-one-slide-out classifier because the available PDAC examples are concentrated in too few slides for a fair slide-held-out estimate.

## Annotation And Tile Export

Regions were annotated in QuPath and exported as coordinate-aware image tiles. Filenames encode the slide, class label, and spatial coordinates:

```text
R4-22_ADM_[x=10142,y=16870].png
```

The training script parses this naming convention to reconstruct slide IDs, labels, and tile coordinates for slide-level splitting and spatial post-processing.

## Training

The main model is a single-stage four-class WideResNet-50-2 classifier implemented in [src/he_ml_pipeline.py](../src/he_ml_pipeline.py). Core training choices:

- leave-one-slide-out validation
- per-fold Macenko stain normalization fitted to held-out-slide reference tiles
- ImageNet normalization
- random flips, rotations, color jitter, and light grayscale augmentation
- weighted focal loss
- slide-stratified balanced sampling
- MixUp augmentation
- early stopping by validation behavior

## Inference And Post-Processing

Inference uses eight-pass test-time augmentation. The pipeline stores class probabilities for every tile and applies soft spatial consensus using neighboring tile predictions. Per-class probability thresholds can then be tuned with [src/threshold_tune.py](../src/threshold_tune.py) to optimize tissue-class macro F1.

## Evaluation

Saved outputs use leave-one-slide-out evaluation across:

```text
R4-22, R4-23, R4-25, R4-27, R4-28, R4-29
```

Primary reported metric is tissue-only macro F1 over `ADM`, `PanIN_LG`, and `PanIN_HG`. `Other` is still modeled and reported, but tissue-only metrics focus the biological lesion-grading task.
