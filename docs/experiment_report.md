# Experiment Report

## Objective

Build a reproducible computer vision pipeline for classifying pancreatic H&E tiles into:

- `ADM`
- `PanIN_LG`
- `PanIN_HG`
- `Other`

The practical goal is to support quantitative review of annotated lesion regions while keeping the public repository safe to share.

## Dataset Snapshot

Development tile counts from the local exported dataset:

| Class | Tiles |
|---|---:|
| ADM | 2,663 |
| PanIN_LG | 486 |
| PanIN_HG | 1,061 |
| PDAC | 3,269 |
| Other | 35,734 |

PDAC was excluded from the final four-class leave-one-slide-out model because its slide distribution was too concentrated for a fair held-out-slide estimate.

## Validation Design

The project uses leave-one-slide-out validation across six slides:

```text
R4-22, R4-23, R4-25, R4-27, R4-28, R4-29
```

This design is more realistic than random tile splitting because neighboring tiles from the same slide can be highly correlated.

## Modeling Components

- WideResNet-50-2 ImageNet backbone
- Per-fold Macenko stain normalization
- Random flip, rotation, color jitter, and grayscale augmentation
- MixUp augmentation
- Weighted focal loss with label smoothing
- Slide-stratified balanced sampler
- 8-pass test-time augmentation
- Confidence-aware spatial consensus over tile coordinates
- Per-class post-hoc threshold tuning

## Aggregate Results

See [results/aggregate_metrics.md](../results/aggregate_metrics.md) for the generated summary.

Headline evaluation from saved outputs:

- Mean original tissue macro F1: 0.413
- Mean tuned tissue macro F1: 0.573
- Mean threshold-tuning gain: +0.160
- Evaluated tiles: 39,944
- Tissue tiles: 4,210

## Interpretation

The project is strongest as a complete ML workflow: data extraction, leakage-aware validation, class imbalance handling, inference post-processing, reproducible reporting, and honest artifact governance. The tuned tissue macro F1 indicates that the post-processing step recovers tissue-class signal, but class-level precision remains limited for rare classes. That limitation is expected given the dataset imbalance and should motivate additional annotation, nested threshold validation, and larger slide coverage before any clinical use.

## Novelty

- Connects QuPath annotation engineering directly to a reproducible PyTorch modeling pipeline.
- Preserves slide ID and tile coordinates from filenames to support leakage-aware validation and spatial post-processing.
- Uses a multi-step imbalance strategy: capped `Other` sampling, class weighting, focal loss, balanced sampling, and macro-F1 reporting.
- Packages a biomedical ML project for public GitHub review without exposing protected raw data or credentials.

## Future Work

- Add more slides and improve rare-class coverage.
- Compare the WideResNet baseline with pathology foundation models and self-supervised encoders.
- Use nested cross-validation for threshold tuning.
- Add probability calibration and uncertainty summaries.
- Aggregate tile predictions into slide-level lesion burden features.
- Publish selected checkpoints through GitHub Releases only if data-sharing approval allows it.
