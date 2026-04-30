# Model Card

## Model

Single-stage four-class H&E tile classifier.

Expected class order:

```text
ADM, PanIN_LG, PanIN_HG, Other
```

Architecture:

- ImageNet-pretrained WideResNet-50-2 backbone
- Dropout and dense classification head
- Weighted focal loss with label smoothing
- Test-time augmentation and spatial consensus during inference

## Intended Use

Research and portfolio demonstration of a pancreatic histopathology ML workflow. The model is not intended for clinical decision-making.

## Inputs

RGB H&E image tiles exported from QuPath. Filenames are expected to encode slide ID, label, and tile coordinates:

```text
<slide_id>_<label>_[x=<x>,y=<y>].png
```

## Outputs

Per-tile class probabilities and predicted labels. Saved CSVs include:

- coordinates
- ground-truth annotation label
- raw predicted class
- class probabilities
- spatially refined prediction
- tuned prediction when threshold tuning is applied

## Evaluation

Validation uses leave-one-slide-out evaluation across six held-out slides. The primary metric is tissue-class macro F1 over `ADM`, `PanIN_LG`, and `PanIN_HG`.

Current saved-output headline:

| Metric | Value |
|---|---:|
| Mean original tissue macro F1 | 0.413 |
| Mean tuned tissue macro F1 | 0.573 |
| Mean threshold-tuning gain | +0.160 |
| Mean tuned tissue accuracy | 77.6% |
| Mean tuned tissue balanced accuracy | 57.0% |
| Evaluated tiles | 39,944 |

## Limitations

- Class imbalance is severe, especially for `PanIN_LG`.
- PDAC is not part of the final public classifier because slide-level representation was not sufficient for fair leave-one-slide-out validation.
- Threshold tuning is post-hoc and should be nested or validated on independent slides before publication-grade claims.
- Tile-level labels inherit uncertainty from region annotations and may not perfectly represent every patch.
- The model has not been externally validated on another cohort or staining protocol.

## Ethical And Data Notes

Raw slides, complete tile datasets, model checkpoints, and private credentials are not included in this repository. Data access should follow Reya Lab/project terms.
