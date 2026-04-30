# Aggregate Metrics

- Mean original tissue macro F1: **0.413**
- Mean tuned tissue macro F1: **0.573**
- Mean threshold-tuning gain: **+0.160**
- Mean tuned tissue accuracy: **77.6%**
- Mean tuned tissue balanced accuracy: **57.0%**
- Held-out slides: **6**
- Evaluated tiles: **39,944**
- Tissue tiles: **4,210**

## Per-Slide Metrics

| slide   |   n_tiles |   n_tissue_tiles |   original_tissue_macro_f1 |   tuned_tissue_macro_f1 |   delta_f1 |   original_tissue_accuracy |   tuned_tissue_accuracy |   delta_accuracy |   original_tissue_balanced_accuracy |   tuned_tissue_balanced_accuracy |   delta_balanced_accuracy |
|:--------|----------:|-----------------:|---------------------------:|------------------------:|-----------:|---------------------------:|------------------------:|-----------------:|------------------------------------:|---------------------------------:|--------------------------:|
| R4-22   |      3267 |              551 |                      0.349 |                   0.530 |      0.181 |                      0.673 |                   0.887 |            0.214 |                               0.468 |                            0.504 |                     0.036 |
| R4-23   |      9126 |             1010 |                      0.488 |                   0.547 |      0.059 |                      0.554 |                   0.727 |            0.172 |                               0.393 |                            0.526 |                     0.133 |
| R4-25   |      2032 |              440 |                      0.144 |                   0.518 |      0.374 |                      0.132 |                   0.609 |            0.477 |                               0.293 |                            0.516 |                     0.222 |
| R4-27   |      6765 |              345 |                      0.565 |                   0.625 |      0.059 |                      0.672 |                   0.867 |            0.194 |                               0.651 |                            0.713 |                     0.062 |
| R4-28   |      7269 |              580 |                      0.350 |                   0.565 |      0.215 |                      0.609 |                   0.817 |            0.209 |                               0.500 |                            0.533 |                     0.033 |
| R4-29   |     11485 |             1284 |                      0.581 |                   0.654 |      0.072 |                      0.569 |                   0.752 |            0.183 |                               0.623 |                            0.629 |                     0.007 |

Detailed per-class diagnostics are saved in `class_metrics_tuned.csv`.
