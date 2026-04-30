# Aggregate Metrics

- Mean original tissue macro F1: **0.413**
- Mean tuned tissue macro F1: **0.573**
- Mean threshold-tuning gain: **+0.160**
- Held-out slides: **6**
- Evaluated tiles: **39,944**
- Tissue tiles: **4,210**

## Per-Slide Metrics

| slide   |   n_tiles |   n_tissue_tiles |   original_tissue_macro_f1 |   tuned_tissue_macro_f1 |   delta_f1 |
|:--------|----------:|-----------------:|---------------------------:|------------------------:|-----------:|
| R4-22   |      3267 |              551 |                      0.349 |                   0.530 |      0.181 |
| R4-23   |      9126 |             1010 |                      0.488 |                   0.547 |      0.059 |
| R4-25   |      2032 |              440 |                      0.144 |                   0.518 |      0.374 |
| R4-27   |      6765 |              345 |                      0.565 |                   0.625 |      0.059 |
| R4-28   |      7269 |              580 |                      0.350 |                   0.565 |      0.215 |
| R4-29   |     11485 |             1284 |                      0.581 |                   0.654 |      0.072 |

## Tuned Class Metrics

| class    |   precision |   recall |   f1_score |   support |
|:---------|------------:|---------:|-----------:|----------:|
| ADM      |       0.158 |    0.876 |      0.267 |      2663 |
| PanIN_LG |       0.037 |    0.362 |      0.068 |       486 |
| PanIN_HG |       0.140 |    0.679 |      0.232 |      1061 |
| Other    |       0.990 |    0.422 |      0.592 |     35734 |
