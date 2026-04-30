# Results

This folder contains lightweight CSV outputs from the leave-one-slide-out evaluation.

- `results_R4-*.csv`: per-tile predictions for each held-out slide
- `results_R4-*_tuned.csv`: predictions after per-class threshold tuning
- `optimal_thresholds.csv`: thresholds and original/tuned tissue macro F1 by slide
- `tile_class_counts.csv`: exported tile counts by class from the local development dataset

Large model checkpoints generated during training are not stored here.
