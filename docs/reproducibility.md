# Reproducibility

## Environment

Use either [requirements.txt](../requirements.txt) or [environment.yml](../environment.yml). A CUDA-enabled PyTorch install is recommended for full training.

```bash
conda env create -f environment.yml
conda activate he-pathology
```

## Training

```bash
python src/he_ml_pipeline.py \
  --data_dir /path/to/spatial_tiles_dataset \
  --output_dir pipeline_outputs \
  --batch_size 32 \
  --epochs 30 \
  --patience 6
```

The script performs leave-one-slide-out training. For each held-out slide, it writes:

- model checkpoint
- per-tile prediction CSV
- aggregate confusion matrices

Checkpoints are intentionally ignored by git. Store them in local storage, Google Drive, or GitHub Releases if public sharing is appropriate.

## Threshold Tuning

```bash
python src/threshold_tune.py --output_dir pipeline_outputs
```

This reads saved prediction CSVs, searches per-class thresholds, writes tuned CSVs, and produces threshold comparison summaries.

## Public Figure Generation

```bash
python scripts/make_public_figures.py
```

The script reads the lightweight CSVs in [results](../results) and, when the local tile dataset exists, refreshes README figures and example tile mosaics.
