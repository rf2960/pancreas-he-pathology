# Figures

Figures in this folder support the public README and methods documentation.

- `pipeline_overview.png`: workflow schematic
- `qupath_to_ml_workflow.png`: QuPath annotation and tile-export workflow
- `qupath_annotation_example.png`: representative QuPath slide thumbnail with annotation inventory
- `model_architecture.png`: modeling and inference architecture
- `example_tile_mosaic.png`: representative tile examples
- `tile_class_distribution.png`: local exported tile counts
- `threshold_tuning_summary.png`: original vs tuned tissue macro F1 by held-out slide
- `per_slide_f1_delta.png`: threshold-tuning gain by held-out slide
- `class_f1_heatmap.png`: aggregate tuned class metrics
- `threshold_heatmap.png`: learned per-class thresholds
- `tissue_accuracy_summary.png`: tuned tissue accuracy and balanced accuracy by held-out slide
- `confusion_matrix_all.png`: aggregate all-class confusion matrix
- `confusion_matrix_tissue.png`: aggregate tissue-only confusion matrix
- `confusion_matrix_tuned_vs_original.png`: original and tuned tissue confusion matrices

Refresh generated figures with:

```bash
python scripts/make_public_figures.py
python scripts/summarize_results.py
```
