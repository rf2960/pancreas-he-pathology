# Resume Positioning

## Recommended Repository Name

Best option:

```text
pancreas-he-pathology-ai
```

Why: it is searchable, specific, and immediately communicates domain plus ML content. `H-E` is concise but too ambiguous for recruiters scanning a GitHub profile.

Other strong options:

- `pancreatic-histology-cv`
- `he-pathology-tile-classifier`
- `pancreas-histopathology-ml`

## Resume Bullet Options

- Built an end-to-end PyTorch computer vision pipeline for pancreatic H&E tile classification, including QuPath export, stain normalization, leave-one-slide-out validation, and spatial post-processing.
- Engineered an imbalanced histopathology ML workflow using focal loss, class weighting, slide-stratified sampling, MixUp augmentation, and test-time augmentation.
- Evaluated 39,944 tile predictions across six held-out slides and improved mean tissue macro F1 from 0.413 to 0.573 using per-class threshold tuning.
- Packaged a research codebase into a reproducible public GitHub repository with model card, data governance notes, CI checks, figures, and metric summaries.

## Interview Talking Points

- Why slide-level validation is more appropriate than random tile splitting.
- How stain normalization reduces batch effects in H&E images.
- Why class imbalance makes accuracy misleading and macro F1 more informative.
- How spatial consensus uses tile coordinates to smooth noisy predictions.
- Why the repo excludes raw data/checkpoints and documents artifact access instead.

## Honest Framing

This is best framed as a research-grade ML engineering project rather than a clinically deployable model. The strongest story is not only the final metric; it is the complete workflow, validation design, reproducibility, and careful handling of sensitive/large biomedical artifacts.
