# QuPath Workflow

## Role In The Project

QuPath is the bridge between raw H&E whole-slide images and the ML-ready tile dataset. This project uses QuPath for:

- whole-slide image review
- region annotation
- lesion-class organization
- coordinate-aware tile export
- reproducible data extraction with Groovy scripts

The important point for reviewers: the dataset was not a generic image folder. It was engineered from pathology annotations and whole-slide spatial coordinates.

## Class Ontology

The public four-class model uses:

- `ADM`
- `PanIN_LG`
- `PanIN_HG`
- `Other`

`PDAC` exists in the development export but is excluded from the final public classifier because available examples are too concentrated in too few slides for fair leave-one-slide-out validation.

## Scripts

The QuPath scripts are stored in [qupath_scripts](../qupath_scripts):

| Script | Purpose |
|---|---|
| `tiling.groovy` | Tile-generation workflow for annotated regions. |
| `export_tiles.groovy` | Exports labeled image tiles from QuPath annotations. |
| `exported_patches.groovy` | Patch export helper retained for provenance. |

## Expected Tile Output

The PyTorch pipeline expects class folders:

```text
spatial_tiles_dataset/
|-- ADM/
|-- PanIN_LG/
|-- PanIN_HG/
|-- PDAC/
`-- Other/
```

Tile filenames encode slide identity and spatial position:

```text
R4-22_ADM_[x=10142,y=16870].png
```

The training pipeline parses:

- `slide_id`
- `label_name`
- `x`
- `y`

Those fields enable leave-one-slide-out validation and spatial consensus.

## Why This Matters For ML

Random image splitting can inflate performance when neighboring tiles from the same slide appear in both train and test sets. The QuPath export preserves slide identity and tile coordinates, making it possible to evaluate with stricter held-out-slide validation.

## Public Release Policy

The repository includes scripts and sample tiles, but not raw whole-slide images or the full QuPath project database. That keeps the GitHub release useful for technical review while respecting data size and access constraints.
