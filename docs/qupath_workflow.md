# QuPath Workflow

## Role In The Project

QuPath is the bridge between raw H&E whole-slide images and the ML-ready tile dataset. This project uses QuPath for:

- whole-slide image review
- region annotation
- lesion-class organization
- coordinate-aware tile export
- reproducible data extraction with Groovy scripts

The important point for reviewers: the dataset was not a generic image folder. It was engineered from pathology annotations and whole-slide spatial coordinates.

## Public QuPath Example

The README includes [figures/qupath_annotation_example.png](../figures/qupath_annotation_example.png), generated from the QuPath project thumbnail and annotation summary for `R4-22.svs`.

It also includes [figures/qupath_r425_annotation_overlay.png](../figures/qupath_r425_annotation_overlay.png), a QuPath-style R4-25 overlay generated from:

- the R4-25 QuPath project thumbnail
- QuPath annotation summary counts
- exported R4-25 tile coordinates and labels

The raw `.svs` file is not committed to GitHub because whole-slide images are large and may be subject to project data-use restrictions. If a higher-quality QuPath screenshot is needed, export a snapshot from the QuPath GUI with annotations visible and replace the generated public figure.

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

## How To Export A Better QuPath Screenshot

1. Open the QuPath project locally.
2. Open a representative slide such as `R4-22.svs`.
3. Turn on annotation overlays and class colors.
4. Zoom to a region where annotations are visible.
5. Export a snapshot as PNG.
6. Add the PNG to `figures/` and reference it in the README.
