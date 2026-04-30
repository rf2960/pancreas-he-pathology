# Data Availability

This repository does not include the raw H&E scans, the full exported tile dataset, or the QuPath project database. These files are too large for git and may be subject to project-specific access controls.

Project Drive folder:

<https://drive.google.com/drive/folders/1bzPsdvUn9KUEjALNeJGkmOVVipVVXvzz>

## Expected Local Layout

The scripts expect exported tiles in class subfolders:

```text
spatial_tiles_dataset/
|-- ADM/
|-- PanIN_LG/
|-- PanIN_HG/
|-- PDAC/
`-- Other/
```

Each tile filename should contain the slide ID, label, and coordinates:

```text
<slide_id>_<label>_[x=<x>,y=<y>].png
```

## Public Examples

The repository includes a small representative subset in [examples/tiles](../examples/tiles) and a derived mosaic in [figures/example_tile_mosaic.png](../figures/example_tile_mosaic.png). These examples are for visual orientation and README display, not for model training.

## Files Intentionally Excluded

- whole-slide images (`.svs`, `.tif`, `.ndpi`, `.scn`, `.qptiff`)
- full tile exports
- QuPath project databases
- model checkpoints
- private keys or cloud credentials

The exclusion rules are captured in [.gitignore](../.gitignore).
