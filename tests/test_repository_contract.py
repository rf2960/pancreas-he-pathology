from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DISALLOWED_SUFFIXES = {".pem", ".pth", ".pt", ".ckpt", ".svs", ".ndpi", ".scn", ".qptiff"}
MAX_GIT_FILE_BYTES = 100 * 1024 * 1024


def test_required_public_files_exist() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "environment.yml",
        "src/he_ml_pipeline.py",
        "src/threshold_tune.py",
        "scripts/make_public_figures.py",
        "scripts/summarize_results.py",
        "docs/model_card.md",
        "docs/experiment_report.md",
        "docs/qupath_workflow.md",
        "docs/storage_cleanup_plan.md",
        "figures/example_tile_mosaic.png",
        "figures/model_architecture.png",
        "figures/qupath_annotation_example.png",
        "figures/qupath_to_ml_workflow.png",
        "figures/tissue_accuracy_summary.png",
        "results/aggregate_metrics.md",
        "results/headline_metrics.csv",
    ]
    missing = [path for path in required if not (REPO_ROOT / path).exists()]
    assert not missing


def test_no_private_or_large_artifacts_are_committed() -> None:
    bad_files = []
    for path in REPO_ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix.lower() in DISALLOWED_SUFFIXES or path.stat().st_size > MAX_GIT_FILE_BYTES:
            bad_files.append(path.relative_to(REPO_ROOT).as_posix())
    assert not bad_files


def test_result_csv_contract() -> None:
    required_columns = {"x", "y", "Actual", "Predicted", "p_ADM", "p_PanIN_LG", "p_PanIN_HG", "p_Other", "Conf", "Refined"}
    result_files = sorted(p for p in (REPO_ROOT / "results").glob("results_R4-*.csv") if not p.name.endswith("_tuned.csv"))
    assert result_files
    for path in result_files:
        with path.open(newline="", encoding="utf-8") as fh:
            columns = set(next(csv.DictReader(fh)).keys())
        assert required_columns.issubset(columns), path.name
