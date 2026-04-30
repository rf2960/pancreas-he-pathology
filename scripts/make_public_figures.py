"""
Create lightweight, publication-style figures for the public GitHub repository.

The script reads local evaluation CSVs and, when available, a local tile dataset.
It avoids copying raw whole-slide images or model checkpoints into the repository.
"""

from __future__ import annotations

import csv
import math
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageStat


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
EXAMPLES_DIR = REPO_ROOT / "examples" / "tiles"

DEFAULT_DATA_DIR = Path(r"C:\Users\ruoch\Desktop\CU\Research\H&E ML\spatial_tiles_dataset")

CLASSES = ["ADM", "PanIN_LG", "PanIN_HG", "Other"]
TISSUE_CLASSES = ["ADM", "PanIN_LG", "PanIN_HG"]
COLORS = {
    "ADM": "#4C78A8",
    "PanIN_LG": "#F58518",
    "PanIN_HG": "#54A24B",
    "Other": "#9D755D",
    "PDAC": "#B279A2",
}


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def read_thresholds() -> list[dict[str, str]]:
    path = RESULTS_DIR / "optimal_thresholds.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot_threshold_summary() -> None:
    rows = read_thresholds()
    slides = [r["slide"] for r in rows]
    original = [float(r["orig_macro_f1"]) for r in rows]
    tuned = [float(r["tuned_macro_f1"]) for r in rows]

    x = range(len(slides))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar([i - 0.18 for i in x], original, width=0.36, label="Original", color="#7A869A")
    ax.bar([i + 0.18 for i in x], tuned, width=0.36, label="Threshold tuned", color="#2F80ED")
    ax.set_xticks(list(x), slides)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Tissue macro F1")
    ax.set_title("Leave-one-slide-out performance after threshold tuning")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_tuning_summary.png", dpi=200)
    plt.close(fig)


def collect_tiles(data_dir: Path) -> dict[str, list[Path]]:
    tiles = {name: [] for name in ["ADM", "PanIN_LG", "PanIN_HG", "PDAC", "Other"]}
    for label_dir in data_dir.iterdir():
        if not label_dir.is_dir() or label_dir.name not in tiles:
            continue
        tiles[label_dir.name] = sorted(
            p for p in label_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
    return tiles


def plot_class_distribution(tiles: dict[str, list[Path]]) -> None:
    labels = [label for label, paths in tiles.items() if paths]
    counts = [len(tiles[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(labels, counts, color=[COLORS.get(label, "#777777") for label in labels])
    ax.set_ylabel("Tiles")
    ax.set_title("Exported tile distribution")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    for i, count in enumerate(counts):
        ax.text(i, count, f"{count:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tile_class_distribution.png", dpi=200)
    plt.close(fig)


def tile_content_score(path: Path) -> float:
    """Score tiles by visual tissue content so README examples avoid blank patches."""
    img = Image.open(path).convert("RGB").resize((96, 96))
    gray = ImageOps.grayscale(img)
    stat_rgb = ImageStat.Stat(img)
    stat_gray = ImageStat.Stat(gray)

    mean_brightness = sum(stat_rgb.mean) / 3.0
    channel_std = sum(stat_rgb.stddev) / 3.0
    contrast = stat_gray.stddev[0]

    # Blank H&E background is usually very bright and low contrast. A useful
    # public example should have both stain variation and structural texture.
    non_white_bonus = max(0.0, 245.0 - mean_brightness)
    return non_white_bonus + (1.8 * channel_std) + (1.5 * contrast)


def choose_informative_tiles(paths: list[Path], n: int) -> list[Path]:
    scored = []
    for path in paths:
        try:
            score = tile_content_score(path)
        except Exception:
            continue
        if math.isfinite(score):
            scored.append((score, path))

    scored.sort(reverse=True, key=lambda item: item[0])
    if len(scored) <= n:
        return [path for _, path in scored]

    # Pick from high-scoring tiles while keeping some visual diversity.
    candidate_pool = scored[: max(n * 12, 24)]
    step = max(1, len(candidate_pool) // n)
    return [candidate_pool[i][1] for i in range(0, min(len(candidate_pool), step * n), step)][:n]


def copy_example_tiles(tiles: dict[str, list[Path]], n_per_class: int = 3) -> dict[str, list[Path]]:
    random.seed(7)
    copied: dict[str, list[Path]] = {}
    for label in CLASSES:
        paths = tiles.get(label, [])
        if not paths:
            continue
        label_dir = EXAMPLES_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for old_tile in label_dir.glob("*"):
            if old_tile.is_file():
                old_tile.unlink()
        chosen = choose_informative_tiles(paths, min(n_per_class, len(paths)))
        copied[label] = []
        for idx, src in enumerate(chosen, start=1):
            dst = label_dir / f"{label}_{idx}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            copied[label].append(dst)
    return copied


def plot_tile_mosaic(copied: dict[str, list[Path]]) -> None:
    labels = [label for label in CLASSES if copied.get(label)]
    if not labels:
        return

    n_cols = max(len(copied[label]) for label in labels)
    thumb_size = 160
    label_h = 34
    pad = 10
    width = n_cols * thumb_size + (n_cols + 1) * pad
    height = len(labels) * (thumb_size + label_h + pad) + pad
    canvas = Image.new("RGB", (width, height), "white")

    for row, label in enumerate(labels):
        y0 = pad + row * (thumb_size + label_h + pad)
        for col, path in enumerate(copied[label]):
            img = Image.open(path).convert("RGB")
            img = ImageOps.fit(img, (thumb_size, thumb_size), method=Image.Resampling.LANCZOS)
            x0 = pad + col * (thumb_size + pad)
            canvas.paste(img, (x0, y0 + label_h))

        fig, ax = plt.subplots(figsize=(1, 1))
        ax.axis("off")
        ax.text(0, 0, label, fontsize=12, fontweight="bold", color=COLORS.get(label, "#333333"))
        fig.canvas.draw()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(width / 120, height / 120))
    ax.imshow(canvas)
    ax.axis("off")
    for row, label in enumerate(labels):
        y = pad + row * (thumb_size + label_h + pad) + 18
        ax.text(pad, y, label, fontsize=13, fontweight="bold", color=COLORS.get(label, "#333333"))
    fig.tight_layout(pad=0)
    fig.savefig(FIGURES_DIR / "example_tile_mosaic.png", dpi=200)
    plt.close(fig)


def plot_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")

    steps = [
        ("QuPath annotations", "Manual region labels\non H&E slides"),
        ("Tile export", "Patch coordinates\nand class folders"),
        ("Training", "WideResNet-50-2\nwith stain normalization"),
        ("Evaluation", "Leave-one-slide-out\nprediction CSVs"),
        ("Tuning", "Per-class thresholds\nfor tissue macro F1"),
    ]

    x_positions = [0.08, 0.29, 0.50, 0.71, 0.90]
    for idx, ((title, body), x) in enumerate(zip(steps, x_positions)):
        ax.text(
            x,
            0.63,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#F7F8FA", ec="#AAB2C0", lw=1),
        )
        ax.text(x, 0.34, body, ha="center", va="center", fontsize=9, color="#384250")
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.075, 0.63),
                xytext=(x + 0.075, 0.63),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="#566176"),
            )

    ax.set_title("H&E tile classification workflow", fontsize=14, pad=18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pipeline_overview.png", dpi=200)
    plt.close(fig)


def plot_model_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axis("off")

    layers = [
        ("H&E tile", "RGB patch\nfrom QuPath"),
        ("Stain norm", "Per-fold\nMacenko fit"),
        ("Augment", "Flip, rotate,\ncolor jitter, MixUp"),
        ("Backbone", "ImageNet\nWideResNet-50-2"),
        ("Head", "Dropout + dense\n4-class logits"),
        ("Inference", "8-pass TTA +\nspatial consensus"),
    ]

    x_positions = [0.08, 0.245, 0.41, 0.58, 0.745, 0.91]
    for idx, ((title, body), x) in enumerate(zip(layers, x_positions)):
        ax.text(
            x,
            0.62,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.48", fc="#F2F7FF", ec="#4C78A8", lw=1.1),
        )
        ax.text(x, 0.34, body, ha="center", va="center", fontsize=9.2, color="#344054")
        if idx < len(layers) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.063, 0.62),
                xytext=(x + 0.063, 0.62),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#475467"),
            )

    ax.text(
        0.5,
        0.08,
        "Training objective: weighted focal loss with label smoothing; validation: leave-one-slide-out tissue macro F1.",
        ha="center",
        va="center",
        fontsize=10,
        color="#1D2939",
    )
    ax.set_title("Modeling pipeline", fontsize=14, pad=18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_architecture.png", dpi=200)
    plt.close(fig)


def write_dataset_summary(tiles: dict[str, list[Path]]) -> None:
    rows = ["class,tile_count"]
    for label in ["ADM", "PanIN_LG", "PanIN_HG", "PDAC", "Other"]:
        rows.append(f"{label},{len(tiles.get(label, []))}")
    (RESULTS_DIR / "tile_class_counts.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    plot_threshold_summary()
    plot_pipeline_overview()
    plot_model_architecture()

    if DEFAULT_DATA_DIR.exists():
        tiles = collect_tiles(DEFAULT_DATA_DIR)
        write_dataset_summary(tiles)
        plot_class_distribution(tiles)
        copied = copy_example_tiles(tiles)
        plot_tile_mosaic(copied)


if __name__ == "__main__":
    main()
