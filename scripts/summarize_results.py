"""
Summarize saved prediction CSVs into interview-friendly ML evaluation artifacts.

Outputs:
- results/slide_metrics.csv
- results/class_metrics_tuned.csv
- results/headline_metrics.csv
- results/aggregate_metrics.md
- figures/per_slide_f1_delta.png
- figures/tissue_accuracy_summary.png
- figures/class_f1_heatmap.png
- figures/threshold_heatmap.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

TISSUE_CLASSES = ["ADM", "PanIN_LG", "PanIN_HG"]
ALL_CLASSES = ["ADM", "PanIN_LG", "PanIN_HG", "Other"]


def tissue_balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    recalls = []
    for label in TISSUE_CLASSES:
        mask = y_true == label
        if mask.sum() == 0:
            continue
        recalls.append((y_pred[mask] == label).mean())
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def read_thresholds() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "optimal_thresholds.csv")


def iter_slide_files() -> list[tuple[str, Path, Path]]:
    pairs = []
    for original in sorted(RESULTS_DIR.glob("results_R4-*.csv")):
        if original.name.endswith("_tuned.csv"):
            continue
        slide = original.stem.replace("results_", "")
        tuned = RESULTS_DIR / f"results_{slide}_tuned.csv"
        if tuned.exists():
            pairs.append((slide, original, tuned))
    return pairs


def compute_slide_metrics() -> pd.DataFrame:
    rows = []
    for slide, original_path, tuned_path in iter_slide_files():
        original = pd.read_csv(original_path)
        tuned = pd.read_csv(tuned_path)

        tissue_original = original[original["Actual"] != "Other"]
        tissue_tuned = tuned[tuned["Actual"] != "Other"]

        orig_f1 = f1_score(
            tissue_original["Actual"],
            tissue_original["Refined"],
            labels=TISSUE_CLASSES,
            average="macro",
            zero_division=0,
        )
        tuned_f1 = f1_score(
            tissue_tuned["Actual"],
            tissue_tuned["Tuned"],
            labels=TISSUE_CLASSES,
            average="macro",
            zero_division=0,
        )
        original_accuracy = accuracy_score(tissue_original["Actual"], tissue_original["Refined"])
        tuned_accuracy = accuracy_score(tissue_tuned["Actual"], tissue_tuned["Tuned"])
        original_balanced_accuracy = tissue_balanced_accuracy(tissue_original["Actual"], tissue_original["Refined"])
        tuned_balanced_accuracy = tissue_balanced_accuracy(tissue_tuned["Actual"], tissue_tuned["Tuned"])

        rows.append(
            {
                "slide": slide,
                "n_tiles": len(original),
                "n_tissue_tiles": len(tissue_original),
                "original_tissue_macro_f1": orig_f1,
                "tuned_tissue_macro_f1": tuned_f1,
                "delta_f1": tuned_f1 - orig_f1,
                "original_tissue_accuracy": original_accuracy,
                "tuned_tissue_accuracy": tuned_accuracy,
                "delta_accuracy": tuned_accuracy - original_accuracy,
                "original_tissue_balanced_accuracy": original_balanced_accuracy,
                "tuned_tissue_balanced_accuracy": tuned_balanced_accuracy,
                "delta_balanced_accuracy": tuned_balanced_accuracy - original_balanced_accuracy,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "slide_metrics.csv", index=False)
    return df


def compute_headline_metrics(slide_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_rows = [
        ("mean_original_tissue_macro_f1", slide_metrics["original_tissue_macro_f1"].mean(), "{:.3f}"),
        ("mean_tuned_tissue_macro_f1", slide_metrics["tuned_tissue_macro_f1"].mean(), "{:.3f}"),
        ("mean_threshold_tuning_f1_gain", slide_metrics["delta_f1"].mean(), "+{:.3f}"),
        ("mean_tuned_tissue_accuracy", slide_metrics["tuned_tissue_accuracy"].mean(), "{:.1%}"),
        ("mean_tuned_tissue_balanced_accuracy", slide_metrics["tuned_tissue_balanced_accuracy"].mean(), "{:.1%}"),
        ("evaluated_tiles", int(slide_metrics["n_tiles"].sum()), "{:,}"),
        ("tissue_tiles", int(slide_metrics["n_tissue_tiles"].sum()), "{:,}"),
        ("held_out_slides", len(slide_metrics), "{}"),
    ]
    rows = [{"metric": name, "value": value, "display": fmt.format(value)} for name, value, fmt in metric_rows]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "headline_metrics.csv", index=False)
    return df


def compute_class_metrics() -> pd.DataFrame:
    tuned_frames = [pd.read_csv(tuned_path) for _, _, tuned_path in iter_slide_files()]
    tuned_all = pd.concat(tuned_frames, ignore_index=True)

    report = classification_report(
        tuned_all["Actual"],
        tuned_all["Tuned"],
        labels=ALL_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in ALL_CLASSES:
        rows.append(
            {
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1_score": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "class_metrics_tuned.csv", index=False)
    return df


def plot_per_slide_delta(slide_metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(slide_metrics["slide"], slide_metrics["delta_f1"], color="#2E90FA")
    ax.axhline(0, color="#344054", linewidth=1)
    ax.set_ylabel("Macro F1 gain")
    ax.set_title("Threshold tuning gain by held-out slide")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(slide_metrics["delta_f1"]):
        ax.text(idx, value + 0.01, f"+{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_slide_f1_delta.png", dpi=200)
    plt.close(fig)


def plot_accuracy_summary(slide_metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = range(len(slide_metrics))
    ax.bar(
        [i - 0.18 for i in x],
        slide_metrics["tuned_tissue_accuracy"],
        width=0.36,
        label="Tissue accuracy",
        color="#12B76A",
    )
    ax.bar(
        [i + 0.18 for i in x],
        slide_metrics["tuned_tissue_balanced_accuracy"],
        width=0.36,
        label="Balanced accuracy",
        color="#2E90FA",
    )
    ax.set_xticks(list(x), slide_metrics["slide"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Tuned tissue accuracy by held-out slide")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tissue_accuracy_summary.png", dpi=200)
    plt.close(fig)


def plot_class_metric_heatmap(class_metrics: pd.DataFrame) -> None:
    metric_cols = ["precision", "recall", "f1_score"]
    matrix = class_metrics.set_index("class")[metric_cols]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_cols)), ["Precision", "Recall", "F1"])
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Tuned aggregate class metrics")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="#101828")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_f1_heatmap.png", dpi=200)
    plt.close(fig)


def plot_threshold_heatmap() -> None:
    thresholds = read_thresholds().set_index("slide")[["ADM", "PanIN_LG", "PanIN_HG", "Other"]]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    im = ax.imshow(thresholds.values, cmap="YlGnBu")
    ax.set_xticks(range(len(thresholds.columns)), thresholds.columns)
    ax.set_yticks(range(len(thresholds.index)), thresholds.index)
    ax.set_title("Learned post-hoc thresholds")
    for i in range(thresholds.shape[0]):
        for j in range(thresholds.shape[1]):
            ax.text(j, i, f"{thresholds.values[i, j]:.2f}", ha="center", va="center", color="#101828")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_heatmap.png", dpi=200)
    plt.close(fig)


def write_markdown_summary(slide_metrics: pd.DataFrame) -> None:
    mean_original = slide_metrics["original_tissue_macro_f1"].mean()
    mean_tuned = slide_metrics["tuned_tissue_macro_f1"].mean()
    mean_delta = slide_metrics["delta_f1"].mean()
    mean_accuracy = slide_metrics["tuned_tissue_accuracy"].mean()
    mean_balanced_accuracy = slide_metrics["tuned_tissue_balanced_accuracy"].mean()

    lines = [
        "# Aggregate Metrics",
        "",
        f"- Mean original tissue macro F1: **{mean_original:.3f}**",
        f"- Mean tuned tissue macro F1: **{mean_tuned:.3f}**",
        f"- Mean threshold-tuning gain: **+{mean_delta:.3f}**",
        f"- Mean tuned tissue accuracy: **{mean_accuracy:.1%}**",
        f"- Mean tuned tissue balanced accuracy: **{mean_balanced_accuracy:.1%}**",
        f"- Held-out slides: **{len(slide_metrics)}**",
        f"- Evaluated tiles: **{slide_metrics['n_tiles'].sum():,}**",
        f"- Tissue tiles: **{slide_metrics['n_tissue_tiles'].sum():,}**",
        "",
        "## Per-Slide Metrics",
        "",
        slide_metrics.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Detailed per-class diagnostics are saved in `class_metrics_tuned.csv`.",
        "",
    ]
    (RESULTS_DIR / "aggregate_metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    slide_metrics = compute_slide_metrics()
    compute_headline_metrics(slide_metrics)
    class_metrics = compute_class_metrics()
    plot_per_slide_delta(slide_metrics)
    plot_accuracy_summary(slide_metrics)
    plot_class_metric_heatmap(class_metrics)
    plot_threshold_heatmap()
    write_markdown_summary(slide_metrics)


if __name__ == "__main__":
    main()
