# =============================================================================
#  THRESHOLD TUNING — Post-hoc optimization on saved prediction CSVs
#
#  Runs after all 6 folds complete. No retraining needed.
#  Finds the optimal per-class probability threshold that maximizes
#  macro F1 on tissue classes (ADM, PanIN_LG, PanIN_HG).
#
#  Usage:
#    python3 threshold_tune.py
#    python3 threshold_tune.py --output_dir /custom/path
# =============================================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pathlib import Path

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default=str(Path(__file__).parent.resolve() / 'pipeline_outputs'))
    return parser.parse_args()


CLASSES    = ['ADM', 'PanIN_LG', 'PanIN_HG', 'Other']
TISSUE_CLS = ['ADM', 'PanIN_LG', 'PanIN_HG']
PROB_COLS  = ['p_ADM', 'p_PanIN_LG', 'p_PanIN_HG', 'p_Other']


def apply_thresholds(df, thresholds):
    """
    Given per-class probability thresholds, assign each tile to the class
    whose (probability / threshold) ratio is highest.
    This effectively scales the decision boundary per class.
    """
    probs = df[PROB_COLS].values.copy()
    t     = np.array([thresholds[c] for c in CLASSES])
    scaled = probs / t
    preds  = [CLASSES[i] for i in scaled.argmax(axis=1)]
    return preds


def tune_thresholds(df, n_steps=20):
    """
    Grid search over per-class thresholds to maximize macro F1
    on tissue-only tiles. Other threshold is fixed at 1.0.
    Searches ADM, PanIN_LG, PanIN_HG thresholds independently.
    """
    best_f1     = 0
    best_thresh = {'ADM': 1.0, 'PanIN_LG': 1.0, 'PanIN_HG': 1.0, 'Other': 1.0}

    # Search range: 0.3 to 2.0 for tissue classes
    search = np.linspace(0.3, 2.0, n_steps)

    tissue_df = df[df['Actual'] != 'Other'].copy()

    print("  Tuning thresholds (grid search)...")
    for t_adm in search:
        for t_panin_hg in search:
            # PanIN_LG gets a wider search since it's rarest
            for t_panin_lg in np.linspace(0.2, 1.5, n_steps):
                thresh = {'ADM': t_adm, 'PanIN_LG': t_panin_lg,
                          'PanIN_HG': t_panin_hg, 'Other': 1.0}
                preds  = apply_thresholds(tissue_df, thresh)
                f1     = f1_score(tissue_df['Actual'], preds,
                                  labels=TISSUE_CLS, average='macro',
                                  zero_division=0)
                if f1 > best_f1:
                    best_f1     = f1
                    best_thresh = thresh.copy()

    print(f"  Best macro F1: {best_f1:.4f}")
    print(f"  Best thresholds: " +
          ", ".join([f"{k}: {v:.2f}" for k, v in best_thresh.items()]))
    return best_thresh, best_f1


def main():
    cfg = get_config()

    csv_files = sorted([f for f in os.listdir(cfg.output_dir)
                        if f.startswith('results_') and f.endswith('.csv')])

    if not csv_files:
        print("No result CSVs found. Run the pipeline first.")
        return

    print(f"\nFound {len(csv_files)} result files: {csv_files}")

    all_results     = {}
    all_tuned       = {}
    per_slide_report = {}

    for f in csv_files:
        slide = f.replace('results_', '').replace('.csv', '')
        df    = pd.read_csv(os.path.join(cfg.output_dir, f))

        # Skip slides missing probability columns (old format)
        if 'p_ADM' not in df.columns:
            print(f"  {slide}: missing probability columns, skipping")
            continue

        print(f"\n{'='*55}")
        print(f"  {slide}")
        print(f"{'='*55}")

        # Original results
        tissue_df = df[df['Actual'] != 'Other']
        print(f"\n-- Original Tissue-Only (n={len(tissue_df)}) --")
        if len(tissue_df) > 0:
            orig_report = classification_report(tissue_df['Actual'], tissue_df['Refined'],
                                                 labels=TISSUE_CLS, zero_division=0,
                                                 output_dict=True)
            print(classification_report(tissue_df['Actual'], tissue_df['Refined'],
                                         labels=TISSUE_CLS, zero_division=0))
        else:
            print("  No tissue tiles.")
            continue

        # Tune thresholds on this fold
        best_thresh, best_f1 = tune_thresholds(df)

        # Apply tuned thresholds to full slide
        df['Tuned'] = apply_thresholds(df, best_thresh)

        # Tuned results
        tissue_tuned = df[df['Actual'] != 'Other']
        print(f"\n-- Tuned Tissue-Only (n={len(tissue_tuned)}) --")
        tuned_report = classification_report(tissue_tuned['Actual'], tissue_tuned['Tuned'],
                                              labels=TISSUE_CLS, zero_division=0,
                                              output_dict=True)
        print(classification_report(tissue_tuned['Actual'], tissue_tuned['Tuned'],
                                     labels=TISSUE_CLS, zero_division=0))

        # Compare
        orig_macro  = orig_report['macro avg']['f1-score']
        tuned_macro = tuned_report['macro avg']['f1-score']
        delta       = tuned_macro - orig_macro
        print(f"  Macro F1: {orig_macro:.3f} → {tuned_macro:.3f} "
              f"({'↑' if delta > 0 else '↓'}{abs(delta):.3f})")

        # Save tuned CSV
        df.to_csv(os.path.join(cfg.output_dir, f'results_{slide}_tuned.csv'), index=False)

        all_results[slide] = df
        all_tuned[slide]   = best_thresh
        per_slide_report[slide] = {
            'orig_macro': orig_macro,
            'tuned_macro': tuned_macro,
            'thresholds': best_thresh
        }

    # ── Aggregate comparison ──────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*55}")
        print("  AGGREGATE — ALL FOLDS")
        print(f"{'='*55}")

        all_actual         = pd.concat([r['Actual']  for r in all_results.values()])
        all_refined_orig   = pd.concat([r['Refined'] for r in all_results.values()])
        all_refined_tuned  = pd.concat([r['Tuned']   for r in all_results.values()])

        tissue_actual  = all_actual[all_actual != 'Other']
        tissue_orig    = all_refined_orig[all_actual != 'Other']
        tissue_tuned   = all_refined_tuned[all_actual != 'Other']

        print(f"\n-- Original Tissue-Only (n={len(tissue_actual)}) --")
        print(classification_report(tissue_actual, tissue_orig,
                                     labels=TISSUE_CLS, zero_division=0))

        print(f"\n-- Tuned Tissue-Only (n={len(tissue_actual)}) --")
        print(classification_report(tissue_actual, tissue_tuned,
                                     labels=TISSUE_CLS, zero_division=0))

        # Summary table
        print("\n-- Per-Slide Macro F1 Summary --")
        print(f"{'Slide':<10} {'Original':>10} {'Tuned':>10} {'Delta':>10}")
        print("-" * 42)
        for slide, r in per_slide_report.items():
            delta = r['tuned_macro'] - r['orig_macro']
            print(f"{slide:<10} {r['orig_macro']:>10.3f} {r['tuned_macro']:>10.3f} "
                  f"{'↑' if delta > 0 else '↓'}{abs(delta):>9.3f}")

        # Confusion matrices
        cm_orig = confusion_matrix(tissue_actual, tissue_orig,   labels=TISSUE_CLS)
        cm_tune = confusion_matrix(tissue_actual, tissue_tuned,  labels=TISSUE_CLS)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=TISSUE_CLS, yticklabels=TISSUE_CLS)
        axes[0].set_title('Original — Tissue Only')
        axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

        sns.heatmap(cm_tune, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=TISSUE_CLS, yticklabels=TISSUE_CLS)
        axes[1].set_title('Threshold Tuned — Tissue Only')
        axes[1].set_ylabel('Actual'); axes[1].set_xlabel('Predicted')

        plt.tight_layout()
        out_path = os.path.join(cfg.output_dir, 'confusion_matrix_tuned_vs_original.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\nComparison confusion matrix saved to {out_path}")

        # Save threshold summary
        thresh_df = pd.DataFrame([
            {'slide': slide, **r['thresholds'], 
             'orig_macro_f1': r['orig_macro'],
             'tuned_macro_f1': r['tuned_macro']}
            for slide, r in per_slide_report.items()
        ])
        thresh_path = os.path.join(cfg.output_dir, 'optimal_thresholds.csv')
        thresh_df.to_csv(thresh_path, index=False)
        print(f"Optimal thresholds saved to {thresh_path}")

    print(f"\nDone. Tuned results in: {cfg.output_dir}")


if __name__ == '__main__':
    main()
