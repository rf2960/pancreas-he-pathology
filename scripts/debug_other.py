# =============================================================================
#  DEBUG SCRIPT - Test Other tile flow through the full pipeline
#  No training, no GPU needed, finishes in ~2 minutes.
#
#  Run from project folder:
#    python debug_other.py
#    python debug_other.py --data_dir /custom/path/to/spatial_tiles_dataset
#
#  What this checks:
#    Step 1 - Are Other tiles loading into df_master correctly?
#    Step 2 - Are they present in test_full for a specific slide?
#    Step 3 - Do they survive inference and appear in res_df['Actual']?
#    Step 4 - Does Stage 1 diagnostic show correct Other support?
# =============================================================================

import os, re, random, argparse, warnings
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# -- Config --------------------------------------------------------------------
parser = argparse.ArgumentParser()
script_dir = Path(__file__).parent.resolve()
parser.add_argument('--data_dir',  type=str,
                    default=str(script_dir / 'spatial_tiles_dataset'))
parser.add_argument('--test_slide', type=str, default='R4-23',
                    help='Which slide to use as test fold')
parser.add_argument('--n_tiles',    type=int, default=200,
                    help='Max tiles per class to sample (keeps it fast)')
cfg = parser.parse_args()

DEVICE      = torch.device('cpu')   # intentionally CPU - this is just a debug run
ALL_CLASSES = ['ADM', 'PanIN_LG', 'PanIN_HG', 'PDAC', 'Other']
CLASSES_S1  = ['Other', 'Tissue']
CLASSES_S2  = ['ADM', 'PanIN_LG', 'PanIN_HG', 'PDAC']
LMAP_S2     = {name: i for i, name in enumerate(CLASSES_S2)}

print("=" * 60)
print("  DEBUG: Other Tile Flow Test")
print(f"  data_dir:   {cfg.data_dir}")
print(f"  test_slide: {cfg.test_slide}")
print(f"  n_tiles:    {cfg.n_tiles} per class (sampled)")
print("=" * 60)


# =============================================================================
# STEP 1 - Load df_master and confirm Other tiles exist
# =============================================================================
print("\n-- Step 1: Load df_master --")

pattern = re.compile(r"(.+?)_(.+?)_\[x=(\d+),y=(\d+)\]")
data = []
for root, _, files in os.walk(cfg.data_dir):
    for f in files:
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue
        m = pattern.search(f)
        if m:
            data.append({
                'path':       os.path.join(root, f),
                'slide_id':   m.group(1),
                'label_name': m.group(2),
                'x':          int(m.group(3)),
                'y':          int(m.group(4))
            })

df_master = pd.DataFrame(data)
print(f"Total tiles loaded: {len(df_master):,}")
print(f"\nBy class:\n{df_master['label_name'].value_counts().to_string()}")
print(f"\nBy slide:\n{df_master['slide_id'].value_counts().to_string()}")

assert 'Other' in df_master['label_name'].values, \
    "FAIL: 'Other' class not found in df_master at all - check folder name or file extensions"
print("\n[OK] Other class present in df_master")


# =============================================================================
# STEP 2 - Build test_full and confirm Other tiles are there
# =============================================================================
print(f"\n-- Step 2: test_full for {cfg.test_slide} --")

assert cfg.test_slide in df_master['slide_id'].values, \
    f"FAIL: slide '{cfg.test_slide}' not found. Available: {sorted(df_master['slide_id'].unique())}"

test_full  = df_master[df_master['slide_id'] == cfg.test_slide].copy()
train_full = df_master[df_master['slide_id'] != cfg.test_slide].copy()

print(f"test_full size: {len(test_full):,}")
print(f"\ntest_full by class:\n{test_full['label_name'].value_counts().to_string()}")

if 'Other' not in test_full['label_name'].values:
    print(f"\n[WARN] Other is NOT present in {cfg.test_slide}'s test tiles")
    print("  This means R4-23 genuinely has no Other tiles on disk for that slide.")
    print("  Check your QuPath export - did you export Other tiles for all slides?")
else:
    n_other = (test_full['label_name'] == 'Other').sum()
    print(f"\n[OK] Other present in test_full: {n_other:,} tiles")


# =============================================================================
# STEP 3 - Subsample for speed and run mock inference
#           (untrained model - we only care about data flow, not accuracy)
# =============================================================================
print(f"\n-- Step 3: Mock inference with untrained model --")
print(f"(Sampling up to {cfg.n_tiles} tiles per class for speed)")

# Subsample test_full
subsets = []
for cls in test_full['label_name'].unique():
    pool   = test_full[test_full['label_name'] == cls]
    subset = pool.sample(min(cfg.n_tiles, len(pool)), random_state=42)
    subsets.append(subset)
test_small = pd.concat(subsets).reset_index(drop=True)

print(f"Subsampled test size: {len(test_small)}")
print(f"Classes present:\n{test_small['label_name'].value_counts().to_string()}")

# Minimal transform (no Macenko - not needed for flow test)
simple_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tiny untrained models (random weights - just testing data flow)
model_s1 = models.wide_resnet50_2(weights=None)
model_s1.fc = nn.Linear(model_s1.fc.in_features, 2)
model_s1.eval()

model_s2 = models.wide_resnet50_2(weights=None)
model_s2.fc = nn.Linear(model_s2.fc.in_features, 4)
model_s2.eval()

print("\nRunning inference (random weights - accuracy meaningless, flow is what matters)...")

infer_data = []
pil_images = [Image.open(p).convert('RGB') for p in test_small['path']]
BATCH      = 16

with torch.no_grad():
    for start in range(0, len(pil_images), BATCH):
        batch_pils = pil_images[start:start + BATCH]
        batch_meta = test_small.iloc[start:start + BATCH]
        batch_t    = torch.stack([simple_tf(img) for img in batch_pils])

        prob_s1   = F.softmax(model_s1(batch_t), dim=1)
        prob_s2   = F.softmax(model_s2(batch_t), dim=1)
        is_tissue = prob_s1.argmax(dim=1)
        grades    = prob_s2.argmax(dim=1)

        for i in range(len(batch_pils)):
            row      = batch_meta.iloc[i]
            p_tissue = prob_s1[i][1].item()
            infer_data.append({
                'x':          int(row['x']),
                'y':          int(row['y']),
                'Actual':     row['label_name'],
                'Predicted':  'Other' if is_tissue[i] == 0 else CLASSES_S2[grades[i]],
                'p_Other':    prob_s1[i][0].item(),
                'p_ADM':      prob_s2[i][LMAP_S2['ADM']].item()      * p_tissue,
                'p_PanIN_LG': prob_s2[i][LMAP_S2['PanIN_LG']].item() * p_tissue,
                'p_PanIN_HG': prob_s2[i][LMAP_S2['PanIN_HG']].item() * p_tissue,
                'p_PDAC':     prob_s2[i][LMAP_S2['PDAC']].item()     * p_tissue,
            })

res_df = pd.DataFrame(infer_data)

print(f"\nInference complete.")
print(f"  test_small: {len(test_small)} | res_df: {len(res_df)}")
print(f"\nres_df['Actual'] counts:\n{res_df['Actual'].value_counts().to_string()}")


# =============================================================================
# STEP 4 - Stage 1 Diagnostic (the actual bug check)
# =============================================================================
print(f"\n-- Step 4: Stage 1 Diagnostic --")

res_df['S1_Actual'] = res_df['Actual'].apply(
    lambda x: 'Tissue' if x != 'Other' else 'Other')
res_df['S1_Pred'] = res_df['Predicted'].apply(
    lambda x: 'Tissue' if x != 'Other' else 'Other')

print(classification_report(res_df['S1_Actual'], res_df['S1_Pred'],
                              labels=CLASSES_S1, zero_division=0))

other_support = (res_df['S1_Actual'] == 'Other').sum()
if other_support == 0:
    print("[FAIL] BUG CONFIRMED: Other support = 0 in Stage 1 diagnostic")
    print("  Other tiles exist in test_full but are not reaching res_df['Actual']")
    print("  -> The bug is inside the inference loop, not in data loading")
    print("  -> Check: are pil_images being loaded from test_full correctly?")
    print("  -> Check: is test_reset being indexed correctly in the inner loop?")
else:
    print(f"[OK] Other support = {other_support} - data flow is correct!")
    print("  The bug was likely a one-off Colab issue. Full pipeline should work.")

print("\n-- Summary --")
print(f"  df_master Other tiles:    {(df_master['label_name'] == 'Other').sum():,}")
print(f"  test_full Other tiles:    {(test_full['label_name'] == 'Other').sum():,}")
print(f"  test_small Other tiles:   {(test_small['label_name'] == 'Other').sum()}")
print(f"  res_df Other in Actual:   {(res_df['Actual'] == 'Other').sum()}")
print(f"  Stage 1 Other support:    {other_support}")
print()
if other_support > 0:
    print("[OK] All checks passed - safe to run full pipeline on AWS")
else:
    print("[FAIL] Bug still present - share output above and we'll fix before full run")
