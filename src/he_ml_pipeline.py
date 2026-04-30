# =============================================================================
#  H&E PANCREAS PATHOLOGY — SINGLE-STAGE PIPELINE v5 (PDAC excluded)
#
#  KEY CHANGE from v3: Collapsed two-stage (S1 filter + S2 grader) into one
#  single 4-class model predicting: ADM / PanIN_LG / PanIN_HG / Other
#
#  PDAC excluded: 66% of PDAC tiles are in one slide (R4-25), making
#  leave-one-slide-out evaluation of PDAC statistically unfair.
#
#  Other fixes retained:
#    - Per-fold Macenko normalization (normalize to test slide each fold)
#    - Weighted Focal Loss for all 5 classes
#    - Slide-stratified balanced sampler
#    - MixUp augmentation
#    - 8-pass TTA inference
#    - Soft spatial consensus
#    - Tissue-only reporting as primary metric
# =============================================================================

import os, re, random, warnings, argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms

import torchstain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG
# =============================================================================

def get_config():
    script_dir = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',   type=str, default=str(script_dir / 'spatial_tiles_dataset'))
    parser.add_argument('--output_dir', type=str, default=str(script_dir / 'pipeline_outputs'))

    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--epochs',       type=int,   default=30)
    parser.add_argument('--patience',     type=int,   default=6)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--mixup_alpha',  type=float, default=0.3)

    parser.add_argument('--confidence_threshold', type=float, default=0.80)
    parser.add_argument('--spatial_window',       type=int,   default=2)

    parser.add_argument('--resume', action='store_true')

    return parser.parse_args()


# =============================================================================
# DEVICE
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Apple MPS GPU")
    else:
        device = torch.device('cpu')
        print("CPU only")
    return device


def get_dataloader_workers():
    return 0 if os.name == 'nt' else 4


# =============================================================================
# STAIN NORMALIZATION — per-fold, normalize TO test slide
# =============================================================================

def fit_macenko_normalizer(df_master, ref_slide_id, n_fit=10):
    ref_paths = df_master[df_master['slide_id'] == ref_slide_id]['path'].tolist()
    if not ref_paths:
        raise ValueError(f"Slide '{ref_slide_id}' not found.")
    sample_paths = random.sample(ref_paths, min(n_fit, len(ref_paths)))
    normalizer   = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    to_tensor    = transforms.ToTensor()
    fitted = 0
    for path in sample_paths:
        try:
            normalizer.fit(to_tensor(Image.open(path).convert('RGB')))
            fitted += 1
        except Exception:
            continue
    print(f"  Macenko normalizer fitted on {fitted} tiles from '{ref_slide_id}'")
    return normalizer


class MacenkoTransform:
    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.to_tensor  = transforms.ToTensor()

    def __call__(self, pil_img):
        try:
            t = self.to_tensor(pil_img)
            t_norm, _, _ = self.normalizer.normalize(t, stains=False)
            arr = (t_norm.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        except Exception:
            return pil_img


def build_train_transform(macenko_tf):
    return transforms.Compose([
        macenko_tf,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.03),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def build_val_transform(macenko_tf):
    return transforms.Compose([
        macenko_tf,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def build_tta_transforms(macenko_tf):
    norm = [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return [
        transforms.Compose([macenko_tf] + norm),
        transforms.Compose([macenko_tf, transforms.RandomHorizontalFlip(p=1)] + norm),
        transforms.Compose([macenko_tf, transforms.RandomVerticalFlip(p=1)]   + norm),
        transforms.Compose([macenko_tf, transforms.Lambda(lambda x: x.rotate(90))]  + norm),
        transforms.Compose([macenko_tf, transforms.Lambda(lambda x: x.rotate(180))] + norm),
        transforms.Compose([macenko_tf, transforms.Lambda(lambda x: x.rotate(270))] + norm),
        transforms.Compose([macenko_tf, transforms.RandomHorizontalFlip(p=1),
                             transforms.RandomVerticalFlip(p=1)] + norm),
        transforms.Compose([macenko_tf,
                             transforms.Lambda(lambda x: x.transpose(Image.TRANSPOSE))] + norm),
    ]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_master_df(data_dir):
    pattern = re.compile(r"(.+?)_(.+?)_\[x=(\d+),y=(\d+)\]")
    data    = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue
            m = pattern.search(f)
            if m:
                data.append({'path': os.path.join(root, f),
                             'slide_id':   m.group(1),
                             'label_name': m.group(2),
                             'x': int(m.group(3)),
                             'y': int(m.group(4))})
    if not data:
        raise FileNotFoundError(f"No tiles found in '{data_dir}'.")
    df = pd.DataFrame(data)
    print(f"\nLoaded {len(df):,} tiles across {df['slide_id'].nunique()} slides")
    df = df[df['label_name'] != 'PDAC'].copy()
    print(f"Note: PDAC excluded (insufficient cross-slide representation).")
    print(f"Remaining: {len(df):,} tiles")
    print(df['label_name'].value_counts().to_string())
    return df


# =============================================================================
# CLASS WEIGHTS — inverse frequency
# =============================================================================

def compute_class_weights(df, label_col, class_names, device):
    n_total   = len(df)
    n_classes = len(class_names)
    weights   = []
    for cls in class_names:
        n_cls = (df[label_col] == cls).sum()
        weights.append(n_total / (n_classes * n_cls) if n_cls > 0 else 1.0)
    w_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print("  Class weights: " +
          ", ".join([f"{c}: {w:.3f}" for c, w in zip(class_names, weights)]))
    return w_tensor


# =============================================================================
# SLIDE-STRATIFIED BALANCED SAMPLER
# =============================================================================

class SlideStratifiedSampler(Sampler):
    def __init__(self, df, n_samples_per_epoch=None):
        self.df    = df.reset_index(drop=True)
        self.index = defaultdict(lambda: defaultdict(list))
        for i, row in self.df.iterrows():
            self.index[row['slide_id']][row['label_name']].append(i)
        self.slides = list(self.index.keys())
        self.n      = n_samples_per_epoch or len(df)

    def __len__(self):
        return self.n

    def __iter__(self):
        samples   = []
        per_slide = self.n // len(self.slides)
        for slide in self.slides:
            classes   = list(self.index[slide].keys())
            per_class = max(1, per_slide // len(classes))
            for cls in classes:
                pool = self.index[slide][cls]
                if pool:
                    samples.extend(random.choices(pool, k=per_class))
        random.shuffle(samples)
        return iter(samples[:self.n])


# =============================================================================
# DATASET
# =============================================================================

class TileDataset(Dataset):
    def __init__(self, df, transform, label_map):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row['path']).convert('RGB')
        img   = self.transform(img)
        label = self.label_map[row['label_name']]
        return img, label, int(row['x']), int(row['y'])


# =============================================================================
# MIXUP
# =============================================================================

def mixup_batch(imgs, labels, alpha):
    if alpha <= 0:
        return imgs, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], labels, labels[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# MODEL
# =============================================================================

class PathologySpecialist(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_feats, 512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 128),      nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# =============================================================================
# WEIGHTED FOCAL LOSS
# =============================================================================

class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=2, label_smoothing=0.1):
        super().__init__()
        self.class_weights   = class_weights
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n = inputs.shape[1]
        if self.label_smoothing > 0:
            smooth_targets = torch.full_like(inputs, self.label_smoothing / (n - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce_loss = -(smooth_targets * F.log_softmax(inputs, dim=1)).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt            = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        focal_weight  = (1 - pt) ** self.gamma
        sample_weight = self.class_weights[targets]
        return (sample_weight * focal_weight * ce_loss).mean()


# =============================================================================
# TRAINING
# =============================================================================

def train_val_split(df, val_fraction=0.15):
    train_rows, val_rows = [], []
    for slide in df['slide_id'].unique():
        for cls in df['label_name'].unique():
            subset = df[(df['slide_id'] == slide) & (df['label_name'] == cls)]
            if len(subset) == 0:
                continue
            n_val     = max(1, int(len(subset) * val_fraction))
            val_idx   = subset.sample(n=n_val, random_state=42).index
            train_idx = subset.index.difference(val_idx)
            val_rows.extend(val_idx.tolist())
            train_rows.extend(train_idx.tolist())
    return df.loc[train_rows].copy(), df.loc[val_rows].copy()


def train_model(model, train_loader, val_loader, criterion,
                epochs, device, cfg):
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_loss, best_state, patience_count = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels, _, _ in tqdm(train_loader,
                                        desc=f"Epoch {epoch+1}/{epochs}",
                                        leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            if cfg.mixup_alpha > 0:
                imgs, la, lb, lam = mixup_batch(imgs, labels, cfg.mixup_alpha)
                optimizer.zero_grad()
                loss = mixup_criterion(criterion, model(imgs), la, lb, lam)
            else:
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_loss    += criterion(model(imgs), labels).item()
        avg_val = val_loss / len(val_loader)

        print(f"  Epoch {epoch+1:2d}/{epochs} "
              f"| Train: {avg_train:.4f} | Val: {avg_val:.4f} "
              f"| LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_val < best_val_loss:
            best_val_loss  = avg_val
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"  Early stopping (best val: {best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    return model


# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def tta_predict_paths(model, paths, tta_transforms, device):
    all_probs = []
    for tf in tta_transforms:
        imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths]).to(device)
        all_probs.append(F.softmax(model(imgs), dim=1))
    return torch.stack(all_probs).mean(dim=0)


# =============================================================================
# SPATIAL CONSENSUS
# =============================================================================

def soft_spatial_consensus(res_df, class_cols, confidence_threshold, window):
    tile_size    = 256
    coord_to_idx = {(r['x'], r['y']): i for i, r in res_df.iterrows()}
    prob_matrix  = res_df[class_cols].values
    max_probs    = prob_matrix.max(axis=1)
    refined      = []
    for i, row in res_df.iterrows():
        if max_probs[i] >= confidence_threshold:
            refined.append(class_cols[prob_matrix[i].argmax()])
            continue
        neighbors = [prob_matrix[i]]
        for dx in range(-window, window + 1):
            for dy in range(-window, window + 1):
                if dx == 0 and dy == 0:
                    continue
                j = coord_to_idx.get((row['x'] + dx * tile_size,
                                      row['y'] + dy * tile_size))
                if j is not None:
                    neighbors.append(prob_matrix[j])
        refined.append(class_cols[np.stack(neighbors).mean(axis=0).argmax()])
    return pd.Series(refined, index=res_df.index)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

ALL_CLASSES = ['ADM', 'PanIN_LG', 'PanIN_HG', 'Other']
CLASSES_S2  = ['ADM', 'PanIN_LG', 'PanIN_HG']
LMAP        = {name: i for i, name in enumerate(ALL_CLASSES)}


def run_pipeline(cfg, device):
    os.makedirs(cfg.output_dir, exist_ok=True)
    nw = get_dataloader_workers()
    pm = device.type == 'cuda'
    print(f"DataLoader workers: {nw} | pin_memory: {pm}")

    df_master = load_master_df(cfg.data_dir)
    slides    = df_master['slide_id'].unique()

    if cfg.resume:
        done   = [f.replace('results_', '').replace('.csv', '')
                  for f in os.listdir(cfg.output_dir) if f.startswith('results_')]
        slides = [s for s in slides if s not in done]
        if done:
            print(f"Resuming -- skipping: {done}")

    all_results = {}

    for test_slide in slides:
        print(f"\n{'='*60}")
        print(f"  FOLD: {test_slide}")
        print(f"{'='*60}")

        train_full = df_master[df_master['slide_id'] != test_slide].copy()
        test_full  = df_master[df_master['slide_id'] == test_slide].copy()
        print(f"  Train: {len(train_full):,} | Test: {len(test_full):,}")
        print(f"  Test breakdown:\n{test_full['label_name'].value_counts().to_string()}")

        # Cap Other tiles to 3x tissue count to reduce imbalance during training
        tissue_train = train_full[train_full['label_name'] != 'Other']
        other_train  = train_full[train_full['label_name'] == 'Other']
        max_other    = min(len(other_train), len(tissue_train) * 3)
        other_sampled = other_train.sample(n=max_other, random_state=42)
        train_full   = pd.concat([tissue_train, other_sampled]).reset_index(drop=True)
        print(f"  Training balance — Tissue: {len(tissue_train):,} | Other: {max_other:,} (capped)")

        # Per-fold Macenko: normalize everything to match the test slide
        print(f"\n  Normalizing to test slide: {test_slide}")
        normalizer = fit_macenko_normalizer(df_master, test_slide)
        macenko_tf = MacenkoTransform(normalizer)
        train_tf   = build_train_transform(macenko_tf)
        val_tf     = build_val_transform(macenko_tf)
        tta_tfs    = build_tta_transforms(macenko_tf)

        # Single 4-class model: ADM / PanIN_LG / PanIN_HG / Other
        print("\n Single-Stage 5-Class Classifier")
        train_full['label'] = train_full['label_name'].map(LMAP)
        tr_df, val_df       = train_val_split(train_full)

        weights   = compute_class_weights(train_full, 'label_name', ALL_CLASSES, device)
        criterion = WeightedFocalLoss(weights, gamma=4, label_smoothing=0.1)

        model = PathologySpecialist(4).to(device)
        model = train_model(
            model,
            DataLoader(TileDataset(tr_df,  train_tf, LMAP), cfg.batch_size,
                       sampler=SlideStratifiedSampler(tr_df),
                       num_workers=nw, pin_memory=pm),
            DataLoader(TileDataset(val_df, val_tf,   LMAP), cfg.batch_size,
                       shuffle=False, num_workers=nw),
            criterion, cfg.epochs, device, cfg
        )

        # Inference
        print("\n Inference (TTA + Soft Spatial Consensus)...")
        model.eval()
        test_reset = test_full.reset_index(drop=True)
        infer_data = []

        for start in tqdm(range(0, len(test_reset), cfg.batch_size), desc='Inference'):
            batch_meta  = test_reset.iloc[start:start + cfg.batch_size]
            batch_paths = batch_meta['path'].tolist()
            probs       = tta_predict_paths(model, batch_paths, tta_tfs, device)

            for i in range(len(batch_paths)):
                row = batch_meta.iloc[i]
                infer_data.append({
                    'x':          int(row['x']),
                    'y':          int(row['y']),
                    'Actual':     row['label_name'],
                    'Predicted':  ALL_CLASSES[probs[i].argmax().item()],
                    'p_ADM':      probs[i][LMAP['ADM']].item(),
                    'p_PanIN_LG': probs[i][LMAP['PanIN_LG']].item(),
                    'p_PanIN_HG': probs[i][LMAP['PanIN_HG']].item(),
                    'p_Other':    probs[i][LMAP['Other']].item(),
                    'Conf':       probs[i].max().item(),
                })

        res_df = pd.DataFrame(infer_data)

        # Spatial consensus
        class_cols = ALL_CLASSES
        prob_cols  = ['p_ADM', 'p_PanIN_LG', 'p_PanIN_HG', 'p_Other']
        prob_df    = res_df[['x', 'y'] + prob_cols].copy()
        prob_df.columns = ['x', 'y'] + class_cols
        res_df['Refined'] = soft_spatial_consensus(
            prob_df, class_cols, cfg.confidence_threshold, cfg.spatial_window).values

        # Reports
        print(f"\n-- Final Report ({test_slide}) — All Classes --")
        print(classification_report(res_df['Actual'], res_df['Refined'],
                                     labels=ALL_CLASSES, zero_division=0))

        tissue_df = res_df[res_df['Actual'] != 'Other']
        print(f"\n-- Tissue-Only ({test_slide}) — n={len(tissue_df)} — PRIMARY METRIC --")
        if len(tissue_df) > 0:
            print(classification_report(tissue_df['Actual'], tissue_df['Refined'],
                                         labels=CLASSES_S2, zero_division=0))

        # Save
        res_df.to_csv(os.path.join(cfg.output_dir, f'results_{test_slide}.csv'), index=False)
        torch.save(model.state_dict(),
                   os.path.join(cfg.output_dir, f'model_{test_slide}.pth'))
        print(f"  Saved to {cfg.output_dir}")

        all_results[test_slide] = res_df
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Aggregate
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  AGGREGATE RESULTS — ALL FOLDS")
        print(f"{'='*60}")
        all_actual  = pd.concat([r['Actual']  for r in all_results.values()])
        all_refined = pd.concat([r['Refined'] for r in all_results.values()])

        print("\n-- All Classes --")
        print(classification_report(all_actual, all_refined,
                                     labels=ALL_CLASSES, zero_division=0))

        tissue_actual  = all_actual[all_actual  != 'Other']
        tissue_refined = all_refined[all_actual != 'Other']
        print(f"\n-- Tissue-Only (n={len(tissue_actual)}) — PRIMARY METRIC --")
        print(classification_report(tissue_actual, tissue_refined,
                                     labels=CLASSES_S2, zero_division=0))

        # Tissue confusion matrix
        cm = confusion_matrix(tissue_actual, tissue_refined, labels=CLASSES_S2)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES_S2, yticklabels=CLASSES_S2)
        plt.title('Aggregate Confusion Matrix — Tissue Only')
        plt.ylabel('Actual'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, 'confusion_matrix_tissue.png'), dpi=150)
        plt.close()

        # Full confusion matrix
        cm_full = confusion_matrix(all_actual, all_refined, labels=ALL_CLASSES)
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues',
                    xticklabels=ALL_CLASSES, yticklabels=ALL_CLASSES)
        plt.title('Aggregate Confusion Matrix — All Classes')
        plt.ylabel('Actual'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, 'confusion_matrix_all.png'), dpi=150)
        plt.close()
        print(f"Confusion matrices saved to {cfg.output_dir}")

    print(f"\nDone. All outputs in: {cfg.output_dir}")
    return all_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    cfg    = get_config()
    device = get_device()
    print("\n-- Config --")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
    print(f"  device: {device}\n")
    run_pipeline(cfg, device)