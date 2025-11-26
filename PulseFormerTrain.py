# train.py
# -*- coding: utf-8 -*-
"""
Multi-Modal ECG Analysis: Hybrid ViT + Gated GNN + Curriculum Learning
======================================================================

1. SYSTEM OVERVIEW
------------------
This script trains a state-of-the-art hierarchical classifier for 19 cardiovascular conditions.
It utilizes a "Dual-Stream" architecture that fuses raw ECG images with Continuous Wavelet 
Transform (CWT) time-frequency maps.

2. ARCHITECTURAL INTUITION
--------------------------
A. The Backbone: Hybrid CNN-ViT (ConvStemViT)
   - PROBLEM: Standard Vision Transformers (ViT) are bad at low-level edge detection (crucial for ECG spikes).
   - SOLUTION: We use a convolutional "Stem" (3-layer CNN) to extract rich texture features first.
     The ViT then treats these feature maps as tokens.
   - RESULT: The model captures both local wave morphology (CNN) and global rhythm context (ViT).

B. The Fusion: Cross-Attention
   - Instead of simple concatenation, the ECG and CWT streams "talk" to each other via Multi-Head Attention.
   - The ECG stream queries the CWT stream to find relevant frequency anomalies that match the spatial features.

C. The Reasoning: Hierarchical Gating (The "Brain")
   - The model has a specific "Superclass Head" that predicts broad categories (MI, CD, HYP, STTC).
   - These probabilities are used to create a "Soft Gate" (Sigmoid Mask).
   - INTUITION: If the model predicts "MI" with 90% confidence, the Gate amplifies MI-related features 
     in the embedding vector and suppresses irrelevant features before the final classification.

D. The Output: LabelGNN (Graph Neural Network)
   - Diseases are not independent (e.g., 'Anterior MI' is a child of 'MI').
   - We use a Graph Attention Network (GATv2) to learn embeddings for the labels themselves.
   - The final prediction is the Dot Product of the [Patient Feature] and the [Label Graph Embeddings].

3. TRAINING DYNAMICS: 3-Stage Curriculum
----------------------------------------
To prevent the model from getting confused by complex subclasses early on, we use Curriculum Learning:
   
   • STAGE 1 (Epochs 1-5):  "Triage Mode"
     - Task: Binary Classification (Normal vs. Abnormal).
     - Goal: Learn basic signal quality and healthy waveforms.
     
   • STAGE 2 (Epochs 6-20): "Differential Diagnosis"
     - Task: Learn the 4 Superclasses (MI, CD, Hypertrophy, ST-Changes).
     - Goal: Learn to distinguish major pathology groups.
     - Loss: Heavy focus on the 'Superclass Head'.
     
   • STAGE 3 (Epochs 21+):  "Full Diagnosis"
     - Task: Predict all 19 Subclasses.
     - Mechanism: The 'Gated' features are fully activated.
     - Sampler: We switch to a specialized 'Subclass Sampler' to oversample rare diseases.

4. AUGMENTATION STRATEGY
------------------------
   - "Augmentation as Prior": We apply Affine transforms (rotation/scaling) and Gaussian Noise 
     to BOTH Train and Validation sets.
   - WHY? ECG signals in the real world are noisy and imperfectly scaled. By testing on augmented data,
     we ensure the model learns robust invariant features, not just memorization of clean datasets.

Author: Sai Koundinya Upadhyayula (and Gemini :/)
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

import torchvision.transforms as T
import timm
from torch_geometric.nn import GATv2Conv

# ───────────────────────── Config ─────────────────────────
CONFIG = {
    'ROOT_DIR': r"C:\Users\uskou",  # <--- UPDATE THIS PATH
    'RESULTS_DIR': "results_hybrid_no_aux",
    'EPOCHS': 50,
    'STAGE1_END': 5,
    'STAGE2_END': 20,
    'BATCH_SIZE': 128,
    'LR': 1e-4,
    'WD': 5e-2,           
    'IMG_SIZE': 224,
    'SEED': 42,
    'NUM_WORKERS': 4
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Taxonomy
SUPER = ['MI', 'CD', 'HYP', 'STTC']
SUB   = [
    'sub_IMI', 'sub_LMI', 'sub_AMI', 'sub_PMI', 'sub_LBBB', 'sub_RBBB',
    'sub_AVB', 'sub_WPW', 'sub_IVCD', 'sub_STTCs', 'sub_LVH', 'sub_RVH',
    'sub_ISC', 'sub_NST'
]
ALL_NAMES = ['NORM'] + SUPER + SUB
NUM_LABELS = len(ALL_NAMES)

# Indices
NORM_IDX = 0
MI_IDX, CD_IDX, HYP_IDX, STTC_IDX = 1, 2, 3, 4
IDX_MAP = {
    'MI': list(range(5, 9)),
    'CD': list(range(9, 15)),
    'HYP': list(range(15, 17)),
    'STTC': list(range(17, 19))
}
SUPER_IDXS = [MI_IDX, CD_IDX, HYP_IDX, STTC_IDX]
SUB_IDXS = [item for sublist in IDX_MAP.values() for item in sublist]

# ───────────────────────── Utilities ──────────────────────────
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

def get_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        # Tweak: Slightly stronger augmentation to combat overfitting
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        T.ToTensor(),
        AddGaussianNoise(mean=0., std=0.03) 
    ])

# ───────────────────────── Dataset ──────────────────────────
class ECGCWTDataset(Dataset):
    GROUP_MAP = {
        'sub_LBBB': ['sub_CLBBB','sub_ILBBB','sub_LAFB/LPFB'],
        'sub_RBBB': ['sub_CRBBB','sub_IRBBB'],
        'sub_LVH':  ['sub_LVH','sub_SEHYP','sub_LAO/LAE'],
        'sub_RVH':  ['sub_RVH','sub_RAO/RAE'],
        'sub_ISC':  ['sub_ISCA','sub_ISC_','sub_ISCI']
    }
    DIRECT_MAP = {
        'sub_IMI':'sub_IMI', 'sub_LMI':'sub_LMI', 'sub_AMI':'sub_AMI', 
        'sub_PMI':'sub_PMI', 'sub_AVB':'sub__AVB', 'sub_WPW':'sub_WPW', 
        'sub_IVCD':'sub_IVCD', 'sub_STTCs':'sub_STTCs', 'sub_NST':'sub_NST_'
    }

    def __init__(self, csv_path, ecg_dir, cwt_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.ecg_dir = ecg_dir
        self.cwt_dir = cwt_dir
        self.transform = transform
        self.labels = self._process_labels()

    def _process_labels(self):
        df_lbl = pd.DataFrame()
        df_lbl['NORM'] = self.df['NORM']
        for s in SUPER: df_lbl[s] = self.df.get(s, 0)
        for new_k, old_list in self.GROUP_MAP.items():
            existing = [c for c in old_list if c in self.df.columns]
            df_lbl[new_k] = (self.df[existing].sum(axis=1) > 0).astype(float) if existing else 0.0
        for new_k, old_k in self.DIRECT_MAP.items():
            df_lbl[new_k] = self.df.get(old_k, 0.0)
        for col in ALL_NAMES:
            if col not in df_lbl: df_lbl[col] = 0.0
        return torch.tensor(df_lbl[ALL_NAMES].values, dtype=torch.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        pid = str(self.df.iloc[i]['ecg_id'])
        ecg = Image.open(os.path.join(self.ecg_dir, f"{pid}.jpg")).convert('RGB')
        cwt = Image.open(os.path.join(self.cwt_dir, f"{pid}.jpg")).convert('RGB')
        return self.transform(ecg), self.transform(cwt), self.labels[i]

def collate_fn(b):
    e, c, l = zip(*b)
    return torch.stack(e), torch.stack(c), torch.stack(l)

def subclass_sampler(ds, ep, s3_start, s3_end, lambda_max=0.65, gamma=0.5):
    if ep < s3_start: return None
    frac = max(0, min(1, (ep - s3_start) / max(1, s3_end - s3_start)))
    current_lambda = lambda_max * (1 - frac)
    
    labs = ds.labels[:, SUB_IDXS]
    pos_counts = labs.sum(0)
    total_pos = pos_counts.sum()
    if total_pos == 0: return None
    
    class_weights = (total_pos / (pos_counts + 1e-6)).pow(gamma)
    sample_weights = (labs * class_weights).sum(1)
    sample_weights[sample_weights == 0] = 1.0
    
    final_weights = (1 - current_lambda) * torch.ones_like(sample_weights) + current_lambda * sample_weights
    return WeightedRandomSampler(final_weights.tolist(), num_samples=len(ds), replacement=True)

# ───────────────────────── Architecture ────────────────────────
# Static Edge Index
edges = []
for idx_list, center in zip([IDX_MAP['MI'], IDX_MAP['CD'], IDX_MAP['HYP'], IDX_MAP['STTC']], SUPER_IDXS):
    for node in idx_list:
        edges.extend([(center, node), (node, center)])
EDGE_INDEX = torch.tensor(edges, dtype=torch.long).t().contiguous()

class ConvStemViT(nn.Module):
    def __init__(self, name="vit_small_patch16_224"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), 
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.vit = timm.create_model(name, pretrained=True, in_chans=64, 
                                   img_size=(112, 112), num_classes=0, global_pool="")
    
    def forward(self, x):
        return self.vit.forward_features(self.stem(x))

class LabelGNN(nn.Module):
    def __init__(self, n_nodes, dim, edge_index):
        super().__init__()
        self.register_buffer('edge_index', edge_index)
        self.embed = nn.Parameter(torch.randn(n_nodes, dim))
        self.g1 = GATv2Conv(dim, dim // 4, heads=4)
        self.g2 = GATv2Conv(dim, dim, heads=1, concat=True)
    
    def forward(self):
        x = self.g1(self.embed, self.edge_index).relu()
        return self.g2(x, self.edge_index)

class ECGLabelGNNClassifier(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        # Backbones
        self.vit_e = ConvStemViT()
        self.vit_c = ConvStemViT()
        
        # Cross Attention
        self.ca_e2c = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.ca_c2e = nn.MultiheadAttention(dim, 8, batch_first=True)
        
        # Fusion
        self.fuse = nn.Sequential(nn.LayerNorm(dim*2), nn.Linear(dim*2, dim), nn.GELU(), nn.Dropout(0.5)) # Increased Dropout
        
        # Heads
        self.superclass_head = nn.Linear(dim, len(SUPER_IDXS))
        self.condition_gate = nn.Linear(len(SUPER_IDXS), dim)
        self.projection_head = nn.Linear(dim, dim)
        self.label_gnn = LabelGNN(NUM_LABELS, dim, EDGE_INDEX)
        
        # REMOVED: aux_e and aux_c

    def forward(self, ecg, cwt):
        # Features
        fe = self.vit_e(ecg).unsqueeze(1).squeeze(2) # Ensure [B, Tokens, Dim]
        fc = self.vit_c(cwt).unsqueeze(1).squeeze(2)
        
        # Attention
        fe_attn, _ = self.ca_e2c(fe, fc, fc)
        fc_attn, _ = self.ca_c2e(fc, fe, fe)
        
        # CLS Tokens (Index 0)
        cls_e, cls_c = fe_attn[:, 0], fc_attn[:, 0]
        
        # Fusion
        fused = self.fuse(torch.cat([cls_e, cls_c], 1))
        
        # Superclass Pred
        super_logits = self.superclass_head(fused)
        super_probs = torch.sigmoid(super_logits)
        
        # Gating
        gate = torch.sigmoid(self.condition_gate(super_probs))
        f_conditioned = fused + (fused * gate)
        
        # Final Projection
        f_projected = F.relu(self.projection_head(f_conditioned))
        label_embeds = self.label_gnn()
        
        final_logits = f_projected @ label_embeds.t()
        
        # REMOVED: Return of aux_e, aux_c
        return final_logits, super_logits

# ───────────────────────── Losses ──────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, g=2., pw=None, red='mean'):
        super().__init__()
        self.g, self.pw, self.red = g, pw, red
    def forward(self, x, y):
        bce = F.binary_cross_entropy_with_logits(x, y, weight=self.pw, reduction='none')
        loss = ((1 - torch.exp(-bce)) ** self.g) * bce
        return loss.mean() if self.red == 'mean' else loss

class ConditionalHierarchicalLoss(nn.Module):
    def __init__(self, pw_full, gamma=2.0):
        super().__init__()
        # Removed aux_w argument
        self.loss_super = FocalLoss(g=gamma, pw=pw_full[SUPER_IDXS])
        self.loss_main = FocalLoss(g=gamma, pw=pw_full)

    def forward(self, final_logits, super_logits, labels):
        # Simplified: Just Superclass Loss + Final GNN Loss
        l_s = self.loss_super(super_logits, labels[:, SUPER_IDXS])
        l_m = self.loss_main(final_logits, labels)
        return l_s + l_m

# ───────────────────────── Training Engine ──────────────────────────
def run_epoch(m, ldr, crit, opt=None, scaler=None, desc=""):
    is_train = opt is not None
    m.train() if is_train else m.eval()
    total_loss, n_samples = 0., 0
    all_logits, all_labels = [], []
    
    pbar = tqdm(ldr, desc=f"{desc:^7}", ncols=100, leave=False)
    for ecg, cwt, y in pbar:
        ecg, cwt, y = ecg.to(DEVICE), cwt.to(DEVICE), y.to(DEVICE)
        if is_train: opt.zero_grad()
        
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # UPDATED: Only expects 2 return values
                f_logits, s_logits = m(ecg, cwt)
                # UPDATED: Loss function call
                loss = crit(f_logits, s_logits, labels=y)
            
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
        
        bs = y.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        all_logits.append(f_logits.detach().cpu())
        all_labels.append(y.cpu())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_samples, torch.cat(all_logits), torch.cat(all_labels)

def compute_and_print_metrics(logits, labels, epoch, phase_name):
    """
    Computes and prints detailed metrics:
    - Global Accuracy (Hamming): % of individual labels predicted correctly across all patients.
    - Subset Accuracy (Exact Match): % of patients where the ENTIRE diagnosis (all 19 labels) is correct.
    """
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    targets = labels.numpy()
    
    print(f"\n{'-'*30} EPOCH {epoch} {phase_name} REPORT {'-'*30}")
    
    # --- 1. GLOBAL METRICS ---
    # Global Accuracy (Hamming Score): The accuracy over all N x 19 predictions flattened.
    global_acc = accuracy_score(targets.ravel(), preds.ravel())
    
    # Exact Match (Subset Accuracy): Strict metric. Patient is correct ONLY if all 19 labels match.
    subset_acc = accuracy_score(targets, preds)
    
    # Macro Metrics
    try: macro_auc = roc_auc_score(targets, probs, average='macro')
    except: macro_auc = 0.5
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    
    print(f"OVERALL ACCURACY (Label-wise): {global_acc:.4f}  <-- (This is the main 'Accuracy')")
    print(f"PERFECT MATCH ACC (Subset):    {subset_acc:.4f}  <-- (Strict diagnosis match)")
    print(f"Macro AUC: {macro_auc:.4f} | Macro F1: {macro_f1:.4f}\n")
    
    # --- 2. CLASS-WISE TABLE ---
    print(f"{'CLASS':<12} | {'AUC':<6} | {'F1':<6} | {'ACC':<6} | {'POS':<4}")
    print("-" * 45)
    
    for i, name in enumerate(ALL_NAMES):
        # Handle AUC edge cases (if a class is missing in the batch)
        try: cls_auc = roc_auc_score(targets[:, i], probs[:, i])
        except ValueError: cls_auc = 0.5
            
        cls_f1 = f1_score(targets[:, i], preds[:, i], zero_division=0)
        cls_acc = accuracy_score(targets[:, i], preds[:, i])
        pos_count = int(targets[:, i].sum())
        
        print(f"{name:<12} | {cls_auc:.4f} | {cls_f1:.4f} | {cls_acc:.4f} | {pos_count:<4}")
        
    print(f"{'-'*45}\n")
    
    return macro_f1
    
# ───────────────────────── Main ──────────────────────────
if __name__ == "__main__":
    # 1. Setup & Reproducibility
    seed_everything(CONFIG['SEED'])
    os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)
    
    # Define paths for saving
    CKPT_PATH = os.path.join(CONFIG['RESULTS_DIR'], "last_checkpoint.pth")
    BEST_PATH = os.path.join(CONFIG['RESULTS_DIR'], "model_best.pth")

    print(f"Device: {DEVICE} | Epochs: {CONFIG['EPOCHS']} | Aux Heads: REMOVED")

    # 2. Data Preparation
    tf_pipeline = get_transforms(CONFIG['IMG_SIZE'])
    train_ds = ECGCWTDataset(f"{CONFIG['ROOT_DIR']}/train_meta.csv", f"{CONFIG['ROOT_DIR']}/ECG_train", f"{CONFIG['ROOT_DIR']}/CWTfast_train", tf_pipeline)
    val_ds = ECGCWTDataset(f"{CONFIG['ROOT_DIR']}/val_meta.csv", f"{CONFIG['ROOT_DIR']}/ECG_val", f"{CONFIG['ROOT_DIR']}/CWTfast_val", tf_pipeline)
    
    # Note: Pin_memory=True speeds up transfer to GPU
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, 
                           num_workers=CONFIG['NUM_WORKERS'], pin_memory=True, collate_fn=collate_fn)

    # 3. Model Initialization
    model = ECGLabelGNNClassifier().to(DEVICE)
    
    # 4. Optimizer & Scheduler
    opt = AdamW([
        {'params': model.vit_e.parameters(), 'lr': CONFIG['LR'] / 20},
        {'params': model.vit_c.parameters(), 'lr': CONFIG['LR'] / 20},
        {'params': model.ca_e2c.parameters()}, {'params': model.ca_c2e.parameters()},
        {'params': model.fuse.parameters()}, {'params': model.superclass_head.parameters()},
        {'params': model.condition_gate.parameters()}, 
        {'params': model.projection_head.parameters()}, {'params': model.label_gnn.parameters()},
    ], lr=CONFIG['LR'], weight_decay=CONFIG['WD'])
    
    scaler = GradScaler()
    sched = CosineAnnealingLR(opt, T_max=CONFIG['EPOCHS'], eta_min=1e-6)

    # 5. Loss Setup
    pos_counts = train_ds.labels.sum(0)
    pos_w = (len(train_ds) - pos_counts) / (pos_counts + 1e-6)
    pos_w = torch.clamp(pos_w, 1.0, 100.0).to(DEVICE)

    s1_crit = lambda fl, *a, labels: F.binary_cross_entropy_with_logits(fl[:, [NORM_IDX]], labels[:, [NORM_IDX]], pos_weight=pos_w[[NORM_IDX]])
    s2_focal_norm = FocalLoss(pw=pos_w[[NORM_IDX]])
    s2_focal_super = FocalLoss(pw=pos_w[SUPER_IDXS])
    
    def s2_crit(fl, sl, labels):
        return s2_focal_norm(fl[:, [NORM_IDX]], labels[:, [NORM_IDX]]) + \
               s2_focal_super(sl, labels[:, SUPER_IDXS]) + \
               0.5 * s2_focal_super(fl[:, SUPER_IDXS], labels[:, SUPER_IDXS])

    s3_crit = ConditionalHierarchicalLoss(pos_w)

    # ───────────────── RESUME LOGIC ─────────────────
    start_epoch = 1
    best_loss = float('inf')

    if os.path.isfile(CKPT_PATH):
        print(f">>> Resuming from checkpoint: {CKPT_PATH}")
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        scaler.load_state_dict(checkpoint['scaler'])
        sched.load_state_dict(checkpoint['sched'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f">>> Resuming at Epoch {start_epoch} (Best Loss so far: {best_loss:.4f})")
    else:
        print(">>> No checkpoint found. Starting fresh training.")

    # ───────────────── TRAINING LOOP ─────────────────
    s3_start = CONFIG['STAGE2_END'] + 1
    
    for ep in range(start_epoch, CONFIG['EPOCHS'] + 1):
        # A. Determine Curriculum Stage
        if ep <= CONFIG['STAGE1_END']: phase, crit = "S1", s1_crit
        elif ep <= CONFIG['STAGE2_END']: phase, crit = "S2", s2_crit
        else: phase, crit = "S3", s3_crit
        
        # B. Sampler Logic (Curriculum Sampling)
        sampler = subclass_sampler(train_ds, ep, s3_start, CONFIG['EPOCHS'])
        train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], 
                                sampler=sampler, shuffle=(sampler is None), 
                                num_workers=CONFIG['NUM_WORKERS'], collate_fn=collate_fn)
        
        print(f"\n{'='*20} Epoch {ep}/{CONFIG['EPOCHS']} [{phase}] {'='*20}")
        
        # C. Run Epochs
        t_loss, _, _ = run_epoch(model, train_loader, crit, opt, scaler, "Train")
        v_loss, v_logits, v_labels = run_epoch(model, val_loader, crit, None, scaler, "Val")
        
        print(f"Loss -> Train: {t_loss:.4f} | Val: {v_loss:.4f}")
        
        # D. Print Detailed Metrics (Using function from previous turn)
        # We move tensors to CPU to avoid GPU bottlenecks during printing
        compute_and_print_metrics(v_logits.cpu(), v_labels.cpu(), ep, "VALIDATION")
        
        # E. Learning Rate Schedule & Warmup
        if ep >= s3_start and ep < s3_start + 3:
            # Manual Warmup for Stage 3 transition
            warm_lr = (CONFIG['LR'] * (ep - s3_start + 1) / 3)
            for pg in opt.param_groups: pg['lr'] = warm_lr
        else:
            sched.step()

        # F. Save Best Model 
        save_condition = v_loss < best_loss
        if ep >= s3_start and save_condition:
            best_loss = v_loss
            torch.save(model.state_dict(), BEST_PATH)
            print("  [+] Saved Best Model (Lowest Loss)")
        elif ep <= CONFIG['STAGE2_END'] and save_condition:
             # Optional: Track best model during early stages too
             best_loss = v_loss
             torch.save(model.state_dict(), BEST_PATH)
             print("  [+] Saved Best Model (Early Stage)")

        # G. Save Checkpoint (EVERY EPOCH)
        # This overwrites the file every epoch so you always have the latest state
        checkpoint = {
            'epoch': ep,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scaler': scaler.state_dict(),
            'sched': sched.state_dict(),
            'best_loss': best_loss
        }
        torch.save(checkpoint, CKPT_PATH)
        print("  [i] Checkpoint Saved")

        # H. End of Stage Notifications
        if ep == CONFIG['STAGE1_END']:
            print(f"\n>>> [STAGE 1 COMPLETE] Binary Triage Phase Done.")
        if ep == CONFIG['STAGE2_END']:
            print(f"\n>>> [STAGE 2 COMPLETE] Superclass Learning Done.")