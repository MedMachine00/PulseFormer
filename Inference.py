# inference.py
# -*- coding: utf-8 -*-
"""
Full Inference & Explainability Script (Hybrid ViT Support)
-----------------------------------------------------------
1. METRICS: 
   - Calculates per-class AUC, F1, Accuracy.
   - Calculates Global Accuracy (Hamming) and Exact Match Ratio.

2. VISUALIZATION (Hybrid Grad-CAM):
   - Hooks into the ViT 'norm' layer.
   - Extracts 7x7 attention maps (from 112x112 CNN features).
   - Projects them back to 224x224 input space to show model focus.

3. ROBUSTNESS:
   - Uses Test-Time Augmentation (Affine + Noise) identical to training
     to ensure the model works on imperfect real-world data.

Usage:
    python inference.py
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm.auto import tqdm

# Import necessary components from your train.py
# Ensure train.py is in the same directory!
from train import (
    ECGLabelGNNClassifier, 
    ECGCWTDataset, 
    get_transforms, 
    collate_fn, 
    CONFIG, 
    ALL_NAMES, 
    DEVICE
)

# ──────────────────────── Config ────────────────────────
INF_CONFIG = {
    'MODEL_PATH': os.path.join(CONFIG['RESULTS_DIR'], "model_best.pth"),
    # Tries to find test_meta.csv, falls back to val_meta.csv if missing
    'TEST_CSV': os.path.join(CONFIG['ROOT_DIR'], "test_meta.csv"),
    'OUTPUT_DIR': "inference_final_results",
    'BATCH_SIZE': 32,
    'IMG_SIZE': 224
}

# ────────────────────── Hybrid Grad-CAM Engine ──────────────────────
class HybridGradCAM:
    """
    Grad-CAM specifically designed for the Hybrid ConvStemViT.
    
    Logic:
    1. Input (224x224) -> CNN Stem -> Feature Map (112x112).
    2. ViT patches this into 7x7 tokens.
    3. We extract gradients from the ViT's final Norm layer.
    4. We resize the 7x7 heatmap directly back to 224x224.
       (Because Token 0,0 corresponds to Top-Left of image).
    """
    def __init__(self, model_module, target_layer):
        self.model = model_module
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, class_idx, original_img_size=(224, 224)):
        # 1. Pool Gradients [B, Tokens, Dim] -> [B, 1, Dim]
        pooled_gradients = torch.mean(self.gradients, dim=1, keepdim=True)
        
        # 2. Weight Activations
        # Skip CLS token (index 0) to get spatial tokens
        spatial_activations = self.activations[:, 1:, :] 
        
        # Weighted sum
        cam = (spatial_activations * pooled_gradients).sum(dim=2)
        
        # 3. Reshape 1D tokens to 2D Grid
        # For 112x112 feature map / 16 patch size = 7x7 grid
        grid_size = int(np.sqrt(cam.shape[1])) 
        cam = cam.view(cam.shape[0], grid_size, grid_size)
        
        # 4. ReLU & Normalize
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        
        heatmaps = []
        for i in range(cam.shape[0]):
            mask = cam[i]
            if mask.max() > 0:
                mask = mask / mask.max()
            
            # 5. PROJECTION: Upsample 7x7 -> 224x224
            # Uses bilinear interpolation to smooth the blocks
            mask = cv2.resize(mask, original_img_size)
            heatmaps.append(mask)
            
        return heatmaps

def save_visualisation(img_tensor, heatmap, class_name, save_path):
    """Overlays heatmap on ECG and saves image."""
    # Denormalize
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    # Heatmap to RGB
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    superimposed = 0.65 * img + 0.35 * heatmap_color
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.title(f"Input: {class_name}"); plt.imshow(img); plt.axis('off')
    plt.subplot(1, 2, 2); plt.title("Model Focus (Grad-CAM)"); plt.imshow(superimposed); plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ──────────────────────── Main Logic ────────────────────────
def main():
    # 1. Setup
    os.makedirs(INF_CONFIG['OUTPUT_DIR'], exist_ok=True)
    vis_dir = os.path.join(INF_CONFIG['OUTPUT_DIR'], "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"--- Inference Started on {DEVICE} ---")
    
    # 2. Data Loading
    csv_path = INF_CONFIG['TEST_CSV']
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Using Val set for demo.")
        csv_path = os.path.join(CONFIG['ROOT_DIR'], "val_meta.csv")
        
    # Note: Using get_transforms() keeps the Affine/Noise augs active
    ds = ECGCWTDataset(
        csv_path, 
        f"{CONFIG['ROOT_DIR']}/ECG_train", 
        f"{CONFIG['ROOT_DIR']}/CWTfast_train", 
        get_transforms(CONFIG['IMG_SIZE']) 
    )
    loader = DataLoader(ds, batch_size=INF_CONFIG['BATCH_SIZE'], 
                        shuffle=False, collate_fn=collate_fn)
    
    # 3. Model Loading
    model = ECGLabelGNNClassifier().to(DEVICE)
    print(f"Loading weights from {INF_CONFIG['MODEL_PATH']}...")
    ckpt = torch.load(INF_CONFIG['MODEL_PATH'], map_location=DEVICE, weights_only=True)
    
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    
    # 4. Initialize Grad-CAM
    # Hook into the ECG ViT's final norm layer (inside the ConvStemViT wrapper)
    # Path: model -> vit_e (ConvStemViT) -> vit (timm model) -> norm
    grad_cam = HybridGradCAM(model, model.vit_e.vit.norm)
    
    # Trackers
    all_preds, all_labels = [], []
    visualized_classes = {name: False for name in ALL_NAMES}
    
    print("Running Inference...")
    
    # 5. Inference Loop
    for batch_idx, (ecg, cwt, labels) in enumerate(tqdm(loader)):
        ecg, cwt, labels = ecg.to(DEVICE), cwt.to(DEVICE), labels.to(DEVICE)
        
        # Turn on gradients only for input (required for Grad-CAM even in eval)
        ecg.requires_grad = True 
        
        # Forward Pass (Updated for No-Aux)
        logits, _ = model(ecg, cwt)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Store metrics
        all_preds.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        
        # --- Visualization Logic ---
        # Find 1 correct positive example per class
        for i in range(ecg.size(0)):
            for cls_idx, cls_name in enumerate(ALL_NAMES):
                if visualized_classes[cls_name]: continue # Skip if already done
                
                # Check: Ground Truth is 1 AND Prediction is 1
                if labels[i, cls_idx] == 1 and preds[i, cls_idx] == 1:
                    
                    # Rerun forward for single sample to get clean gradients
                    model.zero_grad()
                    
                    # Slice single item (keep batch dim)
                    s_ecg, s_cwt = ecg[i:i+1], cwt[i:i+1]
                    
                    # Forward
                    s_logits, _ = model(s_ecg, s_cwt)
                    
                    # Backward on the specific class score
                    s_logits[0, cls_idx].backward()
                    
                    # Generate Map
                    heatmap = grad_cam.generate_heatmap(cls_idx)[0]
                    
                    # Save
                    save_path = os.path.join(vis_dir, f"Correct_{cls_name}.png")
                    save_visualisation(s_ecg[0].detach(), heatmap, cls_name, save_path)
                    
                    visualized_classes[cls_name] = True

    # 6. Compute Metrics
    print("\nComputing Final Metrics...")
    y_probs = np.vstack(all_preds)
    y_preds = (y_probs > 0.5).astype(int)
    y_true = np.vstack(all_labels)
    
    results = []
    
    # Class-wise
    for i, name in enumerate(ALL_NAMES):
        try:
            auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        except ValueError:
            auc = 0.5 # Handle single-class batches
            
        f1 = f1_score(y_true[:, i], y_preds[:, i], zero_division=0)
        acc = accuracy_score(y_true[:, i], y_preds[:, i])
        
        results.append({
            'Class': name,
            'AUC': round(auc, 4),
            'F1': round(f1, 4),
            'Acc': round(acc, 4),
            'Support': int(y_true[:, i].sum())
        })
    
    df_res = pd.DataFrame(results)
    
    # Global Metrics
    global_acc = accuracy_score(y_true.ravel(), y_preds.ravel())
    subset_acc = accuracy_score(y_true, y_preds)
    macro_auc = roc_auc_score(y_true, y_probs, average='macro')
    macro_f1 = f1_score(y_true, y_preds, average='macro', zero_division=0)
    
    # Print Report
    print("\n" + "="*50)
    print(f"FINAL TEST REPORT")
    print("="*50)
    print(f"Global Accuracy (Hamming):  {global_acc:.4f}")
    print(f"Exact Match Accuracy:       {subset_acc:.4f}")
    print(f"Macro AUC:                  {macro_auc:.4f}")
    print(f"Macro F1:                   {macro_f1:.4f}")
    print("-" * 50)
    print(df_res.to_string(index=False))
    print("-" * 50)
    
    # Save
    csv_out = os.path.join(INF_CONFIG['OUTPUT_DIR'], "test_metrics.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\n[+] Metrics saved to: {csv_out}")
    print(f"[+] Visualizations saved to: {vis_dir}")

if __name__ == "__main__":
    main()