from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from ultralytics import YOLO

model_path = "/work/courses/dslab/team7/multi-national/last.pt"
val_root   = Path("/work/courses/dslab/team7/multi-national/yolo_classification_oversample/val")

IMGSZ      = 224
BATCH_SIZE = 64

IMPROVED_LIST     = [
    'Borehole_Tubewell', 'Piped_Water', 'Protected_Spring',
    'Protected_Well', 'Rainwater_Harvesting', 'Sand_or_Sub-surface_Dam',
]
NOT_IMPROVED_LIST = ['Delivered_Water', 'Surface_Water', 'Unprotected_Well']
ALL_CATEGORIES    = IMPROVED_LIST + NOT_IMPROVED_LIST
IMPROVED_SET      = set(IMPROVED_LIST)

AUG_FNS = [
    lambda img: img,                   # original
    lambda img: ImageOps.mirror(img),  # horizontal flip
]

model       = YOLO(model_path)
name_to_idx = {name: i for i, name in model.names.items()}

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

all_samples = [
    (img_path, name_to_idx[img_path.parent.name])
    for cls_name in ALL_CATEGORIES
    for img_path in sorted((val_root / cls_name).iterdir())
    if img_path.suffix.lower() in IMAGE_SUFFIXES
]

y_true      = []
y_pred_top1 = []
top2_hits   = 0
top3_hits   = 0
binary_true = []
binary_pred = []

for start in tqdm(range(0, len(all_samples), BATCH_SIZE), desc="TTA eval"):
    batch  = all_samples[start : start + BATCH_SIZE]
    imgs   = [Image.open(p).convert("RGB") for p, _ in batch]
    labels = [lbl for _, lbl in batch]

    probs_per_aug = []
    for aug_fn in AUG_FNS:
        aug_imgs = [aug_fn(img) for img in imgs]
        results  = model.predict(aug_imgs, verbose=False, imgsz=IMGSZ)
        probs    = torch.stack([r.probs.data for r in results])  # [B, C]
        probs_per_aug.append(probs)

    avg_probs = torch.stack(probs_per_aug).mean(0)  # [B, C]

    for i, label in enumerate(labels):
        top3_idxs = torch.topk(avg_probs[i], k=3).indices.tolist()
        t1, t2, t3 = top3_idxs

        y_true.append(label)
        y_pred_top1.append(t1)

        if label in (t1, t2):
            top2_hits += 1
        if label in (t1, t2, t3):
            top3_hits += 1

        cls_name  = model.names[label]
        pred_name = model.names[t1]
        binary_true.append('improved' if cls_name  in IMPROVED_SET else 'not_improved')
        binary_pred.append('improved' if pred_name in IMPROVED_SET else 'not_improved')

# --- Metrics ---
total           = len(y_true)
y_true_arr      = np.array(y_true)
y_pred_arr      = np.array(y_pred_top1)
binary_true_arr = np.array(binary_true)
binary_pred_arr = np.array(binary_pred)

top1_acc   = (y_true_arr == y_pred_arr).mean()
top2_acc   = top2_hits / total
top3_acc   = top3_hits / total
binary_acc = (binary_true_arr == binary_pred_arr).mean()

improved_mask     = binary_true_arr == 'improved'
not_improved_mask = binary_true_arr == 'not_improved'
binary_correct    = binary_true_arr == binary_pred_arr
improved_binary_acc     = binary_correct[improved_mask].mean()     if improved_mask.any()     else 0.0
not_improved_binary_acc = binary_correct[not_improved_mask].mean() if not_improved_mask.any() else 0.0

print(f"\n--- TTA Final Performance Report (orig + h-flip) ---")
print(f"Total Images Validated:          {total}")
print(f"Top-1 Accuracy:                  {top1_acc:.2%}")
print(f"Top-2 Accuracy:                  {top2_acc:.2%}")
print(f"Top-3 Accuracy:                  {top3_acc:.2%}")
print(f"\n--- Binary Accuracy (Improved vs Not Improved) ---")
print(f"Overall Binary Accuracy:         {binary_acc:.2%}")
print(f"  Improved sources accuracy:     {improved_binary_acc:.2%}")
print(f"  Not-improved sources accuracy: {not_improved_binary_acc:.2%}")

# --- Heatmap ---
ordered_indices = [name_to_idx[n] for n in ALL_CATEGORIES]
cm = confusion_matrix(y_true, y_pred_top1, labels=ordered_indices)

pd.DataFrame(cm, index=ALL_CATEGORIES, columns=ALL_CATEGORIES).to_csv("tta_results_matrix.csv")

plt.figure(figsize=(13, 11))
sns.heatmap(
    np.sqrt(cm),
    annot=cm,
    fmt='d',
    xticklabels=ALL_CATEGORIES,
    yticklabels=ALL_CATEGORIES,
    cmap='YlOrRd',
    square=True,
    cbar_kws={'label': 'Color Intensity (Scaled)'},
)
plt.title(
    f'TTA Confusion Matrix (orig + h-flip)\n'
    f'Top-1: {top1_acc:.2%} | Top-2: {top2_acc:.2%} | Top-3: {top3_acc:.2%} | Binary: {binary_acc:.2%}',
    fontsize=14,
)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("tta_heatmap.png", dpi=300)
print("\nHeatmap saved as: tta_heatmap.png")
