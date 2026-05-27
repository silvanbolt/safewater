import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Required for headless cluster environments
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# --- CONFIGURATION ---
#ROOT_PATH = "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split"
ROOT_PATH = "/work/courses/dslab/team7/yolo/combined-training/data_train_combined"
#MODEL_PATH = "/work/courses/dslab/team7/yolo/weights1/last.pt"
MODEL_PATH = "/work/courses/dslab/team7/yolo/combined-training/waterpoint_cls_comb/weights/last.pt"
VAL_PATH = os.path.join(ROOT_PATH, "val") 
SAVE_NAME = "yolo_calibration_analysis_combined.png"

# 1. Load Model
print(f"Loading model for calibration: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Check model class mapping (usually {0: 'improved', 1: 'not_improved'})
# We will calibrate for the 'improved' class
target_class_idx = 0 
class_name = model.names[target_class_idx]
print(f"Analyzing calibration for class: '{class_name}'")

y_true = []
y_probs = []

# 2. Collect Predictions from Validation Set
for category in ['improved', 'not_improved']:
    folder = os.path.join(VAL_PATH, category)
    if not os.path.exists(folder):
        print(f"Warning: Folder not found: {folder}")
        continue
        
    print(f"Processing category: {category}...")
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in files:
        img_path = os.path.join(folder, img_file)
        
        # Robust image loading check
        test_img = cv2.imread(img_path)
        if test_img is None:
            continue

        # Run Inference
        results = model.predict(img_path, verbose=False)
        
        if len(results) > 0 and results[0].probs is not None:
            # Extract probability for the 'improved' class
            # .data contains the softmax probabilities
            prob = results[0].probs.data[target_class_idx].cpu().item()
            
            # Ground Truth Label: 1 if actual class is 'improved', else 0
            label = 1 if category == class_name else 0
            
            y_probs.append(prob)
            y_true.append(label)

# 3. Calculate Calibration Metrics
# n_bins=10 divides the confidence scores into 10 intervals (0.0-0.1, 0.1-0.2, etc.)
prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)

# Brier Score Loss: measure of how far probabilities are from actual outcomes (0 is perfect)
brier = brier_score_loss(y_true, y_probs)
print(f"Brier Score Loss: {brier:.4f}")

# 4. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})

# Subplot 1: Reliability Diagram
ax1.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
ax1.plot(prob_pred, prob_true, "s-", color="darkblue", label=f"YOLO ({class_name})")
ax1.set_xlabel("Mean Predicted Confidence Score")
ax1.set_ylabel("Actual Accuracy (Fraction of Positives)")
ax1.set_title(f"Reliability Diagram (Brier Score: {brier:.4f})")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Subplot 2: Confidence Histogram
ax2.hist(y_probs, range=(0, 1), bins=10, color="skyblue", histtype="stepfilled", alpha=0.7, edgecolor='black')
ax2.set_xlabel("Predicted Confidence Score")
ax2.set_ylabel("Number of Images")
ax2.set_title("Distribution of Model Confidence")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_NAME, dpi=300)
print(f"Analysis complete. Plot saved as: {SAVE_NAME}")
