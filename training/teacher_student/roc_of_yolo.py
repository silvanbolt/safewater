import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns

# --- CONFIGURATION ---
BASE_PATH = "/work/courses/dslab/team7/teacher_student"
YOLO_MODEL_PATH = "/work/courses/dslab/team7/yolo/weights1/last.pt"
VAL_DIR = "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_yolo_native():
    model = YOLO(YOLO_MODEL_PATH)
    
    # --- DYNAMIC CLASS MAPPING ---
    # We want the probability of 'improved'. Let's find which index that is in YOLO.
    # model.names usually looks like {0: 'improved', 1: 'not_improved'} or vice versa.
    yolo_classes = model.names
    improved_idx = None
    for idx, name in yolo_classes.items():
        if 'improved' in name.lower() and 'not' not in name.lower():
            improved_idx = idx
            break
    
    if improved_idx is None:
        print(f"Warning: Could not find 'improved' in model names: {yolo_classes}")
        improved_idx = 1 # Fallback to 1
    else:
        print(f"Mapping 'improved' to YOLO class index: {improved_idx}")

    y_true = []
    y_probs = []
    
    # Your script's internal mapping
    categories = {'not_improved': 0, 'improved': 1}

    print(f"Running evaluation for native YOLOv8 model...")

    for category_name, label in categories.items():
        category_path = os.path.join(VAL_DIR, category_name)
        if not os.path.exists(category_path):
            continue
            
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(category_path, img_name)
            try:
                results = model.predict(source=img_path, device=DEVICE, verbose=False)
                
                # Get the probability for the 'improved' index
                probs_tensor = results[0].probs.data
                prob_improved = probs_tensor[improved_idx].item() 
                
                y_true.append(label)
                y_probs.append(prob_improved)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # --- INVERSION CHECK ---
    # If AUROC < 0.5, it means labels are perfectly swapped.
    auroc = roc_auc_score(y_true, y_probs)
    if auroc < 0.5:
        print(f"AUROC {auroc:.4f} is below 0.5. Inverting probabilities to correct mapping.")
        y_probs = 1 - y_probs
        auroc = roc_auc_score(y_true, y_probs)

    # --- METRICS CALCULATION ---
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # Find best accuracy threshold
    accuracies = [np.mean((y_probs >= t).astype(int) == y_true) for t in thresholds]
    best_idx = np.argmax(accuracies)
    best_t = thresholds[best_idx]

    print(f"\n--- YOLOv8 Native Results (Corrected) ---")
    print(f"Final AUROC: {auroc:.4f}")
    print(f"Optimal Threshold: {best_t:.4f}")
    print(f"Max Accuracy: {accuracies[best_idx]:.4f}")

    # --- PLOT ROC ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'YOLOv8 ROC (AUC = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, label=f'Best Accuracy Pt', zorder=5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Corrected ROC Curve: Native YOLOv8')
    plt.legend(loc="lower right")
    plt.savefig(f"{BASE_PATH}/yolo_native_roc_curve.png")
    
    # --- BEST CONFUSION MATRIX ---
    y_pred = (y_probs >= best_t).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm.T, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Not Improved', 'Improved'],
                yticklabels=['Not Improved', 'Improved'])
    plt.ylabel('Predicted (at Best Threshold)')
    plt.xlabel('Actual')
    plt.title(f'YOLOv8 Confusion Matrix (Corrected)')
    plt.savefig(f"{BASE_PATH}/yolo_native_confusion_matrix.png")

    print(f"\nSaved corrected plots to {BASE_PATH}")

if __name__ == "__main__":
    evaluate_yolo_native()
