import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def run_top2_evaluation(model_path, data_root):
    # 1. Load Model
    model = YOLO(model_path)
    val_path = Path(data_root) / "val"
    
    # Define our hierarchical order
    improved = ['Borehole_Tubewell', 'Piped_Water', 'Protected_Spring', 'Protected_Well', 'Rainwater_Harvesting']
    not_improved = ['Delivered_Water', 'Surface_Water', 'Unprotected_Well']
    neutral = ['Sand_or_Sub-surface_Dam']
    all_categories = improved + not_improved + neutral

    # Map category names to indices based on model metadata
    name_to_idx = {name: i for i, name in model.names.items()}
    
    y_true = []
    y_pred_top1 = []
    top2_hits = 0
    total_count = 0

    print(f"Loading weights from: {model_path}")
    print("Processing validation images by folder...")

    # Loop through each folder in the val directory
    for class_name in all_categories:
        class_dir = val_path / class_name
        if not class_dir.exists():
            print(f"Warning: Folder {class_name} not found in {val_path}")
            continue
            
        true_idx = name_to_idx[class_name]
        
        # Predict on all images in this class folder
        # verbose=False keeps the console clean; stream=True handles large folders
        results = model.predict(source=str(class_dir), conf=0.0, save=False, verbose=False, stream=True)
        
        for res in results:
            total_count += 1
            
            # Get probabilities and top 2 indices
            probs = res.probs.data # Tensor of probabilities [class_0_prob, class_1_prob, ...]
            top2_values, top2_indices = torch.topk(probs, k=2)
            
            t1 = top2_indices[0].item()
            t2 = top2_indices[1].item()
            
            y_true.append(true_idx)
            y_pred_top1.append(t1)
            
            # Top-2 logic: Check if the ground truth index is either the 1st or 2nd choice
            if true_idx == t1 or true_idx == t2:
                top2_hits += 1

    # 2. Calculate Accuracies
    top1_acc = (np.array(y_true) == np.array(y_pred_top1)).mean()
    top2_acc = top2_hits / total_count

    print(f"\n--- Final Performance Report ---")
    print(f"Total Images Validated: {total_count}")
    print(f"Top-1 Accuracy: {top1_acc:.2%}")
    print(f"Top-2 Accuracy: {top2_acc:.2%}")

    # 3. Create Heatmap (Reordered based on custom_order)
    ordered_indices = [name_to_idx[n] for n in all_categories]
    cm = confusion_matrix(y_true, y_pred_top1, labels=ordered_indices)
    
    # Save the raw numbers
    df_cm = pd.DataFrame(cm, index=all_categories, columns=all_categories)
    df_cm.to_csv("top2_results_matrix.csv")

    # 4. Plotting
    plt.figure(figsize=(12, 10))
    # We use a square-root scale to help see the smaller numbers (errors) better
    sns.heatmap(
        np.sqrt(cm), 
        annot=cm, 
        fmt='d', 
        xticklabels=all_categories, 
        yticklabels=all_categories,
        cmap='YlOrRd', 
        square=True,
        cbar_kws={'label': 'Color Intensity (Scaled)'}
    )
    
    plt.title(f'Hierarchical Confusion Matrix\nTop-1: {top1_acc:.2%} | Top-2: {top2_acc:.2%}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig("final_top2_heatmap.png", dpi=300)
    print("Heatmap saved as: final_top2_heatmap.png")
    plt.show()

if __name__ == "__main__":
    # Updated path as requested
    #MODEL_PATH = "/work/courses/dslab/team7/multi-national/waterpoint_final_approach/weights/best.pt"
    MODEL_PATH = "/work/courses/dslab/team7/multi-national/last.pt"
    #DATA_ROOT = "/work/courses/dslab/team7/multi-national/yolo_classification_dataset"
    DATA_ROOT = "/work/courses/dslab/team7/multi-national/yolo_classification_oversample"
    run_top2_evaluation(MODEL_PATH, DATA_ROOT)
