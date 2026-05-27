import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

def validate_and_export_results(model_path, data_root):
    # 1. Load Model
    model = YOLO(model_path)
    
    # 2. Run Validation
    print("Running validation and extracting predictions...")
    results = model.val(data=data_root, split='val', save=False)
    
    # 3. Define the Logical Order (Grouping similar categories)
    improved = ['Borehole_Tubewell', 'Piped_Water', 'Protected_Spring', 'Protected_Well', 'Rainwater_Harvesting']
    not_improved = ['Delivered_Water', 'Surface_Water', 'Unprotected_Well']
    neutral = ['Sand_or_Sub-surface_Dam']
    
    custom_order = improved + not_improved + neutral
    
    # Map model indices to custom order
    name_to_idx = {v: k for k, v in model.names.items()}
    ordered_indices = [name_to_idx[name] for name in custom_order]

    # 4. Extract and Reorder Matrix
    # YOLO confusion matrix is [True, Predicted]
    raw_cm = results.confusion_matrix.matrix[:9, :9]
    ordered_cm = raw_cm[ordered_indices, :][:, ordered_indices]

    # 5. Export to CSV
    df_cm = pd.DataFrame(ordered_cm, index=custom_order, columns=custom_order)
    csv_path = "waterpoint_confusion_matrix.csv"
    df_cm.to_csv(csv_path)
    print(f"CSV exported successfully to: {csv_path}")

    # 6. Plot Heatmap
    plt.figure(figsize=(12, 10))
    
    # Use square root scaling for color if counts vary wildly
    # This makes small 'punishable' mistakes more visible
    sns.heatmap(
        np.sqrt(ordered_cm), 
        annot=ordered_cm, # Put actual counts in the boxes
        fmt='.0f', 
        xticklabels=custom_order, 
        yticklabels=custom_order,
        cmap='YlOrRd',    # Yellow-Orange-Red style
        cbar_kws={'label': 'Color intensity (Sqrt scale)'},
        square=True,
        linewidths=.5
    )

    plt.title('Hierarchical Heatmap: Improved vs Not Improved Groups', fontsize=16)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('True Category', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    heatmap_path = "waterpoint_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    print(f"Heatmap saved to: {heatmap_path}")
    plt.show()

if __name__ == "__main__":
    MODEL_PATH = "/work/courses/dslab/team7/multi-national/waterpoint_final_approach/weights/last.pt"
    DATA_ROOT = "/work/courses/dslab/team7/multi-national/yolo_classification_dataset"
    
    validate_and_export_results(MODEL_PATH, DATA_ROOT)
