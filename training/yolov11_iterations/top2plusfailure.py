import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import confusion_matrix
from collections import Counter

def run_top2_with_failures(model_path, data_root):
    model = YOLO(model_path)
    val_path = Path(data_root) / "val"

    improved_labels = {
        'Borehole_Tubewell', 'Piped_Water', 'Protected_Spring',
        'Protected_Well', 'Rainwater_Harvesting', 'Sand_or_Sub-surface_Dam'
    }
    not_improved_labels = {
        'Delivered_Water', 'Surface_Water', 'Unprotected_Well'
    }

    improved_list = [
        'Borehole_Tubewell', 'Piped_Water', 'Protected_Spring',
        'Protected_Well', 'Rainwater_Harvesting', 'Sand_or_Sub-surface_Dam'
    ]
    not_improved_list = ['Delivered_Water', 'Surface_Water', 'Unprotected_Well']
    all_categories = improved_list + not_improved_list

    name_to_idx = {name: i for i, name in model.names.items()}

    y_true = []
    y_pred_top1 = []
    top2_hits = 0
    total_count = 0

    binary_true = []
    binary_pred = []

    failures = []

    print(f"Loading weights from: {model_path}")
    print("Processing validation images...")

    for class_name in all_categories:
        class_dir = val_path / class_name
        if not class_dir.exists():
            print(f"Warning: Folder {class_name} not found in {val_path}")
            continue

        true_idx = name_to_idx[class_name]
        true_binary = 'improved' if class_name in improved_labels else 'not_improved'

        results = model.predict(source=str(class_dir), conf=0.0, save=False, verbose=False, stream=True)

        for res in results:
            total_count += 1

            probs = res.probs.data
            top2_values, top2_indices = torch.topk(probs, k=2)

            t1 = top2_indices[0].item()
            t2 = top2_indices[1].item()

            y_true.append(true_idx)
            y_pred_top1.append(t1)

            if true_idx in (t1, t2):
                top2_hits += 1
            else:
                # top-2 failure: record image name and predictions
                top2_names = [model.names[i] for i in (t1, t2)]
                top2_confs = [top2_values[i].item() for i in range(2)]
                failures.append({
                    'image': Path(res.path).name,
                    'true_class': class_name,
                    'top2_predictions': top2_names,
                    'top2_confidences': top2_confs,
                })

            pred_class_name = model.names[t1]
            binary_pred.append('improved' if pred_class_name in improved_labels else 'not_improved')
            binary_true.append(true_binary)

    # --- Accuracies ---
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred_top1)
    binary_true_arr = np.array(binary_true)
    binary_pred_arr = np.array(binary_pred)

    top1_acc = (y_true_arr == y_pred_arr).mean()
    top2_acc = top2_hits / total_count
    binary_acc = (binary_true_arr == binary_pred_arr).mean()

    improved_mask = binary_true_arr == 'improved'
    not_improved_mask = binary_true_arr == 'not_improved'
    binary_correct = binary_true_arr == binary_pred_arr
    improved_binary_acc = binary_correct[improved_mask].mean() if improved_mask.any() else 0.0
    not_improved_binary_acc = binary_correct[not_improved_mask].mean() if not_improved_mask.any() else 0.0

    print(f"\n--- Final Performance Report ---")
    print(f"Total Images Validated:          {total_count}")
    print(f"Top-1 Accuracy:                  {top1_acc:.2%}")
    print(f"Top-2 Accuracy:                  {top2_acc:.2%}")
    print(f"\n--- Binary Class Accuracy (Improved vs Not Improved) ---")
    print(f"Overall Binary Accuracy:         {binary_acc:.2%}")
    print(f"  Improved sources accuracy:     {improved_binary_acc:.2%}")
    print(f"  Not-improved sources accuracy: {not_improved_binary_acc:.2%}")

    # --- Confusion matrix ---
    ordered_indices = [name_to_idx[n] for n in all_categories]
    cm = confusion_matrix(y_true, y_pred_top1, labels=ordered_indices)

    df_cm = pd.DataFrame(cm, index=all_categories, columns=all_categories)
    df_cm.to_csv("top2_results_matrix.csv")
    print("\nConfusion matrix saved as: top2_results_matrix.csv")

    # --- Heatmap ---
    plt.figure(figsize=(13, 11))
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
    plt.title(
        f'Hierarchical Confusion Matrix\n'
        f'Top-1: {top1_acc:.2%} | Top-2: {top2_acc:.2%} | Binary: {binary_acc:.2%}',
        fontsize=14
    )
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("final_top2_heatmap.png", dpi=300)
    print("Heatmap saved as: final_top2_heatmap.png")
    plt.show()

    # --- Failure txt ---
    failure_count = len(failures)
    print(f"\nTop-2 failures: {failure_count} / {total_count} ({failure_count / total_count:.2%})")

    with open("top2_failures.txt", 'w') as f:
        f.write(f"Top-2 Failures: {failure_count} / {total_count} ({failure_count / total_count:.2%})\n")
        f.write("=" * 70 + "\n\n")
        for entry in failures:
            f.write(f"Image:      {entry['image']}\n")
            f.write(f"True class: {entry['true_class']}\n")
            for rank, (name, conf) in enumerate(zip(entry['top2_predictions'], entry['top2_confidences']), 1):
                f.write(f"  Top-{rank}: {name} ({conf:.3f})\n")
            f.write("\n")

    print("Failure list saved as: top2_failures.txt")

    print("\n--- Failures per class ---")
    class_counts = Counter(e['true_class'] for e in failures)
    for cls in all_categories:
        print(f"  {cls:<35} {class_counts.get(cls, 0)}")

if __name__ == "__main__":
    MODEL_PATH = "/work/courses/dslab/team7/multi-national/last.pt"
    DATA_ROOT = "/work/courses/dslab/team7/multi-national/yolo_classification_oversample"
    run_top2_with_failures(MODEL_PATH, DATA_ROOT)
