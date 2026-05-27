import torch
from ultralytics import YOLO
from pathlib import Path

def find_top3_failures(model_path, data_root, output_file="top3_failures.txt"):
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

    failures = []
    total_count = 0

    print(f"Loading weights from: {model_path}")
    print("Scanning validation images for top-3 failures...")

    for class_name in all_categories:
        class_dir = val_path / class_name
        if not class_dir.exists():
            print(f"Warning: Folder {class_name} not found in {val_path}")
            continue

        true_idx = name_to_idx[class_name]

        results = model.predict(source=str(class_dir), conf=0.0, save=False, verbose=False, stream=True)

        for res in results:
            total_count += 1

            probs = res.probs.data
            top3_values, top3_indices = torch.topk(probs, k=3)
            top3_idx_list = [top3_indices[i].item() for i in range(3)]

            if true_idx not in top3_idx_list:
                image_name = Path(res.path).name
                top3_names = [model.names[i] for i in top3_idx_list]
                top3_confs = [top3_values[i].item() for i in range(3)]
                failures.append({
                    'image': image_name,
                    'true_class': class_name,
                    'top3_predictions': top3_names,
                    'top3_confidences': top3_confs,
                })

    failure_count = len(failures)
    print(f"\nTotal images processed: {total_count}")
    print(f"Top-3 failures:         {failure_count} ({failure_count / total_count:.2%})")

    with open(output_file, 'w') as f:
        f.write(f"Top-3 Failures: {failure_count} / {total_count} ({failure_count / total_count:.2%})\n")
        f.write("=" * 70 + "\n\n")
        for entry in failures:
            f.write(f"Image:      {entry['image']}\n")
            f.write(f"True class: {entry['true_class']}\n")
            for rank, (name, conf) in enumerate(zip(entry['top3_predictions'], entry['top3_confidences']), 1):
                f.write(f"  Top-{rank}: {name} ({conf:.3f})\n")
            f.write("\n")

    print(f"Failure list saved to: {output_file}")

    # Also print a compact summary grouped by true class
    print("\n--- Failures per class ---")
    from collections import Counter
    class_counts = Counter(e['true_class'] for e in failures)
    for cls in all_categories:
        count = class_counts.get(cls, 0)
        print(f"  {cls:<35} {count}")

    return failures

if __name__ == "__main__":
    MODEL_PATH = "/work/courses/dslab/team7/multi-national/last.pt"
    DATA_ROOT = "/work/courses/dslab/team7/multi-national/yolo_classification_oversample"
    find_top3_failures(MODEL_PATH, DATA_ROOT)
