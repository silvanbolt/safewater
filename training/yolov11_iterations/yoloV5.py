import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import WeightedRandomSampler

from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.classify import ClassificationTrainer


class BalancedClassificationTrainer(ClassificationTrainer):

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            # yolo_classification_dataset has no val split; use the oversample val set
            dataset_path = "/work/courses/dslab/team7/multi-national/yolo_classification_oversample/val"

        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if mode != "train":
            return loader

        dataset = loader.dataset
        class_counts = Counter(int(s[1]) for s in dataset.samples)
        sample_weights = [1.0 / class_counts[int(s[1])] for s in dataset.samples]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )

        nd = torch.cuda.device_count()
        nw = min(os.cpu_count() // max(nd, 1), self.args.workers)

        target_per_class = len(dataset) / len(class_counts)
        print("\n--- Balanced sampling ---")
        for idx, cnt in sorted(class_counts.items()):
            print(f"  class {idx}: {cnt:6d} real → ×{target_per_class/cnt:.1f} per epoch")

        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            drop_last=self.args.compile and len(dataset) % batch_size != 0,
        )


if __name__ == "__main__":
    data_root  = "/work/courses/dslab/team7/multi-national/yolo_classification_dataset"
    model_path = "/work/courses/dslab/team7/multi-national/best-finetune.pt"

    trainer = BalancedClassificationTrainer(overrides=dict(
        model=model_path,
        data=data_root,
        epochs=10,
        imgsz=224,
        batch=64,
        lr0=2e-4,
        lrf=0.2,
        optimizer='AdamW',
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        patience=15,
        project="/work/courses/dslab/team7/multi-national",
        name="waterpoint_finetune_v5",
        workers=2,
        amp=True,
        cache=False,
        plots=True,
    ))

    trainer.train()

    best_path = Path(trainer.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        print("No best.pt found, skipping evaluation.")
    else:
        val_model   = YOLO(best_path)
        val_root    = Path("/work/courses/dslab/team7/multi-national/yolo_classification_oversample/val")
        save_dir    = Path(trainer.save_dir)

        IMPROVED_LIST     = [
            'Borehole_Tubewell', 'Piped_Water', 'Protected_Spring',
            'Protected_Well', 'Rainwater_Harvesting', 'Sand_or_Sub-surface_Dam',
        ]
        NOT_IMPROVED_LIST = ['Delivered_Water', 'Surface_Water', 'Unprotected_Well']
        ALL_CATEGORIES    = IMPROVED_LIST + NOT_IMPROVED_LIST
        IMPROVED_SET      = set(IMPROVED_LIST)

        name_to_idx = {name: i for i, name in val_model.names.items()}

        y_true      = []
        y_pred_top1 = []
        top2_hits   = 0
        top3_hits   = 0
        binary_true = []
        binary_pred = []
        total_count = 0

        for cls_name in ALL_CATEGORIES:
            cls_dir = val_root / cls_name
            if not cls_dir.exists():
                print(f"Warning: {cls_name} not found in val set")
                continue
            true_idx = name_to_idx[cls_name]
            for res in val_model.predict(source=str(cls_dir), conf=0.0, save=False, verbose=False, stream=True):
                total_count += 1
                top3_idxs = torch.topk(res.probs.data, k=3).indices.tolist()
                t1, t2, t3 = top3_idxs
                y_true.append(true_idx)
                y_pred_top1.append(t1)
                if true_idx in (t1, t2):
                    top2_hits += 1
                if true_idx in (t1, t2, t3):
                    top3_hits += 1
                pred_name = val_model.names[t1]
                binary_true.append('improved' if cls_name  in IMPROVED_SET else 'not_improved')
                binary_pred.append('improved' if pred_name in IMPROVED_SET else 'not_improved')

        y_true_arr      = np.array(y_true)
        y_pred_arr      = np.array(y_pred_top1)
        binary_true_arr = np.array(binary_true)
        binary_pred_arr = np.array(binary_pred)

        top1_acc   = (y_true_arr == y_pred_arr).mean()
        top2_acc   = top2_hits / total_count
        top3_acc   = top3_hits / total_count
        binary_acc = (binary_true_arr == binary_pred_arr).mean()

        improved_mask     = binary_true_arr == 'improved'
        not_improved_mask = binary_true_arr == 'not_improved'
        binary_correct    = binary_true_arr == binary_pred_arr
        improved_binary_acc     = binary_correct[improved_mask].mean()     if improved_mask.any()     else 0.0
        not_improved_binary_acc = binary_correct[not_improved_mask].mean() if not_improved_mask.any() else 0.0

        print(f"\n--- Final Performance Report (val: yolo_classification_oversample) ---")
        print(f"Total Images:                    {total_count}")
        print(f"Top-1 Accuracy:                  {top1_acc:.2%}")
        print(f"Top-2 Accuracy:                  {top2_acc:.2%}")
        print(f"Top-3 Accuracy:                  {top3_acc:.2%}")
        print(f"\n--- Binary Accuracy (Improved vs Not Improved) ---")
        print(f"Overall Binary Accuracy:         {binary_acc:.2%}")
        print(f"  Improved sources accuracy:     {improved_binary_acc:.2%}")
        print(f"  Not-improved sources accuracy: {not_improved_binary_acc:.2%}")

        ordered_indices = [name_to_idx[n] for n in ALL_CATEGORIES]
        cm = confusion_matrix(y_true, y_pred_top1, labels=ordered_indices)

        pd.DataFrame(cm, index=ALL_CATEGORIES, columns=ALL_CATEGORIES).to_csv(save_dir / "v5_results_matrix.csv")

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
            f'YoloV5 Confusion Matrix\n'
            f'Top-1: {top1_acc:.2%} | Top-2: {top2_acc:.2%} | Top-3: {top3_acc:.2%} | Binary: {binary_acc:.2%}',
            fontsize=14,
        )
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_dir / "v5_heatmap.png", dpi=300)
        print(f"\nHeatmap saved to: {save_dir / 'v5_heatmap.png'}")
