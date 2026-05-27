import os
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer

# --- 1. DEFINE HIERARCHICAL + WEIGHTED LOSS --- #

class WaterpointBalancedLoss(nn.Module):
    def __init__(self, model_names, class_weights, penalty_weight=3.0):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.penalty_weight = penalty_weight
        
        # CATEGORY UPDATES:
        improved_labels = {
            'Borehole_Tubewell', 'Piped_Water', 'Protected_Spring', 
            'Protected_Well', 'Rainwater_Harvesting'
        }
        not_improved_labels = {
            'Delivered_Water', 'Surface_Water', 'Unprotected_Well'
        }
        # 'Sand_or_Sub-surface_Dam' is implicitly mapped to -1 (excluded)

        self.category_map = {}
        for idx, name in model_names.items():
            if name in improved_labels:
                self.category_map[idx] = 1
            elif name in not_improved_labels:
                self.category_map[idx] = 0
            else:
                self.category_map[idx] = -1

        num_classes = max(model_names.keys()) + 1
        cat_tensor = torch.full((num_classes,), -1, dtype=torch.long)
        for idx, cat in self.category_map.items():
            cat_tensor[idx] = cat
        self.register_buffer('category_tensor', cat_tensor)

    def forward(self, preds, batch):
        # FIX FOR VALIDATOR TUPLE ERROR:
        # If preds is a tuple (happens during validation), take the first element
        if isinstance(preds, tuple):
            preds = preds[0]

        targets = batch['cls'] if isinstance(batch, dict) else batch
        targets = targets.long()
        
        loss_items = self.base_loss(preds, targets)
        pred_classes = preds.argmax(dim=1)

        t_cats = self.category_tensor[targets]
        p_cats = self.category_tensor[pred_classes]
        cross_cat = (t_cats != -1) & (p_cats != -1) & (t_cats != p_cats)
        penalties = torch.where(cross_cat, torch.full_like(loss_items, self.penalty_weight), torch.ones_like(loss_items))

        total_loss = (loss_items * penalties).mean()
        # Return total loss and detached tensor for Ultralytics logging
        return total_loss, total_loss.detach().unsqueeze(0)

# --- 2. CUSTOM TRAINER --- #

class CustomClassificationTrainer(ClassificationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        device = next(model.parameters()).device
        
        train_path = Path(self.args.data) / "train"
        counts = []
        
        print("\n--- Class Distribution & Weights ---")
        if not train_path.exists():
            print(f"Warning: Train path {train_path} not found. Using default weights.")
            counts = [1.0] * len(model.names)
            actual_names = model.names
        else:
            class_dirs = sorted([d for d in train_path.iterdir() if d.is_dir()])
            actual_names = {i: d.name for i, d in enumerate(class_dirs)}
            for class_dir in class_dirs:
                count = sum(1 for f in class_dir.iterdir() if f.is_file())
                counts.append(max(count, 1))

        ckpt_counts = torch.tensor(counts, dtype=torch.float)
        weights = (ckpt_counts.max() / ckpt_counts).clamp(max=20.0)
        weights = weights.to(device)

        for i, w in enumerate(weights):
            print(f"Index {i} [{actual_names[i]}]: Count={int(counts[i])}, Weight={w:.2f}")

        model.criterion = WaterpointBalancedLoss(
            actual_names,
            class_weights=weights,
            penalty_weight=3.0
        )
        return model

# --- 3. MAIN EXECUTION --- #

if __name__ == "__main__":
    data_root = "/work/courses/dslab/team7/multi-national/yolo_classification_dataset"
    model_path = "yolo11m-cls.pt"
    #model_path = "/work/courses/dslab/team7/multi-national/last.pt"

    train_args = dict(
        model=model_path,
        data=data_root,
        epochs=5,
        imgsz=224,
        batch=1024,
        lr0=0.016,
        optimizer='AdamW',
        #label_smoothing=0.1,
        project="/work/courses/dslab/team7/multi-national",
        name="waterpoint_final_approach",
        workers=1,
        amp=True,
        cache=False,
        plots=True,
        freeze=10,
    )

    trainer = CustomClassificationTrainer(overrides=train_args)
    trainer.train()

    # Automatic evaluation on the best weights found
    best_path = Path(trainer.save_dir) / "weights" / "best.pt"
    if best_path.exists():
        val_model = YOLO(best_path)
        metrics = val_model.val(data=data_root)
        print("\nFinal Metrics Summary:")
        for k, v in metrics.results_dict.items():
            print(f"{k}: {v}")
