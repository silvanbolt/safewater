import torch
import torch.nn as nn
from pathlib import Path
from ultralytics.models.yolo.classify import ClassificationTrainer

# Paths
data_root = str(Path(__file__).parent / "yolo_classification_oversample")


# --- Improved / not-improved penalty loss --- #

class WaterpointPenaltyLoss(nn.Module):
    def __init__(self, model_names, penalty_weight=2.0):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss(reduction='none')
        self.penalty_weight = penalty_weight

        improved_labels = {
            'Borehole_Tubewell', 'Piped_Water', 'Protected_Spring', 'Sand_or_Sub-surface_Dam',
            'Protected_Well', 'Rainwater_Harvesting'
        }
        not_improved_labels = {
            'Delivered_Water', 'Surface_Water', 'Unprotected_Well'
        }

        num_classes = max(model_names.keys()) + 1
        cat_tensor = torch.full((num_classes,), -1, dtype=torch.long)
        for idx, name in model_names.items():
            if name in improved_labels:
                cat_tensor[idx] = 1
            elif name in not_improved_labels:
                cat_tensor[idx] = 0
        self.register_buffer('category_tensor', cat_tensor)

    def forward(self, preds, batch):
        if isinstance(preds, tuple):
            preds = preds[0]

        targets = batch['cls'] if isinstance(batch, dict) else batch
        targets = targets.long()

        loss_items = self.base_loss(preds, targets)
        pred_classes = preds.argmax(dim=1)

        t_cats = self.category_tensor[targets]
        p_cats = self.category_tensor[pred_classes]

        cross_cat = (t_cats != -1) & (p_cats != -1) & (t_cats != p_cats)
        penalties = torch.where(
            cross_cat,
            torch.full_like(loss_items, self.penalty_weight),
            torch.ones_like(loss_items)
        )

        total_loss = (loss_items * penalties).mean()
        return total_loss, total_loss.detach().unsqueeze(0)


class CustomClassificationTrainer(ClassificationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        class_dirs = sorted([d for d in (Path(self.args.data) / "train").iterdir() if d.is_dir()])
        model_names = {i: d.name for i, d in enumerate(class_dirs)}
        model.criterion = WaterpointPenaltyLoss(model_names, penalty_weight=2.0)
        return model


# --- Train --- #

trainer = CustomClassificationTrainer(overrides=dict(
    model=str(Path(__file__).parent / "last.pt"),
    data=data_root,
    task='classify',
    epochs=2,
    imgsz=224,
    batch=64,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    patience=10,
    optimizer='AdamW',
    project="/work/courses/dslab/team7/multi-national",
    name="waterpoint_final_approach",
))

trainer.train()
print("Training finished. Save dir:", trainer.save_dir)
