import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.classify import ClassificationTrainer


# --- 1. CATEGORY PENALTY LOSS (no class weights) --- #

class WaterpointCategoryLoss(nn.Module):
    """Cross-entropy + cross-category penalty. No per-class weighting — the
    sampler handles class balance instead."""

    _IMPROVED = {
        'Borehole_Tubewell', 'Piped_Water', 'Protected_Spring',
        'Protected_Well', 'Rainwater_Harvesting', 'Sand_or_Sub-surface_Dam',
    }
    _NOT_IMPROVED = {'Delivered_Water', 'Surface_Water', 'Unprotected_Well'}

    def __init__(self, model_names: dict, penalty_weight: float = 1.5):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss(reduction='none')
        self.penalty_weight = penalty_weight

        num_classes = max(model_names.keys()) + 1
        cat_tensor = torch.full((num_classes,), -1, dtype=torch.long)
        for idx, name in model_names.items():
            if name in self._IMPROVED:
                cat_tensor[idx] = 1
            elif name in self._NOT_IMPROVED:
                cat_tensor[idx] = 0
        self.register_buffer('category_tensor', cat_tensor)

    def forward(self, preds, batch):
        if isinstance(preds, tuple):
            preds = preds[0]
        targets = (batch['cls'] if isinstance(batch, dict) else batch).long()

        per_sample = self.base_loss(preds, targets)

        pred_cls = preds.argmax(dim=1)
        t_cat = self.category_tensor[targets]
        p_cat = self.category_tensor[pred_cls]
        cross_cat = (t_cat != -1) & (p_cat != -1) & (t_cat != p_cat)
        scale = torch.where(cross_cat,
                            preds.new_full(per_sample.shape, self.penalty_weight),
                            preds.new_ones(per_sample.shape))

        loss = (per_sample * scale).mean()
        return loss, loss.detach().unsqueeze(0)


# --- 2. TRAINER WITH WEIGHTED RANDOM SAMPLER --- #

class OversampledClassificationTrainer(ClassificationTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)

        train_path = Path(self.args.data) / "train"
        class_dirs = sorted(d for d in train_path.iterdir() if d.is_dir())
        names = {i: d.name for i, d in enumerate(class_dirs)}

        print("\n--- Class distribution (oversampling, no loss weights) ---")
        for i, d in enumerate(class_dirs):
            count = sum(1 for f in d.iterdir() if f.is_file())
            print(f"  {i:2d}  {d.name}: {count}")

        model.criterion = WaterpointCategoryLoss(names, penalty_weight=1.5)
        return model

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        # Let the parent build the dataset (handles caching, transforms, nc filtering)
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if mode != "train":
            return loader

        dataset = loader.dataset
        class_counts = Counter(int(s[1]) for s in dataset.samples)
        sample_weights = [1.0 / class_counts[int(s[1])] for s in dataset.samples]

        # num_samples = len(dataset)  →  same epoch length as without oversampling.
        # WeightedRandomSampler balances class frequency; YOLO's stochastic
        # augmentation (randaugment, erasing, fliplr) makes repeated draws look
        # different, so high-repetition classes don't simply memorise.
        num_samples = len(dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
        )

        nd = torch.cuda.device_count()
        nw = min(os.cpu_count() // max(nd, 1), self.args.workers)

        target_per_class = num_samples / len(class_counts)
        print("\n--- Oversampling (~164 balanced batches / epoch) ---")
        for idx, cnt in sorted(class_counts.items()):
            print(f"  class {idx}: {cnt:6d} real → ×{target_per_class/cnt:.1f} per epoch (aug varies each draw)")

        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,          # sampler handles ordering
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            drop_last=self.args.compile and len(dataset) % batch_size != 0,
        )


# --- 3. MAIN --- #

if __name__ == "__main__":
    data_root = "/work/courses/dslab/team7/multi-national/yolo_classification_dataset"
    model_path = "/work/courses/dslab/team7/multi-national/last.pt"

    train_args = dict(
        model=model_path,
        data=data_root,
        epochs=5,
        imgsz=224,
        batch=1024,
        lr0=1e-4,
        lrf=0.1,
        optimizer='AdamW',
        patience=3,
        # freeze=10,  # uncomment to only fine-tune the classification head
        project="/work/courses/dslab/team7/multi-national",
        name="waterpoint_finetune",
        workers=2,
        amp=True,
        cache=False,
        plots=True,
    )

    trainer = OversampledClassificationTrainer(overrides=train_args)
    trainer.train()

    best_path = Path(trainer.save_dir) / "weights" / "best.pt"
    if best_path.exists():
        val_model = YOLO(best_path)
        oversample_root = "/work/courses/dslab/team7/multi-national/yolo_classification_oversample"
        metrics = val_model.val(data=oversample_root)
        print("\nFinal Metrics Summary (val: yolo_classification_oversample):")
        for k, v in metrics.results_dict.items():
            print(f"  {k}: {v}")
