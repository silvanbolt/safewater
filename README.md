# SafeWater

Estimating access to safe water from photos of water sources.

A YOLOv11m image classifier trained to identify water source types in Ethiopia and map them to the WHO **improved / non-improved** binary standard. The final model reaches **~74% top-1 accuracy** across 9 fine-grained source types and **~97.5% binary accuracy** (improved vs. non-improved).

A companion Streamlit desktop app lets field workers upload a photo and get an instant prediction, no internet required after installation.

---

## Repository structure

```
safewater/
├── preprocessing/              Data cleaning and feature engineering
├── training/
│   ├── yolov11_iterations/     Iterative YOLOv11 fine-tuning (V1–V5)
│   └── teacher_student/        Multimodal teacher + distilled student
├── experiments/
│   ├── calibration/            Probability calibration analysis
│   ├── umap/                   UMAP embedding visualisations
│   └── validation/             Confusion-matrix plotting + image review tool
├── model/                      Training artefacts (plots, metrics, config)
├── webapp/                     Streamlit desktop app (macOS & Windows)
└── poster.pdf                  Research poster
```

---

## Pipeline overview

### 1. Preprocessing (`preprocessing/`)

Raw data comes from the Water Point Data Exchange (WPDx) — a CSV of waterpoint metadata with image paths.

| Script               | Purpose                                                           |
| -------------------- | ----------------------------------------------------------------- |
| `download_img.py`    | Downloads waterpoint images from WPDx URLs                        |
| `preprocessing.py`   | Cleans metadata, engineers features (age, rehab age), maps labels |
| `compute_metrics.py` | Computes dataset-level statistics                                 |

The 9 source-type classes and their binary mapping:

| Class                  | Binary       |
| ---------------------- | ------------ |
| Borehole / Tubewell    | Improved     |
| Piped Water            | Improved     |
| Protected Spring       | Improved     |
| Protected Well         | Improved     |
| Rainwater Harvesting   | Improved     |
| Sand / Sub-surface Dam | Improved     |
| Delivered Water        | Non-improved |
| Surface Water          | Non-improved |
| Unprotected Well       | Non-improved |

---

### 2. YOLOv11 training iterations (`training/yolov11_iterations/`)

Five progressive training runs on a multi-national dataset, each addressing shortcomings of the previous:

| Version | Key change                                                     |
| ------- | -------------------------------------------------------------- |
| V1      | Baseline fine-tune from pretrained YOLOv11m                    |
| V2      | Adjusted learning rate and augmentation                        |
| V3      | Extended dataset with cross-country samples                    |
| V4      | Weighted loss for class imbalance                              |
| V5      | **Balanced sampler** (WeightedRandomSampler) — final iteration |

Supporting scripts handle dataset preparation (`prepare.py`), class balancing (`balance.py`), failure analysis (`copy_failures.py`, `top2.py`, `top3.py`), heatmap visualisation (`heatmap.py`), and test-time augmentation evaluation (`tta_eval.py`). Shell scripts (`run_yoloVx.sh`) wrap each run for cluster submission.

---

### 3. Teacher–student approach (`training/teacher_student/`)

An alternative multimodal architecture that fuses image features with tabular metadata (age, rehabilitation year, management type, fecal coliform levels).

- **Teacher** (`train_teacher.py`): YOLOv11 backbone + metadata MLP, fused and trained on binary labels.
- **Student** (`train_student.py`): Lightweight image-only model distilled from the teacher's embeddings.
- Evaluation scripts produce ROC curves and misclassification reports.

---

### 4. Model artefacts (`model/`)

Training outputs from the YOLOv11 V5 run. The model weights themselves are bundled with the webapp at `webapp/app/model/best-finetune.pt`.

| File                    | Description                             |
| ----------------------- | --------------------------------------- |
| `args.yaml`             | Full training configuration             |
| `results.csv`           | Per-epoch training metrics              |
| `confusion_matrix*.png` | Normalised and raw confusion matrices   |
| `v5_heatmap.png`        | Per-class heatmap with accuracy summary |
| `v5_results_matrix.csv` | Raw confusion counts for all 9 classes  |

**Final metrics (V5 val set):**

| Metric                                    | Value  |
| ----------------------------------------- | ------ |
| Top-1 accuracy (9-class)                  | ~73.9% |
| Binary accuracy (improved / non-improved) | ~97.5% |

---

### 5. Experiments (`experiments/`)

#### Calibration (`experiments/calibration/`)

Reliability diagrams and Brier scores for the binary improved/non-improved output. Two scripts cover the single-country and combined-training model variants. Output plots are included alongside the scripts.

#### UMAP (`experiments/umap/`)

UMAP projections of the YOLOv11 backbone embeddings to visualise class separability. Three scripts (`vis.py`, `combined_umap.py`, `combined_umapV2.py`) produce 2D scatter plots coloured by source type and binary label.

#### Validation (`experiments/validation/`)

- `plot_cm.py` — renders confusion matrices from saved predictions.
- `valstar/` — a small CLI tool for manually reviewing ambiguous images and flagging mislabelled ones. See [`experiments/validation/valstar/README.md`](experiments/validation/valstar/README.md).

---

### 6. Web app (`webapp/`)

A Streamlit desktop application for field use. Upload a photo → get the top-3 source-type predictions and a final improved / non-improved verdict.

See [`webapp/README.md`](webapp/README.md) for local development, build instructions (macOS and Windows), and packaging details.

---

## Quickstart (app only)

```bash
cd webapp
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

The model weights are bundled at `webapp/app/model/best-finetune.pt`.

---

## Disclaimer

Predictions are estimates based on visual appearance. They should not replace professional water-quality testing.
