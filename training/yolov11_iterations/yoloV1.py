from ultralytics import YOLO

# Paths
data_root = "/work/courses/dslab/team7/multi-national/yolo_classification_oversample"

# --- 2. Train --- #
#model = YOLO("yolo11m-cls.pt")
#model = YOLO("yolov8n-cls.pt") 
model = YOLO("/work/courses/dslab/team7/multi-national/best-finetune.pt")

results = model.train(
    data=data_root,
    task='classify',
    epochs=15,          # Increase epochs; 10 is very low for classification
    imgsz=224,
    batch=64,
    # Augmentations help, but don't fix imbalance
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    # Add these for better stability:
    patience=10,        # Stop early if validation doesn't improve
    optimizer='AdamW',  # Often better for classification than default SGD
    project="/work/courses/dslab/team7/multi-national",
    name="waterpoint_final_approach", 
)

print("Training finished. Save dir:", results.save_dir)
