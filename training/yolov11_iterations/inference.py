from ultralytics import YOLO

MODEL_PATH = "/work/courses/dslab/team7/multi-national/waterpoint_final_approach/weights/last.pt"
IMAGE_PATH = "/path/to/your/image.jpg"

model = YOLO(MODEL_PATH)

r = model.predict(IMAGE_PATH, verbose=False)[0]

top1_idx = r.probs.top1
top1_conf = r.probs.top1conf
top1_name = model.names[top1_idx]

print(f"Predicted: {top1_name} (class {top1_idx}), confidence: {top1_conf:.3f}")

print("Top-5:")
for idx, conf in zip(r.probs.top5, r.probs.top5conf):
    print(f"  [{idx}] {model.names[idx]}: {conf:.3f}")
