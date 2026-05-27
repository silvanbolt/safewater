import torch
import os
import json
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import torch.nn as nn

# --- 1. MODEL DEFINITION (Keeping your architecture) ---
class VisualStudent(nn.Module):
    def __init__(self, yolo_path, target_dim=512):
        super().__init__()
        yolo_obj = YOLO(yolo_path)
        self.backbone = nn.Sequential(*list(yolo_obj.model.model.children())[:-1])
        with torch.no_grad():
            dummy_out = self.backbone(torch.zeros(1, 3, 640, 640))
            if isinstance(dummy_out, list): dummy_out = dummy_out[-1]
            in_channels = dummy_out.shape[1]
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, target_dim) 
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(target_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, list): feats = feats[-1]
        emb = self.projection(feats)
        return emb, self.classifier(emb)

# --- 2. CONFIGURATION ---
BASE_PATH = "/work/courses/dslab/team7/teacher_student"
YOLO_CHECKPOINT = f"{BASE_PATH}/data/pre-trained-weights.pt"
MODEL_PATH = f"{BASE_PATH}/distilled_student.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the two sets
DATA_DIRS = {
    "train": "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split/train",
    "val": "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split/val"
}

def run_error_audit():
    # Load Model
    model = VisualStudent(YOLO_CHECKPOINT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    categories = {'not_improved': 0, 'improved': 1}
    # This dictionary will store the filenames of errors
    misclassified_report = {"train": [], "val": []}

    for split_name, split_path in DATA_DIRS.items():
        print(f"Auditing {split_name} set...")
        
        for category_name, label in categories.items():
            category_path = os.path.join(split_path, category_name)
            if not os.path.exists(category_path):
                print(f"Warning: Path {category_path} not found.")
                continue

            images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_name in images:
                img_path = os.path.join(category_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        _, prob = model(img_tensor)

                    # Logic: If prob > 0.5, pred is 1. If label is 0, that's an error.
                    prediction = 1 if prob.item() > 0.5 else 0
                    
                    if prediction != label:
                        # Store the name/number and the confidence for analysis
                        misclassified_report[split_name].append({
                            "filename": img_name,
                            "true_label": category_name,
                            "confidence": round(prob.item(), 4)
                        })

                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

    # --- 3. SAVE RESULTS ---
    output_file = f"{BASE_PATH}/misclassified_images.json"
    with open(output_file, 'w') as f:
        json.dump(misclassified_report, f, indent=4)
    
    print("\n" + "="*30)
    print(f"Audit Complete.")
    print(f"Train Errors: {len(misclassified_report['train'])}")
    print(f"Val Errors: {len(misclassified_report['val'])}")
    print(f"Full list saved to: {output_file}")

if __name__ == "__main__":
    run_error_audit()
