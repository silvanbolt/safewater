# eval on only Kenya (in Kenya main folder)
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. MODEL DEFINITION (Must match training) ---
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
# Kenya:
#VAL_DIR = "/work/courses/dslab/team7/Kenya/yolo_waterpoint_data_Kenya_split_split/val" 
# Ethiopia:
VAL_DIR = "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    # Load Model
    model = VisualStudent(YOLO_CHECKPOINT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    y_true = []
    y_pred = []
    
    # Map folder names to labels
    categories = {'not_improved': 0, 'improved': 1}

    print("Running evaluation on validation set...")

    for category_name, label in categories.items():
        category_path = os.path.join(VAL_DIR, category_name)
        if not os.path.exists(category_path):
            continue
            
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    _, prob = model(img_tensor)
                
                # Convert probability to binary prediction
                prediction = 1 if prob.item() > 0.5 else 0
                print(prob.item())
                print(",")
                y_true.append(label)
                y_pred.append(prediction)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # --- 3. PLOT CONFUSION MATRIX ---
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.T
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Improved', 'Improved'], 
                yticklabels=['Not Improved', 'Improved'])
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion Matrix: Distilled Student')
    plt.savefig(f"{BASE_PATH}/confusion_matrix.png")
    print(f"Confusion matrix saved to {BASE_PATH}/confusion_matrix.png")

    # Print precision, recall, and f1-score
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Improved', 'Improved']))

if __name__ == "__main__":
    evaluate()
