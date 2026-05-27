# eval on only Kenya (in Kenya main folder)
"""import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

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
    y_probs = [] # Store raw probabilities for ROC/AUROC
    
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
                
                prob_val = prob.item()
                prediction = 1 if prob_val > 0.5 else 0
                
                y_true.append(label)
                y_pred.append(prediction)
                y_probs.append(prob_val) # Collect probability
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # --- 3. PLOT CONFUSION MATRIX ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm.T, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Improved', 'Improved'], 
                yticklabels=['Not Improved', 'Improved'])
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion Matrix: Distilled Student')
    plt.savefig(f"{BASE_PATH}/confusion_matrix.png")
    print(f"Confusion matrix saved to {BASE_PATH}/confusion_matrix.png")

    # --- 4. ROC CURVE AND AUROC ---
    auroc = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f"{BASE_PATH}/roc_curve.png")
    print(f"ROC curve saved to {BASE_PATH}/roc_curve.png")

    # Print precision, recall, f1-score, and AUROC
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Improved', 'Improved']))
    print(f"AUROC Score: {auroc:.4f}")

if __name__ == "__main__":
    evaluate()"""

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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# --- 1. MODEL DEFINITION (Must match training) ---
class VisualStudent(nn.Module):
    def __init__(self, yolo_path, target_dim=512):
        super().__init__()
        yolo_obj = YOLO(yolo_path)
        # Extract backbone from YOLO
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
# Dataset Path
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
    y_probs = [] # Raw sigmoid probabilities
    
    categories = {'not_improved': 0, 'improved': 1}

    print("Running evaluation on validation set...")

    for category_name, label in categories.items():
        category_path = os.path.join(VAL_DIR, category_name)
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
                
                y_true.append(label)
                y_probs.append(prob.item())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # --- 3. ROC CURVE AND AUROC ---
    auroc = roc_auc_score(y_true, y_probs)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # --- 4. FIND BEST THRESHOLD FOR ACCURACY ---
    accuracies = []
    for t in thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        accuracies.append(np.mean(y_pred_t == y_true))
    
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]

    print("-" * 30)
    print(f"AUROC Score: {auroc:.4f}")
    print(f"Optimal Threshold (Max Accuracy): {best_threshold:.4f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("-" * 30)

    # --- 5. GENERATE BEST CONFUSION MATRIX ---
    y_pred_best = (y_probs >= best_threshold).astype(int)
    cm_best = confusion_matrix(y_true, y_pred_best)

    plt.figure(figsize=(8, 6))
    # We transpose cm (cm.T) so y-axis is 'Predicted' and x-axis is 'Actual' per your original style
    sns.heatmap(cm_best.T, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Not Improved', 'Improved'], 
                yticklabels=['Not Improved', 'Improved'])
    plt.ylabel('Predicted (Optimal Threshold)')
    plt.xlabel('Actual')
    plt.title(f'Confusion Matrix at Best Threshold ({best_threshold:.2f})')
    plt.savefig(f"{BASE_PATH}/best_confusion_matrix.png")
    print(f"Best confusion matrix saved to {BASE_PATH}/best_confusion_matrix.png")

    # --- 6. PLOT ROC CURVE ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # Mark the best accuracy point
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, label=f'Best Accuracy Point', zorder=5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f"{BASE_PATH}/roc_curve.png")
    print(f"ROC curve saved to {BASE_PATH}/roc_curve.png")

    # Print final report
    print("\nClassification Report (at Optimal Threshold):")
    print(classification_report(y_true, y_pred_best, target_names=['Not Improved', 'Improved']))

if __name__ == "__main__":
    evaluate()
