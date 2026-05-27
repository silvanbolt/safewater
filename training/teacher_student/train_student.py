import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from ultralytics import YOLO
from PIL import Image, ImageFile

# Ensure train_teacher.py is in the same directory to import these:
from train_teacher import MultimodalTeacher, WaterpointMultimodalDataset, get_metadata_preprocessor, engineer_features

# --- CONFIGURATION ---
BASE_PATH = "/work/courses/dslab/team7/teacher_student"
CSV_PATH = f"{BASE_PATH}/data/Ethiopia_dataset_clean.csv"
IMAGE_ROOT = "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split/"
YOLO_CHECKPOINT = f"{BASE_PATH}/data/pre-trained-weights.pt"
TEACHER_LOAD_PATH = f"{BASE_PATH}/multimodal_teacher.pth"
STUDENT_SAVE_PATH = f"{BASE_PATH}/distilled_student.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Allow loading of truncated/corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 1. THE STUDENT MODEL ---
class VisualStudent(nn.Module):
    def __init__(self, yolo_path, target_dim=512):
        super().__init__()
        # Load pre-trained YOLO and extract the backbone
        yolo_obj = YOLO(yolo_path)
        self.backbone = nn.Sequential(*list(yolo_obj.model.model.children())[:-1])
        
        # Dynamically find the output channels of the backbone
        with torch.no_grad():
            dummy_out = self.backbone(torch.zeros(1, 3, 640, 640))
            if isinstance(dummy_out, list): dummy_out = dummy_out[-1]
            in_channels = dummy_out.shape[1]

        # Projection Head: Mimics the Teacher's fusion embedding
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, target_dim) 
        )
        
        # Classification Head: Predicts 'improved' (1) vs 'not_improved' (0)
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

# --- 2. MAIN EXECUTION ---
if __name__ == "__main__":
    # A. Load Metadata and Transformer
    df_raw = pd.read_csv(CSV_PATH)
    transformer = get_metadata_preprocessor(df_raw)
    
    # B. RECOVER meta_dim (Option 1)
    # We pass a few rows through the pipeline to see what size the Teacher expects
    print("Recovering meta_dim from transformer...")
    df_engineered = engineer_features(df_raw.head(5))
    dummy_meta = transformer.transform(df_engineered)
    meta_dim = dummy_meta.shape[1]
    print(f"Detected meta_dim: {meta_dim} (Original metadata features expanded to 31 dimensions)")

    # C. Initialize Dataset & Loader
    dataset = WaterpointMultimodalDataset(CSV_PATH, IMAGE_ROOT, transformer)
    
    # Balanced Sampler logic
    labels = dataset.labels.flatten().astype(int)
    counts = np.bincount(labels)
    class_weights = 1.0 / counts
    sample_weights = torch.from_numpy(class_weights[labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=2)

    # D. Load Teacher (Frozen)
    # weights_only=False is used to allow loading the saved Scikit-Learn transformer object
    print(f"Loading Teacher from {TEACHER_LOAD_PATH}...")
    checkpoint = torch.load(TEACHER_LOAD_PATH, map_location=DEVICE, weights_only=False)
    
    teacher = MultimodalTeacher(meta_dim, YOLO_CHECKPOINT).to(DEVICE)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # E. Initialize Student
    student = VisualStudent(YOLO_CHECKPOINT).to(DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    
    # Dual Loss: Align embeddings + Correct classification
    mse_criterion = nn.MSELoss()    # For Knowledge Distillation
    bce_criterion = nn.BCELoss()    # For Task Accuracy

    print("Starting Distillation... Student learning to 'see' metadata in pixels.")

    for epoch in range(20):
        student.train()
        epoch_distill_loss = 0
        epoch_class_loss = 0
        
        for imgs, metas, targets in loader:
            imgs, metas, targets = imgs.to(DEVICE), metas.to(DEVICE), targets.to(DEVICE)
            
            # 1. Get Teacher's Expert Opinion (Uses Image + Metadata)
            with torch.no_grad():
                expert_emb = teacher(imgs, metas, return_distill_emb=True)
            
            # 2. Get Student's Guess (Uses Image ONLY)
            student_emb, student_pred = student(imgs)
            
            # 3. Calculate Loss
            distill_loss = mse_criterion(student_emb, expert_emb)
            class_loss = bce_criterion(student_pred, targets)
            
            # Combined Loss: Adjust weights if one is dominating the other
            total_loss = distill_loss + class_loss 
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_distill_loss += distill_loss.item()
            epoch_class_loss += class_loss.item()
            
        avg_d = epoch_distill_loss / len(loader)
        avg_c = epoch_class_loss / len(loader)
        print(f"Epoch {epoch+1}/20 | Distill Loss: {avg_d:.6f} | Class Loss: {avg_c:.4f}")

    # F. Save the Distilled Student
    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f"Distillation Complete. Student model saved to: {STUDENT_SAVE_PATH}")
