import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import WeightedRandomSampler

# --- CONFIGURATION ---
CSV_PATH = "/work/courses/dslab/team7/teacher_student/data/Ethiopia_dataset_clean.csv"
IMAGE_ROOT = "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split/" 
YOLO_CHECKPOINT = "/work/courses/dslab/team7/teacher_student/data/pre-trained-weights.pt"
SAVE_PATH = "/work/courses/dslab/team7/teacher_student/multimodal_teacher.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15

# --- 1. FEATURE ENGINEERING ---
def engineer_features(df):
    df = df.copy()
    report_yr = pd.to_datetime(df['#report_date'], format='%Y %b %d %I:%M:%S %p', errors='coerce').dt.year
    install_yr = pd.to_numeric(df['#install_year'], errors='coerce')
    rehab_yr = pd.to_numeric(df['#rehab_year'], errors='coerce')
    
    df['age'] = (report_yr - install_yr.fillna(report_yr)).clip(0, 100)
    df['rehab_age'] = (report_yr - rehab_yr.fillna(install_yr).fillna(report_yr)).clip(0, 100)
    df['#fecal_coliform_value'] = pd.to_numeric(df['#fecal_coliform_value'], errors='coerce').fillna(0)
    
    cat_cols = ['water_source_clean', 'water_tech_clean', 'management_clean', 'status_clean']
    for col in cat_cols:
        df[col] = df[col].fillna('unknown')
    return df

def get_metadata_preprocessor(df):
    df = engineer_features(df)
    cat_cols = ['water_source_clean', 'water_tech_clean', 'management_clean', 'status_clean']
    num_cols = ['age', 'rehab_age', '#fecal_coliform_value']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    preprocessor.fit(df)
    return preprocessor

# --- 2. DATASET ---
class WaterpointMultimodalDataset(Dataset):
    def __init__(self, csv_path, img_root, transformer, exclude_val=True):
        raw_df = pd.read_csv(csv_path)
        full_df = engineer_features(raw_df)
        
        # --- NEW: Exclusion Logic ---
        val_root = "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split/val"
        excluded_filenames = set()
        
        if exclude_val:
            for cat in ['improved', 'not_improved']:
                cat_path = os.path.join(val_root, cat)
                if os.path.exists(cat_path):
                    # Get all filenames in the validation subfolders
                    files = [f for f in os.listdir(cat_path) if f.lower().endswith('.jpg')]
                    excluded_filenames.update(files)
            print(f"Found {len(excluded_filenames)} images to exclude from training.")

        self.valid_indices = []
        self.actual_paths = []
        self.labels_list = []
        
        print("Verifying image existence and integrity...")
        for i, row in full_df.iterrows():
            img_filename = os.path.basename(row['image_path'])
            # NEW
            if img_filename in excluded_filenames:
                continue

            found_path = None
            current_label = None
            
            # 1. Find the path
            for split in ['train', 'val']:
                for cat in ['improved', 'not_improved']:
                    test_path = os.path.join(img_root, split, cat, img_filename)
                    if os.path.exists(test_path):
                        found_path = test_path
                        current_label = 1.0 if cat == 'improved' else 0.0
                        break
                if found_path: break
            
            # 2. Check Integrity: Try to open the file now
            if found_path:
                try:
                    with Image.open(found_path) as img:
                        img.load() # Verify file integrity without loading pixels
                    self.actual_paths.append(found_path)
                    self.valid_indices.append(i)
                    self.labels_list.append(current_label)
                except Exception:
                    print(f"Skipping corrupted file: {found_path}")
                    continue

        # Create the filtered dataframe
        self.df = full_df.iloc[self.valid_indices].reset_index(drop=True)
        
        # Ensure metadata and labels stay perfectly aligned with the survivors
        self.meta_features = transformer.transform(self.df)
        self.labels = np.array(self.labels_list, dtype=np.float32)
        
        """self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])"""
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.RandomHorizontalFlip(p=0.5), # New
            transforms.RandomRotation(degrees=15),   # New
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # New
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        # Now we can safely load because we've already verified everything
        image = Image.open(self.actual_paths[idx]).convert('RGB')
        image = self.transform(image)
        meta = torch.FloatTensor(self.meta_features[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return image, meta, label

# --- 3. MULTIMODAL TEACHER MODEL ---
class MultimodalTeacher(nn.Module):
    def __init__(self, meta_dim, yolo_path):
        super().__init__()
        yolo_obj = YOLO(yolo_path)
        # Accessing the internal PyTorch module
        base_model = yolo_obj.model
        
        # Robust Feature Extractor: Grab all layers except the last Classification head
        # This works for YOLOv8-cls and standard backbones
        self.feature_extractor = nn.Sequential(*list(base_model.model.children())[:-1])
        
        # Find input channels dynamically
        with torch.no_grad():
            dummy_out = self.feature_extractor(torch.zeros(1, 3, 640, 640))
            if isinstance(dummy_out, list): dummy_out = dummy_out[-1]
            in_channels = dummy_out.shape[1]

        self.img_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU()
        )
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fusion_layer = nn.Linear(256 + 256, 512)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, img, meta, return_distill_emb=False):
        with torch.no_grad():
            features = self.feature_extractor(img)
            if isinstance(features, list): features = features[-1]
        
        v_f = self.img_proj(features)
        m_f = self.meta_branch(meta)
        combined = torch.cat((v_f, m_f), dim=1)
        emb = self.fusion_layer(combined)
        
        if return_distill_emb: return emb
        return self.classifier(emb)

# --- 4. EXECUTION ---
if __name__ == "__main__":
    df_init = pd.read_csv(CSV_PATH)
    transformer = get_metadata_preprocessor(df_init)
    dataset = WaterpointMultimodalDataset(CSV_PATH, IMAGE_ROOT, transformer)
    
    # Simple Weighting based on instance count
    labels = dataset.labels.flatten().astype(int)
    counts = np.bincount(labels)
    class_weights = 1.0 / counts
    sample_weights = torch.from_numpy(class_weights[labels])
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=4)
    
    model = MultimodalTeacher(dataset.meta_features.shape[1], YOLO_CHECKPOINT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.BCELoss()

    print(f"Verified {len(dataset)} samples. Training Balanced Teacher...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for imgs, metas, targets in loader:
            imgs, metas, targets = imgs.to(DEVICE), metas.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs, metas)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/15 | Loss: {epoch_loss/len(loader):.4f}")

    torch.save({'model_state_dict': model.state_dict(), 'transformer': transformer}, SAVE_PATH)
    print(f"Teacher saved to {SAVE_PATH}")
