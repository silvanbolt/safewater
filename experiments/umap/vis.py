import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Required for cluster environments without a display
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import umap
from sklearn.metrics import silhouette_score

# --- CONFIGURATION ---
ROOT_PATH = "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split"
MODEL_PATH = "/work/courses/dslab/team7/yolo/weights1/last.pt"
CATEGORIES = ['improved', 'not_improved']
SPLITS = ['train', 'val']
SAVE_NAME = "waterpoint_umap_visualization.png"

def get_image_list(root_path):
    """Gathers all image paths and their corresponding labels."""
    path_label_pairs = []
    for split in SPLITS:
        for category in CATEGORIES:
            folder = os.path.join(root_path, split, category)
            if not os.path.exists(folder):
                print(f"Warning: Folder not found: {folder}")
                continue

            print(f"Scanning {split}/{category}...")
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for f in files:
                full_path = os.path.join(folder, f)
                path_label_pairs.append((full_path, f"{split}_{category}"))
    return path_label_pairs

# 1. Load Model
print(f"Loading Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 2. Collect Paths
all_data = get_image_list(ROOT_PATH)
print(f"Found {len(all_data)} total image candidates.")

# 3. Extract Embeddings (Synchronized)
print("Extracting Embeddings (Batch-Processing)...")
final_embeddings = []
final_labels = []
batch_size = 32 

for i in range(0, len(all_data), batch_size):
    batch_subset = all_data[i:i + batch_size]
    batch_paths = [item[0] for item in batch_subset]
    batch_tags = [item[1] for item in batch_subset]
    
    try:
        # model.embed returns a list of Results objects
        results = model.embed(batch_paths, verbose=False)
        
        for idx, r in enumerate(results):
            # Convert torch tensor to numpy
            emb = r.cpu().numpy().flatten()
            
            # Double check the embedding is valid and not empty
            if emb.size > 0:
                final_embeddings.append(emb)
                final_labels.append(batch_tags[idx])
                
    except Exception as e:
        print(f"Error processing batch starting at {batch_paths[0]}: {e}")

X = np.array(final_embeddings)
y = np.array(final_labels)

print(f"Successfully extracted {len(X)} embeddings and {len(y)} labels.")

# 4. Save Raw Data
np.save('waterpoint_embeddings.npy', X)
np.save('waterpoint_labels.npy', y)

# 5. UMAP Dimension Reduction
print("Starting UMAP (reducing to 2D)...")
# Note: random_state=42 makes it reproducible but forces n_jobs=1
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_2d = reducer.fit_transform(X)

# 6. Quality Metric
print("Calculating Silhouette Score...")
# High-dimensional score (on X) is more meaningful for feature quality
score = silhouette_score(X, y)
print(f"Silhouette Score: {score:.4f}")

# 7. Visualization
plt.figure(figsize=(14, 10))
sns.scatterplot(
    x=embedding_2d[:, 0], 
    y=embedding_2d[:, 1], 
    hue=y, 
    palette='Set1', 
    s=60, 
    alpha=0.7,
    edgecolor='w',
    linewidth=0.5
)

plt.title(f'YOLO Waterpoint Embeddings (UMAP)\nModel: last.pt | Silhouette Score: {score:.4f}')
plt.legend(title='Category & Split', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(SAVE_NAME, dpi=300)
print(f"Done! Visualization saved as '{SAVE_NAME}'.")
