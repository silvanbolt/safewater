import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import silhouette_score

# --- Configuration ---
ETHIOPIA_ROOT = "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split"
KENYA_ROOT = "/work/courses/dslab/team7/Kenya/yolo_waterpoint_data_Kenya_split"
MODEL_PATH = "/work/courses/dslab/team7/yolo/weights1/last.pt"
SAVE_NAME = "waterpoint_umap_visualization.png"

def get_combined_image_list():
    """Gathers paths from both Ethiopia (flattened) and Kenya datasets."""
    path_label_pairs = []
    valid_extensions = ('.png', '.jpg', '.jpeg')

    # 2. Process Kenya
    # Folders: Improved, Unimproved
    print("Scanning Kenya data...")
    kenya_map = {
        "Improved": "kenya_improved",
        "Unimproved": "kenya_unimproved"
    }
    
    for folder_name, label in kenya_map.items():
        folder = os.path.join(KENYA_ROOT, folder_name)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
            for f in files:
                path_label_pairs.append((os.path.join(folder, f), label))
        else:
            print(f"Warning: Kenya folder not found: {folder}")

     # 1. Process Ethiopia (Flattening train/val into just category labels)
    # Categories: improved, not_improved
    eth_categories = ['improved', 'not_improved']
    eth_splits = ['train', 'val']
    
    print("Scanning Ethiopia data...")
    for split in eth_splits:
        for cat in eth_categories:
            folder = os.path.join(ETHIOPIA_ROOT, split, cat)
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
                for f in files:
                    # Label is just the category (e.g., 'improved')
                    path_label_pairs.append((os.path.join(folder, f), f"ethiopia_{cat}"))

    # 2. Process Kenya
    # Folders: Improved, Unimproved
    """print("Scanning Kenya data...")
    kenya_map = {
        "Improved": "kenya_improved",
        "Unimproved": "kenya_unimproved"
    }
    
    for folder_name, label in kenya_map.items():
        folder = os.path.join(KENYA_ROOT, folder_name)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
            for f in files:
                path_label_pairs.append((os.path.join(folder, f), label))
        else:
            print(f"Warning: Kenya folder not found: {folder}")"""

    return path_label_pairs

# --- Execution ---

# 1. Load Model
print(f"Loading Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 2. Collect Paths
all_data = get_combined_image_list()
print(f"Found {len(all_data)} total image candidates.")

# 3. Extract Embeddings
print("Extracting Embeddings (Batch-Processing)...")
final_embeddings = []
final_labels = []
batch_size = 32 

for i in range(0, len(all_data), batch_size):
    batch_subset = all_data[i:i + batch_size]
    batch_paths = [item[0] for item in batch_subset]
    batch_tags = [item[1] for item in batch_subset]
    
    try:
        results = model.embed(batch_paths, verbose=False)
        for idx, r in enumerate(results):
            emb = r.cpu().numpy().flatten()
            if emb.size > 0:
                final_embeddings.append(emb)
                final_labels.append(batch_tags[idx])
    except Exception as e:
        print(f"Error processing batch starting at index {i}: {e}")

X = np.array(final_embeddings)
y = np.array(final_labels)

print(f"Successfully extracted {len(X)} embeddings.")

# 4. Save Raw Data
np.save('waterpoint_embeddings.npy', X)
np.save('waterpoint_labels.npy', y)

# 5. UMAP Dimension Reduction
print("Starting UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_2d = reducer.fit_transform(X)

# 6. Quality Metric
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

plt.title(f'YOLO Waterpoint Embeddings (UMAP)\nEthiopia vs Kenya | Silhouette Score: {score:.4f}')
plt.legend(title='Region & Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(SAVE_NAME, dpi=300)
print(f"Done! Visualization saved as '{SAVE_NAME}'.")
