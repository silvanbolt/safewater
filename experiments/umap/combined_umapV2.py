import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from ultralytics import YOLO
from scipy.spatial.distance import cdist

# --- Configuration ---
ETHIOPIA_ROOT = "/work/courses/dslab/team7/yolo/yolo_waterpoint_data_split"
KENYA_ROOT = "/work/courses/dslab/team7/Kenya/yolo_waterpoint_data_Kenya_split"
MODEL_PATH = "/work/courses/dslab/team7/yolo/weights1/last.pt"
SAVE_NAME = "waterpoint_umap_visualization.png"

def get_combined_image_list():
    path_label_pairs = []
    valid_extensions = ('.png', '.jpg', '.jpeg')

    # 1. Process Kenya (Using your exact logic)
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

    # 2. Process Ethiopia (Using your exact logic)
    eth_categories = ['improved', 'not_improved']
    eth_splits = ['train', 'val']
    print("Scanning Ethiopia data...")
    for split in eth_splits:
        for cat in eth_categories:
            folder = os.path.join(ETHIOPIA_ROOT, split, cat)
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
                for f in files:
                    path_label_pairs.append((os.path.join(folder, f), f"ethiopia_{cat}"))
            else:
                print(f"Warning: Ethiopia folder not found: {folder}")
    # 1. Process Kenya
    """for folder_name, label in kenya_map.items():
        folder = os.path.join(KENYA_ROOT, folder_name)
        if os.path.exists(folder):
            # Get all valid files
            all_files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
            
            # SLICE HERE: Only take the first 100
            files = all_files[:20] 
            
            for f in files:
                path_label_pairs.append((os.path.join(folder, f), label))
            print(f"Added {len(files)} images for {label}")

    # 2. Process Ethiopia
    print("Scanning Ethiopia data (limiting to 100 per class per split)...")
    for split in eth_splits:
        for cat in eth_categories:
            folder = os.path.join(ETHIOPIA_ROOT, split, cat)
            if os.path.exists(folder):
                all_files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
                
                # SLICE HERE: Only take 100
                files = all_files[:20]
                
                for f in files:
                    path_label_pairs.append((os.path.join(folder, f), f"ethiopia_{cat}"))"""


    return path_label_pairs

# --- 1. Load Model & Data ---
model = YOLO(MODEL_PATH)
all_data = get_combined_image_list()
print(f"Found {len(all_data)} total images.")

# --- 2. Extraction Loop ---
"""final_embeddings, final_labels, final_probs, final_paths = [], [], [], []
batch_size = 32

print("Extracting features and probabilities...")
for i in range(0, len(all_data), batch_size):
    batch_subset = all_data[i : i + batch_size]
    paths = [item[0] for item in batch_subset]
    tags = [item[1] for item in batch_subset]

    try:
        # Get embeddings and predictions
        emb_results = model.embed(paths, verbose=False)
        pred_results = model.predict(paths, verbose=False)

        for idx, r in enumerate(emb_results):
            # Extract embedding vector
            vector = r.cpu().numpy().flatten()
            # Extract confidence of top prediction
            conf = pred_results[idx].probs.top1conf.item()
            
            final_embeddings.append(vector)
            final_labels.append(tags[idx])
            final_probs.append(conf)
            final_paths.append(paths[idx])

    except Exception as e:
        print(f"Error at batch {i}: {e}")
        if i == 0:
            print("CRITICAL: First batch failed. Check model and paths."); exit()

# Convert lists to numpy arrays for UMAP
X = np.array(final_embeddings)
y = np.array(final_labels)
p = np.array(final_probs)
paths_arr = np.array(final_paths)"""



# --- 2. Extraction Loop ---
final_embeddings, final_labels, final_probs, final_paths = [], [], [], []
batch_size = 32

print("Extracting features and probabilities...")
for i in range(0, len(all_data), batch_size):
    batch_subset = all_data[i : i + batch_size]
    batch_paths = [item[0] for item in batch_subset]  # Defined here
    batch_tags = [item[1] for item in batch_subset]   # Defined here

    try:
        # Get embeddings and predictions (Forcing CPU for GTX 1080 Ti compatibility)
        embeddings = model.embed(batch_paths, verbose=False, device='cpu')
        predictions = model.predict(batch_paths, verbose=False, device='cpu', embed=None)
        """if i == 0:
            # predictions is a list, so we check the length of the list 
            # and the shape of the first tensor inside it
            print(f"Batch 0: Found a list of {len(predictions)} prediction tensors.")
            
            # This checks the dimensions of the first image's prediction
            first_pred_shape = predictions[0].shape
            print(f"Shape of the first prediction tensor: {first_pred_shape}")"""

        for idx, emb_tensor in enumerate(embeddings):
            # 1. Process Embedding
            emb = emb_tensor.cpu().numpy().flatten()
            
            if emb.size > 0:
                pred = predictions[idx]
                #here:
                
                # --- NEW DEBUG PRINT ---
                """if i == 0 and idx == 0: # Just print for the very first image
                    if pred.probs is not None:
                        # This tells you the shape of the probability array
                        print(f"Probabilities shape: {pred.probs.data.shape}")
                        print(f"Raw values: {pred.probs.data.tolist()}")
                    else:
                        print("No probs found in this Results object.")"""

                # 2. Process Probability (Flexible check for object vs tensor)
                if hasattr(pred, 'probs') and pred.probs is not None:
                    conf = pred.probs.top1conf.item()
                    #print("found probs")
                else:
                    # If it's a raw tensor, take the max value
                    probs = F.softmax(pred, dim=-1) 
                    conf = probs.max().item()
                
                final_embeddings.append(emb)
                final_labels.append(batch_tags[idx])
                final_probs.append(conf)
                final_paths.append(batch_paths[idx])

    except Exception as e:
        print(f"Error at batch {i}: {e}")
        if i == 0:
            print("CRITICAL: First batch failed. Check if batch_paths is defined above.")
            exit()

# Final check before UMAP
X = np.array(final_embeddings)
y = np.array(final_labels)
p = np.array(final_probs)
paths_arr = np.array(final_paths)

print(f"Successfully collected {len(X)} samples.")





if X.size == 0:
    print("ERROR: No embeddings were collected."); exit()

# --- 3. UMAP Reduction ---
print("Running UMAP (this may take a few minutes)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_2d = reducer.fit_transform(X)

# --- 4. Visualization ---
plt.figure(figsize=(14, 10))
# Shuffle drawing order so Kenya isn't buried under Ethiopia
indices = np.random.permutation(len(embedding_2d))

sns.scatterplot(
    x=embedding_2d[indices, 0], 
    y=embedding_2d[indices, 1],
    hue=y[indices], 
    size=p[indices], 
    sizes=(1, 40),
    palette='Set1', 
    alpha=0.3, # Transparent dots to see the overlap clearly
    edgecolor=None
)
plt.title("Waterpoint Embeddings: Ethiopia vs Kenya")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(SAVE_NAME, bbox_inches='tight', dpi=300)
print(f"Plot saved: {SAVE_NAME}")

# --- 5. Find Images where Kenya Improved looks like Ethiopia Not Improved ---
def get_outliers(target_class, compare_class, title):
    # Get centroids of the comparison class in original embedding space (X)
    compare_X = X[y == compare_class]
    if len(compare_X) == 0: return
    centroid = compare_X.mean(axis=0)

    # Get target class points
    target_mask = (y == target_class)
    target_X = X[target_mask]
    target_paths = paths_arr[target_mask]

    # Calculate distances
    dists = cdist(target_X, [centroid], 'euclidean').flatten()
    closest_indices = np.argsort(dists)[:10]

    print(f"\n--- {title} ---")
    for idx in closest_indices:
        print(f"Dist: {dists[idx]:.4f} | Path: {target_paths[idx]}")

get_outliers("kenya_improved", "ethiopia_not_improved", "Top 10 Kenya Improved near Ethiopia Not Improved")
get_outliers("kenya_unimproved", "ethiopia_improved", "Top 10 Kenya Unimproved near Ethiopia Improved")
