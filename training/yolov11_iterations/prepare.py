import pandas as pd
import os
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
#csv_file = '/work/courses/dslab/team7/multi-national/luis_upload3/All_plus_half_Ethiopia_clean_dataset.csv'  # Replace with your actual CSV file name
csv_file = '/work/courses/dslab/team7/rest_world_7/file.csv'
source_dir = '/work/courses/dslab/team7/rest_world_7/images'
output_dir = '/work/courses/dslab/team7/multi-national/yolo_classification_dataset/train'

# Create the output root directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the data
df = pd.read_csv(csv_file)

print(f"Processing {len(df)} images...")

# Statistics
copied_count = 0
missing_count = 0

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    label = str(row['label']).strip()
    image_path_raw = str(row['image_path'])
    
    # 1. Extract the final image name (e.g., '88134.jpg')
    image_name = os.path.basename(image_path_raw)
    
    # 2. Define source and target paths
    src_path = os.path.join(source_dir, image_name)
    
    # Sanitize label for folder naming (remove slashes, etc.)
    safe_label = label.replace('/', '_').replace('\\', '_')
    target_folder = os.path.join(output_dir, safe_label)
    dst_path = os.path.join(target_folder, image_name)
    
    # 3. Create label folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 4. Copy the file
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        copied_count += 1
    else:
        missing_count += 1

print(f"\nTask Complete!")
print(f"Successfully copied: {copied_count}")
print(f"Files not found in source: {missing_count}")
print(f"Dataset is ready at: {os.path.abspath(output_dir)}")
