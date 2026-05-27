import os
import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---
train_path = Path("/work/courses/dslab/team7/multi-national/yolo_classification_dataset/train")
target_count = 4000  # We want every class to have at least this many images

def balance_dataset(root_path, target):
    classes = [d for d in root_path.iterdir() if d.is_dir()]
    
    for cls in classes:
        images = list(cls.glob('*')) # Supports .jpg, .png, etc.
        current_count = len(images)
        
        print(f"Class: {cls.name} | Current: {current_count}")
        
        if current_count < target:
            to_add = target - current_count
            print(f"  --> Oversampling: Adding {to_add} images...")
            
            # Randomly pick images from the existing set to duplicate
            for i in range(to_add):
                source_img = random.choice(images)
                # Create a new filename to avoid overwriting
                new_name = f"aug_{i}_{source_img.name}"
                dest_path = cls / new_name
                
                # Copy the file
                shutil.copy2(source_img, dest_path)
        else:
            print("  --> Class is sufficient. Skipping.")

if __name__ == "__main__":
    balance_dataset(train_path, target_count)
