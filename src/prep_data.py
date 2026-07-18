import os
import shutil
import random

source_dir = "dataset_v2"
target_dir = "dataset_split"
classes = ["PET", "HDPE", "PP", "LDPE", "MISC", "BACKGROUND"]
split_ratio = 0.8

print("Organizing dataset for YOLOv8 Classification...")

for c in classes:
    os.makedirs(os.path.join(target_dir, "train", c), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "val", c), exist_ok=True)
    
    class_dir = os.path.join(source_dir, c)
    if not os.path.exists(class_dir):
        print(f"Warning: {class_dir} not found.")
        continue
        
    images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    
    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    print(f"Moving {c} -> {len(train_imgs)} Train | {len(val_imgs)} Val")
    
    for img in train_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(target_dir, "train", c, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(target_dir, "val", c, img))

print("\nDone! Dataset is ready for training!")
