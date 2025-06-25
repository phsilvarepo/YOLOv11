import os
import shutil
import random

def split_yolo_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, move_files=False):
    # Define output directories
    images_output = os.path.join(output_dir, 'images')
    labels_output = os.path.join(output_dir, 'labels')

    # Ensure output directories exist
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(images_output, subset), exist_ok=True)
        os.makedirs(os.path.join(labels_output, subset), exist_ok=True)

    # Get all image files starting with 'rgb_' and having valid extensions
    image_files = [f for f in os.listdir(images_dir)
                   if f.startswith('rgb_') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Filter out images without corresponding label files
    image_files = [f for f in image_files if os.path.exists(os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt'))]

    # Shuffle and split
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    def move_or_copy(file_list, subset):
        for img_file in file_list:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'

            src_img = os.path.join(images_dir, img_file)
            src_lbl = os.path.join(labels_dir, label_file)

            dst_img = os.path.join(images_output, subset, img_file)
            dst_lbl = os.path.join(labels_output, subset, label_file)

            # Copy or move the files
            if move_files:
                shutil.move(src_img, dst_img)
                shutil.move(src_lbl, dst_lbl)
            else:
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)

    # Move or copy to destination
    move_or_copy(train_files, 'train')
    move_or_copy(val_files, 'val')

    print("✅ Split complete.")
    print(f"➡️ {len(train_files)} training images")
    print(f"➡️ {len(val_files)} validation images")

# Example usage
split_yolo_dataset(
    images_dir='/home/rics/omni.replicator_out/leaf_3',
    labels_dir='/home/rics/omni.replicator_out/leaf_3/yolo_labels/',
    output_dir='/home/rics/Desktop/Videira_Dataset',
    train_ratio=0.8,
    move_files=True
)
