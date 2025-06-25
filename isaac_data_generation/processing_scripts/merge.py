import json
import os
import shutil

def load_coco_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_coco_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def merge_coco_datasets(anno1, anno2, image_dir1, image_dir2, output_img_dir):
    os.makedirs(output_img_dir, exist_ok=True)

    merged = {
        'images': [],
        'annotations': [],
        'categories': anno1['categories'],  # assuming same categories
    }

    image_id_map = {}
    ann_id = 1
    img_id = 1

    # Process dataset 1
    for img in anno1['images']:
        old_id = img['id']
        img['id'] = img_id
        image_id_map[old_id] = img_id
        merged['images'].append(img)

        # copy image
        shutil.copy(os.path.join(image_dir1, img['file_name']),
                    os.path.join(output_img_dir, img['file_name']))
        img_id += 1

    for ann in anno1['annotations']:
        ann['id'] = ann_id
        ann['image_id'] = image_id_map[ann['image_id']]
        merged['annotations'].append(ann)
        ann_id += 1

    # Reset map for dataset 2
    image_id_map = {}

    for img in anno2['images']:
        old_id = img['id']
        img['id'] = img_id
        image_id_map[old_id] = img_id
        merged['images'].append(img)

        # copy image
        shutil.copy(os.path.join(image_dir2, img['file_name']),
                    os.path.join(output_img_dir, img['file_name']))
        img_id += 1

    for ann in anno2['annotations']:
        ann['id'] = ann_id
        ann['image_id'] = image_id_map[ann['image_id']]
        merged['annotations'].append(ann)
        ann_id += 1

    return merged

# === USAGE ===
anno_path1 = '/home/rics/Desktop/1041188_dataset 2025-05-08 13-26-52/annotations/instances.json'
img_dir1 = '/home/rics/Desktop/1041188_dataset 2025-05-08 13-26-52/Videira/dataset 2025-05-08 13-26-52'

img_dir2 = '/home/rics/Desktop/J/images'
anno_path2 = '/home/rics/Desktop/J/instances.json'
output_dir = '/home/rics/Desktop/K/images'
output_anno = '/home/rics/Desktop/K/instances.json'

# Load annotations
anno1 = load_coco_json(anno_path1)
anno2 = load_coco_json(anno_path2)

# Merge datasets
merged_anno = merge_coco_datasets(anno1, anno2, img_dir1, img_dir2, output_dir)

# Save merged annotation
os.makedirs(os.path.dirname(output_anno), exist_ok=True)
save_coco_json(merged_anno, output_anno)