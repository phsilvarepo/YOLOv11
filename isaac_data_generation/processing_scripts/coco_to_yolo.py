import json
import os
from tqdm import tqdm

def coco_to_yolov11_segmentation(coco_json_path, output_dir, images_dir=None):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Map image_id to file_name, width, height
    image_info = {img['id']: img for img in coco['images']}

    # Map categories to class indices (0-based)
    cat_id_to_class = {cat['id']: i for i, cat in enumerate(coco['categories'])}

    # Group annotations by image
    from collections import defaultdict
    image_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        if isinstance(ann['segmentation'], list):  # Only keep polygon segmentations
            image_to_anns[ann['image_id']].append(ann)

    for img_id, anns in tqdm(image_to_anns.items()):
        img = image_info[img_id]
        width, height = img['width'], img['height']
        img_name = os.path.splitext(img['file_name'])[0]
        label_path = os.path.join(output_dir, f"{img_name}.txt")

        with open(label_path, 'w') as f:
            for ann in anns:
                class_id = cat_id_to_class[ann['category_id']]
                for seg in ann['segmentation']:
                    # Flatten the polygon and normalize
                    norm_seg = [str(coord / width if i % 2 == 0 else coord / height)
                                for i, coord in enumerate(seg)]
                    f.write(f"{class_id} " + " ".join(norm_seg) + "\n")


coco_to_yolov11_segmentation(
    coco_json_path='/home/rics/Desktop/1043502_dataset 2025-05-20 11-03-20/annotations/last.json',
    output_dir='labels',
    images_dir='images'  # optional, for checking file names
)