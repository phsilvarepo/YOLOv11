import os
import cv2
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
input_dir = "/home/rics/Desktop/test/"
output_label_dir = os.path.join(input_dir, "yolo_labels/")
output_vis_dir = os.path.join(input_dir, "visualizations/")
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_vis_dir, exist_ok=True)

MIN_AREA = 10  # Minimum contour area in pixels

# === Helper to parse RGBA string to tuple ===
def parse_color_str(color_str):
    return tuple(map(int, color_str.strip("()").split(",")))

# === Get list of mask files ===
all_files = os.listdir(input_dir)
png_files = [f for f in all_files if f.startswith("instance_segmentation_") and f.endswith(".png")]

# === Process each file ===
for png_filename in tqdm(png_files, desc="Processing masks"):
    # Derive corresponding json path
    id_str = png_filename.replace("instance_segmentation_", "").replace(".png", "")
    json_filename = f"instance_segmentation_semantics_mapping_{id_str}.json"
    
    png_path = os.path.join(input_dir, png_filename)
    json_path = os.path.join(input_dir, json_filename)
    
    if not os.path.exists(json_path):
        print(f"[WARNING] Missing JSON for: {png_filename}")
        continue
    
    # Load RGBA mask (via PIL for consistency)
    mask_rgba = np.array(Image.open(png_path).convert("RGBA"))
    height, width = mask_rgba.shape[:2]

    # Visualization image (BGR for OpenCV)
    vis_img = cv2.cvtColor(np.array(Image.open(png_path).convert("RGB")), cv2.COLOR_RGB2BGR)

    # Load color map JSON
    with open(json_path, "r") as f:
        color_map = json.load(f)
    
    # Build class name to id map
    class_name_to_id = {}
    class_counter = 0
    for info in color_map.values():
        cls = info["class"]
        if cls not in class_name_to_id:
            class_name_to_id[cls] = class_counter
            class_counter += 1

    # Filter leaf colors only
    leaf_colors = {
        parse_color_str(color_str): class_name_to_id["leaf"]
        for color_str, info in color_map.items()
        if info["class"] == "leaf"
    }

    output_lines = []
    instance_count = 0

    for color_rgba, class_id in leaf_colors.items():
        # Create mask for this color
        mask_single = np.all(mask_rgba == color_rgba, axis=-1).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # Pick largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < MIN_AREA:
            continue

        polygon = largest_contour.squeeze()
        if polygon.ndim != 2 or len(polygon) < 3:
            continue

        # Normalize polygon points to [0,1]
        norm_points = [(x / width, y / height) for x, y in polygon]
        line = [str(0)] + [f"{x:.6f} {y:.6f}" for x, y in norm_points]
        output_lines.append(" ".join(line))
        instance_count += 1

        # Draw polygon on visualization image
        polygon_pts = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
        cv2.polylines(vis_img, [polygon_pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(vis_img, "leaf", tuple(polygon_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Save YOLO labels file
    label_file = os.path.join(output_label_dir, f"rgb_{id_str}.txt")
    with open(label_file, "w") as f:
        f.write("\n".join(output_lines))
    
    # Save visualization image
    vis_file = os.path.join(output_vis_dir, f"overlayed_{id_str}.png")
    cv2.imwrite(vis_file, vis_img)

    print(f"✅ {png_filename}: Saved {instance_count} leaf instances.")

# Save class names file once
classes_file = os.path.join(output_label_dir, "classes.txt")
with open(classes_file, "w") as f:
    for cls in sorted(class_name_to_id, key=lambda k: class_name_to_id[k]):
        f.write(cls + "\n")

print("✅ Done processing all images.")
