# yolov11_ros2

A generic, configuration-driven ROS 2 (Humble) node that runs **YOLOv11** inference on any image topic and publishes annotated images, bounding boxes, and/or pose keypoint markers — all without recompiling.

---

## Topics

### Subscribed

| Topic | Type | Description |
|---|---|---|
| `/image_raw` *(default)* | `sensor_msgs/Image` | Input image stream. Override with `INPUT_TOPIC`. |

### Published (all optional, enabled by setting the env var)

| Env Variable | Topic Type | Description |
|---|---|---|
| `OUTPUT_TOPIC_IMAGE` | `sensor_msgs/Image` | Annotated frame with boxes, masks, or skeletons drawn |
| `OUTPUT_TOPIC_BB` | `vision_msgs/Detection2DArray` | Bounding boxes with class ID and confidence score |
| `OUTPUT_TOPIC_MARKERS` | `visualization_msgs/MarkerArray` | 2D pose keypoints as `SPHERE_LIST` markers (pose models only) |

> Publishers are only created when the corresponding env variable is set. Set only the outputs you need.

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `yolo11n.pt` | Path to the YOLO `.pt` weights file inside the container |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence (0.0 – 1.0) |
| `IMAGE_RESOLUTION` | `640` | Inference resolution passed to YOLO (`imgsz`) |
| `INPUT_TOPIC` | `/image_raw` | ROS 2 topic to subscribe for input images |
| `OUTPUT_TOPIC_IMAGE` | *(unset)* | Publish annotated debug images to this topic |
| `OUTPUT_TOPIC_BB` | *(unset)* | Publish bounding box detections to this topic |
| `OUTPUT_TOPIC_MARKERS` | *(unset)* | Publish pose keypoint markers to this topic |

---

## Supported YOLO Tasks

The node adapts automatically based on the model you load:

| Model file | YOLO Task | Active outputs |
|---|---|---|
| `yolo11n.pt` | Detection | Image, BBoxes |
| `yolo11n-seg.pt` | Segmentation | Image (with masks), BBoxes |
| `yolo11n-pose.pt` | Pose Estimation | Image (with skeleton), BBoxes, Markers |

---

## Quick Start (Docker)

### 1. Build the image

```bash
docker build -t yolov11_ros2 .
```

### 2. Run — detection example

```bash
docker run --rm --network host \
  -e INPUT_TOPIC=/camera/image_raw \
  -e OUTPUT_TOPIC_IMAGE=/yolo/image_annotated \
  -e OUTPUT_TOPIC_BB=/yolo/detections \
  -e CONFIDENCE_THRESHOLD=0.4 \
  yolov11_ros2
```

### 3. Run — pose estimation example

```bash
docker run --rm --network host \
  -e MODEL_PATH=yolo11n-pose.pt \
  -e INPUT_TOPIC=/camera/image_raw \
  -e OUTPUT_TOPIC_IMAGE=/yolo/pose_image \
  -e OUTPUT_TOPIC_MARKERS=/yolo/pose_markers \
  yolov11_ros2
```

### 4. Run — segmentation example

```bash
docker run --rm --network host \
  -e MODEL_PATH=yolo11n-seg.pt \
  -e INPUT_TOPIC=/camera/image_raw \
  -e OUTPUT_TOPIC_IMAGE=/yolo/seg_image \
  -e OUTPUT_TOPIC_BB=/yolo/detections \
  yolov11_ros2
```

> `--network host` is required so the container can communicate with other ROS 2 nodes via Fast DDS UDP multicast.

---

## Using a Custom Model

### Option A — Mount a local `.pt` file

For custom-trained models or air-gapped environments, mount your own `.pt` file into the container:

```bash
docker run --rm --network host \
  -v /path/to/my_model.pt:/models/my_model.pt \
  -e MODEL_PATH=/models/my_model.pt \
  -e INPUT_TOPIC=/camera/image_raw \
  -e OUTPUT_TOPIC_BB=/yolo/detections \
  yolov11_ros2
```

### Option B — Direct URL (e.g. GitHub Release Asset)

You can pass any publicly accessible direct download URL as `MODEL_PATH`. Ultralytics will fetch and load the file automatically:

```bash
docker run --rm --network host \
  -e MODEL_PATH=https://github.com/your-org/your-repo/releases/download/v1.0/my_model.pt \
  -e INPUT_TOPIC=/camera/image_raw \
  -e OUTPUT_TOPIC_BB=/yolo/detections \
  yolov11_ros2
```

This works with GitHub release assets, HuggingFace, or any server hosting a `.pt` file directly.

---

## Building from Source (without Docker)

**Prerequisites:** ROS 2 Humble, Python 3, `cv_bridge`, `vision_msgs`, `visualization_msgs`

```bash
# Install Python dependencies
pip3 install "numpy<2" torch torchvision ultralytics

# Clone into your workspace
cd ~/ros_ws/src
git clone <repo-url> yolov11_ros2

# Build
cd ~/ros_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select yolov11_ros2

# Source and run
source install/setup.bash
export INPUT_TOPIC=/image_raw
export OUTPUT_TOPIC_BB=/yolo/detections
export OUTPUT_TOPIC_IMAGE=/yolo/image
ros2 run yolov11_ros2 yolo_node
```

---

## Output Message Details

### `Detection2DArray` (bounding boxes)

Each `Detection2D` in the array contains:
- `bbox.center.position.x/y` — box centre in pixels
- `bbox.size_x/size_y` — box width and height in pixels
- `results[0].hypothesis.class_id` — class name string (e.g. `"person"`)
- `results[0].hypothesis.score` — confidence score (0.0 – 1.0)

An **empty** `detections` array is published when no objects are detected, so downstream nodes always receive a message every frame.

### `MarkerArray` (pose keypoints)

- One `Marker` of type `SPHERE_LIST` per detected person
- Each sphere represents one keypoint at its 2D pixel coordinate (`z = 0`)
- Keypoints with coordinates `(0, 0)` (untracked) are skipped

---

## Dependencies

| Package | Source |
|---|---|
| ROS 2 Humble | [ros.org](https://docs.ros.org/en/humble/) |
| `ros-humble-cv-bridge` | apt |
| `ros-humble-vision-msgs` | apt |
| `ros-humble-visualization-msgs` | apt |
| `ultralytics` | pip |
| `torch` / `torchvision` | pip |
| `numpy < 2` | pip |

---

## Current Use Cases

# Vine Leaf Segmentation

Segmentation model trained to recognize vineleafs:

https://github.com/phsilvarepo/YOLOv11/releases/download/v1/leaf_seg.pt

---
