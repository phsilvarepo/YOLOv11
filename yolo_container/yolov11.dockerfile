FROM ros:humble

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# 2. Install Python packages
RUN python3 -m pip install --upgrade pip
RUN pip3 install "numpy<2" torch torchvision ultralytics

# 3. Setup ROS workspace
WORKDIR /ros_ws

# 4. Copy the code and weights (Paths relative to Desktop/YOLOv11)
COPY ./yolov11_ros2 /ros_ws/src/yolov11_ros2
COPY ./yolo_container/leaf_seg.pt /ros_ws/src/yolov11_ros2/leaf_seg.pt

# 5. Build the workspace
RUN . /opt/ros/humble/setup.bash && \
    colcon build --packages-select yolov11_ros2

# 6. Define Environment Variables (Defaults)
ENV YOLO_MODEL="/ros_ws/src/yolov11_ros2/leaf_seg.pt"
ENV YOLO_INPUT_TOPIC="/rgb"
ENV YOLO_OUTPUT_TOPIC="/yolo/detections_image"

# 7. Setup sourcing
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros_ws/install/setup.bash" >> ~/.bashrc

# 8. Launch the node
ENTRYPOINT ["/bin/bash", "-c", "source /ros_ws/install/setup.bash && ros2 run yolov11_ros2 yolo_node"]
