FROM ros:humble-ros-base

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and install Python packages
RUN python3 -m pip install --upgrade pip
RUN pip3 install "numpy<2" torch torchvision ultralytics

# 3. Setup ROS workspace
WORKDIR /ros_ws
RUN mkdir -p src

# 4. Copy the ROS2 package and model weights
COPY ./yolov11_ros2 /ros_ws/src/yolov11_ros2
COPY ./yolo_container/leaf_seg.pt /ros_ws/src/yolov11_ros2/leaf_seg.pt

# 5. Build the workspace using bash
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --packages-select yolov11_ros2"

# 6. Define Environment Variables (Defaults)
ENV YOLO_MODEL="/ros_ws/src/yolov11_ros2/leaf_seg.pt"
ENV YOLO_INPUT_TOPIC="/rgb"
ENV YOLO_OUTPUT_TOPIC="/yolo/detections_image"
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# 7. Setup sourcing for interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros_ws/install/setup.bash" >> ~/.bashrc

# 8. Launch the ROS2 node
ENTRYPOINT ["/bin/bash", "-c", "source /ros_ws/install/setup.bash && ros2 run yolov11_ros2 yolo_node"]
