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

# 5. Build the workspace
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --packages-select yolov11_ros2"

# 6. Define Environment Variables
ENV MODEL_PATH="yolo11n.pt"
ENV INPUT_TOPIC="/rgb"
ENV OUTPUT_TOPIC="/yolo/detections_image"
ENV CONFIDENCE_THRESHOLD="0.5"
ENV IMAGE_RESOLUTION="640"
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# 7. Setup sourcing
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros_ws/install/setup.bash" >> ~/.bashrc

# 8. Launch the ROS2 node
ENTRYPOINT ["/bin/bash", "-c", "source /ros_ws/install/setup.bash && ros2 run yolov11_ros2 yolo_node"]
