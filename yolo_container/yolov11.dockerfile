FROM ros:humble

# Install Python tools and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    && apt-get clean

# Upgrade pip and install Python packages (with NumPy downgrade)
RUN python3 -m pip install --upgrade pip
RUN pip3 install "numpy<2" torch torchvision ultralytics

# Setup ROS workspace inside container
WORKDIR /ros_ws

# Copy your ROS workspace and YOLO model weights
COPY leaf_seg.pt /ros_ws/src/yolov11_ros/src/leaf_seg.pt

# Source ROS and your workspace setup.bash automatically when starting the container
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros_ws/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
