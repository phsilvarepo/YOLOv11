FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-visualization-msgs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN pip3 install "numpy<2" torch torchvision ultralytics

RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-seg.pt'); YOLO('yolo11n-pose.pt')"

WORKDIR /ros_ws
COPY ./yolov11_ros2 /ros_ws/src/yolov11_ros2

RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --packages-select yolov11_ros2"

ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros_ws/install/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/bin/bash", "-c", "source /ros_ws/install/setup.bash && ros2 run yolov11_ros2 yolo_node"]
