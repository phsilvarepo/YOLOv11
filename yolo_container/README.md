# YOLO Docker Container

Build **ROS Humble Docker** image and the necessary dependencies to run the **yolov11_ros** package :

 - docker build -t yolov11_ros -f yolov11_ros.dockerfile .

Run the **Docker** container:

 - docker run -it --rm --net=host --gpus all --privileged yolov11_ros

This container utilizes the host's IP and GPU resources to run YOLO, through torch
