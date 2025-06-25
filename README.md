# YOLOv11

This repository contains scripts for the full YOLOv11 pipeline — from data collection and annotation to training and inference. It supports ROS2 integration, and Docker-based deployment

## train
This folder includes a Google Colab notebook to train and validate a custom YOLOv11 model using your Google Drive.

Requirements:

- Dataset in YOLO format
- .yaml configuration file

Note: Trained model weights are not automatically saved to **Google Drive**. Make sure to manually download them after training.

## yolov11_ros

This ROS2 package enables real-time inference using a custom YOLOv11 model. The node subscribes to the sensor_msgs/Image in the **/rgb** topic. And outputs the following topics: 

- **/ultralytics/detection/image** -- (sensor_msg/Image) An image of the the detected bounding boxes, classes and confidence score

Requirements:

- Weight of the model(.pt)

## yolov11_container

This package includes the yolov11_ros ROS package, as well as the necessary files for deploying it in a **Docker** environment.

Requirements:

- Weight of the model(.pt)
