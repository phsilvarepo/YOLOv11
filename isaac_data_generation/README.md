# YOLO + Omniverse Replicator

This package contains the necessary scripts for post processing of the output of Omniverse Replicator for the YOLOv11 input data.

## seg_converter.py
Script to convert synthetic images and annotations from Isaac Sim to the YOLO instance segmentation format. The output is sotred at ./yolo_labels

Parameters:
- **input_dir** : Path to the Isaac Sim dataset directory
