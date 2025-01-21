Light-FDD: Lightweight Fabric Defect Detection
Light-FDD is a lightweight yet efficient method for fabric defect detection, designed to perform high-accuracy defect detection while reducing computational complexity. 
It leverages an innovative three-stage cascaded architecture, integrating an improved FasterNet backbone with a pruned YOLOv8n for accurate and fast defect detection.

Key Features
Improved FasterNet Backbone: Utilizes an enhanced version of FasterNet-T0 for feature extraction, providing better accuracy without compromising speed.
Parallel Dilated Convolution Downsampling (PDCD): The PDCD block helps extract both local and contextual features by applying varying dilation rates.
Global Context and Receptive-Field Attention (GCRF): A novel attention mechanism designed to focus on the most critical regions of the fabric image, improving the detection of small defects.
Lightweight CSP Layer (VoVDualCSP): A Cross-Stage Partial (CSP) module that fuses multi-scale features while keeping the model lightweight and computationally efficient.
Pruned YOLOv8n: A pruned version of the YOLOv8n model is used for object detection, optimizing inference speed and memory consumption.
Installation
To get started with Light-FDD, follow the steps below.

Prerequisites
Python 3.7+
PyTorch 1.7.1+
torchvision
numpy
opencv-python
You can install the necessary dependencies with pip:


Usage
(1) Identify the Files to Replace: Ensure that you have the two files you want to replace. These could be:
ultralytics-main/
├── ultralytics/nn/modules/block.py
├── ultralytics/nn/tasks.py
(2) Add attention.py to the ultralytics/nn/modules/
(3) Add olov8-FasterNet_ContextGuided_RFCBAM_one_vovd.yaml to the ultralytics/cfg/models/v8
(4) Dowload the fabric dataset.
(5) Run split_data.py
(6) Run train.py
(7) Run test.py
