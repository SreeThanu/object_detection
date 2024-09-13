# Object Detection using YOLO and ResNet

## Overview

This project focuses on building an object detection system using the YOLO (You Only Look Once) algorithm and deep learning techniques like Convolutional Neural Networks (CNN) and ResNet architecture. The model is capable of identifying multiple objects within an image and predicting their bounding boxes with high accuracy.

## Key Features

- **YOLO (You Only Look Once)**: A real-time object detection algorithm that predicts both class labels and bounding boxes simultaneously.
- **Convolutional Neural Networks (CNN)**: Used to automatically and adaptively learn spatial hierarchies of features from images.
- **ResNet Architecture**: A deep residual network used to enhance feature extraction and improve detection accuracy.

## Dataset

The model is trained on the PASCAL VOC 2007 dataset, which contains annotated images of objects from 20 different categories. The dataset consists of:

- **Training images**: Used to train the model
- **Validation images**: Used to validate the model's performance

## Dependencies

- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- VOC dataset

## How to Run

1. Clone the repository.
2. Download the VOC dataset and place it in the required folder structure.
3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt

4. Train the model using:
  python train.py

5. To run inference on new images, use:
  python detect.py --image_path /path/to/image

## Results

The model can detect multiple objects within an image and predict bounding boxes and class labels. Below are some examples of the output:

- ![Object Detection Example 1](example1.png)
- ![Object Detection Example 2](example2.png)

## Conclusion

The object detection model built using YOLO and ResNet is capable of accurately identifying and localizing multiple objects in an image in real-time, making it suitable for applications like surveillance, autonomous driving, and more.


   
