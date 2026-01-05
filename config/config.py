"""
Configuration file for object detection project
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
VOCDEVKIT_DIR = BASE_DIR / "VOCdevkit"
VOC2012_DIR = VOCDEVKIT_DIR / "VOC2012"

# VOC Dataset paths
VOC_ANNOTATIONS_DIR = VOC2012_DIR / "Annotations"
VOC_IMAGES_DIR = VOC2012_DIR / "JPEGImages"
VOC_IMAGESETS_DIR = VOC2012_DIR / "ImageSets" / "Main"

# Model configuration
MODEL_CONFIG = {
    "backbone": "resnet18",
    "num_classes": 21,  # 20 VOC classes + background
    "pretrained": True,
    "min_size": 800,
    "max_size": 1333,
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 4,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_step_size": 30,
    "lr_gamma": 0.1,
    "num_workers": 4,
    "device": "cuda",  # Change to "cpu" if no GPU available
}

# Data augmentation
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
    },
}

# Evaluation configuration
EVAL_CONFIG = {
    "iou_threshold": 0.5,
    "score_threshold": 0.5,
    "nms_threshold": 0.5,
}

# VOC class names
VOC_CLASSES = [
    '__background__',  # 0
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Create directories if they don't exist
for dir_path in [PROCESSED_DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

