"""
Utility functions for object detection project
"""
import torch
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import json

def parse_voc_annotation(xml_path: Path) -> Dict:
    """
    Parse VOC XML annotation file
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        Dictionary containing image info and bounding boxes
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    boxes = []
    labels = []
    difficult = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
        difficult.append(int(obj.find('difficult').text))
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'boxes': boxes,
        'labels': labels,
        'difficult': difficult
    }

def get_voc_image_ids(split: str = 'train') -> List[str]:
    """
    Get list of image IDs for a given split
    
    Args:
        split: 'train', 'val', or 'trainval'
        
    Returns:
        List of image IDs
    """
    from config.config import VOC_IMAGESETS_DIR
    
    split_file = VOC_IMAGESETS_DIR / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        image_ids = [line.strip().split()[0] for line in f.readlines()]
    
    return image_ids

def label_to_id(label: str) -> int:
    """
    Convert VOC label string to class ID
    
    Args:
        label: Class name string
        
    Returns:
        Class ID (0-20)
    """
    from config.config import VOC_CLASSES
    return VOC_CLASSES.index(label) if label in VOC_CLASSES else 0

def id_to_label(class_id: int) -> str:
    """
    Convert class ID to VOC label string
    
    Args:
        class_id: Class ID (0-20)
        
    Returns:
        Class name string
    """
    from config.config import VOC_CLASSES
    return VOC_CLASSES[class_id] if 0 <= class_id < len(VOC_CLASSES) else '__background__'

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes
    
    Args:
        box1: Tensor of shape [N, 4] (x1, y1, x2, y2)
        box2: Tensor of shape [M, 4] (x1, y1, x2, y2)
        
    Returns:
        IoU tensor of shape [N, M]
    """
    # Calculate intersection
    x1 = torch.max(box1[:, None, 0], box2[:, 0])
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - intersection
    
    iou = intersection / union
    return iou

def save_checkpoint(model, optimizer, epoch, loss, filepath: Path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: Path, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optional optimizer
        
    Returns:
        Epoch number
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint['epoch']

def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-sized images
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

