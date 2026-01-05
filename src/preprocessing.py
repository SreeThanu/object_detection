"""
Preprocessing functions for VOC dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    VOC_ANNOTATIONS_DIR, VOC_IMAGES_DIR, VOC_CLASSES, 
    TRAIN_CONFIG, AUGMENTATION_CONFIG
)
from src.utils import parse_voc_annotation, get_voc_image_ids, label_to_id


class VOCDataset(Dataset):
    """
    Pascal VOC Dataset for object detection
    """
    def __init__(self, split: str = 'train', transform=None, target_transform=None):
        """
        Args:
            split: 'train', 'val', or 'trainval'
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on targets
        """
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Get image IDs for this split
        self.image_ids = get_voc_image_ids(split)
        
        print(f"Loaded {len(self.image_ids)} images for {split} split")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = VOC_IMAGES_DIR / f"{image_id}.jpg"
        image = Image.open(image_path).convert('RGB')
        
        # Load annotation
        annotation_path = VOC_ANNOTATIONS_DIR / f"{image_id}.xml"
        annotation = parse_voc_annotation(annotation_path)
        
        # Convert boxes and labels to tensors
        boxes = torch.tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.tensor([label_to_id(label) for label in annotation['labels']], dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target


def get_transforms(train: bool = True):
    """
    Get data transforms for training or validation
    
    Args:
        train: Whether to return training transforms (with augmentation)
        
    Returns:
        Transform function
    """
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def get_data_loaders(batch_size: int = None, num_workers: int = None):
    """
    Get DataLoaders for train and validation sets
    
    Args:
        batch_size: Batch size (uses config if None)
        num_workers: Number of workers (uses config if None)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']
    if num_workers is None:
        num_workers = TRAIN_CONFIG['num_workers']
    
    # Create datasets
    train_dataset = VOCDataset(
        split='train',
        transform=get_transforms(train=True)
    )
    
    val_dataset = VOCDataset(
        split='val',
        transform=get_transforms(train=False)
    )
    
    # Create data loaders
    from src.utils import collate_fn
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def prepare_data():
    """
    Prepare and validate data structure
    """
    print("Checking data structure...")
    
    # Check if directories exist
    assert VOC_IMAGES_DIR.exists(), f"Images directory not found: {VOC_IMAGES_DIR}"
    assert VOC_ANNOTATIONS_DIR.exists(), f"Annotations directory not found: {VOC_ANNOTATIONS_DIR}"
    
    # Count files
    num_images = len(list(VOC_IMAGES_DIR.glob("*.jpg")))
    num_annotations = len(list(VOC_ANNOTATIONS_DIR.glob("*.xml")))
    
    print(f"Found {num_images} images")
    print(f"Found {num_annotations} annotations")
    
    # Get splits
    train_ids = get_voc_image_ids('train')
    val_ids = get_voc_image_ids('val')
    
    print(f"Train images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")
    
    return {
        'num_images': num_images,
        'num_annotations': num_annotations,
        'train_size': len(train_ids),
        'val_size': len(val_ids)
    }


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    stats = prepare_data()
    
    print("\nCreating sample data loader...")
    train_loader, val_loader = get_data_loaders(batch_size=2, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test loading one batch
    print("\nLoading one batch...")
    images, targets = next(iter(train_loader))
    print(f"Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Number of boxes in first image: {len(targets[0]['boxes'])}")

