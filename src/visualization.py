"""
Visualization functions for object detection
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import VOC_CLASSES, VOC_IMAGES_DIR
from src.utils import id_to_label
from src.preprocessing import VOCDataset, get_transforms


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        
    Returns:
        Denormalized image array [H, W, C]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image


def draw_boxes(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    class_names: List[str] = None
) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Image array [H, W, C]
        boxes: Bounding boxes tensor [N, 4] (x1, y1, x2, y2)
        labels: Label tensor [N]
        scores: Optional score tensor [N]
        class_names: Optional list of class names
        
    Returns:
        Image with drawn boxes
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    boxes = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    if scores is not None:
        scores = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(VOC_CLASSES)))
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        
        # Get class name
        if class_names is None:
            class_name = id_to_label(int(label))
        else:
            class_name = class_names[int(label)]
        
        # Get color
        color = colors[int(label) % len(colors)]
        
        # Draw box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label_text = class_name
        if scores is not None:
            label_text += f" {scores[i]:.2f}"
        
        ax.text(
            x1, y1 - 5, label_text,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
            fontsize=10, color='white', weight='bold'
        )
    
    ax.axis('off')
    return fig


def visualize_predictions(
    model,
    dataset: VOCDataset,
    device: torch.device,
    num_images: int = 5,
    score_threshold: float = 0.5,
    save_dir: Optional[Path] = None
):
    """
    Visualize model predictions on dataset
    
    Args:
        model: Trained model
        dataset: Dataset to visualize
        device: Device to run model on
        num_images: Number of images to visualize
        score_threshold: Score threshold for detections
        save_dir: Optional directory to save visualizations
    """
    model.eval()
    
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    for idx in indices:
        image, target = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            prediction = model([image.to(device)])[0]
        
        # Filter by score
        keep = prediction['scores'] > score_threshold
        prediction['boxes'] = prediction['boxes'][keep]
        prediction['labels'] = prediction['labels'][keep]
        prediction['scores'] = prediction['scores'][keep]
        
        # Denormalize image
        image_np = denormalize_image(image)
        
        # Draw ground truth
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ground truth
        axes[0].imshow(image_np)
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box.numpy()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].text(
                x1, y1 - 5, id_to_label(label.item()),
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                fontsize=10, color='white', weight='bold'
            )
        axes[0].set_title('Ground Truth', fontsize=14, weight='bold')
        axes[0].axis('off')
        
        # Predictions
        axes[1].imshow(image_np)
        for box, label, score in zip(
            prediction['boxes'].cpu(),
            prediction['labels'].cpu(),
            prediction['scores'].cpu()
        ):
            x1, y1, x2, y2 = box.numpy()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[1].add_patch(rect)
            axes[1].text(
                x1, y1 - 5, f"{id_to_label(label.item())} {score:.2f}",
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                fontsize=10, color='white', weight='bold'
            )
        axes[1].set_title('Predictions', fontsize=14, weight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"prediction_{idx}.png", dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_dir / f'prediction_{idx}.png'}")
        else:
            plt.show()
        
        plt.close()


def visualize_dataset_samples(
    dataset: VOCDataset,
    num_samples: int = 5,
    save_dir: Optional[Path] = None
):
    """
    Visualize random samples from dataset
    
    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to show
        save_dir: Optional directory to save visualizations
    """
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        
        # Denormalize image
        image_np = denormalize_image(image)
        
        axes[i].imshow(image_np)
        
        # Draw boxes
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box.numpy()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[i].add_patch(rect)
            axes[i].text(
                x1, y1 - 5, id_to_label(label.item()),
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                fontsize=10, color='white', weight='bold'
            )
        
        axes[i].set_title(f'Sample {idx} - {len(target["boxes"])} objects', fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "dataset_samples.png", dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_dir / 'dataset_samples.png'}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Path] = None
):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Visualizing dataset samples...")
    train_dataset = VOCDataset(split='train', transform=get_transforms(train=False))
    visualize_dataset_samples(train_dataset, num_samples=3)

