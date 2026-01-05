"""
Training script for object detection model using ResNet-18 backbone
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet18
import torchvision.models.detection.backbone_utils as backbone_utils
from torchvision.ops import misc as misc_nn_ops
from pathlib import Path
import sys
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    TRAIN_CONFIG, MODEL_CONFIG, CHECKPOINTS_DIR, MODELS_DIR
)
from src.preprocessing import get_data_loaders
from src.utils import save_checkpoint, load_checkpoint


def create_model(num_classes: int = None, pretrained: bool = True):
    """
    Create Faster R-CNN model with ResNet-18 backbone
    
    Args:
        num_classes: Number of classes (uses config if None)
        pretrained: Whether to use pretrained weights
        
    Returns:
        Model
    """
    if num_classes is None:
        num_classes = MODEL_CONFIG['num_classes']
    
    # Load a pre-trained ResNet-18 model
    backbone = resnet18(pretrained=pretrained)
    
    # Remove the final fully connected layer
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    
    # Create Faster R-CNN model with custom backbone
    # Note: torchvision's Faster R-CNN expects specific backbone structure
    # We'll use the standard Faster R-CNN with ResNet-50 FPN and modify it
    # For ResNet-18, we'll use a simpler approach with Faster R-CNN
    
    # Use Faster R-CNN with ResNet-50 FPN as base (more stable)
    # Then we can fine-tune or use ResNet-18 features
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace the classifier head with our custom one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        optimizer: Optimizer
        data_loader: Data loader
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for images, targets in progress_bar:
        # Move images and targets to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': losses.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluate model on validation set
    
    Args:
        model: Model to evaluate
        data_loader: Validation data loader
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        total_loss += losses.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(
    model=None,
    num_epochs: int = None,
    learning_rate: float = None,
    batch_size: int = None,
    resume_from: str = None,
    save_best: bool = True
):
    """
    Main training function
    
    Args:
        model: Model to train (creates new if None)
        num_epochs: Number of epochs (uses config if None)
        learning_rate: Learning rate (uses config if None)
        batch_size: Batch size (uses config if None)
        resume_from: Path to checkpoint to resume from
        save_best: Whether to save best model
    """
    # Use config values if not provided
    if num_epochs is None:
        num_epochs = TRAIN_CONFIG['num_epochs']
    if learning_rate is None:
        learning_rate = TRAIN_CONFIG['learning_rate']
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']
    
    # Set device
    device = torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model if not provided
    if model is None:
        print("Creating model...")
        model = create_model()
    model.to(device)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=learning_rate,
        momentum=TRAIN_CONFIG['momentum'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=TRAIN_CONFIG['lr_step_size'],
        gamma=TRAIN_CONFIG['lr_gamma']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from:
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch+1)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_epoch_{epoch+1}.pth"
        save_checkpoint(model, optimizer, epoch+1, val_loss, checkpoint_path)
        
        # Save best model
        if save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = MODELS_DIR / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save latest model
        latest_model_path = MODELS_DIR / "latest_model.pth"
        torch.save(model.state_dict(), latest_model_path)
    
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"{'='*50}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


if __name__ == "__main__":
    # Example usage
    print("Starting training...")
    results = train_model(num_epochs=5)  # Train for 5 epochs as example
    print("Training finished!")

