"""
Evaluation functions for object detection model
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import EVAL_CONFIG, VOC_CLASSES
from src.preprocessing import get_data_loaders, VOCDataset
from src.utils import calculate_iou, id_to_label
from src.training import create_model


def calculate_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) using 11-point interpolation
    
    Args:
        recall: Array of recall values
        precision: Array of precision values
        
    Returns:
        Average Precision
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap


def evaluate_detections(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Evaluate detections against ground truth
    
    Args:
        predictions: List of prediction dictionaries with 'boxes', 'labels', 'scores'
        ground_truths: List of ground truth dictionaries with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Group predictions and ground truths by class
    class_predictions = defaultdict(list)
    class_ground_truths = defaultdict(list)
    
    for pred, gt in zip(predictions, ground_truths):
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            class_predictions[label.item()].append((box, score))
        
        for box, label in zip(gt['boxes'], gt['labels']):
            class_ground_truths[label.item()].append(box)
    
    # Calculate AP for each class
    aps = {}
    for class_id in range(1, len(VOC_CLASSES)):  # Skip background
        if class_id not in class_predictions:
            aps[class_id] = 0.0
            continue
        
        preds = class_predictions[class_id]
        gts = class_ground_truths[class_id]
        
        if len(gts) == 0:
            aps[class_id] = 0.0
            continue
        
        # Sort predictions by score
        preds = sorted(preds, key=lambda x: x[1], reverse=True)
        
        # Match predictions to ground truth
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        gt_matched = [False] * len(gts)
        
        for i, (pred_box, score) in enumerate(preds):
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for j, gt_box in enumerate(gts):
                if gt_matched[j]:
                    continue
                
                # Convert to tensor for IoU calculation
                pred_tensor = torch.tensor([pred_box], dtype=torch.float32)
                gt_tensor = torch.tensor([gt_box], dtype=torch.float32)
                
                iou = calculate_iou(pred_tensor, gt_tensor)[0, 0].item()
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Calculate AP
        ap = calculate_ap(recalls, precisions)
        aps[class_id] = ap
    
    # Calculate mAP
    mean_ap = np.mean(list(aps.values()))
    
    return {
        'mAP': mean_ap,
        'APs': aps,
        'class_names': [id_to_label(i) for i in range(1, len(VOC_CLASSES))]
    }


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device: torch.device,
    score_threshold: float = None,
    nms_threshold: float = None
) -> Dict:
    """
    Evaluate model on a dataset
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        score_threshold: Score threshold for detections
        nms_threshold: NMS threshold
        
    Returns:
        Dictionary with evaluation results
    """
    if score_threshold is None:
        score_threshold = EVAL_CONFIG['score_threshold']
    if nms_threshold is None:
        nms_threshold = EVAL_CONFIG['nms_threshold']
    
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    print("Running evaluation...")
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        
        # Get predictions
        predictions = model(images)
        
        # Filter by score threshold
        for i, pred in enumerate(predictions):
            # Filter by score
            keep = pred['scores'] > score_threshold
            pred['boxes'] = pred['boxes'][keep]
            pred['labels'] = pred['labels'][keep]
            pred['scores'] = pred['scores'][keep]
            
            all_predictions.append(pred)
            all_ground_truths.append(targets[i])
    
    # Evaluate
    results = evaluate_detections(
        all_predictions,
        all_ground_truths,
        iou_threshold=EVAL_CONFIG['iou_threshold']
    )
    
    return results


def print_evaluation_results(results: Dict):
    """
    Print evaluation results in a formatted way
    
    Args:
        results: Results dictionary from evaluate_model
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nMean Average Precision (mAP): {results['mAP']:.4f}\n")
    
    print("Per-class Average Precision:")
    print("-" * 60)
    print(f"{'Class':<20} {'AP':<10}")
    print("-" * 60)
    
    for class_id, ap in results['APs'].items():
        class_name = id_to_label(class_id)
        print(f"{class_name:<20} {ap:.4f}")
    
    print("="*60)


def evaluate_from_checkpoint(
    checkpoint_path: Path,
    split: str = 'val',
    device: str = None
) -> Dict:
    """
    Evaluate model from a checkpoint file
    
    Args:
        checkpoint_path: Path to model checkpoint
        split: Dataset split to evaluate on ('val' or 'test')
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Evaluation results dictionary
    """
    from config.config import TRAIN_CONFIG, MODEL_CONFIG
    
    if device is None:
        device = torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Loading model from {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(num_classes=MODEL_CONFIG['num_classes'])
    model.to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create data loader
    from src.preprocessing import get_data_loaders
    _, val_loader = get_data_loaders()
    
    # Evaluate
    results = evaluate_model(model, val_loader, device)
    
    # Print results
    print_evaluation_results(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    from config.config import MODELS_DIR
    
    model_path = MODELS_DIR / "best_model.pth"
    if model_path.exists():
        results = evaluate_from_checkpoint(model_path)
    else:
        print(f"Model not found at {model_path}")
        print("Please train a model first or provide a valid checkpoint path.")

