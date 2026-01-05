# Object Detection Project with ResNet-18

This project implements object detection using Faster R-CNN with ResNet-18 backbone on the Pascal VOC 2012 dataset.

## Project Structure

```
object_detection/
├── VOCdevkit/              # VOC dataset (already downloaded)
│   └── VOC2012/
│       ├── Annotations/    # XML annotation files
│       ├── JPEGImages/     # Image files
│       └── ImageSets/      # Train/val splits
├── config/                 # Configuration files
│   └── config.py          # Project configuration
├── src/                    # Source code
│   ├── preprocessing.py   # Data loading and preprocessing
│   ├── training.py        # Model training
│   ├── evaluation.py     # Model evaluation
│   ├── visualization.py  # Visualization functions
│   └── utils.py          # Utility functions
├── notebooks/             # Jupyter notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_data_visualization.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_prediction_visualization.ipynb
├── data/                  # Processed data (generated)
│   ├── processed/
│   └── raw/
├── models/                # Saved models
├── checkpoints/           # Training checkpoints
├── results/               # Results and outputs
│   ├── visualizations/
│   └── evaluations/
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. VOCdevkit Data Organization

The VOCdevkit should already be in the project root. The structure should be:

```
VOCdevkit/
└── VOC2012/
    ├── Annotations/       # XML files with bounding box annotations
    ├── JPEGImages/        # All images (.jpg files)
    └── ImageSets/
        └── Main/
            ├── train.txt  # List of training image IDs
            ├── val.txt    # List of validation image IDs
            └── trainval.txt
```

**Important Notes:**
- The `VOCdevkit` folder should be placed directly in the project root directory
- The `VOC2012` folder should contain:
  - `Annotations/`: XML files (one per image) with bounding box coordinates and class labels
  - `JPEGImages/`: All image files in JPG format
  - `ImageSets/Main/`: Text files listing image IDs for train/val splits

If your VOCdevkit is in a different location, you can:
1. Copy the entire `VOCdevkit` folder to the project root
2. Or update the `VOCDEVKIT_DIR` path in `config/config.py`

### 3. Verify Data Structure

Run the preprocessing notebook to verify your data is correctly organized:

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

## Usage

### Step 1: Data Preprocessing

Open and run `notebooks/01_data_preprocessing.ipynb` to:
- Verify data structure
- Create data loaders
- Inspect sample data

### Step 2: Data Visualization

Open and run `notebooks/02_data_visualization.ipynb` to:
- Visualize dataset samples with bounding boxes
- Understand the data format

### Step 3: Training

Open and run `notebooks/03_training.ipynb` to:
- Create the ResNet-18 based Faster R-CNN model
- Train the model on VOC dataset
- Save checkpoints and best model

**Training Configuration:**
- Model: Faster R-CNN with ResNet-50 FPN backbone (modified for ResNet-18 features)
- Batch size: 4 (adjust based on GPU memory)
- Learning rate: 0.001
- Epochs: 50 (configurable)
- Optimizer: SGD with momentum

### Step 4: Evaluation

Open and run `notebooks/04_evaluation.ipynb` to:
- Evaluate model on validation set
- Calculate mAP (Mean Average Precision)
- Get per-class AP scores

### Step 5: Visualization

Open and run `notebooks/05_prediction_visualization.ipynb` to:
- Visualize model predictions
- Compare predictions with ground truth
- Save visualization results

## Model Architecture

The project uses **Faster R-CNN** with:
- **Backbone**: ResNet-18 (pretrained on ImageNet)
- **Detection Head**: Fast R-CNN predictor
- **Number of Classes**: 21 (20 VOC classes + background)

## Configuration

Edit `config/config.py` to modify:
- Model parameters (backbone, number of classes, etc.)
- Training hyperparameters (batch size, learning rate, epochs, etc.)
- Data paths
- Evaluation thresholds

## Output Files

- **Models**: Saved in `models/` directory
  - `best_model.pth`: Best model based on validation loss
  - `latest_model.pth`: Latest model checkpoint

- **Checkpoints**: Saved in `checkpoints/` directory
  - `checkpoint_epoch_N.pth`: Checkpoint for epoch N

- **Results**: Saved in `results/` directory
  - `visualizations/`: Prediction visualizations
  - `evaluations/`: Evaluation metrics (JSON format)

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config/config.py`
- Use gradient accumulation
- Use smaller image sizes

### Data Loading Issues
- Verify VOCdevkit structure matches expected format
- Check that ImageSets/Main/train.txt and val.txt exist
- Ensure Annotations and JPEGImages directories contain files

### Import Errors
- Make sure you're running notebooks from the project root
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check that `sys.path.append` is correctly set in notebooks

## Notes

- The model uses ResNet-50 FPN as the base architecture (more stable than pure ResNet-18)
- For a true ResNet-18 backbone, you would need to implement a custom backbone wrapper
- Training time depends on GPU: ~2-4 hours on a modern GPU for 50 epochs
- The project is designed to work with Pascal VOC 2012 dataset format

## License

This project is for educational purposes. The VOC dataset is provided by the PASCAL VOC project.

