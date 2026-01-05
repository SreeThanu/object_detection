# VOCdevkit Data Organization Guide

This guide explains how to organize your VOCdevkit data in this project.

## Current Structure

Your VOCdevkit is already in the project root. The expected structure is:

```
object_detection/
└── VOCdevkit/
    └── VOC2012/
        ├── Annotations/          # XML annotation files (one per image)
        ├── JPEGImages/           # All image files (.jpg format)
        └── ImageSets/
            └── Main/
                ├── train.txt     # Training image IDs (one per line)
                ├── val.txt       # Validation image IDs (one per line)
                └── trainval.txt  # Combined train+val IDs
```

## What Each Directory Contains

### 1. Annotations/
- **Format**: XML files
- **Naming**: Each XML file corresponds to an image (e.g., `2007_000027.xml`)
- **Content**: Bounding box coordinates, class labels, image dimensions
- **Example structure**:
  ```xml
  <annotation>
    <filename>2007_000027.jpg</filename>
    <size>
      <width>500</width>
      <height>375</height>
    </size>
    <object>
      <name>person</name>
      <bndbox>
        <xmin>174</xmin>
        <ymin>101</ymin>
        <xmax>349</xmax>
        <ymax>351</ymax>
      </bndbox>
    </object>
  </annotation>
  ```

### 2. JPEGImages/
- **Format**: JPG image files
- **Naming**: Image IDs (e.g., `2007_000027.jpg`)
- **Content**: Original images from the dataset

### 3. ImageSets/Main/
- **Format**: Text files
- **Content**: List of image IDs (without extension), one per line
- **train.txt**: Image IDs for training set
- **val.txt**: Image IDs for validation set
- **Example content**:
  ```
  2007_000027
  2007_000032
  2007_000033
  ...
  ```

## How to Verify Your Data

Run the preprocessing notebook to check if your data is correctly organized:

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

The notebook will:
1. Check if all required directories exist
2. Count images and annotations
3. Verify train/val splits
4. Test data loading

## If Your Data is in a Different Location

### Option 1: Move/Copy VOCdevkit
```bash
# If your VOCdevkit is elsewhere, copy it to the project root
cp -r /path/to/your/VOCdevkit /Users/sreethanubhuvaneshgk/Downloads/desktop/home_folder/object_detection/
```

### Option 2: Update Config Path
Edit `config/config.py` and update the `VOCDEVKIT_DIR` path:
```python
VOCDEVKIT_DIR = Path("/path/to/your/VOCdevkit")
```

## Common Issues and Solutions

### Issue: "Split file not found"
**Solution**: Ensure `ImageSets/Main/train.txt` and `val.txt` exist. If not, you may need to create them or download the complete VOCdevkit.

### Issue: "Annotations directory not found"
**Solution**: Check that the `Annotations/` folder exists inside `VOC2012/` and contains XML files.

### Issue: "JPEGImages directory not found"
**Solution**: Check that the `JPEGImages/` folder exists inside `VOC2012/` and contains JPG files.

### Issue: Mismatched image/annotation counts
**Solution**: Ensure every image in `JPEGImages/` has a corresponding XML file in `Annotations/` with the same base name.

## Expected File Counts (VOC 2012)

- **Images**: ~17,125 images
- **Annotations**: ~17,125 XML files
- **Training images**: ~11,540 (in train.txt)
- **Validation images**: ~1,449 (in val.txt)

*Note: Actual counts may vary slightly depending on your specific VOCdevkit version.*

## Additional Notes

- The project automatically handles the data structure once it's correctly organized
- No manual preprocessing is required - the code handles everything
- The `train/` and `val/` folders you see are likely from a previous preprocessing step and are not required for this project
- The project reads directly from `Annotations/`, `JPEGImages/`, and `ImageSets/Main/`

## Next Steps

Once your data is organized:
1. Run `notebooks/01_data_preprocessing.ipynb` to verify
2. Run `notebooks/02_data_visualization.ipynb` to visualize samples
3. Proceed with training in `notebooks/03_training.ipynb`

