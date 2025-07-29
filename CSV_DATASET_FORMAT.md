# CSV Dataset Format Support

This document explains how to use the new CSV-based dataset format in your training pipeline.

## Overview

The training script now supports two dataset formats for local datasets:

1. **Folder-based format** (existing): Class subfolders containing images
2. **CSV-based format** (new): CSV files with image paths and labels

## CSV Dataset Structure

For CSV-based datasets, your folder should be structured as follows:

```
your_dataset_folder/
├── jpeg/  (or any subfolder name)
│   ├── image1.jpg
│   ├── image2.png
│   ├── image3.jpg
│   └── ...
├── train_dataset.csv
└── test_dataset.csv
```

**Note**: The image subfolder can have any name (e.g., "images", "jpeg", "photos") as long as the CSV files contain the correct relative paths.

### CSV File Format

Both `train_dataset.csv` and `test_dataset.csv` must contain the following columns:

- **`image_path`**: The image path relative to the dataset folder (e.g., "jpeg/image1.jpg")
- **`binary_label`**: The class label for the image

Example CSV content:
```csv
image_path,binary_label
jpeg/image1.jpg,benign
jpeg/image2.png,malignant
jpeg/image3.jpg,benign
...
```

## Configuration

To use the CSV format, set the `local_dataset_format` parameter in your JSON config file:

### For Folder-based format (default):
```json
{
    "dataset_host": "local_folder",
    "local_folder_path": "/path/to/your/dataset",
    "local_dataset_format": "folder",
    ...
}
```

### For CSV-based format:
```json
{
    "dataset_host": "local_folder",
    "local_folder_path": "/path/to/your/csv/dataset",
    "local_dataset_format": "csv",
    ...
}
```

## Example Configuration Files

- `config/localDatasetConfig.json`: Example for folder-based format
- `config/localCSVDatasetConfig.json`: Example for CSV-based format

## Data Splitting

The CSV format handles data splitting as follows:

1. **Test set**: Uses the images and labels from `test_dataset.csv`
2. **Training data**: Uses the images and labels from `train_dataset.csv`
3. **Validation set**: Created by splitting the training data (configurable split ratio)

## Requirements

Make sure you have pandas installed:
```bash
pip install pandas>=1.5.0
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Error Handling

The CSV loader includes robust error handling:

- Validates that required CSV files exist
- Checks for required columns (`filename`, `label`)
- Verifies that image files exist in the `images/` folder
- Skips invalid images or missing files with warnings
- Ensures consistent class labels across train and test sets

## Usage Example

```bash
python train.py --config config/localCSVDatasetConfig.json
```

## Benefits of CSV Format

1. **Flexibility**: Easy to modify labels without moving files
2. **Data Management**: Better tracking of train/test splits
3. **Integration**: Works well with data annotation tools
4. **Scalability**: Efficient for large datasets
5. **Version Control**: CSV files can be version controlled easily 