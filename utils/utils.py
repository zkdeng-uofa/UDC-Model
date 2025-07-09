import torch
import argparse
import json
import kagglehub
import os
import datetime
import numpy as np
import torchvision.transforms as transforms
from evaluate import load
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from datasets import load_dataset
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from collections import Counter
from datasets import Dataset
import random

# Import our custom image processor
from utils.image_processor import CustomImageProcessor

# Try to import visualization libraries (they might not be installed)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib and seaborn not available. Visualizations will be skipped.")

def collate_fn(images):
    pixel_values = torch.stack([image["pixel_values"] for image in images])
    labels = torch.tensor([image["label"] for image in images])
    #return pixel_values, labels  # Return as a tuple (image_tensor, label_tensor)
    return {"pixel_values": pixel_values, "labels": labels} 

# Compute metrics for evaluation
def compute_metrics(eval_pred):
    metric_accuracy = load("accuracy")
    metric_f1 = load("f1")
    metric_confusion = load("confusion_matrix")
    outputs, labels = eval_pred
    predictions = np.argmax(outputs, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]

    return {"accuracy": accuracy, "f1": f1}

def compute_metrics_test(eval_pred):
    metric_accuracy = load("accuracy")
    metric_f1 = load("f1")
    metric_confusion = load("confusion_matrix")
    outputs, labels = eval_pred
    predictions = np.argmax(outputs, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]

    confusion = metric_confusion.compute(predictions=predictions, references=labels)
    confusion["confusion_matrix"] = np.array(confusion["confusion_matrix"]).tolist()

    return {"accuracy": accuracy, "f1": f1, "confusion_matrix": confusion}

def compute_metrics_test_no_confusion(eval_pred):
    """
    Compute metrics for testing without confusion matrix to avoid TensorBoard logging warnings.
    The confusion matrix will be computed separately.
    """
    metric_accuracy = load("accuracy")
    metric_f1 = load("f1")
    outputs, labels = eval_pred
    predictions = np.argmax(outputs, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]

    return {"accuracy": accuracy, "f1": f1}

# Split dataset into train, validation, and test
def split_to_train_val_test(dataset):
    split1 = dataset.train_test_split(test_size=0.2)
    train_ds = split1["train"]
    split2 = split1["test"].train_test_split(test_size=0.5)
    val_ds = split2["train"]
    test_ds = split2["test"]
    return train_ds, val_ds, test_ds

def parse_HF_args():
    """
    Parse hugging face arguments from a JSON file
    """
    # **Added argparse to handle the JSON file path as a command line argument**
    parser = argparse.ArgumentParser(description="Run Hugging Face model with JSON config")
    parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
    args = parser.parse_args()

    # **Load the JSON file specified by the command line argument**
    with open(args.config, 'r') as f:
        json_args = json.load(f)
    
    # Handle cost_matrix conversion
    if 'cost_matrix' in json_args and json_args['cost_matrix'] is not None:
        # If cost_matrix is already a list (from JSON), convert it to string for HfArgumentParser
        if isinstance(json_args['cost_matrix'], list):
            json_args['cost_matrix'] = json.dumps(json_args['cost_matrix'])
    
    hf_parser = HfArgumentParser(ScriptTrainingArguments)
    script_args = hf_parser.parse_dict(json_args)
    parsed_args = script_args[0]
    
    # Convert cost_matrix back to list if it exists
    if parsed_args.cost_matrix is not None:
        parsed_args.cost_matrix = json.loads(parsed_args.cost_matrix)
    
    return parsed_args  # **Returns the parsed arguments**

@dataclass
class ScriptTrainingArguments:
    """
    Arguments pertaining to this script
    """
    dataset: str = field(
        default=None,
        metadata={"help": "Name of dataset from HG hub"}
    )
    model: str = field(
        default=None,
        metadata={"help": "Name of model from HG hub"}
    )
    weights: str = field(
        default=None,
        metadata={"help": "Weight of model from HG hub"}
    )
    learning_rate: float = field(  # **Added learning_rate to the dataclass**
        default=5e-5,
        metadata={"help": "Learning rate for training"}
    )
    num_train_epochs: int = field(  # **Added num_train_epochs to the dataclass**
        default=5,
        metadata={"help": "Number of training epochs"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of training epochs"}
    )
    num_labels: int = field(
        default=5,
        metadata={"help": "Number of training labels"}
    )
    wandb: str = field(
        default=False,
        metadata={"help": "Wandb upload"}
    )
    push_to_hub: str = field(
        default=False,
        metadata={"help": "Push to hub"}
    )
    output_dir: str = field(
        default="convnext",
        metadata={"help": "Output directory for model checkpoints"}
    )
    dataset_host: str = field(
        default="huggingface",
        metadata={"help": "Dataset host"}
    )
    local_dataset_name: str = field(
        default=None,
        metadata={"help": "Name of local dataset"}
    )
    local_folder_path: str = field(
        default=None,
        metadata={"help": "Path to local folder containing class subfolders with images"}
    )
    loss_function: str = field(
        default="cross_entropy",
        metadata={"help": "Loss function to use"}
    )
    cost_matrix: Optional[str] = field(
        default=None,
        metadata={"help": "Cost matrix for cost-sensitive loss functions (JSON string)"}
    )
    sweep_mode: bool = field(
        default=False,
        metadata={"help": "Whether we are running in cost matrix sweep mode"}
    )
    sweep_cost_value: float = field(
        default=None,
        metadata={"help": "Current cost value being swept (only used in sweep mode)"}
    )
    sweep_matrix_row: int = field(
        default=None,
        metadata={"help": "Row index of cost matrix cell being swept (only used in sweep mode)"}
    )
    sweep_matrix_col: int = field(
        default=None,
        metadata={"help": "Column index of cost matrix cell being swept (only used in sweep mode)"}
    )

def preprocess_hf_dataset(dataset_name, model_name):
    """
    Preprocess the Hugging Face dataset with the specified model 
    """
    dataset = load_dataset(dataset_name)

    image_processor = CustomImageProcessor.from_pretrained(model_name, use_fast=True)
    # Preprocessing
    image_preprocessor = ImagePreprocessor(dataset, image_processor)

    train_ds, val_ds, test_ds = split_to_train_val_test(dataset['train'])
    train_ds.set_transform(image_preprocessor.preprocess_train)
    val_ds.set_transform(image_preprocessor.preprocess_val)
    test_ds.set_transform(image_preprocessor.preprocess_test)

    return train_ds, val_ds, test_ds

def preprocess_kg_dataset(cloud_dataset_name, local_dataset_name, model_name):
    """ 
    Preprocess the Kaggle dataset
    """
    path = kagglehub.dataset_download(cloud_dataset_name)
    local_path = f"{path}/{local_dataset_name}"
    dataset = load_dataset("imagefolder", data_dir=local_path)

    image_processor = CustomImageProcessor.from_pretrained(model_name, use_fast=True)
    # Preprocessing
    image_preprocessor = ImagePreprocessor(dataset, image_processor)

    train_ds, val_ds, test_ds = split_to_train_val_test(dataset['train'])
    train_ds.set_transform(image_preprocessor.preprocess_train)
    val_ds.set_transform(image_preprocessor.preprocess_val)
    test_ds.set_transform(image_preprocessor.preprocess_test)

    return train_ds, val_ds, test_ds

def preprocess_hf_ros_dataset(dataset_name, model_name):
    """
    Preprocess the Hugging Face dataset with the specified model 
    """
    dataset = load_dataset(dataset_name)

    image_processor = CustomImageProcessor.from_pretrained(model_name, use_fast=True)
    # Preprocessing
    image_preprocessor = ImagePreprocessor(dataset, image_processor)

    class_counts = Counter(dataset["train"]["label"])
    max_count = max(class_counts.values())

    oversampled_data = []

    for label, count in class_counts.items():
        class_samples = [example for example in dataset["train"] if example["label"] == label]
        num_to_add = max_count - count
        oversampled_class_samples = class_samples + random.choices(class_samples, k=num_to_add)
        oversampled_data.extend(oversampled_class_samples)

    oversampled_dataset = Dataset.from_dict({key: [d[key] for d in oversampled_data] for key in oversampled_data[0]})

    train_ds, val_ds, test_ds = split_to_train_val_test(oversampled_dataset)
    train_ds.set_transform(image_preprocessor.preprocess_train)
    val_ds.set_transform(image_preprocessor.preprocess_val)
    test_ds.set_transform(image_preprocessor.preprocess_test)

    return train_ds, val_ds, test_ds

# Enhanced Image Preprocessor for data augmentation that works with CustomImageProcessor
class ImagePreprocessor():
    def __init__(self, dataset, image_processor):
        """
        Initialize ImagePreprocessor with CustomImageProcessor.
        
        Args:
            dataset: The dataset to be processed
            image_processor: CustomImageProcessor instance
        """
        self.image_processor = image_processor
        
        # Get transforms from the custom image processor
        self.train_transforms = image_processor.get_transform_for_training(
            random_resize_crop=True,
            horizontal_flip=True,
            color_jitter=False
        )
        
        self.val_transforms = image_processor.get_transform_for_validation()
        self.test_transforms = image_processor.get_transform_for_test()

    def preprocess_train(self, image_batch):
        """Preprocess training images with data augmentation."""
        image_batch["pixel_values"] = [
            self.train_transforms(image.convert("RGB")) for image in image_batch["image"]
        ]
        return image_batch
    
    def preprocess_val(self, image_batch):
        """Preprocess validation images."""
        image_batch["pixel_values"] = [
            self.val_transforms(image.convert("RGB")) for image in image_batch["image"]
        ]
        return image_batch
    
    def preprocess_test(self, image_batch):
        """Preprocess test images."""
        image_batch["pixel_values"] = [
            self.test_transforms(image.convert("RGB")) for image in image_batch["image"]
        ]
        return image_batch

def save_metrics_to_file(metrics, class_metrics, class_counts, output_dir, filename_suffix=""):
    """
    Save all metrics to JSON and text files for easy analysis.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        class_metrics: Dictionary containing per-class metrics
        class_counts: Dictionary containing sample counts per class
        output_dir: Directory to save the metrics files
        filename_suffix: Optional suffix for the filename
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"metrics_{timestamp}{filename_suffix}"
    
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    # Prepare comprehensive metrics dictionary
    comprehensive_metrics = {
        "timestamp": timestamp,
        "overall_metrics": {
            "accuracy": float(metrics.get("eval_accuracy", 0.0)),
            "f1_score": float(metrics.get("eval_f1", 0.0)),
            "loss": float(metrics.get("eval_loss", 0.0))
        },
        "class_counts": convert_numpy_types(class_counts),
        "per_class_metrics": convert_numpy_types(class_metrics),
        "confusion_matrix": convert_numpy_types(metrics.get("eval_confusion_matrix", {}).get("confusion_matrix", []))
    }
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(comprehensive_metrics, f, indent=2)
    
    # Save as readable text file
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Training Metrics Report - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {comprehensive_metrics['overall_metrics']['accuracy']:.4f}\n")
        f.write(f"F1 Score: {comprehensive_metrics['overall_metrics']['f1_score']:.4f}\n")
        f.write(f"Loss: {comprehensive_metrics['overall_metrics']['loss']:.4f}\n\n")
        
        f.write("CLASS SAMPLE COUNTS:\n")
        f.write("-" * 20 + "\n")
        for class_id, count in sorted(comprehensive_metrics["class_counts"].items()):
            f.write(f"Class {class_id}: {count} samples\n")
        f.write("\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 20 + "\n")
        class_metrics_data = comprehensive_metrics["per_class_metrics"]
        for class_id in sorted(class_metrics_data["accuracy"].keys()):
            f.write(f"Class {class_id}:\n")
            f.write(f"  True Positive Rate (TPR): {class_metrics_data['accuracy'][class_id]:.4f}\n")
            f.write(f"  False Positive Rate (FPR): {class_metrics_data['false_positive_rate'][class_id]:.4f}\n")
            f.write(f"  False Negative Rate (FNR): {class_metrics_data['false_negative_rate'][class_id]:.4f}\n")
            f.write("\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 20 + "\n")
        confusion_matrix = comprehensive_metrics["confusion_matrix"]
        if confusion_matrix:
            for i, row in enumerate(confusion_matrix):
                f.write(f"Row {i}: {row}\n")
    
    print(f"Metrics saved to: {json_path} and {txt_path}")
    return json_path, txt_path

def create_confusion_matrix_visualization(confusion_matrix, class_names=None, output_dir="results", filename_suffix=""):
    """
    Create a professional confusion matrix visualization with three views:
    1. Raw counts
    2. Recall (normalized by true class)
    3. Precision (normalized by predicted class)
    
    Args:
        confusion_matrix: 2D array/list containing the confusion matrix
        class_names: List of class names (optional)
        output_dir: Directory to save the visualization
        filename_suffix: Optional suffix for the filename
    """
    if not VISUALIZATION_AVAILABLE:
        print("Skipping visualization: matplotlib/seaborn not available")
        return None
        
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy array if needed
    cm = np.array(confusion_matrix)
    
    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up the matplotlib figure with professional styling for three subplots
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix - Raw Counts', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 2: Normalized by true class (recall/sensitivity)
    cm_recall = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_recall = np.nan_to_num(cm_recall)  # Handle division by zero
    
    sns.heatmap(cm_recall, annot=True, fmt='.3f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Recall (TPR)'})
    ax2.set_title('Confusion Matrix - Recall\n(Normalized by True Class)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 3: Normalized by predicted class (precision)
    cm_precision = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    cm_precision = np.nan_to_num(cm_precision)  # Handle division by zero
    
    sns.heatmap(cm_precision, annot=True, fmt='.3f', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax3, cbar_kws={'label': 'Precision (PPV)'})
    ax3.set_title('Confusion Matrix - Precision\n(Normalized by Predicted Class)', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust layout and save
    plt.tight_layout(pad=3.0)
    
    # Save the figure
    output_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}{filename_suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"Confusion matrix visualizations (3-panel) saved to: {output_path} and {output_path.replace('.png', '.pdf')}")
    
    # Also create a separate detailed visualization
    create_detailed_confusion_matrix(cm, class_names, output_dir, filename_suffix, timestamp)
    
    plt.close()
    return output_path

def create_detailed_confusion_matrix(cm, class_names, output_dir, filename_suffix, timestamp):
    """
    Create a detailed confusion matrix with additional statistics.
    """
    if not VISUALIZATION_AVAILABLE:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate additional statistics
    total_samples = np.sum(cm)
    accuracy_per_class = np.diag(cm) / np.sum(cm, axis=1)
    precision_per_class = np.diag(cm) / np.sum(cm, axis=0)
    
    # Create the heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Set up the plot
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Detailed Confusion Matrix with Statistics', fontsize=16, fontweight='bold', pad=20)
    
    # Annotate each cell with count and percentage
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            percentage = count / total_samples * 100
            text_color = "white" if count > thresh else "black"
            ax.text(j, i, f'{count}\n({percentage:.1f}%)',
                   ha="center", va="center", color=text_color, fontweight='bold')
    
    # Add statistics text
    stats_text = "Per-Class Statistics:\n"
    for i, class_name in enumerate(class_names):
        recall = accuracy_per_class[i] if not np.isnan(accuracy_per_class[i]) else 0
        precision = precision_per_class[i] if not np.isnan(precision_per_class[i]) else 0
        stats_text += f"{class_name}: Recall={recall:.3f}, Precision={precision:.3f}\n"
    
    # Add text box with statistics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    
    # Save the detailed figure
    detailed_path = os.path.join(output_dir, f"confusion_matrix_detailed_{timestamp}{filename_suffix}.png")
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Detailed confusion matrix saved to: {detailed_path}")
    return detailed_path

def perform_comprehensive_evaluation(trainer, test_ds, script_args, dataset_name=None, class_names=None):
    """
    Perform comprehensive evaluation including confusion matrix, per-class metrics, and visualizations.
    
    Args:
        trainer: The trained model trainer
        test_ds: Test dataset
        script_args: Script arguments containing configuration
        dataset_name: Name of the dataset (optional)
        class_names: List of class names (optional, for local datasets)
    
    Returns:
        str: Path to the results directory
    """
    print("\n" + "="*60)
    print("PERFORMING COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Create results directory with timestamp
    now = datetime.datetime.now()
    date_str = now.strftime("%m-%d")  # Format: MM-DD
    time_str = now.strftime("%H-%M")  # Format: HH-MM
    
    # Build directory name based on sweep mode
    if script_args.sweep_mode and script_args.sweep_cost_value is not None:
        # Sweep mode: {output_dir}__modifier{cost_value}_true{row}_predict{col}_{date}_{time}
        cost_str = str(script_args.sweep_cost_value)  # Keep decimal point as specified
        dir_name = f"{script_args.output_dir}__modifier{cost_str}_true{script_args.sweep_matrix_row}_predict{script_args.sweep_matrix_col}_{date_str}_{time_str}"
    else:
        # Normal mode: {output_dir}_{date}_{time}
        dir_name = f"{script_args.output_dir}_{date_str}_{time_str}"
    
    results_dir = f"results/resnet_test/{dir_name}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Save run configuration
    print("Saving run configuration...")
    save_run_configuration(script_args, results_dir, dataset_name)
    
    # Perform evaluation
    print("Running model evaluation...")
    metrics = trainer.evaluate()
    
    # Get confusion matrix
    print("Computing detailed metrics...")
    metric_confusion = load("confusion_matrix")
    
    # Get predictions
    predictions = trainer.predict(test_ds)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    
    # Compute confusion matrix
    confusion_result = metric_confusion.compute(predictions=y_pred, references=y_true)
    confusion_matrix = confusion_result["confusion_matrix"]
    
    # Add confusion matrix to metrics
    metrics["eval_confusion_matrix"] = confusion_result
    
    # Calculate per-class metrics
    num_classes = len(confusion_matrix)
    total_samples = np.sum(confusion_matrix)
    
    # Count samples per class
    class_counts = {}
    total_samples_per_class = {}
    for i in range(num_classes):
        class_total = sum(confusion_matrix[i])
        class_counts[i] = class_total
        total_samples_per_class[i] = class_total
    
    # Initialize per-class metrics dictionary
    class_metrics = {
        "accuracy": {},
        "false_positive_rate": {},
        "false_negative_rate": {},
        "precision": {},
        "recall": {},
        "f1_score": {}
    }

    # Calculate metrics for each class
    for i in range(num_classes):
        true_positives = confusion_matrix[i][i]
        false_negatives = sum(confusion_matrix[i]) - true_positives
        false_positives = sum(row[i] for row in confusion_matrix) - true_positives
        true_negatives = total_samples - (true_positives + false_negatives + false_positives)

        # Recall (True Positive Rate) - same as accuracy per class
        class_metrics["recall"][i] = true_positives / total_samples_per_class[i] if total_samples_per_class[i] > 0 else 0.0
        class_metrics["accuracy"][i] = class_metrics["recall"][i]  # For backward compatibility

        # Precision
        class_metrics["precision"][i] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

        # F1 Score
        precision = class_metrics["precision"][i]
        recall = class_metrics["recall"][i]
        class_metrics["f1_score"][i] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # False Positive Rate (FPR)
        class_metrics["false_positive_rate"][i] = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0

        # False Negative Rate (FNR)
        class_metrics["false_negative_rate"][i] = false_negatives / total_samples_per_class[i] if total_samples_per_class[i] > 0 else 0.0

    # Save all metrics to files
    print("Saving metrics to files...")
    save_metrics_to_file(metrics, class_metrics, class_counts, results_dir, f"_{script_args.loss_function}")
    
    # Create professional confusion matrix visualizations
    print("Creating confusion matrix visualizations...")
    
    # Use provided class names or try to get from dataset, or fall back to generic names
    if class_names is not None:
        # Use provided class names (from local folder dataset)
        final_class_names = class_names
        print(f"Using provided class names: {final_class_names}")
    else:
        # Try to get class names from dataset if available
        try:
            if dataset_name:
                dataset = load_dataset(dataset_name)
                if hasattr(dataset["train"].features["label"], "names"):
                    final_class_names = dataset["train"].features["label"].names
                else:
                    final_class_names = [f"Class {i}" for i in range(num_classes)]
            else:
                final_class_names = [f"Class {i}" for i in range(num_classes)]
        except:
            final_class_names = [f"Class {i}" for i in range(num_classes)]
        print(f"Using class names: {final_class_names}")
    
    create_confusion_matrix_visualization(
        confusion_matrix, 
        class_names=final_class_names, 
        output_dir=results_dir, 
        filename_suffix=f"_{script_args.loss_function}"
    )
    
    # Print summary to console (minimal output)
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED - RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Output Directory: {results_dir}")
    print(f"Overall Accuracy: {metrics.get('eval_accuracy', 0.0):.4f}")
    print(f"Overall F1 Score: {metrics.get('eval_f1', 0.0):.4f}")
    print(f"Loss Function Used: {script_args.loss_function}")
    print(f"Total Test Samples: {total_samples}")
    if class_names:
        print(f"Classes: {', '.join(final_class_names)}")
    print(f"\nDetailed metrics and visualizations saved to: {results_dir}")
    print(f"{'='*60}")
    
    return results_dir

def preprocess_local_folder_dataset(folder_path, model_name, test_size=0.1, val_size=0.1, random_state=42):
    """
    Load images from a local folder structure where subfolders represent classes.
    
    Args:
        folder_path (str): Path to the root folder containing class subfolders
        model_name (str): Model name for the image processor
        test_size (float): Proportion of data to use for testing (default: 0.1)
        val_size (float): Proportion of remaining data to use for validation (default: 0.1)
        random_state (int): Random seed for reproducible splits
    
    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names) - preprocessed datasets and class names
    """
    print(f"Loading dataset from: {folder_path}")
    
    # Check if path exists
    if not os.path.exists(folder_path):
        raise ValueError(f"Dataset path does not exist: {folder_path}")
    
    # Get all class folders (subdirectories)
    class_folders = [d for d in os.listdir(folder_path) 
                    if os.path.isdir(os.path.join(folder_path, d)) and not d.startswith('.')]
    class_folders.sort()  # Ensure consistent ordering
    
    if not class_folders:
        raise ValueError(f"No class folders found in {folder_path}")
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    
    # Create class to index mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    
    # Load all images and labels
    image_paths = []
    labels = []
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for class_name in class_folders:
        class_folder = os.path.join(folder_path, class_name)
        class_idx = class_to_idx[class_name]
        
        # Get all image files in this class folder
        class_images = []
        for file in os.listdir(class_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(class_folder, file)
                class_images.append(image_path)
        
        print(f"Class '{class_name}' (index {class_idx}): {len(class_images)} images")
        
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))
    
    print(f"Total images loaded: {len(image_paths)}")
    
    # Split the data: first split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Then split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create datasets
    def create_dataset_dict(image_paths, labels):
        """Create a dictionary compatible with HuggingFace datasets"""
        dataset_dict = {
            'image': [],
            'label': labels
        }
        
        # Load images
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                dataset_dict['image'].append(image)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                # Skip this image and corresponding label
                continue
        
        return dataset_dict
    
    # Create dataset dictionaries
    train_dict = create_dataset_dict(X_train, y_train)
    val_dict = create_dataset_dict(X_val, y_val)
    test_dict = create_dataset_dict(X_test, y_test)
    
    # Convert to HuggingFace Dataset objects
    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)
    
    # Set up image processor and preprocessing
    image_processor = CustomImageProcessor.from_pretrained(model_name, use_fast=True)
    
    # Create a dummy dataset object for ImagePreprocessor (it expects a dataset with some structure)
    class DummyDataset:
        def __init__(self):
            pass
    
    dummy_dataset = DummyDataset()
    image_preprocessor = ImagePreprocessor(dummy_dataset, image_processor)
    
    # Apply transforms
    train_dataset.set_transform(image_preprocessor.preprocess_train)
    val_dataset.set_transform(image_preprocessor.preprocess_val)
    test_dataset.set_transform(image_preprocessor.preprocess_test)
    
    # Print dataset info
    print(f"Dataset created successfully!")
    print(f"Class mapping: {idx_to_class}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Return class names as a list in the correct order (index -> class_name)
    class_names = [idx_to_class[i] for i in range(len(class_folders))]
    
    return train_dataset, val_dataset, test_dataset, class_names 

def save_run_configuration(script_args, output_dir, dataset_name=None):
    """
    Save the input configuration for the current run to a JSON file.
    
    Args:
        script_args: Script arguments containing all configuration parameters
        output_dir: Directory to save the configuration file
        dataset_name: Optional dataset name for better identification
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for the configuration
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine dataset path/location based on dataset_host
    dataset_location = None
    if script_args.dataset_host == "huggingface":
        dataset_location = script_args.dataset
    elif script_args.dataset_host == "kaggle":
        dataset_location = f"kaggle:{script_args.dataset}"
        if script_args.local_dataset_name:
            dataset_location += f"/{script_args.local_dataset_name}"
    elif script_args.dataset_host == "local_folder":
        dataset_location = script_args.local_folder_path
    else:
        dataset_location = script_args.dataset if script_args.dataset else "unknown"
    
    # Prepare configuration dictionary
    config = {
        "run_info": {
            "timestamp": timestamp,
            "output_directory": output_dir
        },
        "model_configuration": {
            "model": script_args.model,
            "weights": script_args.weights,
            "num_labels": script_args.num_labels
        },
        "dataset_configuration": {
            "dataset_host": script_args.dataset_host,
            "dataset": script_args.dataset,
            "dataset_location": dataset_location,
            "local_dataset_name": script_args.local_dataset_name,
            "local_folder_path": script_args.local_folder_path
        },
        "training_parameters": {
            "batch_size": script_args.batch_size,
            "learning_rate": script_args.learning_rate,
            "num_train_epochs": script_args.num_train_epochs,
            "loss_function": script_args.loss_function
        },
        "cost_matrix_configuration": {
            "cost_matrix": script_args.cost_matrix,
            "sweep_mode": script_args.sweep_mode,
            "sweep_cost_value": script_args.sweep_cost_value,
            "sweep_matrix_row": script_args.sweep_matrix_row,
            "sweep_matrix_col": script_args.sweep_matrix_col
        },
        "other_settings": {
            "wandb_enabled": script_args.wandb,
            "push_to_hub": script_args.push_to_hub
        }
    }
    
    # Save configuration to JSON file
    config_path = os.path.join(output_dir, f"run_configuration_{timestamp}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Run configuration saved to: {config_path}")
    return config_path 