
import os
import sys

# Add the parent directory of 'models' to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, parent_dir)

import torch
import wandb
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    set_seed, 
    TrainingArguments, 
    Trainer, 
    AutoImageProcessor,
    AutoModelForImageClassification
    )
from transformers.utils import logging
from PIL import Image

# Import from models
from model.ResNetOld import ResNetConfig, ResNetForImageClassification
from utils.loss_functions import LossFunctions
from utils.utils import (
    collate_fn,
    compute_metrics,
    compute_metrics_test,
    parse_HF_args,
    preprocess_hf_dataset,
    preprocess_kg_dataset
)

class CustomTrainer(Trainer):
    def __init__(self, loss_fxn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lossClass = LossFunctions()
        self.loss_fxn = loss_fxn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Unpack inputs dictionary and send to device
        pixel_values = inputs["pixel_values"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device)

        # Forward pass through the model
        outputs = model(pixel_values)

        # Compute loss
        logits = outputs.logits
        loss_fxn = self.lossClass.loss_function(self.loss_fxn)

        loss = loss_fxn(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override the prediction_step to properly handle evaluation (unpacking dict).
        """
        pixel_values = inputs["pixel_values"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device) if "labels" in inputs else None

        with torch.no_grad():
            outputs = model(pixel_values)

        logits = outputs.logits
        loss = None
        if labels is not None:
            #loss = seesaw_loss(logits, labels)
            loss_fxn = self.lossClass.loss_function(self.loss_fxn)
            loss = loss_fxn(logits, labels)
        return (loss, logits, labels)

import torch
from torchvision import transforms


class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),              # Resize the shorter side to 256 pixels
            transforms.CenterCrop(224),          # Center crop to 224x224
            transforms.ToTensor(),               # Convert PIL image to a PyTorch tensor
            transforms.Normalize(                 # Normalize using ImageNet statistics
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, image):
        """
        Process an input image using the defined transformations.
        Args:
            image (PIL.Image or np.array): The input image.
        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        if isinstance(image, Image.Image):
            return self.transform(image)
        elif isinstance(image, torch.Tensor):
            # If already a tensor, apply normalization only
            return transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image)
        else:
            raise ValueError("Input must be a PIL image or torch.Tensor")

# Main function
def main(script_args):

    if script_args.wandb == "True":
        wandb.login(key="e68d14a1a7b3aed71e0455589cde53c783018f5a")
        wandb.init(
            project="convnext",
            name=script_args.output_dir,
        )

        wandb.config.update({
            "model_checkpoint": script_args.output_dir,
            "batch_size": script_args.batch_size,
            "learning_rate": script_args.learning_rate,  # **Updated to use value from JSON**
            "num_train_epochs": script_args.num_train_epochs,  # **Updated to use value from JSON**
        })
    #Load dataset

    image_processor = AutoImageProcessor.from_pretrained(script_args.model)

    if script_args.dataset_host == "huggingface":
        train_ds, val_ds, test_ds = preprocess_hf_dataset(script_args.dataset, script_args.model)
    elif script_args.dataset_host == "kaggle":
        train_ds, val_ds, test_ds = preprocess_kg_dataset(
            script_args.dataset,
            script_args.local_dataset_name,
            script_args.model
        )
    # Pretrained weights
    print(script_args.weights)
    pretrained_weights_path = os.path.abspath(script_args.weights)

    pretrained_weights = torch.load(pretrained_weights_path, map_location="cpu")

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    config = ResNetConfig(num_labels=script_args.num_labels, depths=[3, 4, 6, 3])
    model = ResNetForImageClassification(config)

# Specify the model name or path from the Hugging Face Hub

    dataset = load_dataset(script_args.dataset)

    # labels = dataset["train"].features["label"].names
    # label2id, id2label = dict(), dict()
    # for i, label in enumerate(labels):
    #     label2id[label] = i
    #     id2label[i] = label
    # model_name = "microsoft/resnet-50"

    # Load the pretrained model
    # model = AutoModelForImageClassification.from_pretrained(
    #     model_name,
    #     label2id=label2id,
    #     id2label=id2label,
    #     ignore_mismatched_sizes=True,
    # )


    # (Optional) Load the corresponding image processor (e.g., for preprocessing input images)
    #image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = ResNetForImageClassification(config)
    model.to(device)

    #print(model)
    # Load pretrained weights
    filtered_weights = {k: v for k, v in pretrained_weights.items() if "classifier" not in k}
    #print(f"Filtered weights: {filtered_weights.keys()}")
    missing_keys, unexpected_keys = model.load_state_dict(filtered_weights, strict=False)

    # Check for any missing or unexpected keys
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    print("Pretrained weights loaded successfully!")

    if script_args.wandb == "True":
        training_args = TrainingArguments(
            output_dir=f"checkpoints/{script_args.output_dir}",
            remove_unused_columns=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=script_args.learning_rate,
            per_device_train_batch_size=script_args.batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=script_args.batch_size,
            num_train_epochs=script_args.num_train_epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=(script_args.push_to_hub=="True"),
            report_to="wandb",
        )
    else:
        training_args = TrainingArguments(
            output_dir=f"checkpoints/{script_args.output_dir}",
            remove_unused_columns=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=script_args.learning_rate,
            per_device_train_batch_size=script_args.batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=script_args.batch_size,
            num_train_epochs=script_args.num_train_epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=(script_args.push_to_hub=="True"),
        )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=image_processor,  # AutoImageProcessor works as a tokenizer here for image tasks
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        loss_fxn=script_args.loss_function
    )

    train_results = trainer.train()
    # trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    # trainer.save_metrics("train", train_results.metrics)
    # trainer.save_state()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=image_processor,
        compute_metrics=compute_metrics_test,
        data_collator=collate_fn,
        loss_fxn=script_args.loss_function
    )

    metrics = trainer.evaluate()
    scalar_metrics = {k: v for k, v in metrics.items() if k != "eval_confusion_matrix"}
    # print(metrics.items())
    # print(scalar_metrics)
    trainer.log_metrics("eval", scalar_metrics)

    # Optionally, log the confusion matrix separately

    class_counts = {}
    for sample in test_ds:
        label = sample["label"]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    print("\nTotal Samples Per Class:")
    for class_id, count in sorted(class_counts.items()):
        print(f"Class {class_id}: {count}")

    print(f"Confusion Matrix:")
    for row in metrics["eval_confusion_matrix"]["confusion_matrix"]:
        print(row)

    # Initialize dictionaries to store metrics
    class_metrics = {
        "accuracy": {},
        "false_positive_rate": {},
        "false_negative_rate": {}
    }

    confusion_matrix = metrics["eval_confusion_matrix"]["confusion_matrix"]
    num_classes = len(confusion_matrix)
    # Total samples per class (from confusion matrix rows)
    total_samples_per_class = [sum(row) for row in confusion_matrix]

    # Total samples in the dataset
    total_samples = sum(total_samples_per_class)

    # Calculate metrics for each class
    for i in range(num_classes):
        true_positives = confusion_matrix[i][i]
        false_negatives = sum(confusion_matrix[i]) - true_positives
        false_positives = sum(row[i] for row in confusion_matrix) - true_positives
        true_negatives = total_samples - (true_positives + false_negatives + false_positives)

        # Accuracy for class i
        class_metrics["accuracy"][i] = true_positives / total_samples_per_class[i] if total_samples_per_class[i] > 0 else 0.0

        # False Positive Rate (FPR) for class i
        class_metrics["false_positive_rate"][i] = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0

        # False Negative Rate (FNR) for class i
        class_metrics["false_negative_rate"][i] = false_negatives / total_samples_per_class[i] if total_samples_per_class[i] > 0 else 0.0

    # Log metrics
    print("\nClass-wise Metrics:")
    for class_id in range(num_classes):
        print(f"Class {class_id}:")
        print(f"  True Positive Rate (TPR): {class_metrics['accuracy'][class_id]:.4f}")
        print(f"  False Positive Rate (FPR): {class_metrics['false_positive_rate'][class_id]:.4f}")
        print(f"  False Negative Rate (FNR): {class_metrics['false_negative_rate'][class_id]:.4f}")
    #trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    set_seed(42)
    logger = logging.get_logger(__name__)
    args = parse_HF_args()  # **Updated to use JSON-based arguments**
    main(args)
