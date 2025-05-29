import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, parent_dir)

import torch
import wandb
import torch.nn as nn
from datasets import load_dataset
from transformers import set_seed, TrainingArguments, Trainer, AutoImageProcessor
from transformers.utils import logging

class CustomTrainer(Trainer):
    def __init__(self, loss_fxn, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

def load_pretrained_weights(weights_path, model):
    print(weights_path)
    pretrained_weights_path = os.path.abspath(weights_path)
    pretrained_weights = torch.load(pretrained_weights_path, map_location="cpu")
    filtered_weights = {k: v for k, v in pretrained_weights.items() if "classifier" not in k}
    missing_keys, unexpected_keys = model.load_state_dict(filtered_weights, strict=False)

    # Check for any missing or unexpected keys
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    print("Pretrained weights loaded successfully!")

    return model

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

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load dataset
    if script_args.dataset_host == "huggingface":
        train_ds, val_ds, test_ds = preprocess_hf_dataset(script_args.dataset, script_args.model)

    config = ResNetConfig(num_labels=script_args.num_labels, depths=[3, 3, 9, 3])
    model = ResNet(config)
    model = load_pretrained_weights(model)

