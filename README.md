# UDC-Model

## Custom Image Processor

The project now uses a custom `CustomImageProcessor` that replaces the Hugging Face `AutoImageProcessor`. This provides full control over image preprocessing while maintaining compatibility with the existing codebase.

### Features

- **Model-specific configurations**: Pre-configured settings for ResNet, ViT, ConvNeXt, EfficientNet, and Swin models
- **Easy customization**: Override any parameter for specific requirements
- **Flexible interfaces**: Compatible with existing training pipelines
- **Save/load configurations**: Persist custom settings for reproducibility
- **Multiple transform types**: Separate transforms for training, validation, and testing

### Basic Usage

```python
from utils.image_processor import CustomImageProcessor

# Use with model name (automatically detects model type)
processor = CustomImageProcessor.from_pretrained("resnet50")

# Use with custom configuration
custom_processor = CustomImageProcessor(
    model_name="resnet",
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.25, 0.25, 0.25],
    size={'height': 256, 'width': 256},
    crop_pct=0.9,
    interpolation='bicubic'
)

# Process images (same interface as AutoImageProcessor)
result = processor(images)  # Returns {"pixel_values": tensor}
```

### Supported Model Types

| Model Type | Default Size | Normalization | Interpolation |
|------------|--------------|---------------|---------------|
| ResNet | 224×224 | ImageNet | Bilinear |
| ViT | 224×224 | [-1,1] range | Bicubic |
| ConvNeXt | 224 (shortest edge) | ImageNet | Bicubic |
| EfficientNet | 224 (shortest edge) | ImageNet | Bicubic |
| Swin | 224×224 | ImageNet | Bicubic |

### Custom Configurations

You can create custom configurations for specific needs:

```python
# High-resolution configuration
high_res_config = {
    'image_mean': [0.485, 0.456, 0.406],
    'image_std': [0.229, 0.224, 0.225],
    'size': {'height': 384, 'width': 384},
    'crop_pct': 0.95,
    'interpolation': 'bicubic'
}

processor = CustomImageProcessor(custom_config=high_res_config)
```

### Saving and Loading Configurations

```python
# Save configuration
processor.save_config("my_config.json")

# Load configuration
loaded_processor = CustomImageProcessor.from_config("my_config.json")
```

### Advanced Usage

Get specific transforms for different phases:

```python
# Training with data augmentation
train_transform = processor.get_transform_for_training(
    random_resize_crop=True,
    horizontal_flip=True,
    color_jitter=True
)

# Validation/test without augmentation
val_transform = processor.get_transform_for_validation()
test_transform = processor.get_transform_for_test()
```

### Configuration Files

Example configurations are provided in `config/image_processor_configs.json`. You can modify these or create your own for different models or use cases.

## Cost Matrix Configuration

The `CELossLTV1` loss function now supports configurable cost matrices through the `modelConfig.json` file. This allows you to specify different misclassification costs for different class pairs.

### Configuration

Add a `cost_matrix` field to your `config/modelConfig.json`:

```json
{
    "loss_function": "cost_matrix_cross_entropy",
    "cost_matrix": [
        [1.0, 2.0, 3.0, 2.0, 1.0],
        [2.0, 1.0, 2.0, 3.0, 2.0],
        [3.0, 2.0, 1.0, 2.0, 3.0],
        [2.0, 3.0, 2.0, 1.0, 2.0],
        [1.0, 2.0, 3.0, 2.0, 1.0]
    ]
}
```

### Cost Matrix Format

- The cost matrix should be a square matrix with dimensions equal to the number of classes
- `cost_matrix[i][j]` represents the cost of misclassifying a sample from class `j` as class `i`
- Higher values indicate higher misclassification costs
- Diagonal elements are typically set to 1.0 (correct classification cost)

### Usage

When using the cost-sensitive loss functions (`CELossLTV1` or `CELossLT_LossMult`), the model will automatically use the cost matrix specified in your configuration file. If no cost matrix is provided, it defaults to a uniform cost matrix (all values = 1.0).

### Example

The example above shows higher costs (3.0) for certain misclassifications, encouraging the model to be more careful about specific class confusions.

## Metrics and Visualization Features

The training script now automatically saves comprehensive metrics and creates professional visualizations:

### Output Files Generated

1. **Metrics Files**: Saved in `results/{output_dir}/`
   - `metrics_{timestamp}_{loss_function}.json`: Complete metrics in JSON format
   - `metrics_{timestamp}_{loss_function}.txt`: Human-readable metrics report

2. **Visualization Files**: Professional confusion matrix visualizations
   - `confusion_matrix_{timestamp}_{loss_function}.png/pdf`: Side-by-side raw counts and normalized matrices
   - `confusion_matrix_detailed_{timestamp}_{loss_function}.png`: Detailed matrix with statistics

### Metrics Included

- **Overall Metrics**: Accuracy, F1 Score, Loss
- **Per-Class Metrics**: Precision, Recall, F1 Score, False Positive Rate, False Negative Rate
- **Class Distribution**: Sample counts per class
- **Confusion Matrix**: Raw counts and normalized values

### Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Running Training

```bash
python train.py --config config/modelConfig.json
```

The script will automatically create a results directory and save all metrics and visualizations there. Console output is minimized to show only a summary of results.

## Examples

Run the example script to see the CustomImageProcessor in action:

```bash
python examples/custom_image_processor_example.py
```

## Troubleshooting

### Common Warnings and Solutions

1. **TensorBoard confusion matrix warning**: Fixed by computing confusion matrix separately from scalar metrics
2. **Missing visualization dependencies**: Install with `pip install matplotlib seaborn`
3. **Custom processor import errors**: Ensure `utils/` directory is in your Python path