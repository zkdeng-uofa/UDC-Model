# UDC-Model

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

## Image Processor Configuration

The training script uses fast image processors by default (`use_fast=True`) for better performance. This eliminates warnings about slow processors and provides faster preprocessing.

- **Fast processors**: Optimized implementations with better performance
- **Slow processors**: Legacy implementations (can be used with `use_fast=False` if needed)

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

## Troubleshooting

### Common Warnings and Solutions

1. **TensorBoard confusion matrix warning**: Fixed by computing confusion matrix separately from scalar metrics
2. **Slow image processor warning**: Resolved by using `use_fast=True` in AutoImageProcessor calls
3. **Missing visualization dependencies**: Install with `pip install matplotlib seaborn`