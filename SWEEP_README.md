# Cost Matrix Sweep Configuration Guide

This guide explains how to use the configurable cost matrix sweep system.

## Quick Start

Run the sweep with the default configuration:
```bash
./run_cost_matrix_sweep.sh
```

Run with a custom configuration:
```bash
./run_cost_matrix_sweep.sh sweep_config_examples/quick_test.json
```

## Configuration File Structure

The sweep configuration is defined in a JSON file with the following structure:

```json
{
    "cost_range": {
        "min": 0.0,
        "max": 10.0,
        "step": 0.5
    },
    "matrix_cell": {
        "row": 1,
        "col": 2,
        "description": "Matrix cell [row, col] to modify during sweep"
    },
    "experiment": {
        "output_dir": "cost_matrix_sweep_results",
        "base_config": "config/localDatasetConfig.json",
        "description": "Cost matrix sweep experiment description"
    },
    "analysis": {
        "update_graphs_each_iteration": true,
        "generate_final_report": true
    }
}
```

## Configuration Parameters

### `cost_range`
- **`min`**: Minimum cost value (float)
- **`max`**: Maximum cost value (float)
- **`step`**: Step size between values (float)

### `matrix_cell`
- **`row`**: Row index of the cost matrix cell to modify (0-based integer)
- **`col`**: Column index of the cost matrix cell to modify (0-based integer)
- **`description`**: Human-readable description of the experiment

### `experiment`
- **`output_dir`**: Directory name where results will be saved
- **`base_config`**: Path to the base training configuration JSON file
- **`description`**: Description of the experiment

### `analysis`
- **`update_graphs_each_iteration`**: Whether to update graphs after each training run (boolean)
- **`generate_final_report`**: Whether to generate a final comprehensive report (boolean)

## Example Configurations

### Quick Test (3 values)
```json
{
    "cost_range": {"min": 0.0, "max": 1.0, "step": 0.5},
    "matrix_cell": {"row": 1, "col": 2},
    "experiment": {
        "output_dir": "quick_test_results",
        "base_config": "config/localDatasetConfig.json"
    },
    "analysis": {
        "update_graphs_each_iteration": true,
        "generate_final_report": true
    }
}
```

### Fine-Grained Sweep (21 values)
```json
{
    "cost_range": {"min": 0.0, "max": 5.0, "step": 0.25},
    "matrix_cell": {"row": 0, "col": 0},
    "experiment": {
        "output_dir": "fine_sweep_results",
        "base_config": "config/localDatasetConfig.json"
    },
    "analysis": {
        "update_graphs_each_iteration": false,
        "generate_final_report": true
    }
}
```

## File Structure

After running a sweep, the following structure is created:

```
{output_dir}/
├── metrics_summary.csv          # Aggregated metrics from all runs
├── logs/                        # Training logs for each iteration
│   ├── training_cost_0.0.log
│   ├── training_cost_0.5.log
│   └── ...
└── graphs/                      # Analysis visualizations
    ├── overall_metrics.png
    ├── class1_metrics.png
    ├── confusion_cell_metrics.png
    └── comprehensive_dashboard.png
```

## Metrics Collected

For each cost value, the following metrics are collected:
- Overall accuracy, precision, recall, F1 score
- Class 1 specific accuracy, precision, recall
- Confusion matrix cell values (raw count, precision, recall)

## Tips

1. **For quick testing**: Use a small range (e.g., 0.0 to 1.0 with step 0.5)
2. **For production runs**: Disable `update_graphs_each_iteration` to save time
3. **For different cells**: Experiment with different `row` and `col` values
4. **For different datasets**: Update the `base_config` path

## Troubleshooting

1. **Configuration not found**: Ensure the config file path is correct
2. **Training failures**: Check the log files in `{output_dir}/logs/`
3. **Missing metrics**: Verify the base config file is valid and accessible 