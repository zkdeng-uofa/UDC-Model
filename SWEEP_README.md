# Cost Matrix Sweep Configuration Guide

This guide explains how to use the configurable cost matrix sweep system.

## Quick Start

Run the sweep with the default configuration:
```bash
./run_cost_matrix_sweep.sh
```

Run with a custom configuration:
```bash
./run_cost_matrix_sweep.sh config/quick_test.json
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
- **`output_dir`**: Directory path where results will be saved (should include "results/" prefix)
- **`base_config`**: Path to the base training configuration JSON file
- **`description`**: Description of the experiment

### `analysis`
- **`update_graphs_each_iteration`**: Whether to update graphs after each training run (boolean)
- **`generate_final_report`**: Whether to generate a final comprehensive report (boolean)

## Example Configurations

### Quick Test (3 values)
Available as `config/quick_test.json`:
```json
{
    "cost_range": {"min": 0.0, "max": 1.0, "step": 0.5},
    "matrix_cell": {"row": 1, "col": 2},
    "experiment": {
        "output_dir": "results/quick_test_results",
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
        "output_dir": "results/fine_sweep_results",
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
results/{output_dir}/
├── metrics_summary.csv          # Aggregated metrics from all runs
├── logs/                        # Training logs for each iteration
│   ├── training_cost_0.0.log
│   ├── training_cost_0.5.log
│   └── ...
└── graphs/                      # Analysis visualizations
    ├── overall_metrics.png
    ├── class{X}_metrics.png
    ├── confusion_cell_{row}_{col}_metrics.png
    └── comprehensive_dashboard_{row}_{col}.png
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

## File Organization

The repository now uses the following structure:
- **Configuration files**: `config/` directory
- **Analysis scripts**: `utils/` directory  
- **Results**: `results/` directory with experiment subfolders
- **Main scripts**: Root directory

## Troubleshooting

1. **Configuration not found**: Ensure the config file path is correct (should be in `config/` directory)
2. **Training failures**: Check the log files in `results/{output_dir}/logs/`
3. **Missing metrics**: Verify the base config file is valid and accessible
4. **Script errors**: Analysis scripts are now in `utils/` directory 