#!/bin/bash

# Cost Matrix Sweep Experiment Script
# This script iterates through different cost matrix values for a specified cell
# and collects comprehensive metrics for analysis

set -e  # Exit on any error

# Default sweep configuration file
SWEEP_CONFIG_FILE="sweep_config.json"

# Check if custom config file is provided
if [[ $# -eq 1 ]]; then
    SWEEP_CONFIG_FILE="$1"
fi

# Check if sweep config file exists
if [[ ! -f "$SWEEP_CONFIG_FILE" ]]; then
    echo "ERROR: Sweep configuration file not found: $SWEEP_CONFIG_FILE"
    echo "Usage: $0 [sweep_config.json]"
    exit 1
fi

echo "Using sweep configuration: $SWEEP_CONFIG_FILE"

# Read configuration from JSON file
read_config() {
    python3 -c "
import json
import sys

try:
    with open('$SWEEP_CONFIG_FILE', 'r') as f:
        config = json.load(f)
    
    print('CONFIG_FILE=' + config['experiment']['base_config'])
    print('RESULTS_DIR=' + config['experiment']['output_dir'])
    print('COST_MIN=' + str(config['cost_range']['min']))
    print('COST_MAX=' + str(config['cost_range']['max']))
    print('COST_STEP=' + str(config['cost_range']['step']))
    print('MATRIX_ROW=' + str(config['matrix_cell']['row']))
    print('MATRIX_COL=' + str(config['matrix_cell']['col']))
    print('UPDATE_GRAPHS=' + str(config['analysis']['update_graphs_each_iteration']).lower())
    print('GENERATE_FINAL=' + str(config['analysis']['generate_final_report']).lower())
    
except Exception as e:
    print(f'ERROR: Failed to read configuration: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Load configuration
eval $(read_config)

# Derived configuration
BACKUP_CONFIG="${CONFIG_FILE}_backup.json"
METRICS_CSV="$RESULTS_DIR/metrics_summary.csv"
LOGS_DIR="$RESULTS_DIR/logs"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Backup original config
echo "Backing up original config..."
cp "$CONFIG_FILE" "$BACKUP_CONFIG"

# Initialize CSV file with headers
echo "Creating metrics CSV file..."
cat > "$METRICS_CSV" << EOF
cost_matrix_value,overall_accuracy,overall_precision,overall_recall,overall_f1,class1_accuracy,class1_precision,class1_recall,confusion_1_2_raw,confusion_1_2_precision,confusion_1_2_recall,results_dir,timestamp
EOF

echo "=========================================="
echo "STARTING COST MATRIX SWEEP EXPERIMENT"
echo "=========================================="
echo "Configuration: $SWEEP_CONFIG_FILE"
echo "Cost matrix cell [$MATRIX_ROW,$MATRIX_COL] will be varied from $COST_MIN to $COST_MAX in steps of $COST_STEP"
echo "Base config: $CONFIG_FILE"
echo "Results will be saved to: $RESULTS_DIR"
echo "=========================================="

# Function to update cost matrix in JSON file
update_cost_matrix() {
    local cost_value=$1
    
    # Use Python to update the JSON file precisely
    python3 -c "
import json
import sys

# Read the JSON file
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

# Update the specific cell using the configured row and column
config['cost_matrix'][$MATRIX_ROW][$MATRIX_COL] = float($cost_value)

# Write back to file
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=4)

print(f'Updated cost_matrix[$MATRIX_ROW][$MATRIX_COL] to {$cost_value}')
"
}

# Function to extract metrics from results
extract_metrics() {
    local results_dir=$1
    local cost_value=$2
    
    echo "Extracting metrics from: $results_dir"
    
    # Use the separate Python script to extract metrics
    if python3 extract_metrics.py "$results_dir" "$cost_value" --output "$METRICS_CSV"; then
        return 0
    else
        echo "WARNING: Failed to extract metrics for cost value $cost_value"
        return 1
    fi
}

# Main iteration loop
cost_values=($(python3 -c "import numpy as np; print(' '.join([str(x) for x in np.arange($COST_MIN, $COST_MAX + $COST_STEP/2, $COST_STEP)]))"))

total_iterations=${#cost_values[@]}
current_iteration=0

echo "Generated cost values: ${cost_values[@]}"
echo "Total iterations: $total_iterations"
echo ""

for cost_value in "${cost_values[@]}"; do
    current_iteration=$((current_iteration + 1))
    
    echo ""
    echo "=========================================="
    echo "ITERATION $current_iteration/$total_iterations"
    echo "Cost Matrix Value: $cost_value"
    echo "=========================================="
    
    # Update the cost matrix
    echo "Updating cost matrix value to $cost_value..."
    update_cost_matrix "$cost_value"
    
    # Run training
    echo "Running training..."
    log_file="$LOGS_DIR/training_cost_${cost_value}.log"
    
    if python train.py --config "$CONFIG_FILE" > "$log_file" 2>&1; then
        echo "Training completed successfully."
        
        # Extract the results directory from the log
        results_dir=$(grep "Results will be saved to:" "$log_file" | sed 's/Results will be saved to: //')
        
        if [[ -n "$results_dir" && -d "$results_dir" ]]; then
            echo "Results saved to: $results_dir"
            
            # Extract metrics
            echo "Extracting metrics..."
            if extract_metrics "$results_dir" "$cost_value"; then
                echo "Metrics extracted successfully."
            else
                echo "WARNING: Failed to extract metrics for cost value $cost_value"
            fi
            
            # Update graphs if configured
            if [[ "$UPDATE_GRAPHS" == "true" ]]; then
                echo "Updating analysis graphs..."
                if python3 analyze_cost_matrix_results.py "$RESULTS_DIR"; then
                    echo "Graphs updated successfully."
                else
                    echo "WARNING: Failed to update graphs"
                fi
            fi
        else
            echo "WARNING: Could not find results directory for cost value $cost_value"
        fi
    else
        echo "ERROR: Training failed for cost value $cost_value"
        echo "Check log file: $log_file"
    fi
    
    echo "Iteration $current_iteration/$total_iterations completed."
done

# Restore original config
echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETED"
echo "=========================================="
echo "Restoring original configuration..."
cp "$BACKUP_CONFIG" "$CONFIG_FILE"

if [[ "$GENERATE_FINAL" == "true" ]]; then
    echo "Generating final analysis and summary..."
    python3 analyze_cost_matrix_results.py "$RESULTS_DIR" --final
else
    echo "Skipping final analysis (configured to false)"
fi

echo ""
echo "=========================================="
echo "COST MATRIX SWEEP EXPERIMENT COMPLETED"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo "Metrics summary: $METRICS_CSV"
echo "Training logs: $LOGS_DIR"
echo "Analysis graphs: $RESULTS_DIR/graphs"
echo "==========================================" 