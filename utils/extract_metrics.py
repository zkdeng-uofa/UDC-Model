#!/usr/bin/env python3
"""
Metrics Extraction Script

This script extracts metrics from training results and outputs them in CSV format
for the cost matrix sweep experiment.
"""

import json
import sys
import numpy as np
import os
import argparse
from datetime import datetime
from pathlib import Path

def load_sweep_config(sweep_config_file=None):
    """Load the sweep configuration to get matrix cell info."""
    # If a specific config file is provided, use it first
    config_files = []
    if sweep_config_file:
        config_files.append(sweep_config_file)
    
    # Add fallback config files
    config_files.extend(["config/sweep_config.json", "config/quick_test.json"])
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config
            except Exception as e:
                print(f"WARNING: Could not load {config_file}: {e}", file=sys.stderr)
                continue
    
    # Return default config if no config file found
    print("WARNING: No sweep config found, using default values", file=sys.stderr)
    return {
        "matrix_cell": {"row": 1, "col": 2}
    }

def find_metrics_file(results_dir):
    """Find the most recent metrics JSON file in the results directory."""
    metrics_file = None
    
    # Look for metrics files
    for file in Path(results_dir).glob("metrics_*.json"):
        if metrics_file is None or file.stat().st_mtime > metrics_file.stat().st_mtime:
            metrics_file = file
    
    return str(metrics_file) if metrics_file else None

def extract_metrics(results_dir, cost_value, sweep_config_file=None):
    """
    Extract metrics from the results directory and return CSV line.
    
    Args:
        results_dir (str): Path to the results directory
        cost_value (float): The cost matrix value used for this run
        sweep_config_file (str): Path to the sweep configuration file
    
    Returns:
        str: CSV line with extracted metrics, or None if extraction failed
    """
    # Load sweep configuration to get matrix cell info
    sweep_config = load_sweep_config(sweep_config_file)
    matrix_row = sweep_config['matrix_cell']['row']
    matrix_col = sweep_config['matrix_cell']['col']
    target_class = matrix_col  # Target class is determined by the column being modified
    
    # Find the metrics file
    metrics_file = find_metrics_file(results_dir)
    
    if not metrics_file or not os.path.exists(metrics_file):
        print(f"WARNING: No metrics file found in {results_dir}", file=sys.stderr)
        return None
    
    try:
        # Read metrics file
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # Extract overall metrics
        overall_accuracy = data['overall_metrics']['accuracy']
        overall_f1 = data['overall_metrics']['f1_score']
        
        # Extract confusion matrix
        confusion_matrix = data['confusion_matrix']
        cm = np.array(confusion_matrix)
        
        # Calculate overall precision and recall
        if cm.size > 0:
            # Overall precision (macro average)
            precisions = []
            recalls = []
            for i in range(len(cm)):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
            
            overall_precision = np.mean(precisions)
            overall_recall = np.mean(recalls)
            
            # Target class specific metrics
            if len(cm) > target_class:
                class_tp = cm[target_class, target_class] if cm.shape[0] > target_class and cm.shape[1] > target_class else 0
                class_fp = np.sum(cm[:, target_class]) - class_tp if cm.shape[1] > target_class else 0
                class_fn = np.sum(cm[target_class, :]) - class_tp if cm.shape[0] > target_class else 0
                
                class1_precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0.0
                class1_recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) > 0 else 0.0
                class1_accuracy = class1_recall  # Same as recall for individual class
                
                # Confusion matrix cell values using configured row/col
                confusion_matrix_raw = cm[matrix_row, matrix_col] if cm.shape[0] > matrix_row and cm.shape[1] > matrix_col else 0
                
                # Calculate precision and recall matrices
                cm_precision = cm.astype(float) / np.sum(cm, axis=0)[np.newaxis, :]
                cm_precision = np.nan_to_num(cm_precision)
                
                cm_recall = cm.astype(float) / np.sum(cm, axis=1)[:, np.newaxis]
                cm_recall = np.nan_to_num(cm_recall)
                
                confusion_matrix_precision = cm_precision[matrix_row, matrix_col] if cm_precision.shape[0] > matrix_row and cm_precision.shape[1] > matrix_col else 0.0
                confusion_matrix_recall = cm_recall[matrix_row, matrix_col] if cm_recall.shape[0] > matrix_row and cm_recall.shape[1] > matrix_col else 0.0
            else:
                class1_precision = 0.0
                class1_recall = 0.0
                class1_accuracy = 0.0
                confusion_matrix_raw = 0
                confusion_matrix_precision = 0.0
                confusion_matrix_recall = 0.0
        else:
            overall_precision = 0.0
            overall_recall = 0.0
            class1_precision = 0.0
            class1_recall = 0.0
            class1_accuracy = 0.0
            confusion_matrix_raw = 0
            confusion_matrix_precision = 0.0
            confusion_matrix_recall = 0.0
        
        # Get timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create CSV line with dynamic column names
        csv_line = f'{cost_value},{overall_accuracy:.6f},{overall_precision:.6f},{overall_recall:.6f},{overall_f1:.6f},{class1_accuracy:.6f},{class1_precision:.6f},{class1_recall:.6f},{confusion_matrix_raw},{confusion_matrix_precision:.6f},{confusion_matrix_recall:.6f},{results_dir},{timestamp}'
        
        return csv_line
        
    except Exception as e:
        print(f"ERROR extracting metrics from {metrics_file}: {e}", file=sys.stderr)
        return None

def get_csv_header(sweep_config_file=None):
    """Get the CSV header with dynamic column names based on matrix cell."""
    sweep_config = load_sweep_config(sweep_config_file)
    matrix_row = sweep_config['matrix_cell']['row']
    matrix_col = sweep_config['matrix_cell']['col']
    
    return f'cost_matrix_value,overall_accuracy,overall_precision,overall_recall,overall_f1,class1_accuracy,class1_precision,class1_recall,confusion_{matrix_row}_{matrix_col}_raw,confusion_{matrix_row}_{matrix_col}_precision,confusion_{matrix_row}_{matrix_col}_recall,results_dir,timestamp'

def main():
    parser = argparse.ArgumentParser(description='Extract metrics from training results')
    parser.add_argument('results_dir', help='Path to the results directory')
    parser.add_argument('cost_value', type=float, help='Cost matrix value used for this run')
    parser.add_argument('--output', help='Output CSV file (if not specified, prints to stdout)')
    parser.add_argument('--print-header', action='store_true', help='Print CSV header only')
    parser.add_argument('--sweep-config', help='Path to the sweep configuration file')
    
    args = parser.parse_args()
    
    # If only header is requested, print it and exit
    if args.print_header:
        print(get_csv_header(args.sweep_config))
        return
    
    # Extract metrics
    csv_line = extract_metrics(args.results_dir, args.cost_value, args.sweep_config)
    
    if csv_line is None:
        sys.exit(1)  # Exit with error code
    
    # Output results
    if args.output:
        # Append to file
        with open(args.output, 'a') as f:
            f.write(csv_line + '\n')
        print(f"Metrics appended to {args.output}", file=sys.stderr)
    else:
        # Print to stdout
        print(csv_line)

if __name__ == "__main__":
    main() 