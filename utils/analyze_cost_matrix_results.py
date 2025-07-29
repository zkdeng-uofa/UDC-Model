#!/usr/bin/env python3
"""
Cost Matrix Analysis Script

This script analyzes the results from the cost matrix sweep experiment
and creates comprehensive visualizations showing how different metrics
vary with the cost matrix value.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

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
                    print(f"Successfully loaded sweep config from: {config_file}")
                    return config
            except Exception as e:
                print(f"WARNING: Could not load {config_file}: {e}")
                continue
    
    # Return default config if no config file found
    print("WARNING: No sweep config found, using default values")
    return {
        "matrix_cell": {"row": 1, "col": 2},
        "experiment": {"description": "Cost matrix sweep"}
    }

def parse_class_metrics(df, metric_column):
    """Parse pipe-separated class metrics into separate columns."""
    class_data = {}
    
    for idx, row in df.iterrows():
        if pd.notna(row[metric_column]) and row[metric_column].strip():
            values = row[metric_column].split('|')
            for i, value in enumerate(values):
                if f'class_{i}' not in class_data:
                    class_data[f'class_{i}'] = []
                try:
                    class_data[f'class_{i}'].append(float(value))
                except ValueError:
                    class_data[f'class_{i}'].append(0.0)
        else:
            # Handle missing data
            if class_data:  # If we already have some data, pad with zeros
                num_classes = len(class_data)
                for i in range(num_classes):
                    class_data[f'class_{i}'].append(0.0)
    
    return class_data

def parse_other_cells(df, matrix_row, matrix_col, num_classes=3):
    """Parse pipe-separated other cell values into position-labeled data."""
    other_cells_data = {}
    
    for idx, row in df.iterrows():
        if pd.notna(row['other_cells_raw']) and row['other_cells_raw'].strip():
            values = row['other_cells_raw'].split('|')
            cell_idx = 0
            
            for i in range(num_classes):
                for j in range(num_classes):
                    if i != matrix_row or j != matrix_col:  # Skip sweep cell
                        cell_name = f'cell_{i}_{j}'
                        if cell_name not in other_cells_data:
                            other_cells_data[cell_name] = []
                        
                        if cell_idx < len(values):
                            try:
                                other_cells_data[cell_name].append(float(values[cell_idx]))
                            except ValueError:
                                other_cells_data[cell_name].append(0.0)
                        else:
                            other_cells_data[cell_name].append(0.0)
                        
                        cell_idx += 1
        else:
            # Handle missing data
            if other_cells_data:
                for cell_name in other_cells_data:
                    other_cells_data[cell_name].append(0.0)
    
    return other_cells_data

def load_metrics_data(results_dir, matrix_row, matrix_col):
    """Load the metrics CSV file and return a pandas DataFrame."""
    metrics_file = os.path.join(results_dir, "metrics_summary.csv")
    
    if not os.path.exists(metrics_file):
        print(f"WARNING: Metrics file not found: {metrics_file}")
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        print(f"Loaded {len(df)} data points from {metrics_file}")
        return df
    except Exception as e:
        print(f"ERROR loading metrics file: {e}")
        return None

def create_overall_metrics_graph(df, output_dir, matrix_row, matrix_col):
    """Create Overall Metrics graph - overall accuracy and F1 score."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot overall accuracy and F1 score
    ax.plot(df['cost_matrix_value'], df['overall_accuracy'], 'b-o', linewidth=2, markersize=6, label='Overall Accuracy')
    ax.plot(df['cost_matrix_value'], df['overall_f1'], 'r-o', linewidth=2, markersize=6, label='Overall F1 Score')
    
    ax.set_title(f'Overall Metrics vs Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set dynamic y-axis limits based on data range with padding
    all_values = pd.concat([df['overall_accuracy'], df['overall_f1']])
    min_val = all_values.min()
    max_val = all_values.max()
    range_val = max_val - min_val
    padding = max(range_val * 0.1, 0.02)  # 10% padding or minimum 2% padding
    y_min = max(0, min_val - padding)
    y_max = min(1, max_val + padding)
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'overall_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Overall metrics graph saved to: {output_path}")

def create_class_accuracy_graph(df, output_dir, matrix_row, matrix_col):
    """Create Class Accuracy graph - accuracies of all classes."""
    class_data = parse_class_metrics(df, 'class_accuracies')
    
    if not class_data:
        print("WARNING: No class accuracy data found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot each class accuracy and collect all values for dynamic scaling
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_data)))
    all_values = []
    for i, (class_name, values) in enumerate(class_data.items()):
        ax.plot(df['cost_matrix_value'][:len(values)], values, '-o', linewidth=2, markersize=6, 
                label=f'Class {class_name.split("_")[1]}', color=colors[i])
        all_values.extend(values)
    
    ax.set_title(f'Class Accuracies vs Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set dynamic y-axis limits based on data range with padding
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val
        padding = max(range_val * 0.1, 0.02)  # 10% padding or minimum 2% padding
        y_min = max(0, min_val - padding)
        y_max = min(1, max_val + padding)
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'class_accuracies.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Class accuracies graph saved to: {output_path}")

def create_class_f1_graph(df, output_dir, matrix_row, matrix_col):
    """Create Class F1 Score graph - F1 scores of all classes."""
    class_data = parse_class_metrics(df, 'class_f1_scores')
    
    if not class_data:
        print("WARNING: No class F1 data found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot each class F1 score and collect all values for dynamic scaling
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_data)))
    all_values = []
    for i, (class_name, values) in enumerate(class_data.items()):
        ax.plot(df['cost_matrix_value'][:len(values)], values, '-o', linewidth=2, markersize=6, 
                label=f'Class {class_name.split("_")[1]}', color=colors[i])
        all_values.extend(values)
    
    ax.set_title(f'Class F1 Scores vs Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set dynamic y-axis limits based on data range with padding
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val
        padding = max(range_val * 0.1, 0.02)  # 10% padding or minimum 2% padding
        y_min = max(0, min_val - padding)
        y_max = min(1, max_val + padding)
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'class_f1_scores.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Class F1 scores graph saved to: {output_path}")

def create_sweep_cell_raw_count_graph(df, output_dir, matrix_row, matrix_col):
    """Create Raw Count Sweep Cell graph - raw count of the sweep cell."""
    sweep_col = f'confusion_{matrix_row}_{matrix_col}_raw'
    
    if sweep_col not in df.columns:
        print(f"WARNING: Sweep cell column {sweep_col} not found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(df['cost_matrix_value'], df[sweep_col], 'purple', marker='o', linewidth=2, markersize=6, 
            label=f'Raw Count Cell [{matrix_row},{matrix_col}]')
    
    ax.set_title(f'Raw Count in Sweep Cell [{matrix_row},{matrix_col}] vs Cost Matrix Value', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=12)
    ax.set_ylabel('Raw Count', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'sweep_cell_raw_count.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Sweep cell raw count graph saved to: {output_path}")

def create_count_percentage_graph(df, output_dir, matrix_row, matrix_col):
    """Create Count Percentage graph - percentage of sweep cell / row total."""
    percentage_col = f'confusion_{matrix_row}_{matrix_col}_percentage'
    
    if percentage_col not in df.columns:
        print(f"WARNING: Percentage column {percentage_col} not found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(df['cost_matrix_value'], df[percentage_col], 'orange', marker='o', linewidth=2, markersize=6,
            label=f'Percentage of Cell [{matrix_row},{matrix_col}] / Row {matrix_row} Total')
    
    ax.set_title(f'Count Percentage in Sweep Cell [{matrix_row},{matrix_col}] vs Cost Matrix Value', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set dynamic y-axis limits based on data range with padding
    min_val = df[percentage_col].min()
    max_val = df[percentage_col].max()
    range_val = max_val - min_val
    padding = max(range_val * 0.1, 2.0)  # 10% padding or minimum 2% padding
    y_min = max(0, min_val - padding)
    y_max = min(100, max_val + padding)
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'count_percentage.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Count percentage graph saved to: {output_path}")

def create_other_cells_raw_count_graph(df, output_dir, matrix_row, matrix_col, num_classes=3):
    """Create Raw Count Other Cells graph - raw counts of all other cells."""
    other_cells_data = parse_other_cells(df, matrix_row, matrix_col, num_classes)
    
    if not other_cells_data:
        print("WARNING: No other cells data found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot each other cell
    colors = plt.cm.tab10(np.linspace(0, 1, len(other_cells_data)))
    for i, (cell_name, values) in enumerate(other_cells_data.items()):
        row, col = cell_name.split('_')[1], cell_name.split('_')[2]
        ax.plot(df['cost_matrix_value'][:len(values)], values, '-o', linewidth=2, markersize=4,
                label=f'Cell [{row},{col}]', color=colors[i])
    
    ax.set_title(f'Raw Counts in Other Confusion Matrix Cells vs Cost Matrix Value [{matrix_row},{matrix_col}]', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=12)
    ax.set_ylabel('Raw Count', fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'other_cells_raw_counts.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Other cells raw counts graph saved to: {output_path}")

def create_comprehensive_dashboard(df, output_dir, matrix_row, matrix_col, num_classes=3):
    """Create a comprehensive dashboard with all 7 graphs in subplots."""
    fig = plt.figure(figsize=(24, 30))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Cost Matrix Sweep [{matrix_row},{matrix_col}]: Comprehensive Analysis Dashboard', 
                fontsize=24, fontweight='bold', y=0.98)
    
    # 1. Overall Metrics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['cost_matrix_value'], df['overall_accuracy'], 'b-o', linewidth=2, markersize=5, label='Overall Accuracy')
    ax1.plot(df['cost_matrix_value'], df['overall_f1'], 'r-o', linewidth=2, markersize=5, label='Overall F1 Score')
    ax1.set_title('Overall Metrics', fontweight='bold', fontsize=14)
    ax1.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
    ax1.set_ylabel('Metric Value', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Set dynamic y-axis limits for overall metrics
    all_values = pd.concat([df['overall_accuracy'], df['overall_f1']])
    min_val = all_values.min()
    max_val = all_values.max()
    range_val = max_val - min_val
    padding = max(range_val * 0.1, 0.02)  # 10% padding or minimum 2% padding
    y_min = max(0, min_val - padding)
    y_max = min(1, max_val + padding)
    ax1.set_ylim([y_min, y_max])
    
    # 2. Class Accuracies (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    class_acc_data = parse_class_metrics(df, 'class_accuracies')
    all_acc_values = []
    if class_acc_data:
        colors = plt.cm.Set1(np.linspace(0, 1, len(class_acc_data)))
        for i, (class_name, values) in enumerate(class_acc_data.items()):
            ax2.plot(df['cost_matrix_value'][:len(values)], values, '-o', linewidth=2, markersize=5, 
                    label=f'Class {class_name.split("_")[1]}', color=colors[i])
            all_acc_values.extend(values)
        ax2.legend(fontsize=10)
    ax2.set_title('Class Accuracies', fontweight='bold', fontsize=14)
    ax2.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Set dynamic y-axis limits for class accuracies
    if all_acc_values:
        min_val = min(all_acc_values)
        max_val = max(all_acc_values)
        range_val = max_val - min_val
        padding = max(range_val * 0.1, 0.02)  # 10% padding or minimum 2% padding
        y_min = max(0, min_val - padding)
        y_max = min(1, max_val + padding)
        ax2.set_ylim([y_min, y_max])
    else:
        ax2.set_ylim([0, 1])
    
    # 3. Class F1 Scores (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    class_f1_data = parse_class_metrics(df, 'class_f1_scores')
    all_f1_values = []
    if class_f1_data:
        colors = plt.cm.Set2(np.linspace(0, 1, len(class_f1_data)))
        for i, (class_name, values) in enumerate(class_f1_data.items()):
            ax3.plot(df['cost_matrix_value'][:len(values)], values, '-o', linewidth=2, markersize=5, 
                    label=f'Class {class_name.split("_")[1]}', color=colors[i])
            all_f1_values.extend(values)
        ax3.legend(fontsize=10)
    ax3.set_title('Class F1 Scores', fontweight='bold', fontsize=14)
    ax3.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
    ax3.set_ylabel('F1 Score', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Set dynamic y-axis limits for class F1 scores
    if all_f1_values:
        min_val = min(all_f1_values)
        max_val = max(all_f1_values)
        range_val = max_val - min_val
        padding = max(range_val * 0.1, 0.02)  # 10% padding or minimum 2% padding
        y_min = max(0, min_val - padding)
        y_max = min(1, max_val + padding)
        ax3.set_ylim([y_min, y_max])
    else:
        ax3.set_ylim([0, 1])
    
    # 4. Sweep Cell Raw Count (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    sweep_col = f'confusion_{matrix_row}_{matrix_col}_raw'
    if sweep_col in df.columns:
        ax4.plot(df['cost_matrix_value'], df[sweep_col], 'purple', marker='o', linewidth=2, markersize=5, 
                label=f'Raw Count Cell [{matrix_row},{matrix_col}]')
        ax4.legend(fontsize=10)
    ax4.set_title(f'Raw Count Sweep Cell [{matrix_row},{matrix_col}]', fontweight='bold', fontsize=14)
    ax4.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
    ax4.set_ylabel('Raw Count', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Count Percentage (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    percentage_col = f'confusion_{matrix_row}_{matrix_col}_percentage'
    if percentage_col in df.columns:
        ax5.plot(df['cost_matrix_value'], df[percentage_col], 'orange', marker='o', linewidth=2, markersize=5,
                label=f'Percentage of Cell [{matrix_row},{matrix_col}] / Row {matrix_row} Total')
        ax5.legend(fontsize=10)
        
        # Set dynamic y-axis limits based on data range with padding
        min_val = df[percentage_col].min()
        max_val = df[percentage_col].max()
        range_val = max_val - min_val
        padding = max(range_val * 0.1, 2.0)  # 10% padding or minimum 2% padding
        y_min = max(0, min_val - padding)
        y_max = min(100, max_val + padding)
        ax5.set_ylim([y_min, y_max])
    ax5.set_title(f'Count Percentage Cell [{matrix_row},{matrix_col}]', fontweight='bold', fontsize=14)
    ax5.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
    ax5.set_ylabel('Percentage (%)', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Other Cells Raw Counts (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    other_cells_data = parse_other_cells(df, matrix_row, matrix_col, num_classes)
    if other_cells_data:
        colors = plt.cm.tab10(np.linspace(0, 1, len(other_cells_data)))
        for i, (cell_name, values) in enumerate(other_cells_data.items()):
            row, col = cell_name.split('_')[1], cell_name.split('_')[2]
            ax6.plot(df['cost_matrix_value'][:len(values)], values, '-o', linewidth=1.5, markersize=4,
                    label=f'[{row},{col}]', color=colors[i])
        ax6.legend(fontsize=8, ncol=2)
    ax6.set_title('Raw Counts Other Cells', fontweight='bold', fontsize=14)
    ax6.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
    ax6.set_ylabel('Raw Count', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. FPR/FNR Scatter Plot (bottom center, spanning two columns)
    ax7 = fig.add_subplot(gs[2, 0:2])
    
    # Calculate FPR and FNR values
    fpr_values, fnr_values = calculate_fpr_fnr_for_swept_class(df, matrix_row, matrix_col, num_classes)
    
    if fpr_values and fnr_values:
        # Create scatter plot with color gradient based on cost values
        cost_values = df['cost_matrix_value'].values
        
        # Normalize cost values for color mapping (0 to 1)
        cost_min, cost_max = cost_values.min(), cost_values.max()
        normalized_costs = (cost_values - cost_min) / (cost_max - cost_min) if cost_max > cost_min else np.zeros_like(cost_values)
        
        # Create scatter plot with blue color gradient
        scatter = ax7.scatter(fnr_values, fpr_values, c=normalized_costs, cmap='Blues', 
                            s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax7)
        cbar.set_label(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
        
        # Set dynamic axis limits
        if fnr_values:
            x_min, x_max = min(fnr_values), max(fnr_values)
            x_range = x_max - x_min
            x_padding = max(x_range * 0.1, 2.0)
            ax7.set_xlim([max(0, x_min - x_padding), min(100, x_max + x_padding)])
        
        if fpr_values:
            y_min, y_max = min(fpr_values), max(fpr_values)
            y_range = y_max - y_min
            y_padding = max(y_range * 0.1, 2.0)
            ax7.set_ylim([max(0, y_min - y_padding), min(100, y_max + y_padding)])
    
    ax7.set_title(f'ROC-like Analysis: Class {matrix_row}', fontweight='bold', fontsize=14)
    ax7.set_xlabel(f'False Negative Rate (%) - Class {matrix_row}', fontsize=10)
    ax7.set_ylabel(f'1 - False Positive Rate (%) - Class {matrix_row}', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # 8. Empty space (bottom right) - can be used for additional analysis or kept empty
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.text(0.5, 0.5, 'Additional Analysis\nSpace Reserved', 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=12, style='italic', alpha=0.5)
    ax8.set_xlim([0, 1])
    ax8.set_ylim([0, 1])
    ax8.axis('off')
    
    plt.tight_layout()
    
    # Save the dashboard
    output_path = os.path.join(output_dir, f'comprehensive_dashboard_{matrix_row}_{matrix_col}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Comprehensive dashboard saved to: {output_path}")

def create_summary_statistics(df, output_dir, target_class, matrix_row, matrix_col):
    """Create and save summary statistics."""
    # Calculate summary statistics
    summary_stats = df.describe()
    
    # Find optimal values
    best_overall_accuracy = df.loc[df['overall_accuracy'].idxmax()]
    best_overall_f1 = df.loc[df['overall_f1'].idxmax()]
    
    # Calculate step size from data
    if len(df) > 1:
        step_size = df['cost_matrix_value'].iloc[1] - df['cost_matrix_value'].iloc[0]
    else:
        step_size = "N/A"
    
    # Create summary report
    summary_file = os.path.join(output_dir, f'summary_report_{matrix_row}_{matrix_col}.txt')
    with open(summary_file, 'w') as f:
        f.write("COST MATRIX SWEEP EXPERIMENT - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write(f"Cost Matrix Cell [{matrix_row},{matrix_col}] varied from {df['cost_matrix_value'].min()} to {df['cost_matrix_value'].max()}\n")
        f.write(f"Target Class: {target_class}\n")
        f.write(f"Total iterations: {len(df)}\n")
        f.write(f"Step size: {step_size}\n\n")
        
        f.write("OPTIMAL CONFIGURATIONS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Overall Accuracy: {best_overall_accuracy['overall_accuracy']:.4f} at cost value {best_overall_accuracy['cost_matrix_value']}\n")
        f.write(f"Best Overall F1 Score: {best_overall_f1['overall_f1']:.4f} at cost value {best_overall_f1['cost_matrix_value']}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(summary_stats.to_string())
        f.write("\n\n")
        
        f.write("CORRELATION ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        # Calculate correlations with cost matrix value (only numeric columns)
        numeric_df = df.select_dtypes(include=[np.number])
        if 'cost_matrix_value' in numeric_df.columns and len(numeric_df.columns) > 1:
            correlations = numeric_df.corr()['cost_matrix_value'].sort_values(key=abs, ascending=False)
            f.write("Correlations with cost matrix value:\n")
            for metric, correlation in correlations.items():
                if metric != 'cost_matrix_value' and not np.isnan(correlation):
                    f.write(f"  {metric}: {correlation:.4f}\n")
        else:
            f.write("No numeric data available for correlation analysis.\n")
    
    print(f"Summary report saved to: {summary_file}")

def calculate_fpr_fnr_for_swept_class(df, matrix_row, matrix_col, num_classes=3):
    """
    Calculate False Positive Rate and False Negative Rate for the swept class.
    
    For the swept class (matrix_row):
    - FPR = FP / (FP + TN)
    - FNR = FN / (FN + TP)
    - Y-axis: 1 - FPR (specificity as percentage)
    - X-axis: FNR (as percentage)
    
    Returns:
        fpr_values: List of (1 - FPR) * 100 values (for y-axis)
        fnr_values: List of FNR * 100 values (for x-axis)
    """
    fpr_values = []
    fnr_values = []
    
    # Get other cells data
    other_cells_data = parse_other_cells(df, matrix_row, matrix_col, num_classes)
    
    for idx, row in df.iterrows():
        # Build confusion matrix for this row
        confusion = {}
        
        # Add the swept cell
        swept_cell_col = f'confusion_{matrix_row}_{matrix_col}_raw'
        if swept_cell_col in df.columns:
            confusion[(matrix_row, matrix_col)] = row[swept_cell_col]
        
        # Add other cells from other_cells_data
        for cell_name, values in other_cells_data.items():
            if idx < len(values):
                parts = cell_name.split('_')
                if len(parts) >= 3:
                    r, c = int(parts[1]), int(parts[2])
                    confusion[(r, c)] = values[idx]
        
        # Calculate metrics for the swept class (matrix_row)
        tp = confusion.get((matrix_row, matrix_row), 0)
        
        # FP: other classes predicted as swept class
        fp = sum(confusion.get((i, matrix_row), 0) for i in range(num_classes) if i != matrix_row)
        
        # FN: swept class predicted as other classes
        fn = sum(confusion.get((matrix_row, j), 0) for j in range(num_classes) if j != matrix_row)
        
        # TN: other classes correctly not predicted as swept class
        tn = sum(confusion.get((i, j), 0) for i in range(num_classes) for j in range(num_classes) 
                if i != matrix_row and j != matrix_row)
        
        # Calculate rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Y-axis: 1 - FPR (specificity as percentage)
        # X-axis: FNR (as percentage)
        fpr_values.append((1 - fpr) * 100)
        fnr_values.append(fnr * 100)
    
    return fpr_values, fnr_values

def create_fpr_fnr_scatter_plot(df, output_dir, matrix_row, matrix_col, num_classes=3):
    """Create FPR vs FNR scatter plot for the swept class."""
    
    # Calculate FPR and FNR values
    fpr_values, fnr_values = calculate_fpr_fnr_for_swept_class(df, matrix_row, matrix_col, num_classes)
    
    if not fpr_values or not fnr_values:
        print("WARNING: No FPR/FNR data available for scatter plot")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create scatter plot with color gradient based on cost values
    cost_values = df['cost_matrix_value'].values
    
    # Normalize cost values for color mapping (0 to 1)
    cost_min, cost_max = cost_values.min(), cost_values.max()
    normalized_costs = (cost_values - cost_min) / (cost_max - cost_min) if cost_max > cost_min else np.zeros_like(cost_values)
    
    # Create scatter plot with blue color gradient
    scatter = ax.scatter(fnr_values, fpr_values, c=normalized_costs, cmap='Blues', 
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=12)
    
    # Set titles and labels
    ax.set_title(f'ROC-like Analysis: Class {matrix_row} vs Cost Matrix Value [{matrix_row},{matrix_col}]', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel(f'False Negative Rate (%) - Class {matrix_row}', fontsize=12)
    ax.set_ylabel(f'1 - False Positive Rate (%) - Class {matrix_row}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set dynamic axis limits
    if fnr_values:
        x_min, x_max = min(fnr_values), max(fnr_values)
        x_range = x_max - x_min
        x_padding = max(x_range * 0.1, 2.0)
        ax.set_xlim([max(0, x_min - x_padding), min(100, x_max + x_padding)])
    
    if fpr_values:
        y_min, y_max = min(fpr_values), max(fpr_values)
        y_range = y_max - y_min
        y_padding = max(y_range * 0.1, 2.0)
        ax.set_ylim([max(0, y_min - y_padding), min(100, y_max + y_padding)])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'fpr_fnr_scatter_{matrix_row}_{matrix_col}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"FPR/FNR scatter plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze cost matrix sweep results')
    parser.add_argument('results_dir', help='Directory containing the results')
    parser.add_argument('--final', action='store_true', help='Generate final comprehensive report')
    parser.add_argument('--sweep-config', help='Path to the sweep configuration file')
    
    args = parser.parse_args()
    
    # Load sweep configuration
    sweep_config = load_sweep_config(args.sweep_config)
    matrix_row = sweep_config['matrix_cell']['row']
    matrix_col = sweep_config['matrix_cell']['col']
    target_class = matrix_col  # Target class is determined by the column being modified
    
    print(f"Configuration loaded: Matrix cell [{matrix_row},{matrix_col}], Target class: {target_class}")
    
    # Create graphs directory
    graphs_dir = os.path.join(args.results_dir, 'graphs')
    Path(graphs_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_metrics_data(args.results_dir, matrix_row, matrix_col)
    if df is None or len(df) == 0:
        print("No data available for analysis.")
        return
    
    print(f"Analyzing {len(df)} experiments...")
    
    # Generate the 7 specific graphs
    print("Creating visualizations...")
    
    try:
        create_overall_metrics_graph(df, graphs_dir, matrix_row, matrix_col)
        create_class_accuracy_graph(df, graphs_dir, matrix_row, matrix_col)
        create_class_f1_graph(df, graphs_dir, matrix_row, matrix_col)
        create_sweep_cell_raw_count_graph(df, graphs_dir, matrix_row, matrix_col)
        create_count_percentage_graph(df, graphs_dir, matrix_row, matrix_col)
        create_other_cells_raw_count_graph(df, graphs_dir, matrix_row, matrix_col)
        create_fpr_fnr_scatter_plot(df, graphs_dir, matrix_row, matrix_col)
        
        # Create comprehensive dashboard
        create_comprehensive_dashboard(df, graphs_dir, matrix_row, matrix_col)
        
        if args.final:
            create_summary_statistics(df, args.results_dir, target_class, matrix_row, matrix_col)
            print("Final comprehensive analysis completed.")
        
        print(f"All 7 visualizations + comprehensive dashboard saved to: {graphs_dir}")
        
    except Exception as e:
        print(f"ERROR creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 