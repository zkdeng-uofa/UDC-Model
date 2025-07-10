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
    ax.set_ylim([0, 1])
    
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
    
    # Plot each class accuracy
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_data)))
    for i, (class_name, values) in enumerate(class_data.items()):
        ax.plot(df['cost_matrix_value'][:len(values)], values, '-o', linewidth=2, markersize=6, 
                label=f'Class {class_name.split("_")[1]}', color=colors[i])
    
    ax.set_title(f'Class Accuracies vs Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
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
    
    # Plot each class F1 score
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_data)))
    for i, (class_name, values) in enumerate(class_data.items()):
        ax.plot(df['cost_matrix_value'][:len(values)], values, '-o', linewidth=2, markersize=6, 
                label=f'Class {class_name.split("_")[1]}', color=colors[i])
    
    ax.set_title(f'Class F1 Scores vs Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
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
    """Create a comprehensive dashboard with all 6 graphs in subplots."""
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Cost Matrix Sweep [{matrix_row},{matrix_col}]: Comprehensive Analysis Dashboard', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Overall Metrics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['cost_matrix_value'], df['overall_accuracy'], 'b-o', linewidth=2, markersize=5, label='Overall Accuracy')
    ax1.plot(df['cost_matrix_value'], df['overall_f1'], 'r-o', linewidth=2, markersize=5, label='Overall F1 Score')
    ax1.set_title('Overall Metrics', fontweight='bold', fontsize=14)
    ax1.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
    ax1.set_ylabel('Metric Value', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 2. Class Accuracies (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    class_acc_data = parse_class_metrics(df, 'class_accuracies')
    if class_acc_data:
        colors = plt.cm.Set1(np.linspace(0, 1, len(class_acc_data)))
        for i, (class_name, values) in enumerate(class_acc_data.items()):
            ax2.plot(df['cost_matrix_value'][:len(values)], values, '-o', linewidth=2, markersize=5, 
                    label=f'Class {class_name.split("_")[1]}', color=colors[i])
        ax2.legend(fontsize=10)
    ax2.set_title('Class Accuracies', fontweight='bold', fontsize=14)
    ax2.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. Class F1 Scores (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    class_f1_data = parse_class_metrics(df, 'class_f1_scores')
    if class_f1_data:
        colors = plt.cm.Set2(np.linspace(0, 1, len(class_f1_data)))
        for i, (class_name, values) in enumerate(class_f1_data.items()):
            ax3.plot(df['cost_matrix_value'][:len(values)], values, '-o', linewidth=2, markersize=5, 
                    label=f'Class {class_name.split("_")[1]}', color=colors[i])
        ax3.legend(fontsize=10)
    ax3.set_title('Class F1 Scores', fontweight='bold', fontsize=14)
    ax3.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
    ax3.set_ylabel('F1 Score', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Sweep Cell Raw Count (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    sweep_col = f'confusion_{matrix_row}_{matrix_col}_raw'
    if sweep_col in df.columns:
        ax4.plot(df['cost_matrix_value'], df[sweep_col], 'purple', marker='o', linewidth=2, markersize=5, 
                label=f'Raw Count Cell [{matrix_row},{matrix_col}]')
        ax4.legend(fontsize=10)
    ax4.set_title(f'Raw Count Sweep Cell [{matrix_row},{matrix_col}]', fontweight='bold', fontsize=14)
    ax4.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=10)
    ax4.set_ylabel('Raw Count', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Count Percentage (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
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
    
    # 6. Other Cells Raw Counts (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
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
    
    # Generate the 6 specific graphs
    print("Creating visualizations...")
    
    try:
        create_overall_metrics_graph(df, graphs_dir, matrix_row, matrix_col)
        create_class_accuracy_graph(df, graphs_dir, matrix_row, matrix_col)
        create_class_f1_graph(df, graphs_dir, matrix_row, matrix_col)
        create_sweep_cell_raw_count_graph(df, graphs_dir, matrix_row, matrix_col)
        create_count_percentage_graph(df, graphs_dir, matrix_row, matrix_col)
        create_other_cells_raw_count_graph(df, graphs_dir, matrix_row, matrix_col)
        
        # Create comprehensive dashboard
        create_comprehensive_dashboard(df, graphs_dir, matrix_row, matrix_col)
        
        if args.final:
            create_summary_statistics(df, args.results_dir, target_class, matrix_row, matrix_col)
            print("Final comprehensive analysis completed.")
        
        print(f"All 6 visualizations + comprehensive dashboard saved to: {graphs_dir}")
        
    except Exception as e:
        print(f"ERROR creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 