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

def get_dynamic_column_names(matrix_row, matrix_col):
    """Get the dynamic column names based on matrix cell."""
    return {
        'confusion_raw': f'confusion_{matrix_row}_{matrix_col}_raw',
        'confusion_precision': f'confusion_{matrix_row}_{matrix_col}_precision',
        'confusion_recall': f'confusion_{matrix_row}_{matrix_col}_recall'
    }

def load_metrics_data(results_dir, matrix_row, matrix_col):
    """Load the metrics CSV file and return a pandas DataFrame."""
    metrics_file = os.path.join(results_dir, "metrics_summary.csv")
    
    if not os.path.exists(metrics_file):
        print(f"WARNING: Metrics file not found: {metrics_file}")
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        print(f"Loaded {len(df)} data points from {metrics_file}")
        
        # Get dynamic column names
        col_names = get_dynamic_column_names(matrix_row, matrix_col)
        
        # Check if the expected columns exist
        expected_columns = [col_names['confusion_raw'], col_names['confusion_precision'], col_names['confusion_recall']]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"WARNING: Missing columns in CSV: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            # Add missing columns with default values
            for col in missing_columns:
                df[col] = 0
        
        return df
    except Exception as e:
        print(f"ERROR loading metrics file: {e}")
        return None

def create_overall_metrics_plot(df, output_dir, matrix_row, matrix_col):
    """Create plots for overall performance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Overall Performance Metrics vs Cost Matrix Value', fontsize=16, fontweight='bold')
    
    # Overall Accuracy
    axes[0,0].plot(df['cost_matrix_value'], df['overall_accuracy'], 'b-o', linewidth=2, markersize=6)
    axes[0,0].set_title('Overall Accuracy', fontweight='bold')
    axes[0,0].set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim([0, 1])
    
    # Overall Precision
    axes[0,1].plot(df['cost_matrix_value'], df['overall_precision'], 'g-o', linewidth=2, markersize=6)
    axes[0,1].set_title('Overall Precision', fontweight='bold')
    axes[0,1].set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    axes[0,1].set_ylabel('Precision')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim([0, 1])
    
    # Overall Recall
    axes[1,0].plot(df['cost_matrix_value'], df['overall_recall'], 'r-o', linewidth=2, markersize=6)
    axes[1,0].set_title('Overall Recall', fontweight='bold')
    axes[1,0].set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    axes[1,0].set_ylabel('Recall')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim([0, 1])
    
    # Overall F1 Score
    axes[1,1].plot(df['cost_matrix_value'], df['overall_f1'], 'm-o', linewidth=2, markersize=6)
    axes[1,1].set_title('Overall F1 Score', fontweight='bold')
    axes[1,1].set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    axes[1,1].set_ylabel('F1 Score')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'overall_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Overall metrics plot saved to: {output_path}")

def create_target_class_metrics_plot(df, output_dir, target_class, matrix_row, matrix_col):
    """Create plots for target class specific metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Class {target_class} Performance Metrics vs Cost Matrix Value [{matrix_row},{matrix_col}]', fontsize=16, fontweight='bold')
    
    # Target Class Accuracy
    axes[0].plot(df['cost_matrix_value'], df['class1_accuracy'], 'b-o', linewidth=2, markersize=6)
    axes[0].set_title(f'Class {target_class} Accuracy', fontweight='bold')
    axes[0].set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Target Class Precision
    axes[1].plot(df['cost_matrix_value'], df['class1_precision'], 'g-o', linewidth=2, markersize=6)
    axes[1].set_title(f'Class {target_class} Precision', fontweight='bold')
    axes[1].set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    axes[1].set_ylabel('Precision')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Target Class Recall
    axes[2].plot(df['cost_matrix_value'], df['class1_recall'], 'r-o', linewidth=2, markersize=6)
    axes[2].set_title(f'Class {target_class} Recall', fontweight='bold')
    axes[2].set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    axes[2].set_ylabel('Recall')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'class{target_class}_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Class {target_class} metrics plot saved to: {output_path}")

def create_confusion_cell_plot(df, output_dir, matrix_row, matrix_col):
    """Create plots for confusion matrix cell metrics."""
    col_names = get_dynamic_column_names(matrix_row, matrix_col)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Confusion Matrix Cell [{matrix_row},{matrix_col}] Metrics vs Cost Matrix Value', fontsize=16, fontweight='bold')
    
    # Raw Count
    axes[0].plot(df['cost_matrix_value'], df[col_names['confusion_raw']], 'orange', marker='o', linewidth=2, markersize=6)
    axes[0].set_title(f'Raw Count in Cell [{matrix_row},{matrix_col}]', fontweight='bold')
    axes[0].set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    axes[0].set_ylabel('Raw Count')
    axes[0].grid(True, alpha=0.3)
    
    # Precision Value
    axes[1].plot(df['cost_matrix_value'], df[col_names['confusion_precision']], 'purple', marker='o', linewidth=2, markersize=6)
    axes[1].set_title(f'Precision Value in Cell [{matrix_row},{matrix_col}]', fontweight='bold')
    axes[1].set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    axes[1].set_ylabel('Precision Value')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Recall Value
    axes[2].plot(df['cost_matrix_value'], df[col_names['confusion_recall']], 'brown', marker='o', linewidth=2, markersize=6)
    axes[2].set_title(f'Recall Value in Cell [{matrix_row},{matrix_col}]', fontweight='bold')
    axes[2].set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    axes[2].set_ylabel('Recall Value')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'confusion_cell_{matrix_row}_{matrix_col}_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Confusion cell [{matrix_row},{matrix_col}] metrics plot saved to: {output_path}")

def create_comprehensive_dashboard(df, output_dir, target_class, matrix_row, matrix_col):
    """Create a comprehensive dashboard with all metrics."""
    col_names = get_dynamic_column_names(matrix_row, matrix_col)
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Cost Matrix Sweep [{matrix_row},{matrix_col}]: Comprehensive Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    
    # Row 1: Overall Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['cost_matrix_value'], df['overall_accuracy'], 'b-o', linewidth=2, markersize=4)
    ax1.set_title('Overall Accuracy', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['cost_matrix_value'], df['overall_precision'], 'g-o', linewidth=2, markersize=4)
    ax2.set_title('Overall Precision', fontweight='bold')
    ax2.set_ylabel('Precision')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df['cost_matrix_value'], df['overall_recall'], 'r-o', linewidth=2, markersize=4)
    ax3.set_title('Overall Recall', fontweight='bold')
    ax3.set_ylabel('Recall')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Row 2: F1 and Class 1 Accuracy
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(df['cost_matrix_value'], df['overall_f1'], 'm-o', linewidth=2, markersize=4)
    ax4.set_title('Overall F1 Score', fontweight='bold')
    ax4.set_ylabel('F1 Score')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(df['cost_matrix_value'], df['class1_accuracy'], 'cyan', marker='o', linewidth=2, markersize=4)
    ax5.set_title(f'Class {target_class} Accuracy', fontweight='bold')
    ax5.set_ylabel(f'Class {target_class} Accuracy')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(df['cost_matrix_value'], df['class1_precision'], 'orange', marker='o', linewidth=2, markersize=4)
    ax6.set_title(f'Class {target_class} Precision', fontweight='bold')
    ax6.set_ylabel(f'Class {target_class} Precision')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    # Row 3: Target Class Recall and Confusion Raw
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(df['cost_matrix_value'], df['class1_recall'], 'purple', marker='o', linewidth=2, markersize=4)
    ax7.set_title(f'Class {target_class} Recall', fontweight='bold')
    ax7.set_ylabel(f'Class {target_class} Recall')
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 1])
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(df['cost_matrix_value'], df[col_names['confusion_raw']], 'brown', marker='o', linewidth=2, markersize=4)
    ax8.set_title(f'Raw Count Cell [{matrix_row},{matrix_col}]', fontweight='bold')
    ax8.set_ylabel('Raw Count')
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(df['cost_matrix_value'], df[col_names['confusion_precision']], 'pink', marker='o', linewidth=2, markersize=4)
    ax9.set_title(f'Precision Cell [{matrix_row},{matrix_col}]', fontweight='bold')
    ax9.set_ylabel('Precision Value')
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim([0, 1])
    
    # Row 4: Recall Cell and Combined Analysis
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(df['cost_matrix_value'], df[col_names['confusion_recall']], 'navy', marker='o', linewidth=2, markersize=4)
    ax10.set_title(f'Recall Cell [{matrix_row},{matrix_col}]', fontweight='bold')
    ax10.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    ax10.set_ylabel('Recall Value')
    ax10.grid(True, alpha=0.3)
    ax10.set_ylim([0, 1])
    
    # Combined metrics comparison
    ax11 = fig.add_subplot(gs[3, 1:])
    ax11.plot(df['cost_matrix_value'], df['overall_accuracy'], 'b-', label='Overall Accuracy', linewidth=2)
    ax11.plot(df['cost_matrix_value'], df['overall_precision'], 'g-', label='Overall Precision', linewidth=2)
    ax11.plot(df['cost_matrix_value'], df['overall_recall'], 'r-', label='Overall Recall', linewidth=2)
    ax11.plot(df['cost_matrix_value'], df['overall_f1'], 'm-', label='Overall F1', linewidth=2)
    ax11.set_title('Combined Overall Metrics Comparison', fontweight='bold')
    ax11.set_xlabel(f'Cost Matrix Value [{matrix_row},{matrix_col}]')
    ax11.set_ylabel('Metric Value')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    ax11.set_ylim([0, 1])
    
    # Save the dashboard
    output_path = os.path.join(output_dir, f'comprehensive_dashboard_{matrix_row}_{matrix_col}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Comprehensive dashboard saved to: {output_path}")

def create_summary_statistics(df, output_dir, target_class, matrix_row, matrix_col):
    """Create and save summary statistics."""
    col_names = get_dynamic_column_names(matrix_row, matrix_col)
    
    # Calculate summary statistics
    summary_stats = df.describe()
    
    # Find optimal values
    best_overall_accuracy = df.loc[df['overall_accuracy'].idxmax()]
    best_overall_f1 = df.loc[df['overall_f1'].idxmax()]
    best_class1_accuracy = df.loc[df['class1_accuracy'].idxmax()]
    
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
        f.write(f"Best Overall F1 Score: {best_overall_f1['overall_f1']:.4f} at cost value {best_overall_f1['cost_matrix_value']}\n")
        f.write(f"Best Class {target_class} Accuracy: {best_class1_accuracy['class1_accuracy']:.4f} at cost value {best_class1_accuracy['cost_matrix_value']}\n\n")
        
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
    
    # Generate plots
    print("Creating visualizations...")
    
    try:
        create_overall_metrics_plot(df, graphs_dir, matrix_row, matrix_col)
        create_target_class_metrics_plot(df, graphs_dir, target_class, matrix_row, matrix_col)
        create_confusion_cell_plot(df, graphs_dir, matrix_row, matrix_col)
        create_comprehensive_dashboard(df, graphs_dir, target_class, matrix_row, matrix_col)
        
        if args.final:
            create_summary_statistics(df, args.results_dir, target_class, matrix_row, matrix_col)
            print("Final comprehensive analysis completed.")
        
        print(f"All visualizations saved to: {graphs_dir}")
        
    except Exception as e:
        print(f"ERROR creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 