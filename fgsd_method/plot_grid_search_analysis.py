"""
Grid Search Analysis Visualization for FGSD Experiments

This script creates plots analyzing the grid search results:
1. Accuracy vs Binwidth (bins = range / binwidth)
2. Accuracy vs Number of Bins
3. F1 Score vs Binwidth
4. Generation Time vs Binwidth
5. Memory Usage vs Binwidth
6. Accuracy-Time Trade-off (Pareto analysis)
7. Heatmap of Accuracy by Function and Binwidth
8. Combined summary plot

Grid Search Methodology:
- binwidth (h) values: [0.05, 0.1, 0.2, 0.5, 1.0]
- bins = range / binwidth (minimum 10 bins)
- Classifier: Random Forest (n_estimators=500, max_depth=20)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# File paths for grid search results
FILES = {
    'ENZYMES': {
        'preprocessed': 'src/results/fgsd_enzymes_grid_search.csv',
        'raw': 'src/results/raw_embeddings/fgsd_enzymes_raw_grid_search.csv'
    },
    'IMDB-MULTI': {
        'preprocessed': 'src/results/fgsd_imdb_grid_search.csv',
        'raw': 'src/results/raw_embeddings/fgsd_imdb_raw_grid_search.csv'
    },
    'REDDIT-MULTI-12K': {
        'preprocessed': 'src/results/fgsd_reddit_grid_search.csv',
        'raw': 'src/results/raw_embeddings/fgsd_reddit_raw_grid_search.csv'
    },
}

# Output directories
OUTPUT_DIR = 'plots/grid_search_analysis'
OUTPUT_DIR_RAW = 'plots/grid_search_analysis/raw_embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_RAW, exist_ok=True)

# Plot settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Color palettes
FUNC_COLORS = {
    'harmonic': '#2ecc71', 
    'polynomial': '#3498db', 
    'biharmonic': '#9b59b6'
}

def get_func_color(func_name):
    """Get color for function type."""
    return FUNC_COLORS.get(func_name, '#333333')

def load_data(path):
    """Load grid search data if exists."""
    if not os.path.exists(path):
        print(f"  ⚠️  File not found: {path}")
        return None
    
    df = pd.read_csv(path)
    
    # Ensure required columns exist
    required = ['func', 'binwidth', 'bins', 'accuracy']
    if not all(col in df.columns for col in required):
        print(f"  ⚠️  Missing required columns in {path}")
        return None
    
    return df


def get_output_dir(raw_mode=False):
    """Get output directory based on mode."""
    return OUTPUT_DIR_RAW if raw_mode else OUTPUT_DIR


def plot_accuracy_vs_binwidth(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 1: Accuracy vs Binwidth for each function type.
    Shows how accuracy changes with histogram resolution.
    """
    if df is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for func in df['func'].unique():
        func_df = df[df['func'] == func].sort_values('binwidth')
        color = get_func_color(func)
        
        ax.plot(func_df['binwidth'], func_df['accuracy'],
               marker='o', linewidth=2.5, markersize=10, 
               label=func.capitalize(), color=color)
        
        # Annotate with bins
        for _, row in func_df.iterrows():
            ax.annotate(f'b={int(row["bins"])}',
                       (row['binwidth'], row['accuracy']),
                       textcoords="offset points", xytext=(0, 8), 
                       fontsize=8, ha='center')
    
    mode_suffix = " (Raw Embeddings)" if raw_mode else ""
    ax.set_xlabel('Binwidth (h)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{dataset_name}: Accuracy vs Binwidth{mode_suffix}\n(bins = range / binwidth)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Smaller binwidth = more bins = right side
    
    # Add annotation explaining axis
    ax.text(0.02, 0.02, '← More bins (finer resolution)', 
            transform=ax.transAxes, fontsize=9, style='italic', alpha=0.7)
    
    filename = f'{dataset_name.replace("-", "_").lower()}_acc_vs_binwidth{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def plot_accuracy_vs_bins(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 2: Accuracy vs Number of Bins.
    Direct relationship between embedding dimension and accuracy.
    """
    if df is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for func in df['func'].unique():
        func_df = df[df['func'] == func].sort_values('bins')
        color = get_func_color(func)
        
        ax.plot(func_df['bins'], func_df['accuracy'],
               marker='s', linewidth=2.5, markersize=10,
               label=func.capitalize(), color=color)
        
        # Annotate with binwidth
        for _, row in func_df.iterrows():
            ax.annotate(f'h={row["binwidth"]}',
                       (row['bins'], row['accuracy']),
                       textcoords="offset points", xytext=(0, 8),
                       fontsize=8, ha='center')
    
    mode_suffix = " (Raw Embeddings)" if raw_mode else ""
    ax.set_xlabel('Number of Bins (Embedding Dimension)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{dataset_name}: Accuracy vs Embedding Dimension{mode_suffix}',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    filename = f'{dataset_name.replace("-", "_").lower()}_acc_vs_bins{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def plot_f1_vs_binwidth(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 3: F1 Score vs Binwidth.
    """
    if df is None or 'f1_score' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for func in df['func'].unique():
        func_df = df[df['func'] == func].sort_values('binwidth')
        color = get_func_color(func)
        
        ax.plot(func_df['binwidth'], func_df['f1_score'],
               marker='^', linewidth=2.5, markersize=10,
               label=func.capitalize(), color=color)
    
    mode_suffix = " (Raw Embeddings)" if raw_mode else ""
    ax.set_xlabel('Binwidth (h)', fontsize=12)
    ax.set_ylabel('F1 Score (weighted)', fontsize=12)
    ax.set_title(f'{dataset_name}: F1 Score vs Binwidth{mode_suffix}',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    filename = f'{dataset_name.replace("-", "_").lower()}_f1_vs_binwidth{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def plot_time_vs_binwidth(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 4: Generation Time vs Binwidth.
    Shows computational cost at different resolutions.
    """
    if df is None or 'generation_time' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for func in df['func'].unique():
        func_df = df[df['func'] == func].sort_values('binwidth')
        color = get_func_color(func)
        
        ax.plot(func_df['binwidth'], func_df['generation_time'],
               marker='o', linewidth=2.5, markersize=10,
               label=func.capitalize(), color=color)
    
    mode_suffix = " (Raw Embeddings)" if raw_mode else ""
    ax.set_xlabel('Binwidth (h)', fontsize=12)
    ax.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax.set_title(f'{dataset_name}: Computational Cost vs Binwidth{mode_suffix}\n(Smaller binwidth = More bins = Higher cost)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    filename = f'{dataset_name.replace("-", "_").lower()}_time_vs_binwidth{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def plot_memory_vs_binwidth(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 5: Memory Usage vs Binwidth.
    """
    if df is None or 'memory_mb' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for func in df['func'].unique():
        func_df = df[df['func'] == func].sort_values('binwidth')
        color = get_func_color(func)
        
        ax.plot(func_df['binwidth'], func_df['memory_mb'],
               marker='s', linewidth=2.5, markersize=10,
               label=func.capitalize(), color=color)
    
    mode_suffix = " (Raw Embeddings)" if raw_mode else ""
    ax.set_xlabel('Binwidth (h)', fontsize=12)
    ax.set_ylabel('Peak Memory Usage (MB)', fontsize=12)
    ax.set_title(f'{dataset_name}: Memory Usage vs Binwidth{mode_suffix}',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    filename = f'{dataset_name.replace("-", "_").lower()}_memory_vs_binwidth{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def plot_accuracy_time_tradeoff(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 6: Accuracy vs Generation Time (Pareto analysis).
    Shows efficiency of different configurations.
    """
    if df is None or 'generation_time' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for func in df['func'].unique():
        func_df = df[df['func'] == func]
        color = get_func_color(func)
        
        scatter = ax.scatter(func_df['generation_time'], func_df['accuracy'],
                           s=150, alpha=0.8, color=color, 
                           label=func.capitalize(), edgecolors='black', linewidths=0.5)
        
        # Annotate with binwidth (collision avoidance)
        func_df = func_df.sort_values('generation_time')
        
        x_range = func_df['generation_time'].max() - func_df['generation_time'].min()
        y_range = func_df['accuracy'].max() - func_df['accuracy'].min()
        if x_range == 0: x_range = 1
        if y_range == 0: y_range = 1
        
        prev_x, prev_y = -9999, -9999
        offset_dir = 1
        
        for _, row in func_df.iterrows():
            x, y = row['generation_time'], row['accuracy']
            
            dx = abs(x - prev_x) / x_range
            dy = abs(y - prev_y) / y_range
            
            if dx < 0.1 and dy < 0.1:
                offset_dir *= -1
            else:
                offset_dir = 1
            
            xytext = (5, 5) if offset_dir > 0 else (5, -15)
            
            ax.annotate(f'h={row["binwidth"]}',
                       (x, y),
                       textcoords="offset points", xytext=xytext,
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.3))
            
            prev_x, prev_y = x, y
    
    ax.set_xlabel('Generation Time (seconds)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{dataset_name}: Accuracy vs Computational Cost\n(Upper-left is optimal)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add Pareto frontier annotation
    ax.text(0.98, 0.02, 'Pareto optimal: High accuracy, Low time',
           transform=ax.transAxes, fontsize=9, style='italic', 
           ha='right', alpha=0.7)
    
    filename = f'{dataset_name.replace("-", "_").lower()}_pareto{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def plot_heatmap(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 7: Heatmap of Accuracy by Function and Binwidth.
    """
    if df is None:
        return
    
    # Create pivot table
    pivot = df.pivot_table(values='accuracy', index='func', columns='binwidth', aggfunc='first')
    
    # Sort columns (binwidth) in descending order for better visualization
    pivot = pivot[sorted(pivot.columns, reverse=True)]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': 'Accuracy'}, linewidths=0.5,
                vmin=pivot.values.min() - 0.02, vmax=pivot.values.max() + 0.02)
    
    mode_suffix = " (Raw Embeddings)" if raw_mode else ""
    ax.set_xlabel('Binwidth (h) → Fewer bins', fontsize=11)
    ax.set_ylabel('Function Type', fontsize=11)
    ax.set_title(f'{dataset_name}: Grid Search Accuracy Heatmap{mode_suffix}\n(bins = range / binwidth)',
                fontsize=13, fontweight='bold')
    
    # Add bins annotation below
    bins_text = "Bins: " + ", ".join([f"h={h}→{int(df[df['binwidth']==h]['bins'].iloc[0])}" 
                                       for h in sorted(df['binwidth'].unique(), reverse=True)])
    fig.text(0.5, -0.02, bins_text, ha='center', fontsize=9, style='italic')
    
    filename = f'{dataset_name.replace("-", "_").lower()}_heatmap{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def plot_combined_summary(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 8: Combined 2x2 summary plot.
    """
    if df is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Accuracy vs Binwidth
    ax1 = axes[0, 0]
    for func in df['func'].unique():
        func_df = df[df['func'] == func].sort_values('binwidth')
        color = get_func_color(func)
        ax1.plot(func_df['binwidth'], func_df['accuracy'],
                marker='o', linewidth=2, markersize=8, label=func.capitalize(), color=color)
    ax1.set_xlabel('Binwidth (h)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Binwidth', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Top-right: Accuracy vs Bins
    ax2 = axes[0, 1]
    for func in df['func'].unique():
        func_df = df[df['func'] == func].sort_values('bins')
        color = get_func_color(func)
        ax2.plot(func_df['bins'], func_df['accuracy'],
                marker='s', linewidth=2, markersize=8, label=func.capitalize(), color=color)
    ax2.set_xlabel('Number of Bins')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Embedding Dimension', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Time vs Binwidth (if available)
    ax3 = axes[1, 0]
    if 'generation_time' in df.columns:
        for func in df['func'].unique():
            func_df = df[df['func'] == func].sort_values('binwidth')
            color = get_func_color(func)
            ax3.plot(func_df['binwidth'], func_df['generation_time'],
                    marker='^', linewidth=2, markersize=8, label=func.capitalize(), color=color)
        ax3.set_xlabel('Binwidth (h)')
        ax3.set_ylabel('Generation Time (s)')
        ax3.set_title('Computational Cost vs Binwidth', fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.invert_xaxis()
    else:
        ax3.text(0.5, 0.5, 'No time data', ha='center', va='center', fontsize=12)
        ax3.set_title('Computational Cost', fontweight='bold')
    
    # Bottom-right: Best configuration bar chart
    ax4 = axes[1, 1]
    best_per_func = df.loc[df.groupby('func')['accuracy'].idxmax()]
    
    x = np.arange(len(best_per_func))
    colors = [get_func_color(f) for f in best_per_func['func']]
    bars = ax4.bar(x, best_per_func['accuracy'], color=colors, alpha=0.8, edgecolor='black')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([f.capitalize() for f in best_per_func['func']])
    ax4.set_ylabel('Best Accuracy')
    ax4.set_title('Best Accuracy per Function', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add annotations with binwidth
    for i, (bar, (_, row)) in enumerate(zip(bars, best_per_func.iterrows())):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'h={row["binwidth"]}\nb={int(row["bins"])}',
                ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(f'{dataset_name}: Grid Search Summary\n(Classifier: Random Forest)',
                fontsize=15, fontweight='bold', y=1.02)
    
    filename = f'{dataset_name.replace("-", "_").lower()}_summary{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def create_grid_search_summary_table(all_data, output_dir, raw_mode=False):
    """Create summary table of grid search results."""
    summary_rows = []
    
    for dataset_name, df in all_data.items():
        if df is None:
            continue
        
        for func in df['func'].unique():
            func_df = df[df['func'] == func]
            
            # Best configuration
            best_idx = func_df['accuracy'].idxmax()
            best_row = func_df.loc[best_idx]
            
            row = {
                'Dataset': dataset_name,
                'Function': func,
                'Best Binwidth': best_row['binwidth'],
                'Best Bins': int(best_row['bins']),
                'Best Accuracy': best_row['accuracy'],
                'Best F1': best_row.get('f1_score', None),
                'Range': best_row.get('range', None),
            }
            
            if 'generation_time' in func_df.columns:
                row['Gen Time (s)'] = best_row['generation_time']
            if 'memory_mb' in func_df.columns:
                row['Memory (MB)'] = best_row['memory_mb']
            
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{output_dir}/grid_search_summary.csv', index=False)
    
    print("\n" + "="*120)
    print("GRID SEARCH SUMMARY")
    print("="*120)
    print("\nBest configurations per dataset and function:")
    print(summary_df.to_string(index=False))
    print("\n" + "="*120)
    
    return summary_df


def plot_cross_dataset_comparison(all_data_prep, all_data_raw, output_dir):
    """
    Plot comparing grid search results across all datasets.
    """
    # Check how many datasets have data
    valid_data = {k: v for k, v in all_data_prep.items() if v is not None}
    if len(valid_data) < 2:
        print("  ⚠️ Need at least 2 datasets for cross-dataset comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Best accuracy per function across datasets
    ax1 = axes[0]
    
    datasets = list(valid_data.keys())
    funcs = set()
    for df in valid_data.values():
        funcs.update(df['func'].unique())
    funcs = sorted(funcs)
    
    x = np.arange(len(funcs))
    width = 0.8 / len(datasets)
    
    for i, (dataset_name, df) in enumerate(valid_data.items()):
        best_acc = []
        for func in funcs:
            func_df = df[df['func'] == func]
            if len(func_df) > 0:
                best_acc.append(func_df['accuracy'].max())
            else:
                best_acc.append(0)
        
        ax1.bar(x + i*width, best_acc, width, label=dataset_name, alpha=0.8)
    
    ax1.set_xlabel('Function Type', fontsize=11)
    ax1.set_ylabel('Best Accuracy', fontsize=11)
    ax1.set_title('Best Accuracy by Function\n(Across Datasets)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax1.set_xticklabels([f.capitalize() for f in funcs])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Optimal binwidth comparison
    ax2 = axes[1]
    
    opt_binwidths = []
    for dataset_name, df in valid_data.items():
        for func in df['func'].unique():
            func_df = df[df['func'] == func]
            best_idx = func_df['accuracy'].idxmax()
            opt_binwidths.append({
                'Dataset': dataset_name,
                'Function': func,
                'Optimal Binwidth': func_df.loc[best_idx, 'binwidth']
            })
    
    opt_df = pd.DataFrame(opt_binwidths)
    pivot = opt_df.pivot_table(values='Optimal Binwidth', index='Function', 
                               columns='Dataset', aggfunc='first')
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd_r', ax=ax2,
                cbar_kws={'label': 'Optimal Binwidth'}, linewidths=0.5)
    ax2.set_title('Optimal Binwidth per Configuration', fontsize=12, fontweight='bold')
    
    fig.suptitle('Cross-Dataset Grid Search Comparison', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cross_dataset_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Saved: cross_dataset_comparison.png")


def main():
    print("="*80)
    print("FGSD GRID SEARCH ANALYSIS")
    print("="*80)
    
    all_data_preprocessed = {}
    all_data_raw = {}
    
    for dataset_name, paths in FILES.items():
        # Preprocessed
        print(f"\n📊 Processing {dataset_name} (Preprocessed)...")
        df_prep = load_data(paths['preprocessed'])
        all_data_preprocessed[dataset_name] = df_prep
        
        if df_prep is not None:
            output_dir = get_output_dir(raw_mode=False)
            plot_accuracy_vs_binwidth(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_accuracy_vs_bins(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_f1_vs_binwidth(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_time_vs_binwidth(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_memory_vs_binwidth(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_accuracy_time_tradeoff(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_heatmap(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_combined_summary(df_prep, dataset_name, output_dir, raw_mode=False)
        
        # Raw
        print(f"\n📊 Processing {dataset_name} (Raw Embeddings)...")
        df_raw = load_data(paths['raw'])
        all_data_raw[dataset_name] = df_raw
        
        if df_raw is not None:
            output_dir_raw = get_output_dir(raw_mode=True)
            plot_accuracy_vs_binwidth(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_accuracy_vs_bins(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_f1_vs_binwidth(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_time_vs_binwidth(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_memory_vs_binwidth(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_accuracy_time_tradeoff(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_heatmap(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_combined_summary(df_raw, dataset_name, output_dir_raw, raw_mode=True)
    
    # Cross-dataset comparisons
    print("\n📊 Creating cross-dataset comparisons...")
    plot_cross_dataset_comparison(all_data_preprocessed, OUTPUT_DIR, raw_mode=False)
    plot_cross_dataset_comparison(all_data_raw, OUTPUT_DIR_RAW, raw_mode=True)
    
    # Summary tables
    create_grid_search_summary_table(all_data_preprocessed, OUTPUT_DIR, raw_mode=False)
    create_grid_search_summary_table(all_data_raw, OUTPUT_DIR_RAW, raw_mode=True)
    
    # Comparison
    plot_raw_vs_preprocessed_accuracy(all_data_preprocessed, all_data_raw, OUTPUT_DIR)
    
    print(f"\n✅ All plots saved!")
    print(f"  📁 Preprocessed: {OUTPUT_DIR}/")
    print(f"  📁 Raw: {OUTPUT_DIR_RAW}/")


if __name__ == "__main__":
    main()
