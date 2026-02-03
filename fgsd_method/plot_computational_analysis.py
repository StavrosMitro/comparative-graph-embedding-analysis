"""
Computational Cost Analysis for FGSD Experiments

This script creates plots analyzing:
1. Generation Time vs Embedding Dimension
2. Memory Usage vs Embedding Dimension  
3. Time-Memory Trade-off
4. Generation Time vs Accuracy (Pareto analysis)
5. Cross-dataset Comparison
6. Function Type Comparison

Available Data:
- IMDB-MULTI: generation_time, memory_mb per (func, bins)
- ENZYMES: generation_time, memory_mb per (func, bins)
- REDDIT-MULTI-12K: generation_time per (func, bins) - large scale (~4000s)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# File paths
FILES = {
    'ENZYMES': {
        'dimension': 'src/results/fgsd_enzymes_dimension_analysis.csv',
        'grid': None,
    },
    'IMDB-MULTI': {
        'dimension': 'src/results/fgsd_imdb_dimension_analysis.csv',
        'grid': 'src/results/fgsd_imdb_grid_search.csv',
    },
    'REDDIT-MULTI-12K': {
        'dimension': 'src/results/fgsd_reddit_dimension_analysis.csv',
        'grid': None,
    },
}

# Output directory
OUTPUT_DIR = 'plots/computational_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Colors
FUNC_COLORS = {'harmonic': '#2ecc71', 'polynomial': '#3498db', 'biharmonic': '#9b59b6'}
DATASET_COLORS = {'ENZYMES': '#e74c3c', 'IMDB-MULTI': '#3498db', 'REDDIT-MULTI-12K': '#2ecc71'}


def load_data(path):
    """Load CSV data if exists."""
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return None


def plot_time_vs_dimension(all_data, output_dir):
    """
    Plot 1: Generation Time vs Embedding Dimension for each dataset.
    Shows how computational cost scales with embedding size.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (dataset_name, data) in zip(axes, all_data.items()):
        df = data.get('dimension')
        if df is None or 'generation_time' not in df.columns:
            ax.text(0.5, 0.5, f'{dataset_name}\nNo time data', ha='center', va='center')
            ax.set_title(dataset_name)
            continue
        
        # Get unique generation times per (func, bins)
        time_data = df.groupby(['func', 'embedding_dim'])['generation_time'].first().reset_index()
        
        for func in time_data['func'].unique():
            func_df = time_data[time_data['func'] == func].sort_values('embedding_dim')
            color = FUNC_COLORS.get(func, '#333')
            ax.plot(func_df['embedding_dim'], func_df['generation_time'],
                   marker='o', linewidth=2, markersize=8, label=func.capitalize(), color=color)
        
        ax.set_xlabel('Embedding Dimension (bins)', fontsize=11)
        ax.set_ylabel('Generation Time (seconds)', fontsize=11)
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Generation Time vs Embedding Dimension', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_vs_dimension.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Saved: time_vs_dimension.png")


def plot_memory_vs_dimension(all_data, output_dir):
    """
    Plot 2: Memory Usage vs Embedding Dimension.
    Shows memory footprint scaling.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (dataset_name, data) in zip(axes, all_data.items()):
        df = data.get('dimension')
        if df is None or 'memory_mb' not in df.columns:
            ax.text(0.5, 0.5, f'{dataset_name}\nNo memory data', ha='center', va='center')
            ax.set_title(dataset_name)
            continue
        
        # Get unique memory per (func, bins)
        mem_data = df.groupby(['func', 'embedding_dim'])['memory_mb'].first().reset_index()
        
        for func in mem_data['func'].unique():
            func_df = mem_data[mem_data['func'] == func].sort_values('embedding_dim')
            color = FUNC_COLORS.get(func, '#333')
            ax.plot(func_df['embedding_dim'], func_df['memory_mb'],
                   marker='s', linewidth=2, markersize=8, label=func.capitalize(), color=color)
        
        ax.set_xlabel('Embedding Dimension (bins)', fontsize=11)
        ax.set_ylabel('Memory Usage (MB)', fontsize=11)
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Memory Usage vs Embedding Dimension', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/memory_vs_dimension.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Saved: memory_vs_dimension.png")


def plot_time_memory_tradeoff(all_data, output_dir):
    """
    Plot 3: Time vs Memory Trade-off scatter.
    Shows relationship between computational costs.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for dataset_name, data in all_data.items():
        df = data.get('dimension')
        if df is None or 'generation_time' not in df.columns or 'memory_mb' not in df.columns:
            continue
        
        # Get unique (func, bins) combinations
        cost_data = df.groupby(['func', 'embedding_dim']).agg({
            'generation_time': 'first',
            'memory_mb': 'first'
        }).reset_index()
        
        color = DATASET_COLORS.get(dataset_name, '#333')
        
        for func in cost_data['func'].unique():
            func_df = cost_data[cost_data['func'] == func]
            marker = 'o' if func == 'harmonic' else 's' if func == 'polynomial' else '^'
            
            ax.scatter(func_df['generation_time'], func_df['memory_mb'],
                      s=100, alpha=0.7, color=color, marker=marker,
                      label=f'{dataset_name} - {func}')
            
            # Annotate with dimension
            for _, row in func_df.iterrows():
                ax.annotate(f'd={int(row["embedding_dim"])}',
                           (row['generation_time'], row['memory_mb']),
                           textcoords="offset points", xytext=(3, 3), fontsize=7)
    
    ax.set_xlabel('Generation Time (seconds)', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Time vs Memory Trade-off\n(Labels show embedding dimension)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_memory_tradeoff.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Saved: time_memory_tradeoff.png")


def plot_time_vs_accuracy(all_data, output_dir):
    """
    Plot 4: Generation Time vs Accuracy (Pareto frontier analysis).
    Shows efficiency of different configurations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (dataset_name, data) in zip(axes, all_data.items()):
        df = data.get('dimension')
        if df is None or 'generation_time' not in df.columns:
            ax.text(0.5, 0.5, f'{dataset_name}\nNo data', ha='center', va='center')
            ax.set_title(dataset_name)
            continue
        
        # Get best accuracy per (func, bins) config
        best_acc = df.groupby(['func', 'embedding_dim']).agg({
            'accuracy': 'max',
            'generation_time': 'first'
        }).reset_index()
        
        for func in best_acc['func'].unique():
            func_df = best_acc[best_acc['func'] == func]
            color = FUNC_COLORS.get(func, '#333')
            
            ax.scatter(func_df['generation_time'], func_df['accuracy'],
                      s=120, alpha=0.7, color=color, label=func.capitalize(),
                      edgecolors='black', linewidths=0.5)
            
            # Annotate with dimension
            for _, row in func_df.iterrows():
                ax.annotate(f'd={int(row["embedding_dim"])}',
                           (row['generation_time'], row['accuracy']),
                           textcoords="offset points", xytext=(3, 3), fontsize=8)
        
        ax.set_xlabel('Generation Time (seconds)', fontsize=11)
        ax.set_ylabel('Best Accuracy', fontsize=11)
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Generation Time vs Accuracy Trade-off\n(Pareto Analysis - Upper-left is better)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_vs_accuracy.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Saved: time_vs_accuracy.png")


def plot_dataset_comparison(all_data, output_dir):
    """
    Plot 5: Cross-dataset comparison of generation times.
    Bar chart comparing datasets at similar dimensions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Absolute times
    ax1 = axes[0]
    comparison_data = []
    
    for dataset_name, data in all_data.items():
        df = data.get('dimension')
        if df is None or 'generation_time' not in df.columns:
            continue
        
        for func in df['func'].unique():
            func_df = df[df['func'] == func]
            avg_time = func_df.groupby('embedding_dim')['generation_time'].first().mean()
            min_time = func_df.groupby('embedding_dim')['generation_time'].first().min()
            max_time = func_df.groupby('embedding_dim')['generation_time'].first().max()
            
            comparison_data.append({
                'Dataset': dataset_name,
                'Function': func,
                'Avg Time': avg_time,
                'Min Time': min_time,
                'Max Time': max_time
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # Group by dataset
        datasets = comp_df['Dataset'].unique()
        x = np.arange(len(datasets))
        width = 0.25
        
        funcs = comp_df['Function'].unique()
        for i, func in enumerate(funcs):
            func_data = comp_df[comp_df['Function'] == func]
            times = [func_data[func_data['Dataset'] == d]['Avg Time'].values[0] 
                    if d in func_data['Dataset'].values else 0 for d in datasets]
            color = FUNC_COLORS.get(func, '#333')
            ax1.bar(x + i*width, times, width, label=func.capitalize(), color=color, alpha=0.8)
        
        ax1.set_xlabel('Dataset', fontsize=11)
        ax1.set_ylabel('Average Generation Time (seconds)', fontsize=11)
        ax1.set_title('Generation Time by Dataset & Function', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(datasets, rotation=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add log scale note if needed
        if comp_df['Avg Time'].max() / comp_df['Avg Time'].min() > 100:
            ax1.set_yscale('log')
            ax1.set_title('Generation Time by Dataset & Function (Log Scale)', fontsize=12, fontweight='bold')
    
    # Right: Time per graph (normalized by dataset size)
    ax2 = axes[1]
    dataset_sizes = {'ENZYMES': 600, 'IMDB-MULTI': 1500, 'REDDIT-MULTI-12K': 12000}
    
    normalized_data = []
    for dataset_name, data in all_data.items():
        df = data.get('dimension')
        if df is None or 'generation_time' not in df.columns:
            continue
        
        n_graphs = dataset_sizes.get(dataset_name, 1000)
        
        for func in df['func'].unique():
            func_df = df[df['func'] == func]
            avg_time = func_df.groupby('embedding_dim')['generation_time'].first().mean()
            time_per_graph = avg_time / n_graphs * 1000  # ms per graph
            
            normalized_data.append({
                'Dataset': dataset_name,
                'Function': func,
                'Time per Graph (ms)': time_per_graph
            })
    
    if normalized_data:
        norm_df = pd.DataFrame(normalized_data)
        
        datasets = norm_df['Dataset'].unique()
        x = np.arange(len(datasets))
        
        for i, func in enumerate(norm_df['Function'].unique()):
            func_data = norm_df[norm_df['Function'] == func]
            times = [func_data[func_data['Dataset'] == d]['Time per Graph (ms)'].values[0]
                    if d in func_data['Dataset'].values else 0 for d in datasets]
            color = FUNC_COLORS.get(func, '#333')
            ax2.bar(x + i*width, times, width, label=func.capitalize(), color=color, alpha=0.8)
        
        ax2.set_xlabel('Dataset', fontsize=11)
        ax2.set_ylabel('Time per Graph (ms)', fontsize=11)
        ax2.set_title('Normalized Generation Time\n(milliseconds per graph)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(datasets, rotation=15)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Saved: dataset_comparison.png")


def plot_binwidth_analysis(all_data, output_dir):
    """
    Plot 6: Binwidth impact on generation time (from grid search).
    Shows how binwidth parameter affects computational cost.
    """
    # Check if grid search data exists
    grid_data = None
    for dataset_name, data in all_data.items():
        if data.get('grid') is not None:
            grid_data = data['grid']
            grid_dataset = dataset_name
            break
    
    if grid_data is None:
        print("  ⚠️ No grid search data available for binwidth analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Time vs binwidth
    ax1 = axes[0]
    for func in grid_data['func'].unique():
        func_df = grid_data[grid_data['func'] == func].sort_values('binwidth')
        color = FUNC_COLORS.get(func, '#333')
        ax1.plot(func_df['binwidth'], func_df['generation_time'],
                marker='o', linewidth=2, markersize=8, label=func.capitalize(), color=color)
    
    ax1.set_xlabel('Binwidth (h)', fontsize=11)
    ax1.set_ylabel('Generation Time (seconds)', fontsize=11)
    ax1.set_title(f'{grid_dataset}: Time vs Binwidth', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Smaller binwidth = more bins
    
    # Right: Accuracy vs binwidth
    ax2 = axes[1]
    for func in grid_data['func'].unique():
        func_df = grid_data[grid_data['func'] == func].sort_values('binwidth')
        color = FUNC_COLORS.get(func, '#333')
        ax2.plot(func_df['binwidth'], func_df['accuracy'],
                marker='s', linewidth=2, markersize=8, label=func.capitalize(), color=color)
    
    ax2.set_xlabel('Binwidth (h)', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title(f'{grid_dataset}: Accuracy vs Binwidth', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    fig.suptitle('Binwidth Analysis (bins = range / binwidth)\nSmaller binwidth → More bins → Higher dimension',
                fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/binwidth_analysis.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Saved: binwidth_analysis.png")


def plot_function_comparison(all_data, output_dir):
    """
    Plot 7: Function type comparison (harmonic vs polynomial vs biharmonic).
    Box plots showing distribution of times and memory.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Collect all time data
    time_records = []
    memory_records = []
    
    for dataset_name, data in all_data.items():
        df = data.get('dimension')
        if df is None:
            continue
        
        for _, row in df.iterrows():
            if 'generation_time' in df.columns:
                time_records.append({
                    'Dataset': dataset_name,
                    'Function': row['func'],
                    'Generation Time (s)': row['generation_time']
                })
            if 'memory_mb' in df.columns:
                memory_records.append({
                    'Dataset': dataset_name,
                    'Function': row['func'],
                    'Memory (MB)': row['memory_mb']
                })
    
    # Left: Time comparison
    ax1 = axes[0]
    if time_records:
        time_df = pd.DataFrame(time_records)
        # Filter to datasets with reasonable times for visualization
        time_df_filtered = time_df[time_df['Generation Time (s)'] < 100]  # Exclude REDDIT outliers
        
        if len(time_df_filtered) > 0:
            palette = {f: FUNC_COLORS.get(f, '#333') for f in time_df_filtered['Function'].unique()}
            sns.boxplot(data=time_df_filtered, x='Function', y='Generation Time (s)',
                       ax=ax1, palette=palette)
            ax1.set_title('Generation Time by Function Type\n(excluding large-scale datasets)',
                         fontsize=12, fontweight='bold')
        else:
            sns.boxplot(data=time_df, x='Function', y='Generation Time (s)', ax=ax1)
            ax1.set_title('Generation Time by Function Type', fontsize=12, fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Memory comparison
    ax2 = axes[1]
    if memory_records:
        mem_df = pd.DataFrame(memory_records)
        palette = {f: FUNC_COLORS.get(f, '#333') for f in mem_df['Function'].unique()}
        sns.boxplot(data=mem_df, x='Function', y='Memory (MB)', ax=ax2, palette=palette)
        ax2.set_title('Memory Usage by Function Type', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No memory data available', ha='center', va='center')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/function_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Saved: function_comparison.png")


def create_computational_summary(all_data, output_dir):
    """Create summary table of computational costs."""
    summary_rows = []
    
    dataset_sizes = {'ENZYMES': 600, 'IMDB-MULTI': 1500, 'REDDIT-MULTI-12K': 12000}
    
    for dataset_name, data in all_data.items():
        df = data.get('dimension')
        if df is None:
            continue
        
        n_graphs = dataset_sizes.get(dataset_name, 1000)
        
        for func in df['func'].unique():
            func_df = df[df['func'] == func]
            
            # Get time/memory stats
            time_stats = func_df.groupby('embedding_dim')['generation_time'].first()
            
            row = {
                'Dataset': dataset_name,
                'Function': func,
                'Num Graphs': n_graphs,
                'Min Time (s)': time_stats.min() if 'generation_time' in func_df.columns else None,
                'Max Time (s)': time_stats.max() if 'generation_time' in func_df.columns else None,
                'Avg Time (s)': time_stats.mean() if 'generation_time' in func_df.columns else None,
            }
            
            if 'memory_mb' in func_df.columns:
                mem_stats = func_df.groupby('embedding_dim')['memory_mb'].first()
                row['Min Memory (MB)'] = mem_stats.min()
                row['Max Memory (MB)'] = mem_stats.max()
            
            # Time per graph
            if row['Avg Time (s)']:
                row['Avg Time/Graph (ms)'] = row['Avg Time (s)'] / n_graphs * 1000
            
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{output_dir}/computational_summary.csv', index=False)
    
    print("\n" + "="*100)
    print("COMPUTATIONAL COST SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)
    
    return summary_df


def main():
    print("="*80)
    print("FGSD COMPUTATIONAL COST ANALYSIS")
    print("="*80)
    
    # Load all data
    all_data = {}
    for dataset_name, paths in FILES.items():
        print(f"\n📊 Loading {dataset_name}...")
        all_data[dataset_name] = {
            'dimension': load_data(paths['dimension']),
            'grid': load_data(paths['grid']) if paths['grid'] else None
        }
        
        df = all_data[dataset_name]['dimension']
        if df is not None:
            print(f"  Dimension analysis: {len(df)} rows")
            print(f"  Columns: {df.columns.tolist()}")
            if 'generation_time' in df.columns:
                print(f"  Time range: {df['generation_time'].min():.2f} - {df['generation_time'].max():.2f}s")
            if 'memory_mb' in df.columns:
                print(f"  Memory range: {df['memory_mb'].min():.2f} - {df['memory_mb'].max():.2f} MB")
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    plot_time_vs_dimension(all_data, OUTPUT_DIR)
    plot_memory_vs_dimension(all_data, OUTPUT_DIR)
    plot_time_memory_tradeoff(all_data, OUTPUT_DIR)
    plot_time_vs_accuracy(all_data, OUTPUT_DIR)
    plot_dataset_comparison(all_data, OUTPUT_DIR)
    plot_binwidth_analysis(all_data, OUTPUT_DIR)
    plot_function_comparison(all_data, OUTPUT_DIR)
    
    # Create summary
    create_computational_summary(all_data, OUTPUT_DIR)
    
    print(f"\n✅ All plots saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  📈 {f}")


if __name__ == "__main__":
    main()
