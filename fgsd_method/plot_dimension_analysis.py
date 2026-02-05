"""
Dimension Analysis Visualization for FGSD Experiments

This script creates comprehensive plots showing:
1. Accuracy vs Embedding Dimension (per function type and classifier)
2. Accuracy vs Generation Time trade-off
3. Best classifier comparison across dimensions
4. Summary heatmaps

Classifier Hyperparameters (from code):
- SVM (RBF): C=100, gamma='scale', StandardScaler preprocessing
- Random Forest: n_estimators=500, max_depth=20
- MLP: hidden_layer_sizes=(256, 128, 64), max_iter=1000, early_stopping=True, StandardScaler preprocessing
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Classifier hyperparameters (extracted from code)
CLASSIFIER_HYPERPARAMS = {
    'SVM (RBF)': 'C=100, gamma=scale, StandardScaler',
    'Random Forest': 'n_estimators=500, max_depth=20',
    'MLP': 'layers=(256,128,64), max_iter=1000, early_stop=True, StandardScaler'
}

# File paths
FILES = {
    'ENZYMES': {
        'preprocessed': 'src/results/fgsd_enzymes_dimension_analysis.csv',
        'raw': 'src/results/raw_embeddings/fgsd_enzymes_raw_dimension_analysis.csv'
    },
    'IMDB-MULTI': {
        'preprocessed': 'src/results/fgsd_imdb_dimension_analysis.csv',
        'raw': 'src/results/raw_embeddings/fgsd_imdb_raw_dimension_analysis.csv'
    },
    'REDDIT-MULTI-12K': {
        'preprocessed': 'src/results/fgsd_reddit_dimension_analysis.csv',
        'raw': 'src/results/raw_embeddings/fgsd_reddit_raw_dimension_analysis.csv'
    },
}

# Output directories
OUTPUT_DIR = 'plots/dimension_analysis'
OUTPUT_DIR_RAW = 'plots/dimension_analysis/raw_embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_RAW, exist_ok=True)

# Plot settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Color palettes
FUNC_COLORS = {'harmonic': '#2ecc71', 'polynomial': '#3498db', 'biharmonic': '#9b59b6'}
CLF_COLORS = {
    'SVM (RBF)': '#e74c3c', 
    'Random Forest': '#27ae60', 
    'MLP': '#f39c12',
    # Raw variants - same colors but with alpha
    'SVM (RBF) Raw': '#e74c3c',
    'Random Forest Raw': '#27ae60',
    'MLP Raw': '#f39c12'
}


def get_classifier_color(clf_name):
    """Get color for classifier, handling Raw variants."""
    # Remove 'Raw' suffix for color lookup
    base_name = clf_name.replace(' Raw', '')
    return CLF_COLORS.get(base_name, CLF_COLORS.get(clf_name, '#333333'))


def load_data(path):
    """Load and preprocess dimension analysis data."""
    if not os.path.exists(path):
        print(f"  ⚠️  File not found: {path}")
        return None
    
    df = pd.read_csv(path)
    
    # Standardize column names
    if 'bins' in df.columns and 'embedding_dim' not in df.columns:
        df['embedding_dim'] = df['bins']
    
    return df


def get_output_dir(raw_mode=False):
    """Get output directory based on mode."""
    return OUTPUT_DIR_RAW if raw_mode else OUTPUT_DIR


def plot_accuracy_vs_dimension(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 1: Accuracy vs Embedding Dimension
    - Separate subplot for each function type
    - Lines for each classifier
    """
    if df is None or len(df) == 0:
        return
    
    funcs = df['func'].unique()
    n_funcs = len(funcs)
    
    fig, axes = plt.subplots(1, n_funcs, figsize=(5*n_funcs, 5), sharey=True)
    if n_funcs == 1:
        axes = [axes]
    
    for ax, func in zip(axes, funcs):
        func_df = df[df['func'] == func]
        
        for clf in func_df['classifier'].unique():
            clf_df = func_df[func_df['classifier'] == clf]
            clf_df_sorted = clf_df.sort_values('embedding_dim')
            
            # Use the new color getter function
            color = get_classifier_color(clf)
            # Add different linestyle for Raw vs non-Raw
            linestyle = '--' if 'Raw' in clf else '-'
            alpha = 0.8 if 'Raw' in clf else 1.0
            
            ax.plot(clf_df_sorted['embedding_dim'], clf_df_sorted['accuracy'], 
                   marker='o', linewidth=2, markersize=8, label=clf, 
                   color=color, linestyle=linestyle, alpha=alpha)
        
        ax.set_xlabel('Embedding Dimension (bins)', fontsize=11)
        ax.set_title(f'{func.capitalize()}', fontsize=12, fontweight='bold',
                    color=FUNC_COLORS.get(func, '#333333'))
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel('Test Accuracy', fontsize=11)
    
    mode_suffix = " (Raw Embeddings)" if raw_mode else ""
    fig.suptitle(f'{dataset_name}: Accuracy vs Embedding Dimension{mode_suffix}', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    filename = f'{dataset_name.replace("-", "_").lower()}_acc_vs_dim{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def get_pareto_frontier(xs, ys):
    """
    Finds the Pareto frontier for minimizing X (Time/Mem) and maximizing Y (Accuracy).
    Returns sorted lists of x and y coordinates on the frontier.
    """
    # Combine and sort by X (ascending) - assume lower cost is better
    points = sorted(zip(xs, ys), key=lambda k: k[0])
    
    pareto_x = []
    pareto_y = []
    
    current_max_y = -float('inf')
    
    for x, y in points:
        # If we find a higher Y, it's a Pareto point 
        # (since X is increasing, this new point costs more but yields better accuracy)
        if y > current_max_y:
            pareto_x.append(x)
            pareto_y.append(y)
            current_max_y = y
            
    return pareto_x, pareto_y


def plot_accuracy_vs_time(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 2: Accuracy vs Generation Time (Pareto Frontier)
    - Shows which configurations give best accuracy per compute cost
    - Bullets are distinct bin sizes (embedding dimensions)
    - Includes per-function Pareto frontiers AND overall Pareto frontier
    """
    if df is None or 'generation_time' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect all points for overall Pareto frontier
    all_times = []
    all_accs = []
    
    for func in df['func'].unique():
        func_df = df[df['func'] == func]
        color = FUNC_COLORS.get(func, '#333333')
        
        # Get best accuracy per dimension (since dims map to bins)
        best_per_dim = func_df.groupby('embedding_dim').agg({
            'accuracy': 'max',
            'generation_time': 'first'
        }).reset_index()
        
        # Collect for overall Pareto
        all_times.extend(best_per_dim['generation_time'].tolist())
        all_accs.extend(best_per_dim['accuracy'].tolist())
        
        # Scatter plot for points
        ax.scatter(best_per_dim['generation_time'], best_per_dim['accuracy'], 
                  s=80, alpha=0.7, label=func.capitalize(), color=color, edgecolors='white')
        
        # Calculate and draw per-function Pareto Frontier Curve
        px, py = get_pareto_frontier(best_per_dim['generation_time'], best_per_dim['accuracy'])
        ax.plot(px, py, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

        # Annotate with dimension/bins (collision avoidance)
        best_per_dim = best_per_dim.sort_values('generation_time')
        
        x_range = best_per_dim['generation_time'].max() - best_per_dim['generation_time'].min()
        y_range = best_per_dim['accuracy'].max() - best_per_dim['accuracy'].min()
        if x_range == 0: x_range = 1
        if y_range == 0: y_range = 1
        
        prev_x, prev_y = -9999, -9999
        offset_dir = 1
        
        for _, row in best_per_dim.iterrows():
            x, y = row['generation_time'], row['accuracy']
            
            dx = abs(x - prev_x) / x_range
            dy = abs(y - prev_y) / y_range
            
            if dx < 0.1 and dy < 0.1:
                offset_dir *= -1
            else:
                offset_dir = 1
            
            xytext = (0, 5) if offset_dir > 0 else (0, -15)
            
            ax.annotate(f'{int(row["embedding_dim"])}', 
                       (x, y),
                       textcoords="offset points", xytext=xytext, fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.3))
            
            prev_x, prev_y = x, y
    
    # Calculate and draw OVERALL Pareto Frontier (all functions combined)
    if len(all_times) > 0:
        overall_px, overall_py = get_pareto_frontier(all_times, all_accs)
        ax.plot(overall_px, overall_py, color='black', linestyle='-', linewidth=2.5, 
                alpha=0.9, label='Overall Pareto', zorder=10)
        # Add markers on overall Pareto points
        ax.scatter(overall_px, overall_py, s=120, color='black', marker='*', 
                  zorder=11, alpha=0.8, edgecolors='white', linewidths=0.5)
    
    ax.set_xlabel('Embedding Generation Time (seconds) [Min is better]', fontsize=11)
    ax.set_ylabel('Best Test Accuracy [Max is better]', fontsize=11)
    
    mode_suffix = " (Raw)" if raw_mode else ""
    ax.set_title(f'{dataset_name}: Accuracy vs Time (Pareto Frontier){mode_suffix}', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{dataset_name.replace("-", "_").lower()}_acc_vs_time{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def plot_accuracy_vs_memory(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot: Accuracy vs Memory Usage (Pareto Frontier)
    - Bullets are distinct bin sizes (embedding dimensions)
    - Includes per-function Pareto frontiers AND overall Pareto frontier
    """
    if df is None or 'memory_mb' not in df.columns:
        return
     
    # Check if we have valid memory data
    if df['memory_mb'].isna().all() or (df['memory_mb'] == 0).all():
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect all points for overall Pareto frontier
    all_mem = []
    all_accs = []
    
    for func in df['func'].unique():
        func_df = df[df['func'] == func]
        color = FUNC_COLORS.get(func, '#333333')
        
        # Get best accuracy per dimension
        best_per_dim = func_df.groupby('embedding_dim').agg({
            'accuracy': 'max',
            'memory_mb': 'first'
        }).reset_index()
        
        # Collect for overall Pareto
        all_mem.extend(best_per_dim['memory_mb'].tolist())
        all_accs.extend(best_per_dim['accuracy'].tolist())
        
        # Scatter plot
        ax.scatter(best_per_dim['memory_mb'], best_per_dim['accuracy'], 
                  s=80, alpha=0.7, label=func.capitalize(), color=color, edgecolors='white')
        
        # Calculate and draw per-function Pareto Frontier Curve
        px, py = get_pareto_frontier(best_per_dim['memory_mb'], best_per_dim['accuracy'])
        ax.plot(px, py, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Annotate with dimension/bins (collision avoidance)
        best_per_dim = best_per_dim.sort_values('memory_mb')
        
        x_range = best_per_dim['memory_mb'].max() - best_per_dim['memory_mb'].min()
        y_range = best_per_dim['accuracy'].max() - best_per_dim['accuracy'].min()
        if x_range == 0: x_range = 1
        if y_range == 0: y_range = 1
        
        prev_x, prev_y = -9999, -9999
        offset_dir = 1
        
        for _, row in best_per_dim.iterrows():
            x, y = row['memory_mb'], row['accuracy']
            
            dx = abs(x - prev_x) / x_range
            dy = abs(y - prev_y) / y_range
            
            if dx < 0.1 and dy < 0.1:
                offset_dir *= -1
            else:
                offset_dir = 1
            
            xytext = (0, 5) if offset_dir > 0 else (0, -15)
            
            ax.annotate(f'{int(row["embedding_dim"])}', 
                       (x, y),
                       textcoords="offset points", xytext=xytext, fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.3))
            
            prev_x, prev_y = x, y
    
    # Calculate and draw OVERALL Pareto Frontier (all functions combined)
    if len(all_mem) > 0:
        overall_px, overall_py = get_pareto_frontier(all_mem, all_accs)
        ax.plot(overall_px, overall_py, color='black', linestyle='-', linewidth=2.5, 
                alpha=0.9, label='Overall Pareto', zorder=10)
        # Add markers on overall Pareto points
        ax.scatter(overall_px, overall_py, s=120, color='black', marker='*', 
                  zorder=11, alpha=0.8, edgecolors='white', linewidths=0.5)
    
    ax.set_xlabel('Peak Memory Usage (MB) [Min is better]', fontsize=11)
    ax.set_ylabel('Best Test Accuracy [Max is better]', fontsize=11)
    
    mode_suffix = " (Raw)" if raw_mode else ""
    ax.set_title(f'{dataset_name}: Accuracy vs Memory (Pareto Frontier){mode_suffix}', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{dataset_name.replace("-", "_").lower()}_acc_vs_memory{"_raw" if raw_mode else ""}.png'
    plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {filename}")


def plot_classifier_comparison_heatmap(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 3: Heatmap showing best accuracy for each (function, classifier) combination
    """
    if df is None:
        return
    
    # Pivot to get best accuracy per func/classifier
    pivot = df.pivot_table(values='accuracy', index='func', columns='classifier', aggfunc='max')
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': 'Test Accuracy'}, vmin=pivot.values.min()-0.02,
                linewidths=0.5)
    
    ax.set_title(f'{dataset_name}: Best Accuracy by Function & Classifier', 
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Classifier', fontsize=11)
    ax.set_ylabel('Function Type', fontsize=11)
    
    # Add hyperparameter info
    fig.text(0.5, -0.05, 
             'Hyperparams: SVM(C=100), RF(n=500,d=20), MLP(256-128-64)',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_heatmap.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_heatmap.png")


def plot_dimension_impact_summary(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 4: Bar chart showing accuracy improvement from min to max dimension
    """
    if df is None:
        return
    
    summary_data = []
    
    for func in df['func'].unique():
        func_df = df[df['func'] == func]
        
        for clf in func_df['classifier'].unique():
            clf_df = func_df[func_df['classifier'] == clf]
            
            min_dim = clf_df['embedding_dim'].min()
            max_dim = clf_df['embedding_dim'].max()
            
            acc_at_min = clf_df[clf_df['embedding_dim'] == min_dim]['accuracy'].values[0]
            acc_at_max = clf_df[clf_df['embedding_dim'] == max_dim]['accuracy'].values[0]
            best_acc = clf_df['accuracy'].max()
            best_dim = clf_df.loc[clf_df['accuracy'].idxmax(), 'embedding_dim']
            
            summary_data.append({
                'func': func,
                'classifier': clf,
                'min_dim': min_dim,
                'max_dim': max_dim,
                'acc_min_dim': acc_at_min,
                'acc_max_dim': acc_at_max,
                'best_acc': best_acc,
                'best_dim': best_dim,
                'improvement': acc_at_max - acc_at_min
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Best dimension per configuration
    ax1 = axes[0]
    x = np.arange(len(summary_df))
    labels = [f"{row['func'][:4]}\n{row['classifier'][:3]}" for _, row in summary_df.iterrows()]
    colors = [FUNC_COLORS.get(row['func'], '#333') for _, row in summary_df.iterrows()]
    
    bars = ax1.bar(x, summary_df['best_dim'], color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('Best Embedding Dimension', fontsize=11)
    ax1.set_title('Optimal Dimension per Configuration', fontsize=12, fontweight='bold')
    
    # Add accuracy annotations
    for i, (bar, acc) in enumerate(zip(bars, summary_df['best_acc'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Right: Accuracy at min vs max dimension
    ax2 = axes[1]
    width = 0.35
    x = np.arange(len(summary_df))
    
    ax2.bar(x - width/2, summary_df['acc_min_dim'], width, label=f'Min dim', color='#3498db', alpha=0.7)
    ax2.bar(x + width/2, summary_df['acc_max_dim'], width, label=f'Max dim', color='#e74c3c', alpha=0.7)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('Test Accuracy', fontsize=11)
    ax2.set_title('Accuracy: Min vs Max Dimension', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    
    fig.suptitle(f'{dataset_name}: Dimension Impact Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_dim_impact.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_dim_impact.png")


def plot_f1_auc_comparison(df, dataset_name, output_dir, raw_mode=False):
    """
    Plot 5: F1 and AUC scores comparison across dimensions
    """
    if df is None or 'f1_score' not in df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 Score
    ax1 = axes[0]
    for func in df['func'].unique():
        func_df = df[df['func'] == func]
        best_per_dim = func_df.groupby('embedding_dim')['f1_score'].max().reset_index()
        ax1.plot(best_per_dim['embedding_dim'], best_per_dim['f1_score'], 
                marker='o', linewidth=2, label=func.capitalize(),
                color=FUNC_COLORS.get(func, '#333'))
    
    ax1.set_xlabel('Embedding Dimension', fontsize=11)
    ax1.set_ylabel('F1 Score (weighted)', fontsize=11)
    ax1.set_title('Best F1 Score vs Dimension', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUC
    ax2 = axes[1]
    if 'auc' in df.columns and df['auc'].notna().any():
        for func in df['func'].unique():
            func_df = df[df['func'] == func]
            func_df_valid = func_df[func_df['auc'].notna()]
            if len(func_df_valid) > 0:
                best_per_dim = func_df_valid.groupby('embedding_dim')['auc'].max().reset_index()
                ax2.plot(best_per_dim['embedding_dim'], best_per_dim['auc'], 
                        marker='s', linewidth=2, label=func.capitalize(),
                        color=FUNC_COLORS.get(func, '#333'))
        
        ax2.set_xlabel('Embedding Dimension', fontsize=11)
        ax2.set_ylabel('AUC (weighted OvR)', fontsize=11)
        ax2.set_title('Best AUC vs Dimension', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'AUC data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('AUC vs Dimension', fontsize=12)
    
    fig.suptitle(f'{dataset_name}: F1 and AUC Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_f1_auc.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_f1_auc.png")


def plot_raw_vs_preprocessed_dimensions(all_data_prep, all_data_raw, output_dir):
    """
    NEW: Compare raw vs preprocessed across dimensions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, dataset_name in zip(axes, all_data_prep.keys()):
        df_prep = all_data_prep.get(dataset_name)
        df_raw = all_data_raw.get(dataset_name)
        
        if df_prep is None or df_raw is None:
            ax.text(0.5, 0.5, f'{dataset_name}\nNo comparison data', ha='center', va='center')
            ax.set_title(dataset_name)
            continue
        
        # Get best accuracy per dimension for each mode
        for func in set(df_prep['func'].unique()) & set(df_raw['func'].unique()):
            func_prep = df_prep[df_prep['func'] == func]
            best_prep = func_prep.groupby('embedding_dim')['accuracy'].max().reset_index()
            
            func_raw = df_raw[df_raw['func'] == func]
            best_raw = func_raw.groupby('embedding_dim')['accuracy'].max().reset_index()
            
            color = FUNC_COLORS.get(func, '#333')
            ax.plot(best_prep['embedding_dim'], best_prep['accuracy'],
                   marker='o', linewidth=2, label=f'{func} (Prep)', color=color, linestyle='-')
            ax.plot(best_raw['embedding_dim'], best_raw['accuracy'],
                   marker='s', linewidth=2, label=f'{func} (Raw)', color=color, linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Embedding Dimension', fontsize=11)
        ax.set_ylabel('Best Accuracy', fontsize=11)
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Dimension Analysis: Raw vs Preprocessed', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dimension_raw_vs_preprocessed.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Saved: dimension_raw_vs_preprocessed.png")


def create_summary_table(all_data, output_dir, raw_mode=False):
    """Create a summary table with best configurations for each dataset."""
    summary_rows = []
    
    for dataset_name, df in all_data.items():
        if df is None:
            continue
        
        # Best overall
        best_idx = df['accuracy'].idxmax()
        best_row = df.loc[best_idx]
        
        summary_rows.append({
            'Dataset': dataset_name,
            'Best Function': best_row['func'],
            'Best Classifier': best_row['classifier'],
            'Best Dimension': int(best_row['embedding_dim']),
            'Best Accuracy': best_row['accuracy'],
            'Best F1': best_row['f1_score'],
            'Generation Time (s)': best_row.get('generation_time', 'N/A')
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save as CSV
    summary_df.to_csv(f'{output_dir}/dimension_analysis_summary.csv', index=False)
    
    # Print table
    print("\n" + "="*100)
    print("DIMENSION ANALYSIS SUMMARY")
    print("="*100)
    print("\nClassifier Hyperparameters:")
    for clf, params in CLASSIFIER_HYPERPARAMS.items():
        print(f"  {clf}: {params}")
    print("\nBest Configurations:")
    print(summary_df.to_string(index=False))
    print("="*100)
    
    return summary_df


def main():
    print("="*80)
    print("FGSD DIMENSION ANALYSIS VISUALIZATION")
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
            plot_accuracy_vs_dimension(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_accuracy_vs_time(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_accuracy_vs_memory(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_classifier_comparison_heatmap(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_dimension_impact_summary(df_prep, dataset_name, output_dir, raw_mode=False)
            plot_f1_auc_comparison(df_prep, dataset_name, output_dir, raw_mode=False)
        
        # Raw
        print(f"\n📊 Processing {dataset_name} (Raw Embeddings)...")
        df_raw = load_data(paths['raw'])
        all_data_raw[dataset_name] = df_raw
        
        if df_raw is not None:
            output_dir_raw = get_output_dir(raw_mode=True)
            plot_accuracy_vs_dimension(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_accuracy_vs_time(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_accuracy_vs_memory(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_classifier_comparison_heatmap(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_dimension_impact_summary(df_raw, dataset_name, output_dir_raw, raw_mode=True)
            plot_f1_auc_comparison(df_raw, dataset_name, output_dir_raw, raw_mode=True)
    
    # Summary tables
    create_summary_table(all_data_preprocessed, OUTPUT_DIR, raw_mode=False)
    create_summary_table(all_data_raw, OUTPUT_DIR_RAW, raw_mode=True)
    
    # Comparison
    plot_raw_vs_preprocessed_dimensions(all_data_preprocessed, all_data_raw, OUTPUT_DIR)
    
    print(f"\n✅ All plots saved!")
    print(f"  📁 Preprocessed: {OUTPUT_DIR}/")
    print(f"  📁 Raw: {OUTPUT_DIR_RAW}/")


if __name__ == "__main__":
    main()
