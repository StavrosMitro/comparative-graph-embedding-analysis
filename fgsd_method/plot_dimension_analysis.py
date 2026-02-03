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
    'ENZYMES': 'src/results/fgsd_enzymes_dimension_analysis.csv',
    'IMDB-MULTI': 'src/results/fgsd_imdb_dimension_analysis.csv',
    'REDDIT-MULTI-12K': 'src/results/fgsd_reddit_dimension_analysis.csv',
}

# Output directory
OUTPUT_DIR = 'plots/dimension_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Color palettes
FUNC_COLORS = {'harmonic': '#2ecc71', 'polynomial': '#3498db', 'biharmonic': '#9b59b6'}
CLF_COLORS = {'SVM (RBF)': '#e74c3c', 'Random Forest': '#27ae60', 'MLP': '#f39c12'}


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


def plot_accuracy_vs_dimension(df, dataset_name, output_dir):
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
            
            color = CLF_COLORS.get(clf, '#333333')
            ax.plot(clf_df_sorted['embedding_dim'], clf_df_sorted['accuracy'], 
                   marker='o', linewidth=2, markersize=8, label=clf, color=color)
        
        ax.set_xlabel('Embedding Dimension (bins)', fontsize=11)
        ax.set_title(f'{func.capitalize()}', fontsize=12, fontweight='bold',
                    color=FUNC_COLORS.get(func, '#333333'))
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel('Test Accuracy', fontsize=11)
    
    fig.suptitle(f'{dataset_name}: Accuracy vs Embedding Dimension\n'
                 f'(Hyperparams: RF={CLASSIFIER_HYPERPARAMS["Random Forest"][:30]}...)', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_acc_vs_dim.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_acc_vs_dim.png")


def plot_accuracy_vs_time(df, dataset_name, output_dir):
    """
    Plot 2: Accuracy vs Generation Time (efficiency trade-off)
    - Shows which configurations give best accuracy per compute cost
    """
    if df is None or 'generation_time' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for func in df['func'].unique():
        func_df = df[df['func'] == func]
        color = FUNC_COLORS.get(func, '#333333')
        
        # Get best accuracy per dimension
        best_per_dim = func_df.groupby('embedding_dim').agg({
            'accuracy': 'max',
            'generation_time': 'first'
        }).reset_index()
        
        ax.scatter(best_per_dim['generation_time'], best_per_dim['accuracy'], 
                  s=100, alpha=0.7, label=func.capitalize(), color=color)
        
        # Annotate with dimension
        for _, row in best_per_dim.iterrows():
            ax.annotate(f'd={int(row["embedding_dim"])}', 
                       (row['generation_time'], row['accuracy']),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel('Embedding Generation Time (seconds)', fontsize=11)
    ax.set_ylabel('Best Test Accuracy', fontsize=11)
    ax.set_title(f'{dataset_name}: Accuracy vs Compute Cost Trade-off', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_acc_vs_time.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_acc_vs_time.png")


def plot_classifier_comparison_heatmap(df, dataset_name, output_dir):
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


def plot_dimension_impact_summary(df, dataset_name, output_dir):
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


def plot_f1_auc_comparison(df, dataset_name, output_dir):
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


def create_summary_table(all_data, output_dir):
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
    print("\nClassifier Hyperparameters (from code):")
    for clf, params in CLASSIFIER_HYPERPARAMS.items():
        print(f"  • {clf}: {params}")
    
    all_data = {}
    
    for dataset_name, path in FILES.items():
        print(f"\n📊 Processing {dataset_name}...")
        
        df = load_data(path)
        all_data[dataset_name] = df
        
        if df is None:
            continue
        
        print(f"  Loaded {len(df)} rows")
        print(f"  Functions: {df['func'].unique().tolist()}")
        print(f"  Dimensions: {sorted(df['embedding_dim'].unique())}")
        print(f"  Classifiers: {df['classifier'].unique().tolist()}")
        
        # Generate all plots
        plot_accuracy_vs_dimension(df, dataset_name, OUTPUT_DIR)
        plot_accuracy_vs_time(df, dataset_name, OUTPUT_DIR)
        plot_classifier_comparison_heatmap(df, dataset_name, OUTPUT_DIR)
        plot_dimension_impact_summary(df, dataset_name, OUTPUT_DIR)
        plot_f1_auc_comparison(df, dataset_name, OUTPUT_DIR)
    
    # Create summary
    create_summary_table(all_data, OUTPUT_DIR)
    
    print(f"\n✅ All plots saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  📈 {f}")


if __name__ == "__main__":
    main()
