"""
Stability Analysis Visualization for FGSD Experiments

This script creates comprehensive plots showing:
1. Embedding stability (cosine similarity) vs perturbation ratio
2. Classification accuracy drop under perturbation
3. Relative embedding change vs perturbation
4. Stability comparison across function types
5. Trade-off between accuracy and stability

Stability Metrics:
- Cosine Similarity: How similar embeddings remain after perturbation (higher = more stable)
- Relative Change: L2 distance normalized by original norm (lower = more stable)
- Accuracy Drop %: Classification accuracy degradation (lower = more robust)

Perturbation Modes:
- default: Mixed add/remove edges
- remove: Only remove edges
- add: Only add edges
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# File paths for stability results
FILES = {
    'ENZYMES': 'src/results/fgsd_enzymes_stability_results.csv',
    'IMDB-MULTI': 'src/results/fgsd_imdb_stability_results.csv',
    'REDDIT-MULTI-12K': 'src/results/fgsd_reddit_stability_results.csv',
}

# Output directory
OUTPUT_DIR = 'plots/stability_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Color palettes
FUNC_COLORS = {
    'harmonic': '#2ecc71', 
    'polynomial': '#3498db', 
    'biharmonic': '#9b59b6',
    'naive_hybrid': '#e74c3c',
    'hybrid': '#e74c3c',
    'biharmonic_hybrid': '#f39c12',
    'harmonic_with_labels': '#27ae60',
    'polynomial_with_labels': '#2980b9',
    'naive_hybrid_with_labels': '#c0392b',
}

MODE_MARKERS = {'default': 'o', 'remove': 's', 'add': '^'}
MODE_STYLES = {'default': '-', 'remove': '--', 'add': ':'}


def load_data(path):
    """Load stability analysis data."""
    if not os.path.exists(path):
        print(f"  ⚠️  File not found: {path}")
        return None
    
    df = pd.read_csv(path)
    
    # Standardize column names
    if 'ratio' not in df.columns and 'perturbation_ratio' in df.columns:
        df['ratio'] = df['perturbation_ratio']
    
    # Convert ratio to percentage for display
    if 'ratio' in df.columns:
        df['ratio_pct'] = df['ratio'] * 100
    
    return df


def get_func_color(func_name):
    """Get color for function type, handling variants."""
    for key, color in FUNC_COLORS.items():
        if key in func_name.lower():
            return color
    return '#333333'


def plot_cosine_similarity_vs_perturbation(df, dataset_name, output_dir):
    """
    Plot 1: Cosine Similarity vs Perturbation Ratio
    Shows how embedding stability degrades with increasing perturbation.
    """
    if df is None or 'mean_cosine_similarity' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique functions
    funcs = df['func'].unique()
    
    for func in funcs:
        func_df = df[df['func'] == func]
        
        # Group by ratio and mode if available
        if 'mode' in func_df.columns:
            for mode in func_df['mode'].unique():
                mode_df = func_df[func_df['mode'] == mode]
                mode_df_sorted = mode_df.sort_values('ratio')
                
                color = get_func_color(func)
                marker = MODE_MARKERS.get(mode, 'o')
                linestyle = MODE_STYLES.get(mode, '-')
                
                label = f"{func} ({mode})" if len(func_df['mode'].unique()) > 1 else func
                ax.plot(mode_df_sorted['ratio_pct'], mode_df_sorted['mean_cosine_similarity'],
                       marker=marker, linewidth=2, markersize=8, label=label,
                       color=color, linestyle=linestyle, alpha=0.8)
        else:
            func_df_sorted = func_df.sort_values('ratio')
            color = get_func_color(func)
            ax.plot(func_df_sorted['ratio_pct'], func_df_sorted['mean_cosine_similarity'],
                   marker='o', linewidth=2, markersize=8, label=func, color=color)
    
    ax.set_xlabel('Perturbation Ratio (%)', fontsize=12)
    ax.set_ylabel('Mean Cosine Similarity', fontsize=12)
    ax.set_title(f'{dataset_name}: Embedding Stability vs Perturbation\n(Higher = More Stable)', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(0.5, df['mean_cosine_similarity'].min() - 0.05), 1.02])
    
    # Add reference line at 1.0 (perfect stability)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect stability')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_cosine_stability.png',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_cosine_stability.png")


def plot_accuracy_drop_vs_perturbation(df, dataset_name, output_dir):
    """
    Plot 2: Classification Accuracy Drop vs Perturbation
    Shows how robust classification is to graph perturbations.
    """
    if df is None:
        return
    
    # Check for accuracy drop columns
    acc_drop_col = None
    for col in ['rf_acc_drop_pct', 'svm_acc_drop_pct', 'acc_drop_pct']:
        if col in df.columns:
            acc_drop_col = col
            break
    
    if acc_drop_col is None:
        print(f"  ⚠️  No accuracy drop column found for {dataset_name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    funcs = df['func'].unique()
    
    for func in funcs:
        func_df = df[df['func'] == func]
        func_df_sorted = func_df.groupby('ratio').agg({
            acc_drop_col: 'mean',
            'ratio_pct': 'first'
        }).reset_index()
        func_df_sorted = func_df_sorted.sort_values('ratio')
        
        color = get_func_color(func)
        ax.plot(func_df_sorted['ratio_pct'], func_df_sorted[acc_drop_col],
               marker='o', linewidth=2, markersize=8, label=func, color=color)
    
    ax.set_xlabel('Perturbation Ratio (%)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title(f'{dataset_name}: Classification Robustness\n(Lower Drop = More Robust)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add reference line at 0 (no drop)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_accuracy_drop.png',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_accuracy_drop.png")


def plot_relative_change_vs_perturbation(df, dataset_name, output_dir):
    """
    Plot 3: Relative Embedding Change vs Perturbation
    Shows normalized L2 distance between original and perturbed embeddings.
    """
    if df is None or 'mean_relative_change' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    funcs = df['func'].unique()
    
    for func in funcs:
        func_df = df[df['func'] == func]
        func_df_sorted = func_df.groupby('ratio').agg({
            'mean_relative_change': 'mean',
            'ratio_pct': 'first'
        }).reset_index()
        func_df_sorted = func_df_sorted.sort_values('ratio')
        
        color = get_func_color(func)
        ax.plot(func_df_sorted['ratio_pct'], func_df_sorted['mean_relative_change'],
               marker='s', linewidth=2, markersize=8, label=func, color=color)
    
    ax.set_xlabel('Perturbation Ratio (%)', fontsize=12)
    ax.set_ylabel('Mean Relative Change (L2/||x||)', fontsize=12)
    ax.set_title(f'{dataset_name}: Embedding Sensitivity\n(Lower = Less Sensitive)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_relative_change.png',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_relative_change.png")


def plot_stability_heatmap(df, dataset_name, output_dir):
    """
    Plot 4: Heatmap of stability metrics by function and perturbation ratio.
    """
    if df is None or 'mean_cosine_similarity' not in df.columns:
        return
    
    # Pivot for cosine similarity
    pivot = df.pivot_table(
        values='mean_cosine_similarity', 
        index='func', 
        columns='ratio',
        aggfunc='mean'
    )
    
    # Rename columns to percentages
    pivot.columns = [f'{int(c*100)}%' for c in pivot.columns]
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6)))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': 'Cosine Similarity'}, vmin=0.7, vmax=1.0,
                linewidths=0.5)
    
    ax.set_title(f'{dataset_name}: Stability Heatmap\n(Cosine Similarity by Function & Perturbation)',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Perturbation Ratio', fontsize=11)
    ax.set_ylabel('Function Type', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_stability_heatmap.png',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_stability_heatmap.png")


def plot_accuracy_original_vs_perturbed(df, dataset_name, output_dir):
    """
    Plot 5: Original vs Perturbed Accuracy scatter plot.
    Shows trade-off between original performance and robustness.
    """
    if df is None:
        return
    
    # Check for accuracy columns
    orig_col = None
    pert_col = None
    for orig, pert in [('rf_acc_original', 'rf_acc_perturbed'), 
                       ('svm_acc_original', 'svm_acc_perturbed')]:
        if orig in df.columns and pert in df.columns:
            orig_col, pert_col = orig, pert
            break
    
    if orig_col is None:
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    funcs = df['func'].unique()
    ratios = sorted(df['ratio'].unique())
    
    for func in funcs:
        func_df = df[df['func'] == func]
        color = get_func_color(func)
        
        for i, ratio in enumerate(ratios):
            ratio_df = func_df[func_df['ratio'] == ratio]
            if len(ratio_df) > 0:
                marker_size = 100 + i * 50  # Larger markers for higher perturbation
                ax.scatter(ratio_df[orig_col].mean(), ratio_df[pert_col].mean(),
                          s=marker_size, color=color, alpha=0.7,
                          label=f"{func} ({int(ratio*100)}%)" if i == 0 else None,
                          edgecolors='black', linewidths=0.5)
    
    # Add diagonal line (perfect robustness)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='No degradation')
    
    ax.set_xlabel('Original Accuracy', fontsize=12)
    ax.set_ylabel('Perturbed Accuracy', fontsize=12)
    ax.set_title(f'{dataset_name}: Original vs Perturbed Accuracy\n(Closer to diagonal = More Robust)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_acc_scatter.png',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_acc_scatter.png")


def plot_stability_by_mode(df, dataset_name, output_dir):
    """
    Plot 6: Compare stability across perturbation modes (add/remove/default).
    """
    if df is None or 'mode' not in df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Cosine similarity by mode
    ax1 = axes[0]
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        mode_agg = mode_df.groupby('ratio_pct')['mean_cosine_similarity'].mean().reset_index()
        ax1.plot(mode_agg['ratio_pct'], mode_agg['mean_cosine_similarity'],
                marker=MODE_MARKERS.get(mode, 'o'), linewidth=2, markersize=8,
                label=mode.capitalize(), linestyle=MODE_STYLES.get(mode, '-'))
    
    ax1.set_xlabel('Perturbation Ratio (%)', fontsize=11)
    ax1.set_ylabel('Mean Cosine Similarity', fontsize=11)
    ax1.set_title('Embedding Stability by Mode', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Accuracy drop by mode
    ax2 = axes[1]
    acc_drop_col = 'rf_acc_drop_pct' if 'rf_acc_drop_pct' in df.columns else None
    
    if acc_drop_col:
        for mode in df['mode'].unique():
            mode_df = df[df['mode'] == mode]
            mode_agg = mode_df.groupby('ratio_pct')[acc_drop_col].mean().reset_index()
            ax2.plot(mode_agg['ratio_pct'], mode_agg[acc_drop_col],
                    marker=MODE_MARKERS.get(mode, 'o'), linewidth=2, markersize=8,
                    label=mode.capitalize(), linestyle=MODE_STYLES.get(mode, '-'))
        
        ax2.set_xlabel('Perturbation Ratio (%)', fontsize=11)
        ax2.set_ylabel('Accuracy Drop (%)', fontsize=11)
        ax2.set_title('Classification Robustness by Mode', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    
    fig.suptitle(f'{dataset_name}: Stability by Perturbation Mode', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_mode_comparison.png',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_mode_comparison.png")


def plot_combined_stability_summary(df, dataset_name, output_dir):
    """
    Plot 7: Combined summary showing multiple stability metrics.
    """
    if df is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    funcs = df['func'].unique()
    
    # Top-left: Cosine similarity
    ax1 = axes[0, 0]
    for func in funcs:
        func_df = df[df['func'] == func]
        agg = func_df.groupby('ratio_pct')['mean_cosine_similarity'].mean().reset_index()
        ax1.plot(agg['ratio_pct'], agg['mean_cosine_similarity'],
                marker='o', linewidth=2, label=func, color=get_func_color(func))
    ax1.set_xlabel('Perturbation (%)')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Embedding Stability', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Relative change
    ax2 = axes[0, 1]
    if 'mean_relative_change' in df.columns:
        for func in funcs:
            func_df = df[df['func'] == func]
            agg = func_df.groupby('ratio_pct')['mean_relative_change'].mean().reset_index()
            ax2.plot(agg['ratio_pct'], agg['mean_relative_change'],
                    marker='s', linewidth=2, label=func, color=get_func_color(func))
        ax2.set_xlabel('Perturbation (%)')
        ax2.set_ylabel('Relative Change')
        ax2.set_title('Embedding Sensitivity', fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Accuracy drop
    ax3 = axes[1, 0]
    acc_drop_col = 'rf_acc_drop_pct' if 'rf_acc_drop_pct' in df.columns else None
    if acc_drop_col:
        for func in funcs:
            func_df = df[df['func'] == func]
            agg = func_df.groupby('ratio_pct')[acc_drop_col].mean().reset_index()
            ax3.plot(agg['ratio_pct'], agg[acc_drop_col],
                    marker='^', linewidth=2, label=func, color=get_func_color(func))
        ax3.set_xlabel('Perturbation (%)')
        ax3.set_ylabel('Accuracy Drop (%)')
        ax3.set_title('Classification Robustness', fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='green', linestyle='--', alpha=0.3)
    
    # Bottom-right: Bar chart of stability at 10% perturbation
    ax4 = axes[1, 1]
    target_ratio = 10  # 10% perturbation
    df_10pct = df[df['ratio_pct'].round() == target_ratio]
    
    if len(df_10pct) > 0:
        stability_10 = df_10pct.groupby('func')['mean_cosine_similarity'].mean().sort_values(ascending=False)
        colors = [get_func_color(f) for f in stability_10.index]
        bars = ax4.bar(range(len(stability_10)), stability_10.values, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_xticks(range(len(stability_10)))
        ax4.set_xticklabels(stability_10.index, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('Cosine Similarity')
        ax4.set_title(f'Stability Ranking at {target_ratio}% Perturbation', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, stability_10.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle(f'{dataset_name}: Stability Analysis Summary', fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.replace("-", "_").lower()}_stability_summary.png',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {dataset_name}_stability_summary.png")


def create_stability_summary_table(all_data, output_dir):
    """Create a summary table of stability metrics across datasets."""
    summary_rows = []
    
    for dataset_name, df in all_data.items():
        if df is None:
            continue
        
        for func in df['func'].unique():
            func_df = df[df['func'] == func]
            
            # Get metrics at different perturbation levels
            for ratio in [0.05, 0.10, 0.20]:
                ratio_df = func_df[func_df['ratio'].round(2) == ratio]
                if len(ratio_df) == 0:
                    continue
                
                row = {
                    'Dataset': dataset_name,
                    'Function': func,
                    'Perturbation': f'{int(ratio*100)}%',
                    'Cosine Similarity': ratio_df['mean_cosine_similarity'].mean(),
                    'Relative Change': ratio_df['mean_relative_change'].mean() if 'mean_relative_change' in ratio_df.columns else None,
                }
                
                # Add accuracy metrics if available
                if 'rf_acc_original' in ratio_df.columns:
                    row['RF Accuracy (Orig)'] = ratio_df['rf_acc_original'].mean()
                    row['RF Accuracy (Pert)'] = ratio_df['rf_acc_perturbed'].mean()
                    row['RF Accuracy Drop %'] = ratio_df['rf_acc_drop_pct'].mean()
                
                summary_rows.append(row)
    
    if not summary_rows:
        return None
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save as CSV
    summary_df.to_csv(f'{output_dir}/stability_analysis_summary.csv', index=False)
    
    # Print table
    print("\n" + "="*120)
    print("STABILITY ANALYSIS SUMMARY")
    print("="*120)
    print("\nKey Metrics (at 10% perturbation):")
    
    df_10 = summary_df[summary_df['Perturbation'] == '10%']
    if len(df_10) > 0:
        print(df_10.to_string(index=False))
    
    print("\n" + "="*120)
    
    return summary_df


def main():
    print("="*80)
    print("FGSD STABILITY ANALYSIS VISUALIZATION")
    print("="*80)
    
    all_data = {}
    
    for dataset_name, path in FILES.items():
        print(f"\n📊 Processing {dataset_name}...")
        
        df = load_data(path)
        all_data[dataset_name] = df
        
        if df is None:
            continue
        
        print(f"  Loaded {len(df)} rows")
        print(f"  Functions: {df['func'].unique().tolist()}")
        print(f"  Perturbation ratios: {sorted(df['ratio'].unique())}")
        if 'mode' in df.columns:
            print(f"  Modes: {df['mode'].unique().tolist()}")
        
        # Generate all plots
        plot_cosine_similarity_vs_perturbation(df, dataset_name, OUTPUT_DIR)
        plot_accuracy_drop_vs_perturbation(df, dataset_name, OUTPUT_DIR)
        plot_relative_change_vs_perturbation(df, dataset_name, OUTPUT_DIR)
        plot_stability_heatmap(df, dataset_name, OUTPUT_DIR)
        plot_accuracy_original_vs_perturbed(df, dataset_name, OUTPUT_DIR)
        plot_stability_by_mode(df, dataset_name, OUTPUT_DIR)
        plot_combined_stability_summary(df, dataset_name, OUTPUT_DIR)
    
    # Create summary table
    create_stability_summary_table(all_data, OUTPUT_DIR)
    
    print(f"\n✅ All plots saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  📈 {f}")


if __name__ == "__main__":
    main()
