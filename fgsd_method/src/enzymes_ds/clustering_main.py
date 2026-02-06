"""
FGSD Clustering on ENZYMES
Main entry point script.

Usage:
    python -m enzymes_ds.clustering_main
    python -m enzymes_ds.clustering_main --no-node-labels
    python -m enzymes_ds.clustering_main --no-grid-search
"""

import os
import sys
import warnings
import gc
import argparse

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

warnings.filterwarnings('ignore')

from enzymes_ds.config import RESULTS_DIR, DATASET_DIR
from enzymes_ds.data_loader import ensure_dataset_ready, load_all_graphs
from enzymes_ds.clustering import (
    run_clustering_experiment,
    print_clustering_summary
)


# =============================================================================
# DEFAULT CONFIGURATIONS - Based on classification results
# From fgsd_enzymes_final_results.csv - top 3 bins per function by accuracy
# =============================================================================
# ENZYMES Classification Results:
#   harmonic:   range=30.3, best bins: 150 (0.456 acc), 50 (0.433), 100 (0.444)
#   polynomial: range=3.81, best bins: 150 (0.544 acc), 100 (0.533), 50 (0.500)
# =============================================================================

DEFAULT_CONFIGS = [
    # Harmonic - top 3 bins from classification (range=30.3)
    {'name': 'harmonic_150_30.3', 'func': 'harmonic', 'bins': 150, 'range': 30.3},
    {'name': 'harmonic_100_30.3', 'func': 'harmonic', 'bins': 100, 'range': 30.3},
    {'name': 'harmonic_50_30.3', 'func': 'harmonic', 'bins': 50, 'range': 30.3},
    
    # Polynomial - top 3 bins from classification (range=3.81)
    {'name': 'polynomial_150_3.81', 'func': 'polynomial', 'bins': 150, 'range': 3.81},
    {'name': 'polynomial_100_3.81', 'func': 'polynomial', 'bins': 100, 'range': 3.81},
    {'name': 'polynomial_50_3.81', 'func': 'polynomial', 'bins': 50, 'range': 3.81},
    
    # Naive hybrid - best harmonic (150) + best polynomial (100) from classification
    {
        'name': 'naive_hybrid_150_30.3_100_3.81',
        'func': 'hybrid',
        'harm_bins': 150, 'harm_range': 30.3,
        'pol_bins': 100, 'pol_range': 3.81
    },
]


def main(configs=None, neighbor_values=None, use_node_labels=True, run_grid_search=True):
    if configs is None:
        configs = DEFAULT_CONFIGS
    if neighbor_values is None:
        neighbor_values = [5, 10, 15, 20]
    
    # Check for UMAP
    try:
        import umap
        has_umap = True
        umap_version = umap.__version__
    except ImportError:
        has_umap = False
        umap_version = None
    
    print("="*80)
    print("FGSD CLUSTERING ON ENZYMES")
    print(f"Node labels: {'ENABLED' if use_node_labels else 'DISABLED'}")
    print(f"Grid search: {'ENABLED' if run_grid_search else 'DISABLED'}")
    print(f"Normalizations tested: l2, standard, none (raw)")
    print(f"UMAP: {'ENABLED (v' + umap_version + ')' if has_umap else 'DISABLED'}")
    print("="*80)
    
    ensure_dataset_ready()
    graphs, labels, node_labels_list = load_all_graphs(DATASET_DIR)
    print(f"\nLoaded {len(graphs)} graphs with {len(np.unique(labels))} classes")
    
    results = run_clustering_experiment(
        graphs=graphs,
        labels=labels,
        configs=configs,
        node_labels_list=node_labels_list if use_node_labels else None,
        neighbor_values=neighbor_values,
        visualize=True,
        run_grid_search=run_grid_search,
        save_umap_coords=True  # Save UMAP coordinates
    )
    
    print_clustering_summary(results)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    suffix = '_with_labels' if use_node_labels else '_spectral_only'
    if run_grid_search:
        suffix += '_gridsearch'
    output_path = os.path.join(RESULTS_DIR, f'fgsd_enzymes_clustering_results{suffix}.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary of 'none' normalization results (raw embeddings)
    none_results = df[df['normalization'] == 'none'] if 'normalization' in df.columns else pd.DataFrame()
    if len(none_results) > 0:
        print("\n" + "="*80)
        print("RAW EMBEDDINGS (no normalization) RESULTS:")
        print("="*80)
        best_none = none_results.loc[none_results['ari'].idxmax()]
        print(f"  Best ARI: {best_none['ari']:.4f}")
        print(f"  Method: {best_none['method']}")
        print(f"  Config: {best_none['config_name']}")
    
    del graphs
    gc.collect()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FGSD Clustering on ENZYMES')
    parser.add_argument('--no-node-labels', action='store_true', help='Disable node labels')
    parser.add_argument('--no-grid-search', action='store_true', help='Disable clustering grid search')
    args = parser.parse_args()
    
    main(use_node_labels=not args.no_node_labels, run_grid_search=not args.no_grid_search)
