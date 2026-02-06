"""
FGSD Clustering on IMDB-MULTI
Main entry point script.

Usage:
    python -m imbd_ds.clustering_main
    python -m imbd_ds.clustering_main --no-grid-search
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

from imbd_ds.config import RESULTS_DIR, DATASET_DIR
from imbd_ds.data_loader import ensure_dataset_ready, load_all_graphs
from imbd_ds.clustering import (
    run_clustering_experiment,
    print_clustering_summary
)


# =============================================================================
# DEFAULT CONFIGURATIONS - Based on classification results
# From fgsd_imdb_final_results.csv - top 3 bins per function by accuracy
# =============================================================================
# IMDB Classification Results:
#   harmonic:   range=3.52, best bins: 150 (0.489 acc), 200 (0.484), 100 (0.471)
#   polynomial: range=3.13, best bins: 200 (0.498 acc), 150 (0.493), 100 (0.480)
# =============================================================================

DEFAULT_CONFIGS = [
    # Harmonic - top 3 bins from classification (range=3.52)
    {'name': 'harmonic_150_3.52', 'func': 'harmonic', 'bins': 150, 'range': 3.52},
    {'name': 'harmonic_200_3.52', 'func': 'harmonic', 'bins': 200, 'range': 3.52},
    {'name': 'harmonic_100_3.52', 'func': 'harmonic', 'bins': 100, 'range': 3.52},
    
    # Polynomial - top 3 bins from classification (range=3.13)
    {'name': 'polynomial_200_3.13', 'func': 'polynomial', 'bins': 200, 'range': 3.13},
    {'name': 'polynomial_150_3.13', 'func': 'polynomial', 'bins': 150, 'range': 3.13},
    {'name': 'polynomial_100_3.13', 'func': 'polynomial', 'bins': 100, 'range': 3.13},
    
    # Naive hybrid - best harmonic (150) + best polynomial (200) from classification
    {
        'name': 'naive_hybrid_150_3.52_200_3.13',
        'func': 'hybrid',
        'harm_bins': 150, 'harm_range': 3.52,
        'pol_bins': 200, 'pol_range': 3.13
    },
]


def main(configs=None, neighbor_values=None, run_grid_search=True):
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
    print("FGSD CLUSTERING ON IMDB-MULTI")
    print(f"Grid search: {'ENABLED' if run_grid_search else 'DISABLED'}")
    print(f"Normalizations tested: l2, standard, none (raw)")
    print(f"UMAP: {'ENABLED (v' + umap_version + ')' if has_umap else 'DISABLED (install umap-learn)'}")
    print("="*80)
    
    ensure_dataset_ready()
    graphs, labels = load_all_graphs(DATASET_DIR)
    print(f"\nLoaded {len(graphs)} graphs with {len(np.unique(labels))} classes")
    
    results = run_clustering_experiment(
        graphs=graphs,
        labels=labels,
        configs=configs,
        neighbor_values=neighbor_values,
        visualize=True,
        run_grid_search=run_grid_search,
        save_umap_coords=True  # Save UMAP coordinates
    )
    
    print_clustering_summary(results)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    suffix = '_gridsearch' if run_grid_search else ''
    output_path = os.path.join(RESULTS_DIR, f'fgsd_imdb_clustering_results{suffix}.csv')
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
    parser = argparse.ArgumentParser(description='FGSD Clustering on IMDB-MULTI')
    parser.add_argument('--no-grid-search', action='store_true', help='Disable clustering grid search')
    args = parser.parse_args()
    
    main(run_grid_search=not args.no_grid_search)
