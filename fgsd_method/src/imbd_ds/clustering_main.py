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


# Default configurations based on preanalysis results
# IMDB: harmonic range ~3.5, polynomial range ~3.1
DEFAULT_CONFIGS = [
    # Harmonic configurations
    {'name': 'harmonic_35_3.5', 'func': 'harmonic', 'bins': 35, 'range': 3.5},
    {'name': 'harmonic_70_3.5', 'func': 'harmonic', 'bins': 70, 'range': 3.5},
    # Polynomial configurations
    {'name': 'polynomial_31_3.1', 'func': 'polynomial', 'bins': 31, 'range': 3.1},
    {'name': 'polynomial_62_3.1', 'func': 'polynomial', 'bins': 62, 'range': 3.1},
    # Naive hybrid (harmonic + polynomial)
    {
        'name': 'naive_hybrid_70_3.5_62_3.1',
        'func': 'hybrid',
        'harm_bins': 70, 'harm_range': 3.5,
        'pol_bins': 62, 'pol_range': 3.1
    },
]


def main(configs=None, neighbor_values=None, run_grid_search=True):
    if configs is None:
        configs = DEFAULT_CONFIGS
    if neighbor_values is None:
        neighbor_values = [5, 10, 15, 20]
    
    print("="*80)
    print("FGSD CLUSTERING ON IMDB-MULTI")
    print(f"Grid search: {'ENABLED' if run_grid_search else 'DISABLED'}")
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
        run_grid_search=run_grid_search
    )
    
    print_clustering_summary(results)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    suffix = '_gridsearch' if run_grid_search else ''
    output_path = os.path.join(RESULTS_DIR, f'fgsd_imdb_clustering_results{suffix}.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    del graphs
    gc.collect()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FGSD Clustering on IMDB-MULTI')
    parser.add_argument('--no-grid-search', action='store_true', help='Disable clustering grid search')
    args = parser.parse_args()
    
    main(run_grid_search=not args.no_grid_search)
