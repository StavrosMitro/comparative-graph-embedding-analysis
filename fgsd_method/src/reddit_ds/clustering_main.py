"""
FGSD Clustering on REDDIT-MULTI-12K
Main entry point script.

Usage:
    python -m reddit_ds.clustering_main
"""

import os
import sys
import warnings
import gc

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

warnings.filterwarnings('ignore')

from reddit_ds.config import RESULTS_DIR, DATASET_DIR
from reddit_ds.data_loader import ensure_dataset_ready, load_all_graphs
from reddit_ds.clustering import (
    run_clustering_experiment,
    print_clustering_summary
)


# Default configurations based on preanalysis results
# REDDIT: harmonic range ~14.6, polynomial range ~3.5, biharmonic range ~500 (1/λ² gives much larger values)
DEFAULT_CONFIGS = [
    # Harmonic configurations (range ~15)
    {'name': 'harmonic_100_15', 'func': 'harmonic', 'bins': 100, 'range': 15.0},
    {'name': 'harmonic_200_15', 'func': 'harmonic', 'bins': 200, 'range': 15.0},
    # Polynomial configurations (range ~3.5)
    {'name': 'polynomial_50_3.5', 'func': 'polynomial', 'bins': 50, 'range': 3.5},
    {'name': 'polynomial_100_3.5', 'func': 'polynomial', 'bins': 100, 'range': 3.5},
    # Biharmonic configurations (range ~500, because 1/λ² >> 1/λ)
    {'name': 'biharmonic_100_500', 'func': 'biharmonic', 'bins': 100, 'range': 500.0},
    {'name': 'biharmonic_200_500', 'func': 'biharmonic', 'bins': 200, 'range': 500.0},
    # Naive hybrid (harmonic + polynomial) - use 'naive_hybrid' consistently
    {
        'name': 'naive_hybrid_200_15_100_3.5',
        'func': 'naive_hybrid',  # Changed from 'hybrid' for consistency
        'harm_bins': 200, 'harm_range': 15.0,
        'pol_bins': 100, 'pol_range': 3.5
    },
    # Biharmonic hybrid (biharmonic + polynomial)
    {
        'name': 'biharmonic_hybrid_200_500_100_3.5',
        'func': 'biharmonic_hybrid',
        'biharm_bins': 200, 'biharm_range': 500.0,
        'pol_bins': 100, 'pol_range': 3.5
    },
]


def main(configs=None, neighbor_values=None):
    """
    Run clustering experiments on REDDIT-MULTI-12K.
    
    Args:
        configs: List of configuration dicts (uses DEFAULT_CONFIGS if None)
        neighbor_values: List of n_neighbors values to try
    """
    if configs is None:
        configs = DEFAULT_CONFIGS
    if neighbor_values is None:
        neighbor_values = [10, 20, 30]  # Larger values for bigger dataset
    
    print("="*80)
    print("FGSD CLUSTERING ON REDDIT-MULTI-12K")
    print("="*80)
    
    # Ensure dataset is ready
    ensure_dataset_ready()
    
    # Load all graphs
    graphs, labels = load_all_graphs(DATASET_DIR)
    print(f"\nLoaded {len(graphs)} graphs with {len(np.unique(labels))} classes")
    
    # Run clustering experiments
    results = run_clustering_experiment(
        graphs=graphs,
        labels=labels,
        configs=configs,
        neighbor_values=neighbor_values,
        visualize=True
    )
    
    # Print summary
    print_clustering_summary(results)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_clustering_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Cleanup
    del graphs
    gc.collect()
    
    return results


if __name__ == "__main__":
    main()
