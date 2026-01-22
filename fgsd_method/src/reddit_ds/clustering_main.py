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

# Setup paths
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


# Default configurations
DEFAULT_CONFIGS = [
    {
        'name': 'hybrid_500_14.6_500_3.5',
        'func': 'hybrid',
        'harm_bins': 500,
        'harm_range': 14.6,
        'pol_bins': 500,
        'pol_range': 3.5
    },
    {
        'name': 'polynomial_500_3.5',
        'func': 'polynomial',
        'bins': 500,
        'range': 3.5
    },
    {
        'name': 'harmonic_500_14.6',
        'func': 'harmonic',
        'bins': 500,
        'range': 14.6
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
        neighbor_values = [10, 20]
    
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
