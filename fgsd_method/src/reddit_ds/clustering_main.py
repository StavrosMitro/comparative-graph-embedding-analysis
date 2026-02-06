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


# =============================================================================
# DEFAULT CONFIGURATIONS - Based on classification results
# From fgsd_reddit_final_results.csv - top 3 bins per function by accuracy
# =============================================================================
# REDDIT Classification Results:
#   harmonic:   range=14.61, best bins: 500 (0.457 acc), 100 (0.447), 200 (0.444)
#   polynomial: range=3.48, best bins: 500 (0.443 acc), 200 (0.441), 100 (0.432)
#   biharmonic: range=500.0, best bins: 200 (0.406 acc), 500 (0.407), 100 (0.377)
# =============================================================================

DEFAULT_CONFIGS = [
    # Harmonic - top 3 bins from classification (range=14.61)
    {'name': 'harmonic_500_14.61', 'func': 'harmonic', 'bins': 500, 'range': 14.61},
    {'name': 'harmonic_100_14.61', 'func': 'harmonic', 'bins': 100, 'range': 14.61},
    {'name': 'harmonic_200_14.61', 'func': 'harmonic', 'bins': 200, 'range': 14.61},
    
    # Polynomial - top 3 bins from classification (range=3.48)
    {'name': 'polynomial_500_3.48', 'func': 'polynomial', 'bins': 500, 'range': 3.48},
    {'name': 'polynomial_200_3.48', 'func': 'polynomial', 'bins': 200, 'range': 3.48},
    {'name': 'polynomial_100_3.48', 'func': 'polynomial', 'bins': 100, 'range': 3.48},
    
    # Biharmonic - top 3 bins from classification (range=500.0)
    {'name': 'biharmonic_500_500', 'func': 'biharmonic', 'bins': 500, 'range': 500.0},
    {'name': 'biharmonic_200_500', 'func': 'biharmonic', 'bins': 200, 'range': 500.0},
    {'name': 'biharmonic_100_500', 'func': 'biharmonic', 'bins': 100, 'range': 500.0},
    
    # Naive hybrid - best harmonic (500) + best polynomial (500) from classification
    {
        'name': 'naive_hybrid_500_14.61_500_3.48',
        'func': 'naive_hybrid',
        'harm_bins': 500, 'harm_range': 14.61,
        'pol_bins': 500, 'pol_range': 3.48
    },
    
    # Biharmonic hybrid - best biharmonic (500) + best polynomial (500)
    {
        'name': 'biharmonic_hybrid_500_500_500_3.48',
        'func': 'biharmonic_hybrid',
        'biharm_bins': 500, 'biharm_range': 500.0,
        'pol_bins': 500, 'pol_range': 3.48
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
