"""
FGSD Clustering on ENZYMES
Main entry point script.

Usage:
    python -m enzymes_ds.clustering_main
    python -m enzymes_ds.clustering_main --no-node-labels
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


# Default configurations based on preanalysis results
# ENZYMES: harmonic range ~30, polynomial range ~4, biharmonic range ~200 (smaller graphs than Reddit)
DEFAULT_CONFIGS = [
    # Harmonic configurations
    {'name': 'harmonic_50_30', 'func': 'harmonic', 'bins': 50, 'range': 30.0},
    {'name': 'harmonic_100_30', 'func': 'harmonic', 'bins': 100, 'range': 30.0},
    # Polynomial configurations
    {'name': 'polynomial_50_4', 'func': 'polynomial', 'bins': 50, 'range': 4.0},
    {'name': 'polynomial_100_4', 'func': 'polynomial', 'bins': 100, 'range': 4.0},
    # Biharmonic configurations (1/λ² - larger range than harmonic)
    {'name': 'biharmonic_50_200', 'func': 'biharmonic', 'bins': 50, 'range': 200.0},
    {'name': 'biharmonic_100_200', 'func': 'biharmonic', 'bins': 100, 'range': 200.0},
    # Naive hybrid (harmonic + polynomial)
    {
        'name': 'naive_hybrid_100_30_100_4',
        'func': 'hybrid',
        'harm_bins': 100, 'harm_range': 30.0,
        'pol_bins': 100, 'pol_range': 4.0
    },
]


def main(configs=None, neighbor_values=None, use_node_labels=True):
    if configs is None:
        configs = DEFAULT_CONFIGS
    if neighbor_values is None:
        neighbor_values = [5, 10, 15, 20]
    
    print("="*80)
    print("FGSD CLUSTERING ON ENZYMES")
    print(f"Node labels: {'ENABLED' if use_node_labels else 'DISABLED'}")
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
        visualize=True
    )
    
    print_clustering_summary(results)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    suffix = '_with_labels' if use_node_labels else '_spectral_only'
    output_path = os.path.join(RESULTS_DIR, f'fgsd_enzymes_clustering_results{suffix}.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    del graphs
    gc.collect()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FGSD Clustering on ENZYMES')
    parser.add_argument('--no-node-labels', action='store_true', help='Disable node labels')
    args = parser.parse_args()
    
    main(use_node_labels=not args.no_node_labels)
