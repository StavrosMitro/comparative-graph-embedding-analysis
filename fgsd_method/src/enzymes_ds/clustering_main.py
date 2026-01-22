"""
FGSD Clustering on ENZYMES
Main entry point script.

Usage:
    python -m enzymes_ds.clustering_main
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

from enzymes_ds.config import RESULTS_DIR, DATASET_DIR
from enzymes_ds.data_loader import ensure_dataset_ready, load_all_graphs
from enzymes_ds.clustering import (
    run_clustering_experiment,
    print_clustering_summary
)


DEFAULT_CONFIGS = [
    {
        'name': 'hybrid_100_33_100_4.1',
        'func': 'hybrid',
        'harm_bins': 100,
        'harm_range': 33,
        'pol_bins': 100,
        'pol_range': 4.1
    },
    {
        'name': 'polynomial_100_4.1',
        'func': 'polynomial',
        'bins': 100,
        'range': 4.1
    },
    {
        'name': 'harmonic_100_33',
        'func': 'harmonic',
        'bins': 100,
        'range': 33
    },
]


def main(configs=None, neighbor_values=None, use_node_labels=True):
    if configs is None:
        configs = DEFAULT_CONFIGS
    if neighbor_values is None:
        neighbor_values = [10, 20]
    
    print("="*80)
    print("FGSD CLUSTERING ON ENZYMES")
    if use_node_labels:
        print("*** NODE LABELS: ENABLED ***")
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
    output_path = os.path.join(RESULTS_DIR, 'fgsd_enzymes_clustering_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    del graphs
    gc.collect()
    
    return results


if __name__ == "__main__":
    main()
