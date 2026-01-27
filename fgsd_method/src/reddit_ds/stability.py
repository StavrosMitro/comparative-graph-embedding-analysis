"""
Stability Analysis for FGSD embeddings on REDDIT-MULTI-12K.
"""

import gc
import copy
from typing import List, Dict, Any

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD


# Default perturbation ratios
DEFAULT_PERTURBATION_RATIOS = [0.01, 0.05, 0.10, 0.20]


def perturb_graph_edges(
    graph: nx.Graph, 
    perturbation_ratio: float = 0.05,
    seed: int = 42
) -> nx.Graph:
    """Perturb a graph by randomly adding/removing edges."""
    np.random.seed(seed)
    G = copy.deepcopy(graph)
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_nodes < 2 or n_edges < 1:
        return G
    
    n_perturb = max(1, int(n_edges * perturbation_ratio))
    n_remove = n_perturb // 2
    n_add = n_perturb - n_remove
    
    edges = list(G.edges())
    if n_remove > 0 and len(edges) > 0:
        remove_indices = np.random.choice(len(edges), min(n_remove, len(edges)), replace=False)
        for idx in remove_indices:
            G.remove_edge(*edges[idx])
    
    nodes = list(G.nodes())
    added = 0
    max_attempts = n_add * 10
    attempts = 0
    while added < n_add and attempts < max_attempts:
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            added += 1
        attempts += 1
    
    return G


def perturb_graphs_batch(
    graphs: List[nx.Graph],
    perturbation_ratio: float = 0.05,
    seed: int = 42
) -> List[nx.Graph]:
    """Perturb a batch of graphs."""
    perturbed = []
    for i, g in enumerate(graphs):
        perturbed.append(perturb_graph_edges(g, perturbation_ratio, seed=seed + i))
    return perturbed


def compute_embedding_stability(
    X_original: np.ndarray,
    X_perturbed: np.ndarray,
    metric: str = 'cosine'
) -> Dict[str, float]:
    """Compute stability metrics between original and perturbed embeddings."""
    
    if metric == 'cosine':
        # OPTIMIZED: Vectorized computation instead of loop
        # Compute row-wise cosine similarity efficiently
        norms_orig = np.linalg.norm(X_original, axis=1, keepdims=True)
        norms_pert = np.linalg.norm(X_perturbed, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms_orig = np.maximum(norms_orig, 1e-10)
        norms_pert = np.maximum(norms_pert, 1e-10)
        
        # Row-wise dot product divided by norms = cosine similarity per row
        similarities = np.sum(X_original * X_perturbed, axis=1) / (norms_orig.flatten() * norms_pert.flatten())
        
        return {
            'mean_cosine_similarity': float(np.mean(similarities)),
            'std_cosine_similarity': float(np.std(similarities)),
            'min_cosine_similarity': float(np.min(similarities)),
            'median_cosine_similarity': float(np.median(similarities)),
        }
    
    elif metric == 'euclidean':
        distances = np.linalg.norm(X_original - X_perturbed, axis=1)
        original_norms = np.linalg.norm(X_original, axis=1)
        relative_changes = distances / (original_norms + 1e-10)
        
        return {
            'mean_l2_distance': float(np.mean(distances)),
            'mean_relative_change': float(np.mean(relative_changes)),
            'std_relative_change': float(np.std(relative_changes)),
        }
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _generate_base_embedding(graphs: List[nx.Graph], func_type: str, bins: int, range_val: float, seed: int) -> np.ndarray:
    """Generate embedding for a single function type."""
    model = FlexibleFGSD(
        hist_bins=bins,
        hist_range=range_val,
        func_type=func_type,
        seed=seed
    )
    model.fit(graphs)
    return model.get_embedding()


def generate_embeddings_for_graphs(
    graphs: List[nx.Graph],
    config: Dict[str, Any],
    seed: int = 42
) -> np.ndarray:
    """
    Generate FGSD embeddings for a list of graphs.
    OPTIMIZED: For hybrids, concatenates base embeddings instead of recomputing.
    """
    func = config['func']
    
    if func in ['hybrid', 'naive_hybrid']:
        # Generate harmonic and polynomial separately, then concatenate
        X_harm = _generate_base_embedding(
            graphs, 'harmonic', 
            config['harm_bins'], config['harm_range'], seed
        )
        X_poly = _generate_base_embedding(
            graphs, 'polynomial',
            config['pol_bins'], config['pol_range'], seed
        )
        return np.hstack([X_harm, X_poly])
    
    elif func == 'biharmonic_hybrid':
        # Generate biharmonic and polynomial separately, then concatenate
        X_biharm = _generate_base_embedding(
            graphs, 'biharmonic',
            config['biharm_bins'], config['biharm_range'], seed
        )
        X_poly = _generate_base_embedding(
            graphs, 'polynomial',
            config['pol_bins'], config['pol_range'], seed
        )
        return np.hstack([X_biharm, X_poly])
    
    else:
        # Single function type: harmonic, polynomial, biharmonic
        return _generate_base_embedding(
            graphs, func, config['bins'], config['range'], seed
        )


def compute_classification_stability(
    X_original_train: np.ndarray,
    X_original_test: np.ndarray,
    X_perturbed_train: np.ndarray,
    X_perturbed_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42
) -> Dict[str, float]:
    """Compare classification accuracy using Random Forest only."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    results = {}
    
    clf_rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1)
    clf_rf.fit(X_original_train, y_train)
    acc_orig_rf = accuracy_score(y_test, clf_rf.predict(X_original_test))
    
    clf_rf_pert = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1)
    clf_rf_pert.fit(X_perturbed_train, y_train)
    acc_pert_rf = accuracy_score(y_test, clf_rf_pert.predict(X_perturbed_test))
    
    results['rf_acc_original'] = acc_orig_rf
    results['rf_acc_perturbed'] = acc_pert_rf
    results['rf_acc_drop'] = acc_orig_rf - acc_pert_rf
    results['rf_acc_drop_pct'] = (acc_orig_rf - acc_pert_rf) / max(acc_orig_rf, 1e-10) * 100
    
    return results


def print_stability_summary(all_results: List[Dict[str, Any]]):
    """Print summary (RF only)."""
    print("\n" + "="*100)
    print("STABILITY ANALYSIS SUMMARY (Random Forest)")
    print("="*100)
    print(f"{'Config':<25} {'Perturb%':<10} {'CosineSim':<12} {'RelChange':<12} "
          f"{'RF_Orig':<10} {'RF_Pert':<10} {'RF_Drop%':<10}")
    print("-"*100)
    
    for result in all_results:
        config_name = result['config'].get('name', result['config']['func'])
        for pr in result['perturbation_results']:
            print(f"{config_name:<25} {pr['ratio']*100:<10.0f} "
                  f"{pr['mean_cosine_similarity']:<12.4f} {pr['mean_relative_change']:<12.4f} "
                  f"{pr.get('rf_acc_original', 0):<10.4f} {pr.get('rf_acc_perturbed', 0):<10.4f} "
                  f"{pr.get('rf_acc_drop_pct', 0):<10.2f}")
