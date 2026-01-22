"""
Stability Analysis for FGSD embeddings on REDDIT-MULTI-12K.
"""

import gc
import time
import copy
from typing import List, Dict, Any, Tuple

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from optimized_method import HybridFGSD


# Default perturbation ratios
DEFAULT_PERTURBATION_RATIOS = [0.01, 0.05, 0.10, 0.20]


def perturb_graph_edges(
    graph: nx.Graph, 
    perturbation_ratio: float = 0.05,
    seed: int = 42
) -> nx.Graph:
    """
    Perturb a graph by randomly adding/removing edges.
    
    Args:
        graph: Original NetworkX graph
        perturbation_ratio: Fraction of edges to perturb (add + remove)
        seed: Random seed for reproducibility
    
    Returns:
        Perturbed copy of the graph
    """
    np.random.seed(seed)
    G = copy.deepcopy(graph)
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_nodes < 2 or n_edges < 1:
        return G
    
    n_perturb = max(1, int(n_edges * perturbation_ratio))
    n_remove = n_perturb // 2
    n_add = n_perturb - n_remove
    
    # Remove random edges
    edges = list(G.edges())
    if n_remove > 0 and len(edges) > 0:
        remove_indices = np.random.choice(len(edges), min(n_remove, len(edges)), replace=False)
        for idx in remove_indices:
            G.remove_edge(*edges[idx])
    
    # Add random edges (non-existing)
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
    """
    Compute stability metrics between original and perturbed embeddings.
    
    Args:
        X_original: Original embeddings (n_samples, n_features)
        X_perturbed: Perturbed embeddings (n_samples, n_features)
        metric: Similarity metric ('cosine', 'euclidean')
    
    Returns:
        Dictionary with stability metrics
    """
    n_samples = X_original.shape[0]
    
    if metric == 'cosine':
        # Compute pairwise cosine similarity for corresponding samples
        similarities = []
        for i in range(n_samples):
            sim = cosine_similarity(
                X_original[i:i+1], 
                X_perturbed[i:i+1]
            )[0, 0]
            similarities.append(sim)
        similarities = np.array(similarities)
        
        return {
            'mean_cosine_similarity': float(np.mean(similarities)),
            'std_cosine_similarity': float(np.std(similarities)),
            'min_cosine_similarity': float(np.min(similarities)),
            'median_cosine_similarity': float(np.median(similarities)),
        }
    
    elif metric == 'euclidean':
        # Compute L2 distance
        distances = np.linalg.norm(X_original - X_perturbed, axis=1)
        
        # Normalize by original embedding norm
        original_norms = np.linalg.norm(X_original, axis=1)
        relative_changes = distances / (original_norms + 1e-10)
        
        return {
            'mean_l2_distance': float(np.mean(distances)),
            'mean_relative_change': float(np.mean(relative_changes)),
            'std_relative_change': float(np.std(relative_changes)),
        }
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def generate_embeddings_for_graphs(
    graphs: List[nx.Graph],
    config: Dict[str, Any],
    seed: int = 42
) -> np.ndarray:
    """Generate FGSD embeddings for a list of graphs."""
    func = config['func']
    
    if func in ['hybrid', 'naive_hybrid']:
        model = HybridFGSD(
            harm_bins=config['harm_bins'],
            harm_range=config['harm_range'],
            pol_bins=config['pol_bins'],
            pol_range=config['pol_range'],
            func_type='hybrid',
            seed=seed
        )
    else:
        model = FlexibleFGSD(
            hist_bins=config['bins'],
            hist_range=config['range'],
            func_type=func,
            seed=seed
        )
    
    model.fit(graphs)
    return model.get_embedding()


def run_stability_analysis(
    graphs: List[nx.Graph],
    labels: np.ndarray,
    config: Dict[str, Any],
    perturbation_ratios: List[float] = [0.01, 0.05, 0.10],
    X_original: np.ndarray = None,
    seed: int = 42
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Run stability analysis on a configuration.
    
    Args:
        graphs: List of original graphs
        labels: Ground truth labels
        config: FGSD configuration
        perturbation_ratios: List of perturbation ratios to test
        X_original: Pre-computed original embeddings (optional, will compute if None)
        seed: Random seed
    
    Returns:
        Tuple of (stability_results dict, original_embeddings if computed)
    """
    print(f"\n{'='*60}")
    print(f"STABILITY ANALYSIS: {config.get('name', config['func'])}")
    print(f"{'='*60}")
    
    # Compute original embeddings if not provided
    if X_original is None:
        print("Computing original embeddings...")
        start_time = time.time()
        X_original = generate_embeddings_for_graphs(graphs, config, seed)
        orig_time = time.time() - start_time
        print(f"  -> Shape: {X_original.shape}, Time: {orig_time:.2f}s")
    else:
        print(f"Using pre-computed embeddings. Shape: {X_original.shape}")
    
    results = {
        'config': config,
        'original_shape': X_original.shape,
        'perturbation_results': []
    }
    
    for ratio in perturbation_ratios:
        print(f"\n--- Perturbation Ratio: {ratio*100:.0f}% ---")
        
        # Perturb graphs
        print(f"  Perturbing {len(graphs)} graphs...")
        perturbed_graphs = perturb_graphs_batch(graphs, ratio, seed)
        
        # Compute perturbed embeddings
        print(f"  Computing perturbed embeddings...")
        start_time = time.time()
        X_perturbed = generate_embeddings_for_graphs(perturbed_graphs, config, seed)
        perturb_time = time.time() - start_time
        
        # Compute stability metrics
        stability_cosine = compute_embedding_stability(X_original, X_perturbed, 'cosine')
        stability_euclidean = compute_embedding_stability(X_original, X_perturbed, 'euclidean')
        
        print(f"  Stability Metrics:")
        print(f"    -> Mean Cosine Similarity: {stability_cosine['mean_cosine_similarity']:.4f}")
        print(f"    -> Std Cosine Similarity:  {stability_cosine['std_cosine_similarity']:.4f}")
        print(f"    -> Mean Relative Change:   {stability_euclidean['mean_relative_change']:.4f}")
        
        results['perturbation_results'].append({
            'ratio': ratio,
            'perturb_time': perturb_time,
            **stability_cosine,
            **stability_euclidean,
            'X_perturbed': X_perturbed  # Store for classification comparison
        })
        
        # Cleanup
        del perturbed_graphs
        gc.collect()
    
    return results, X_original


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
