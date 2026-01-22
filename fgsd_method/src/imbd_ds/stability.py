"""
Stability Analysis for FGSD embeddings on IMDB-MULTI.
Measures how embeddings change under graph perturbations.
"""

import gc
import time
import copy
from typing import List, Dict, Any, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

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
    n_samples = X_original.shape[0]
    
    if metric == 'cosine':
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


def run_stability_analysis(
    graphs: List[nx.Graph],
    labels: np.ndarray,
    config: Dict[str, Any],
    perturbation_ratios: List[float] = None,
    X_original: np.ndarray = None,
    seed: int = 42,
    test_size: float = 0.15,
    compute_classification: bool = True
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Run stability analysis with classification comparison."""
    if perturbation_ratios is None:
        perturbation_ratios = DEFAULT_PERTURBATION_RATIOS
    
    print(f"\n{'='*60}")
    print(f"STABILITY ANALYSIS: {config.get('name', config['func'])}")
    print(f"Perturbation ratios: {[f'{r*100:.0f}%' for r in perturbation_ratios]}")
    print(f"{'='*60}")
    
    if X_original is None:
        X_original = generate_embeddings_for_graphs(graphs, config, seed)
    
    # Split for classification
    if compute_classification:
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(graphs))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=labels)
        X_orig_train, X_orig_test = X_original[train_idx], X_original[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
    
    results = {'config': config, 'original_shape': X_original.shape, 'perturbation_results': []}
    
    for ratio in perturbation_ratios:
        print(f"\n--- Perturbation: {ratio*100:.0f}% ---")
        
        perturbed_graphs = perturb_graphs_batch(graphs, ratio, seed)
        X_perturbed = generate_embeddings_for_graphs(perturbed_graphs, config, seed)
        
        stability_cosine = compute_embedding_stability(X_original, X_perturbed, 'cosine')
        stability_euclidean = compute_embedding_stability(X_original, X_perturbed, 'euclidean')
        
        result_entry = {'ratio': ratio, **stability_cosine, **stability_euclidean, 'X_perturbed': X_perturbed}
        
        if compute_classification:
            clf_stability = compute_classification_stability(
                X_orig_train, X_orig_test,
                X_perturbed[train_idx], X_perturbed[test_idx],
                y_train, y_test, seed
            )
            result_entry.update(clf_stability)
            print(f"  SVM: {clf_stability['svm_acc_original']:.4f} -> {clf_stability['svm_acc_perturbed']:.4f} (drop: {clf_stability['svm_acc_drop_pct']:.2f}%)")
            print(f"  RF:  {clf_stability['rf_acc_original']:.4f} -> {clf_stability['rf_acc_perturbed']:.4f} (drop: {clf_stability['rf_acc_drop_pct']:.2f}%)")
        
        results['perturbation_results'].append(result_entry)
        del perturbed_graphs
        gc.collect()
    
    return results, X_original


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
