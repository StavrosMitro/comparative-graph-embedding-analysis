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


def perturb_graph_edges_add_only(
    graph: nx.Graph, 
    perturbation_ratio: float = 0.05,
    seed: int = 42
) -> nx.Graph:
    """Perturb a graph by ONLY adding edges."""
    np.random.seed(seed)
    G = copy.deepcopy(graph)
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_nodes < 2:
        return G
    
    n_add = max(1, int(n_edges * perturbation_ratio))
    
    nodes = list(G.nodes())
    added = 0
    max_attempts = n_add * 20
    attempts = 0
    while added < n_add and attempts < max_attempts:
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            added += 1
        attempts += 1
    
    return G


def perturb_graph_edges_remove_only(
    graph: nx.Graph, 
    perturbation_ratio: float = 0.05,
    seed: int = 42
) -> nx.Graph:
    """Perturb a graph by ONLY removing edges."""
    np.random.seed(seed)
    G = copy.deepcopy(graph)
    
    n_edges = G.number_of_edges()
    
    if n_edges < 1:
        return G
    
    n_remove = max(1, int(n_edges * perturbation_ratio))
    
    edges = list(G.edges())
    if n_remove > 0 and len(edges) > 0:
        remove_indices = np.random.choice(len(edges), min(n_remove, len(edges)), replace=False)
        for idx in remove_indices:
            G.remove_edge(*edges[idx])
    
    return G


def perturb_graphs_batch_add_only(
    graphs: List[nx.Graph],
    perturbation_ratio: float = 0.05,
    seed: int = 42
) -> List[nx.Graph]:
    """Perturb a batch of graphs by adding edges only."""
    perturbed = []
    for i, g in enumerate(graphs):
        perturbed.append(perturb_graph_edges_add_only(g, perturbation_ratio, seed=seed + i))
    return perturbed


def perturb_graphs_batch_remove_only(
    graphs: List[nx.Graph],
    perturbation_ratio: float = 0.05,
    seed: int = 42
) -> List[nx.Graph]:
    """Perturb a batch of graphs by removing edges only."""
    perturbed = []
    for i, g in enumerate(graphs):
        perturbed.append(perturb_graph_edges_remove_only(g, perturbation_ratio, seed=seed + i))
    return perturbed


def compute_embedding_stability(
    X_original: np.ndarray,
    X_perturbed: np.ndarray,
    metric: str = 'cosine'
) -> Dict[str, float]:
    """Compute stability metrics between original and perturbed embeddings."""
    
    if metric == 'cosine':
        # OPTIMIZED: Vectorized computation instead of loop
        norms_orig = np.linalg.norm(X_original, axis=1, keepdims=True)
        norms_pert = np.linalg.norm(X_perturbed, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms_orig = np.maximum(norms_orig, 1e-10)
        norms_pert = np.maximum(norms_pert, 1e-10)
        
        # Row-wise cosine similarity
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
    X_train_orig, X_test_orig, X_train_pert, X_test_pert, 
    y_train, y_test, random_state=42, use_raw_classifiers=False
) -> Dict[str, float]:
    """
    Compute classification stability between original and perturbed embeddings.
    
    Args:
        use_raw_classifiers: If True, use classifiers without StandardScaler preprocessing
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    if use_raw_classifiers:
        clf_orig = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1)
        clf_pert = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1)
    else:
        clf_orig = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1))
        clf_pert = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1))
    
    clf_orig.fit(X_train_orig, y_train)
    clf_pert.fit(X_train_pert, y_train)
    
    acc_orig = accuracy_score(y_test, clf_orig.predict(X_test_orig))
    acc_pert = accuracy_score(y_test, clf_pert.predict(X_test_pert))
    
    # Use rf_ prefix for consistency with print_stability_summary
    return {
        'rf_acc_original': acc_orig,
        'rf_acc_perturbed': acc_pert,
        'rf_acc_drop': acc_orig - acc_pert,
        'rf_acc_drop_pct': (acc_orig - acc_pert) / acc_orig * 100 if acc_orig > 0 else 0
    }


def run_stability_analysis(
    graphs: List[nx.Graph],
    labels: np.ndarray,
    config: Dict[str, Any],
    perturbation_ratios: List[float] = None,
    X_original: np.ndarray = None,
    seed: int = 42,
    test_size: float = 0.15,
    compute_classification: bool = True,
    use_raw_classifiers: bool = False
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Run stability analysis on a configuration.
    
    Perturbation order: Edge Add → Edge Remove
    (No node shuffle since IMDB has no node labels)
    
    Args:
        graphs: List of original graphs
        labels: Ground truth labels
        config: FGSD configuration
        perturbation_ratios: List of perturbation ratios (default: [0.01, 0.05, 0.10, 0.20])
        X_original: Pre-computed original embeddings (optional)
        seed: Random seed
        test_size: Test set fraction for classification comparison
        compute_classification: Whether to compute classification stability
        use_raw_classifiers: If True, use classifiers without StandardScaler
    """
    from sklearn.model_selection import train_test_split
    
    if perturbation_ratios is None:
        perturbation_ratios = DEFAULT_PERTURBATION_RATIOS
    
    print(f"\n{'='*60}")
    print(f"STABILITY ANALYSIS: {config.get('name', config['func'])}")
    print(f"Perturbation ratios: {[f'{r*100:.0f}%' for r in perturbation_ratios]}")
    print(f"Perturbation types: edge_add, edge_remove")
    print(f"(No node shuffle - IMDB has no node labels)")
    print(f"{'='*60}")
    
    if X_original is None:
        print("Computing original embeddings...")
        start_time = time.time()
        X_original = generate_embeddings_for_graphs(graphs, config, seed)
        orig_time = time.time() - start_time
        print(f"  -> Shape: {X_original.shape}, Time: {orig_time:.2f}s")
    else:
        print(f"Using pre-computed embeddings. Shape: {X_original.shape}")
    
    # Split for classification comparison
    if compute_classification:
        indices = np.arange(len(graphs))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=labels
        )
        X_orig_train = X_original[train_idx]
        X_orig_test = X_original[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
    
    results = {
        'config': config,
        'original_shape': X_original.shape,
        'perturbation_results': []
    }
    
    # ===== PHASE 1: EDGE ADD (all ratios) =====
    print(f"\n{'='*40}")
    print("PHASE 1: EDGE ADD PERTURBATIONS")
    print(f"{'='*40}")
    
    for ratio in perturbation_ratios:
        print(f"\n--- Edge Add: {ratio*100:.0f}% ---")
        
        start_time = time.time()
        perturbed_graphs = perturb_graphs_batch_add_only(graphs, ratio, seed)
        perturb_time = time.time() - start_time
        
        start_time = time.time()
        X_perturbed = generate_embeddings_for_graphs(perturbed_graphs, config, seed)
        embed_time = time.time() - start_time
        
        stability_cosine = compute_embedding_stability(X_original, X_perturbed, 'cosine')
        stability_euclidean = compute_embedding_stability(X_original, X_perturbed, 'euclidean')
        
        print(f"  Cosine Sim: {stability_cosine['mean_cosine_similarity']:.4f}, "
              f"Rel Change: {stability_euclidean['mean_relative_change']:.4f}")
        
        result_entry = {
            'ratio': ratio,
            'perturbation_type': 'edge_add',
            'perturb_time': perturb_time,
            'embed_time': embed_time,
            **stability_cosine,
            **stability_euclidean,
        }
        
        if compute_classification:
            clf_stability = compute_classification_stability(
                X_orig_train, X_orig_test,
                X_perturbed[train_idx], X_perturbed[test_idx],
                y_train, y_test,
                random_state=seed,
                use_raw_classifiers=use_raw_classifiers
            )
            result_entry.update(clf_stability)
            print(f"  RF: Orig={clf_stability['rf_acc_original']:.4f}, "
                  f"Pert={clf_stability['rf_acc_perturbed']:.4f}, "
                  f"Drop={clf_stability['rf_acc_drop_pct']:.2f}%")
        
        results['perturbation_results'].append(result_entry)
        del perturbed_graphs
        gc.collect()
    
    # ===== PHASE 2: EDGE REMOVE (all ratios) =====
    print(f"\n{'='*40}")
    print("PHASE 2: EDGE REMOVE PERTURBATIONS")
    print(f"{'='*40}")
    
    for ratio in perturbation_ratios:
        print(f"\n--- Edge Remove: {ratio*100:.0f}% ---")
        
        start_time = time.time()
        perturbed_graphs = perturb_graphs_batch_remove_only(graphs, ratio, seed)
        perturb_time = time.time() - start_time
        
        start_time = time.time()
        X_perturbed = generate_embeddings_for_graphs(perturbed_graphs, config, seed)
        embed_time = time.time() - start_time
        
        stability_cosine = compute_embedding_stability(X_original, X_perturbed, 'cosine')
        stability_euclidean = compute_embedding_stability(X_original, X_perturbed, 'euclidean')
        
        print(f"  Cosine Sim: {stability_cosine['mean_cosine_similarity']:.4f}, "
              f"Rel Change: {stability_euclidean['mean_relative_change']:.4f}")
        
        result_entry = {
            'ratio': ratio,
            'perturbation_type': 'edge_remove',
            'perturb_time': perturb_time,
            'embed_time': embed_time,
            **stability_cosine,
            **stability_euclidean,
        }
        
        if compute_classification:
            clf_stability = compute_classification_stability(
                X_orig_train, X_orig_test,
                X_perturbed[train_idx], X_perturbed[test_idx],
                y_train, y_test,
                random_state=seed,
                use_raw_classifiers=use_raw_classifiers
            )
            result_entry.update(clf_stability)
            print(f"  RF: Orig={clf_stability['rf_acc_original']:.4f}, "
                  f"Pert={clf_stability['rf_acc_perturbed']:.4f}, "
                  f"Drop={clf_stability['rf_acc_drop_pct']:.2f}%")
        
        results['perturbation_results'].append(result_entry)
        del perturbed_graphs
        gc.collect()
    
    return results, X_original


def print_stability_summary(all_results: List[Dict[str, Any]]):
    """Print summary of stability analysis (RF only)."""
    print("\n" + "="*130)
    print("STABILITY ANALYSIS SUMMARY (Random Forest)")
    print("="*130)
    print(f"{'Config':<25} {'Type':<15} {'Ratio%':<8} {'Cosine Sim':<12} {'Rel Change':<12} "
          f"{'RF Orig':<10} {'RF Pert':<10} {'RF Drop%':<10}")
    print("-"*130)
    
    for result in all_results:
        config_name = result['config'].get('name', result['config']['func'])
        for pr in result['perturbation_results']:
            pert_type = pr.get('perturbation_type', 'unknown')
            ratio_pct = pr['ratio'] * 100
            rf_orig = pr.get('rf_acc_original', 0)
            rf_pert = pr.get('rf_acc_perturbed', 0)
            rf_drop = pr.get('rf_acc_drop_pct', 0)
            
            print(f"{config_name:<25} {pert_type:<15} {ratio_pct:<8.0f} "
                  f"{pr['mean_cosine_similarity']:<12.4f} "
                  f"{pr['mean_relative_change']:<12.4f} "
                  f"{rf_orig:<10.4f} {rf_pert:<10.4f} {rf_drop:<10.2f}")
