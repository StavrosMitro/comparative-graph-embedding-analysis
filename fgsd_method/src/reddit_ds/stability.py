"""
Stability Analysis for FGSD embeddings on REDDIT-MULTI-12K.
(Modified: No Reshuffle, No Hybrids, Restricted Ratios)
"""

import gc
import copy
from typing import List, Dict, Any

import numpy as np
import networkx as nx
from tqdm import tqdm

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD


# Default perturbation ratios (Left unchanged as requested)
DEFAULT_PERTURBATION_RATIOS = [0.01, 0.05, 0.10, 0.20]
# Modified: Only 0.05 (5%) and 0.10 (10%) for add/remove
EXTRA_PERTURB_RATIOS = [0.05, 0.10]
# Modified: Removed 'reshuffle'
PERTURBATION_MODES = ['default', 'remove', 'add']


def perturb_graph_edges(
    graph: nx.Graph, 
    perturbation_ratio: float = 0.05,
    seed: int = 42,
    mode: str = 'default'
) -> nx.Graph:
    """Perturb a graph by randomly adding/removing edges."""
    np.random.seed(seed)
    G = copy.deepcopy(graph)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes < 2 or n_edges < 1:
        return G

    if mode == 'default':
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

    elif mode == 'remove':
        n_remove = max(1, int(n_edges * perturbation_ratio))
        edges = list(G.edges())
        if n_remove > 0 and len(edges) > 0:
            remove_indices = np.random.choice(len(edges), min(n_remove, len(edges)), replace=False)
            for idx in remove_indices:
                G.remove_edge(*edges[idx])
        return G

    elif mode == 'add':
        n_add = max(1, int(n_edges * perturbation_ratio))
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

    # Removed 'reshuffle' block entirely

    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")


def perturb_graphs_batch(
    graphs: List[nx.Graph],
    perturbation_ratio: float = 0.05,
    seed: int = 42,
    mode: str = 'default'
) -> List[nx.Graph]:
    """Perturb a batch of graphs with a given mode."""
    perturbed = []
    for i, g in enumerate(graphs):
        perturbed.append(perturb_graph_edges(g, perturbation_ratio, seed=seed + i, mode=mode))
    return perturbed


def compute_embedding_stability(
    X_original: np.ndarray,
    X_perturbed: np.ndarray,
    metric: str = 'cosine'
) -> Dict[str, float]:
    """Compute stability metrics between original and perturbed embeddings."""
    
    if metric == 'cosine':
        norms_orig = np.linalg.norm(X_original, axis=1, keepdims=True)
        norms_pert = np.linalg.norm(X_perturbed, axis=1, keepdims=True)
        
        norms_orig = np.maximum(norms_orig, 1e-10)
        norms_pert = np.maximum(norms_pert, 1e-10)
        
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
    """Generate embedding for a single function type with progress bar."""
    model = FlexibleFGSD(
        hist_bins=bins,
        hist_range=range_val,
        func_type=func_type,
        seed=seed
    )
    
    # Process in batches for large datasets
    batch_size = 500
    n_graphs = len(graphs)
    
    if n_graphs > batch_size:
        print(f"      Processing {n_graphs} graphs in batches of {batch_size}...")
        embeddings_list = []
        for i in tqdm(range(0, n_graphs, batch_size), desc=f"      {func_type}"):
            batch = graphs[i:i+batch_size]
            model.fit(batch)
            embeddings_list.append(model.get_embedding())
        return np.vstack(embeddings_list)
    else:
        model.fit(graphs)
        return model.get_embedding()


def generate_embeddings_for_graphs(
    graphs: List[nx.Graph],
    config: Dict[str, Any],
    seed: int = 42
) -> np.ndarray:
    """
    Generate FGSD embeddings for a list of graphs.
    Modified: Removed Hybrid logic.
    """
    func = config['func']
    
    # Removed hybrid checks. Assuming only single function types now.
    
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
    print(f"{'Config':<25} {'Mode':<10} {'Perturb%':<10} {'CosineSim':<12} {'RelChange':<12} "
          f"{'RF_Orig':<10} {'RF_Pert':<10} {'RF_Drop%':<10}")
    print("-"*100)
    for result in all_results:
        config_name = result['config'].get('name', result['config']['func'])
        for pr in result['perturbation_results']:
            print(f"{config_name:<25} {pr.get('mode', 'default'):<10} {pr['ratio']*100:<10.0f} "
                  f"{pr['mean_cosine_similarity']:<12.4f} {pr['mean_relative_change']:<12.4f} "
                  f"{pr.get('rf_acc_original', 0):<10.4f} {pr.get('rf_acc_perturbed', 0):<10.4f} "
                  f"{pr.get('rf_acc_drop_pct', 0):<10.2f}")


def run_full_stability_analysis(
    graphs: List[nx.Graph],
    labels: np.ndarray,
    configs_to_test: List[Dict[str, Any]],
    seed: int = 42,
    test_size: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Run stability analysis.
    Modified: No reshuffle, skips hybrid configs.
    """
    all_results = []
    
    for config in configs_to_test:
        # Check to skip hybrid configs
        if 'hybrid' in config['func']:
            print(f"Skipping hybrid config: {config['func']}")
            continue

        config_results = {'config': config, 'perturbation_results': []}
        
        # Default (add+remove) ratios
        for ratio in DEFAULT_PERTURBATION_RATIOS:
            perturbed_graphs = perturb_graphs_batch(graphs, ratio, seed=seed, mode='default')
            X_original = generate_embeddings_for_graphs(graphs, config, seed=seed)
            X_perturbed = generate_embeddings_for_graphs(perturbed_graphs, config, seed=seed)
            stability_cosine = compute_embedding_stability(X_original, X_perturbed, 'cosine')
            stability_euclidean = compute_embedding_stability(X_original, X_perturbed, 'euclidean')
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(graphs))
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=labels)
            clf_stability = compute_classification_stability(
                X_original[train_idx], X_original[test_idx],
                X_perturbed[train_idx], X_perturbed[test_idx],
                labels[train_idx], labels[test_idx], seed
            )
            result_entry = {'mode': 'default', 'ratio': ratio, **stability_cosine, **stability_euclidean, **clf_stability}
            config_results['perturbation_results'].append(result_entry)
            del perturbed_graphs, X_perturbed
            gc.collect()

        # Remove only (Only 0.05, 0.10)
        for ratio in EXTRA_PERTURB_RATIOS:
            perturbed_graphs = perturb_graphs_batch(graphs, ratio, seed=seed, mode='remove')
            X_original = generate_embeddings_for_graphs(graphs, config, seed=seed)
            X_perturbed = generate_embeddings_for_graphs(perturbed_graphs, config, seed=seed)
            stability_cosine = compute_embedding_stability(X_original, X_perturbed, 'cosine')
            stability_euclidean = compute_embedding_stability(X_original, X_perturbed, 'euclidean')
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(graphs))
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=labels)
            clf_stability = compute_classification_stability(
                X_original[train_idx], X_original[test_idx],
                X_perturbed[train_idx], X_perturbed[test_idx],
                labels[train_idx], labels[test_idx], seed
            )
            result_entry = {'mode': 'remove', 'ratio': ratio, **stability_cosine, **stability_euclidean, **clf_stability}
            config_results['perturbation_results'].append(result_entry)
            del perturbed_graphs, X_perturbed
            gc.collect()

        # Add only (Only 0.05, 0.10)
        for ratio in EXTRA_PERTURB_RATIOS:
            perturbed_graphs = perturb_graphs_batch(graphs, ratio, seed=seed, mode='add')
            X_original = generate_embeddings_for_graphs(graphs, config, seed=seed)
            X_perturbed = generate_embeddings_for_graphs(perturbed_graphs, config, seed=seed)
            stability_cosine = compute_embedding_stability(X_original, X_perturbed, 'cosine')
            stability_euclidean = compute_embedding_stability(X_original, X_perturbed, 'euclidean')
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(graphs))
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=labels)
            clf_stability = compute_classification_stability(
                X_original[train_idx], X_original[test_idx],
                X_perturbed[train_idx], X_perturbed[test_idx],
                labels[train_idx], labels[test_idx], seed
            )
            result_entry = {'mode': 'add', 'ratio': ratio, **stability_cosine, **stability_euclidean, **clf_stability}
            config_results['perturbation_results'].append(result_entry)
            del perturbed_graphs, X_perturbed
            gc.collect()

        # Reshuffle loop removed completely

        all_results.append(config_results)
    return all_results