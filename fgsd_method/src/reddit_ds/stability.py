"""
Stability Analysis for FGSD embeddings on REDDIT-MULTI-12K.
Modified: Batch-wise processing to minimize RAM usage.
- Phase 1: Generate all embeddings (control + all perturbations) in one pass.
- Phase 2: Compute metrics on the collected embeddings
"""

import gc
import copy
import os
import pickle
from typing import List, Dict, Any, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from .config import DATASET_DIR, BATCH_SIZE
from .data_loader import load_metadata, iter_graph_batches
import time  # Add this import at the top with others


# Perturbation ratios: only 5% and 10%
PERTURBATION_RATIOS = [0.05, 0.10]
# Perturbation modes: only add and remove
PERTURBATION_MODES = ['remove', 'add']

# Cache directory for control embeddings
CACHE_DIR = os.path.join(parent_dir, 'cache')


def get_control_embedding_cache_path(func_type: str, bins: int, range_val: float) -> str:
    """Get cache file path for control embeddings."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f'reddit_control_embedding_{func_type}_bins{bins}_range{range_val:.2f}.pkl')


def load_control_embedding(func_type: str, bins: int, range_val: float) -> np.ndarray:
    """Load cached control embedding if exists."""
    cache_path = get_control_embedding_cache_path(func_type, bins, range_val)
    if os.path.exists(cache_path):
        print(f"  Loading cached control embedding: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


def save_control_embedding(X: np.ndarray, func_type: str, bins: int, range_val: float):
    """Save control embedding to cache."""
    cache_path = get_control_embedding_cache_path(func_type, bins, range_val)
    print(f"  Saving control embedding to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(X, f)


def perturb_graph_edges(
    graph: nx.Graph, 
    perturbation_ratio: float = 0.05,
    seed: int = 42,
    mode: str = 'remove'
) -> nx.Graph:
    """Perturb a graph by adding or removing edges."""
    np.random.seed(seed)
    G = copy.deepcopy(graph)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_nodes < 2 or n_edges < 1:
        return G

    n_perturb = max(1, int(n_edges * perturbation_ratio))

    if mode == 'remove':
        edges = list(G.edges())
        if n_perturb > 0 and len(edges) > 0:
            remove_indices = np.random.choice(len(edges), min(n_perturb, len(edges)), replace=False)
            for idx in remove_indices:
                G.remove_edge(*edges[idx])
        return G

    elif mode == 'add':
        nodes = list(G.nodes())
        added = 0
        max_attempts = n_perturb * 10
        attempts = 0
        while added < n_perturb and attempts < max_attempts:
            u, v = np.random.choice(nodes, 2, replace=False)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1
            attempts += 1
        return G

    else:
        raise ValueError(f"Unknown perturbation mode: {mode}. Use 'add' or 'remove'.")


def perturb_graphs_batch(
    graphs: List[nx.Graph],
    perturbation_ratio: float = 0.05,
    seed: int = 42,
    mode: str = 'remove'
) -> List[nx.Graph]:
    """Perturb a batch of graphs with a given mode."""
    perturbed = []
    for i, g in enumerate(graphs):
        perturbed.append(perturb_graph_edges(g, perturbation_ratio, seed=seed + i, mode=mode))
    return perturbed


def compute_embedding_stability(
    X_original: np.ndarray,
    X_perturbed: np.ndarray
) -> Dict[str, float]:
    """Compute cosine similarity between original and perturbed embeddings."""
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


def compute_classification_accuracy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42
) -> Dict[str, float]:
    """Compute classification accuracy using both RF and SVM."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    
    results = {}
    
    # Random Forest
    clf_rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    results['rf_accuracy'] = accuracy_score(y_test, y_pred_rf)
    results['rf_f1'] = f1_score(y_test, y_pred_rf, average='weighted')
    
    # SVM (RBF) with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf_svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=random_state)
    clf_svm.fit(X_train_scaled, y_train)
    y_pred_svm = clf_svm.predict(X_test_scaled)
    results['svm_accuracy'] = accuracy_score(y_test, y_pred_svm)
    results['svm_f1'] = f1_score(y_test, y_pred_svm, average='weighted')
    
    return results


def generate_all_embeddings_batchwise(
    func_type: str,
    bins: int,
    range_val: float,
    seed: int = 42,
    batch_size: int = 500
) -> Dict[str, np.ndarray]:
    """
    PHASE 1: Generate ALL embeddings (control + all perturbations) in one pass.
    
    Process graphs batch-by-batch to minimize RAM:
    - Load batch of graphs
    - Generate control embedding for batch
    - Generate all perturbed embeddings for batch
    - Free batch, keep only embeddings
    
    Returns:
        Dict with keys: 'control', 'remove_5', 'remove_10', 'add_5', 'add_10'
        Each value is the full embedding matrix (n_graphs, bins)
    """
    phase1_start = time.time()
    print(f"\n  PHASE 1: Generating all embeddings batch-wise for {func_type}")
    print(f"  (bins={bins}, range={range_val}, batch_size={batch_size})")
    
    # Check if control embedding is cached
    X_control_cached = load_control_embedding(func_type, bins, range_val)
    
    # Initialize collectors for each perturbation type
    perturbation_keys = ['control', 'remove_5', 'remove_10', 'add_5', 'add_10']
    embeddings_collectors = {key: [] for key in perturbation_keys}
    
    # If control is cached, we still need to generate perturbed ones
    control_cached = X_control_cached is not None
    if control_cached:
        print(f"    Control embedding cached, will generate perturbations only")
    
    # Create FGSD model (reused for all)
    model = FlexibleFGSD(
        hist_bins=bins,
        hist_range=range_val,
        func_type=func_type,
        seed=seed
    )
    
    # Get total batches for progress bar
    records = load_metadata(DATASET_DIR)
    total_batches = (len(records) + batch_size - 1) // batch_size
    total_graphs = len(records)
    
    print(f"    Total graphs: {total_graphs}, Batches: {total_batches}")
    print(f"    Generating: control + 4 perturbations = 5 embeddings per graph")
    
    batch_idx = 0
    batch_start = time.time()
    
    for graphs, labels, gids in tqdm(iter_graph_batches(DATASET_DIR, batch_size), 
                                      total=total_batches,
                                      desc=f"    {func_type} batches"):
        
        base_seed = seed + batch_idx * 1000  # Different seed per batch
        
        # === Control embedding ===
        if not control_cached:
            model.fit(graphs)
            embeddings_collectors['control'].append(model.get_embedding())
        
        # === Perturbed embeddings ===
        for mode in PERTURBATION_MODES:
            for ratio in PERTURBATION_RATIOS:
                key = f"{mode}_{int(ratio*100)}"
                
                # Perturb this batch
                perturbed_batch = perturb_graphs_batch(
                    graphs, ratio, seed=base_seed, mode=mode
                )
                
                # Generate embedding
                model.fit(perturbed_batch)
                embeddings_collectors[key].append(model.get_embedding())
                
                # Free perturbed batch immediately
                del perturbed_batch
        
        # Free original batch
        del graphs
        batch_idx += 1
        
        # Log progress every 5 batches
        if batch_idx % 5 == 0 and batch_idx > 0:
            elapsed = time.time() - batch_start
            graphs_done = batch_idx * batch_size
            rate = graphs_done / elapsed
            remaining = (total_graphs - graphs_done) / rate if rate > 0 else 0
            # tqdm handles this, but we add extra info
        
        # Periodic garbage collection
        if batch_idx % 5 == 0:
            gc.collect()
    
    gc.collect()
    
    # Stack all batch embeddings into full matrices
    print(f"    Stacking embeddings...")
    all_embeddings = {}
    
    if control_cached:
        all_embeddings['control'] = X_control_cached
    else:
        all_embeddings['control'] = np.vstack(embeddings_collectors['control'])
        # Cache the control embedding
        save_control_embedding(all_embeddings['control'], func_type, bins, range_val)
    
    for key in ['remove_5', 'remove_10', 'add_5', 'add_10']:
        all_embeddings[key] = np.vstack(embeddings_collectors[key])
    
    # Free collectors
    del embeddings_collectors
    gc.collect()
    
    print(f"    Generated embeddings shapes:")
    for key, X in all_embeddings.items():
        size_mb = X.nbytes / 1024 / 1024
        print(f"      {key}: {X.shape} ({size_mb:.1f} MB)")
    
    phase1_time = time.time() - phase1_start
    print(f"    ✓ Phase 1 complete in {phase1_time:.1f}s ({phase1_time/60:.1f} min)")
    
    return all_embeddings


def run_stability_analysis(
    graphs: List[nx.Graph],  # Not used in batch mode, kept for API compatibility
    labels: np.ndarray,
    configs: List[Dict[str, Any]],
    seed: int = 42,
    test_size: float = 0.15
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Run stability analysis for given configurations using batch-wise processing.
    
    Two-phase approach:
    - Phase 1: Generate all embeddings batch-by-batch (RAM efficient)
    - Phase 2: Compute metrics on collected embeddings
    """
    from sklearn.model_selection import train_test_split
    
    total_start = time.time()
    
    print("\n" + "="*100)
    print("STABILITY ANALYSIS (Batch-wise, RAM-efficient)")
    print("Modes: add/remove at 5% and 10%")
    print(f"Configs to process: {len(configs)}")
    print("="*100)
    
    all_results = []
    control_embeddings = {}
    
    # Load labels from metadata if not provided
    if labels is None:
        records = load_metadata(DATASET_DIR)
        labels = np.array([r.label for r in records])
    
    # Create train/test split indices
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    for config_idx, config in enumerate(configs):
        config_start = time.time()
        func_type = config['func']
        bins = config['bins']
        range_val = config['range']
        
        print(f"\n{'='*80}")
        print(f"[{config_idx+1}/{len(configs)}] Configuration: {func_type.upper()}, bins={bins}, range={range_val}")
        print(f"{'='*80}")
        
        # === PHASE 1: Generate all embeddings batch-wise ===
        all_embeddings = generate_all_embeddings_batchwise(
            func_type, bins, range_val, seed, batch_size=BATCH_SIZE
        )
        
        X_control = all_embeddings['control']
        control_embeddings[f"{func_type}_{bins}"] = X_control
        
        # === PHASE 2: Compute metrics ===
        phase2_start = time.time()
        print(f"\n  PHASE 2: Computing metrics...")
        
        # Control (baseline) classification accuracy
        X_train_ctrl = X_control[train_idx]
        X_test_ctrl = X_control[test_idx]
        
        ctrl_acc = compute_classification_accuracy(X_train_ctrl, X_test_ctrl, y_train, y_test, seed)
        print(f"\n  CONTROL (no perturbation):")
        print(f"    RF:  Accuracy={ctrl_acc['rf_accuracy']:.4f}, F1={ctrl_acc['rf_f1']:.4f}")
        print(f"    SVM: Accuracy={ctrl_acc['svm_accuracy']:.4f}, F1={ctrl_acc['svm_f1']:.4f}")
        
        # Store control result
        control_result = {
            'func': func_type,
            'bins': bins,
            'range': range_val,
            'mode': 'control',
            'ratio': 0.0,
            'mean_cosine_similarity': 1.0,
            'std_cosine_similarity': 0.0,
            'rf_accuracy': ctrl_acc['rf_accuracy'],
            'rf_f1': ctrl_acc['rf_f1'],
            'svm_accuracy': ctrl_acc['svm_accuracy'],
            'svm_f1': ctrl_acc['svm_f1'],
            'rf_acc_drop_pct': 0.0,
            'svm_acc_drop_pct': 0.0,
        }
        all_results.append(control_result)
        
        # Process each perturbation
        for mode in PERTURBATION_MODES:
            print(f"\n  Mode: {mode.upper()}")
            
            for ratio in PERTURBATION_RATIOS:
                pert_start = time.time()
                key = f"{mode}_{int(ratio*100)}"
                X_perturbed = all_embeddings[key]
                
                print(f"\n    --- Perturbation: {ratio*100:.0f}% ---")
                
                # Compute cosine similarity
                stability = compute_embedding_stability(X_control, X_perturbed)
                print(f"      Cosine Similarity: {stability['mean_cosine_similarity']:.4f} "
                      f"(std={stability['std_cosine_similarity']:.4f})")
                
                # Compute classification accuracy on perturbed
                X_train_pert = X_perturbed[train_idx]
                X_test_pert = X_perturbed[test_idx]
                
                pert_acc = compute_classification_accuracy(
                    X_train_pert, X_test_pert, y_train, y_test, seed
                )
                
                # Compute accuracy drops
                rf_drop_pct = (ctrl_acc['rf_accuracy'] - pert_acc['rf_accuracy']) / max(ctrl_acc['rf_accuracy'], 1e-10) * 100
                svm_drop_pct = (ctrl_acc['svm_accuracy'] - pert_acc['svm_accuracy']) / max(ctrl_acc['svm_accuracy'], 1e-10) * 100
                
                print(f"      RF:  Accuracy={pert_acc['rf_accuracy']:.4f} (drop={rf_drop_pct:.2f}%)")
                print(f"      SVM: Accuracy={pert_acc['svm_accuracy']:.4f} (drop={svm_drop_pct:.2f}%)")
                print(f"      (computed in {time.time() - pert_start:.1f}s)")
                
                result = {
                    'func': func_type,
                    'bins': bins,
                    'range': range_val,
                    'mode': mode,
                    'ratio': ratio,
                    **stability,
                    'rf_accuracy': pert_acc['rf_accuracy'],
                    'rf_f1': pert_acc['rf_f1'],
                    'svm_accuracy': pert_acc['svm_accuracy'],
                    'svm_f1': pert_acc['svm_f1'],
                    'rf_acc_drop_pct': rf_drop_pct,
                    'svm_acc_drop_pct': svm_drop_pct,
                    'rf_acc_control': ctrl_acc['rf_accuracy'],
                    'svm_acc_control': ctrl_acc['svm_accuracy'],
                }
                all_results.append(result)
        
        config_time = time.time() - config_start
        print(f"\n  ✓ Config {func_type} complete in {config_time:.1f}s ({config_time/60:.1f} min)")
        
        # Free embeddings for this config
        del all_embeddings
        gc.collect()
    
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"✓ TOTAL STABILITY ANALYSIS TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*80}")
    
    return all_results, control_embeddings


def print_stability_summary(results: List[Dict[str, Any]]):
    """Print summary of stability analysis."""
    print("\n" + "="*120)
    print("STABILITY ANALYSIS SUMMARY")
    print("="*120)
    print(f"{'Func':<12} {'Bins':<6} {'Mode':<8} {'Ratio':<8} {'CosineSim':<12} "
          f"{'RF_Acc':<10} {'RF_Drop%':<10} {'SVM_Acc':<10} {'SVM_Drop%':<10}")
    print("-"*120)
    
    for r in results:
        ratio_str = f"{r['ratio']*100:.0f}%" if r['ratio'] > 0 else "control"
        print(f"{r['func']:<12} {r['bins']:<6} {r['mode']:<8} {ratio_str:<8} "
              f"{r['mean_cosine_similarity']:<12.4f} "
              f"{r['rf_accuracy']:<10.4f} {r['rf_acc_drop_pct']:<10.2f} "
              f"{r['svm_accuracy']:<10.4f} {r['svm_acc_drop_pct']:<10.2f}")


def load_best_configs_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load best harmonic and polynomial configs from final results CSV."""
    import pandas as pd
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    configs = []
    
    for func_type in ['harmonic', 'polynomial']:
        func_df = df[df['func'] == func_type]
        
        if len(func_df) == 0:
            print(f"  Warning: No results found for {func_type}")
            continue
        
        rf_df = func_df[func_df['classifier'] == 'Random Forest']
        if len(rf_df) == 0:
            rf_df = func_df
        
        best_row = rf_df.loc[rf_df['accuracy'].idxmax()]
        
        config = {
            'func': func_type,
            'bins': int(best_row['bins']),
            'range': float(best_row['range']),
        }
        configs.append(config)
        print(f"  Best {func_type}: bins={config['bins']}, range={config['range']:.2f}")
    
    return configs