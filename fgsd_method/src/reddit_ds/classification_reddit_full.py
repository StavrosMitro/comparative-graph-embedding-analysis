"""
FGSD Classification on REDDIT-MULTI-12K
- Full dataset pre-analysis to determine optimal bins and range
- Automatic parameter selection based on spectral distance distribution
- Classification with multiple models
"""

import os
import sys
import warnings
import gc
import time
import tracemalloc
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Optional, Dict, Any

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from optimized_method import HybridFGSD

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================
DATASET_DIR = '/tmp/REDDIT-MULTI-12K'
BATCH_SIZE = 200
RESULTS_DIR = os.path.join(os.path.dirname(current_dir), 'results')
PREANALYSIS_SAMPLE_SIZE = 700  # Reduced from 2000 to save memory
MAX_NODES_FOR_PREANALYSIS = 500  # Skip graphs larger than this in pre-analysis

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class GraphRecord:
    """Metadata for a single graph."""
    graph_id: int
    label: int
    node_start: int
    node_end: int

@dataclass
class OptimalParams:
    """Optimal parameters determined from pre-analysis."""
    func_type: str
    bins: int
    range_val: float
    p99: float
    recommended_bins: int

# =============================================================================
# DATASET LOADING FUNCTIONS
# =============================================================================
def ensure_dataset_ready() -> str:
    """Ensure the dataset is downloaded and extracted."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    base_url = 'https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-MULTI-12K.zip'
    zip_path = os.path.join(DATASET_DIR, 'REDDIT-MULTI-12K.zip')
    dataset_path = os.path.join(DATASET_DIR, 'REDDIT-MULTI-12K')
    
    if not os.path.exists(dataset_path):
        print("Downloading REDDIT-MULTI-12K dataset...")
        urllib.request.urlretrieve(base_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("Download complete.")
    return dataset_path


def load_metadata(dataset_dir: str) -> List[GraphRecord]:
    """Load graph metadata without loading full graphs into memory."""
    dataset_path = os.path.join(dataset_dir, 'REDDIT-MULTI-12K')
    
    graph_indicator = pd.read_csv(
        os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_indicator.txt'), 
        header=None
    )[0].values
    graph_labels = pd.read_csv(
        os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_labels.txt'), 
        header=None
    )[0].values
    
    records = []
    num_graphs = len(graph_labels)
    
    for gid in range(1, num_graphs + 1):
        mask = graph_indicator == gid
        indices = np.where(mask)[0]
        if len(indices) > 0:
            node_start = indices[0]
            node_end = indices[-1] + 1
        else:
            node_start = node_end = 0
        
        records.append(GraphRecord(
            graph_id=gid,
            label=graph_labels[gid - 1] - 1,
            node_start=node_start,
            node_end=node_end
        ))
    
    return records


def iter_graph_batches(
    dataset_dir: str, 
    batch_size: int, 
    records: Optional[List[GraphRecord]] = None
) -> Iterator[Tuple[List[nx.Graph], np.ndarray, List[int]]]:
    """Iterate over graphs in batches."""
    dataset_path = os.path.join(dataset_dir, 'REDDIT-MULTI-12K')
    
    if records is None:
        records = load_metadata(dataset_dir)
    
    edges_df = pd.read_csv(
        os.path.join(dataset_path, 'REDDIT-MULTI-12K_A.txt'),
        header=None, names=['src', 'dst']
    )
    graph_indicator = pd.read_csv(
        os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_indicator.txt'),
        header=None
    )[0].values
    
    for i in range(0, len(records), batch_size):
        batch_records = records[i:i + batch_size]
        graphs = []
        labels = []
        gids = []
        
        for rec in batch_records:
            G = nx.Graph()
            node_ids = np.where(graph_indicator == rec.graph_id)[0] + 1
            for nid in node_ids:
                G.add_node(nid)
            
            mask = graph_indicator[edges_df['src'].values - 1] == rec.graph_id
            graph_edges = edges_df[mask][['src', 'dst']].values
            G.add_edges_from(graph_edges)
            G = nx.convert_node_labels_to_integers(G)
            
            graphs.append(G)
            labels.append(rec.label)
            gids.append(rec.graph_id)
        
        yield graphs, np.array(labels), gids


def load_all_graphs(dataset_dir: str) -> Tuple[List[nx.Graph], np.ndarray]:
    """Load all graphs into memory."""
    print("Loading all graphs into memory...")
    all_graphs = []
    all_labels = []
    
    for graphs, labels, _ in tqdm(iter_graph_batches(dataset_dir, BATCH_SIZE), 
                                   desc="Loading graphs"):
        all_graphs.extend(graphs)
        all_labels.extend(labels)
    
    return all_graphs, np.array(all_labels)


# =============================================================================
# PRE-ANALYSIS FUNCTIONS
# =============================================================================
def compute_spectral_distances_sampled(
    graphs: List[nx.Graph], 
    func_type: str,
    sample_size: int = PREANALYSIS_SAMPLE_SIZE,
    max_nodes: int = MAX_NODES_FOR_PREANALYSIS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spectral distances for a SAMPLE of graphs.
    Sampling is sufficient for determining optimal parameters.
    Skips very large graphs to avoid OOM.
    """
    # Filter out very large graphs first (they cause OOM)
    valid_indices = [i for i, g in enumerate(graphs) if g.number_of_nodes() <= max_nodes and g.number_of_nodes() >= 2]
    
    # Sample from valid graphs
    np.random.seed(42)
    if len(valid_indices) > sample_size:
        sample_indices = np.random.choice(valid_indices, sample_size, replace=False)
    else:
        sample_indices = valid_indices
    
    print(f"\nComputing spectral distances for {len(sample_indices)} graphs (max {max_nodes} nodes) using func='{func_type}'...")
    print(f"  (Filtered from {len(graphs)} total, {len(valid_indices)} valid size)")
    
    all_distances = []
    node_counts = []
    skipped = 0
    
    for idx in tqdm(sample_indices, desc=f"Processing {func_type}"):
        G = graphs[idx]
        n_nodes = G.number_of_nodes()
        
        if n_nodes < 2:
            skipped += 1
            continue
        
        node_counts.append(n_nodes)
        
        try:
            L = np.asarray(nx.normalized_laplacian_matrix(G).todense())
            w, v = np.linalg.eigh(L)
            
            if func_type == 'harmonic':
                with np.errstate(divide='ignore', invalid='ignore'):
                    func_w = np.where(w > 1e-9, 1.0 / w, 0)
            elif func_type == 'polynomial':
                func_w = w ** 2
            elif func_type == 'biharmonic':
                with np.errstate(divide='ignore', invalid='ignore'):
                    func_w = np.where(w > 1e-9, 1.0 / (w**2), 0)
            else:
                func_w = w
            
            fL = v @ np.diag(func_w) @ v.T
            ones = np.ones(L.shape[0])
            S = np.outer(np.diag(fL), ones) + np.outer(ones, np.diag(fL)) - 2 * fL
            S = np.asarray(S)
            
            triu_indices = np.triu_indices_from(S, k=1)
            distances = S[triu_indices]
            all_distances.extend(distances.flatten().tolist())
            
            # Explicitly free memory
            del L, w, v, fL, S, distances
            
        except Exception as e:
            skipped += 1
        
        # Periodic garbage collection
        if len(node_counts) % 100 == 0:
            gc.collect()
    
    if skipped > 0:
        print(f"  Skipped {skipped} graphs (errors or too small)")
    
    gc.collect()
    return np.array(all_distances), np.array(node_counts)


def determine_optimal_params(
    distances: np.ndarray, 
    node_counts: np.ndarray, 
    func_type: str,
    save_plot: bool = True
) -> OptimalParams:
    """
    Analyze the distribution and determine optimal bins and range.
    
    Strategy:
    - Range: Use 99th percentile (captures most data, ignores outliers)
    - Bins: Choose bins where avg_hits_per_graph >= 1.0 (low sparsity risk)
    """
    print(f"\n{'='*60}")
    print(f"PRE-ANALYSIS (SAMPLED): {func_type.upper()}")
    print(f"{'='*60}")
    
    min_val = np.min(distances)
    max_val = np.max(distances)
    p95 = np.percentile(distances, 95)
    p99 = np.percentile(distances, 99)
    
    print(f"Total distance values analyzed: {len(distances):,}")
    print(f"Graphs in sample: {len(node_counts):,}")
    print(f"\n1. SPECTRAL DISTANCE STATISTICS:")
    print(f"   Min:  {min_val:.4f}")
    print(f"   Max:  {max_val:.4f}")
    print(f"   95th Percentile: {p95:.4f}")
    print(f"   99th Percentile: {p99:.4f}")
    
    # Optimal range is 99th percentile
    optimal_range = p99
    
    # Determine optimal bins
    print(f"\n2. BINS ANALYSIS (Range = [0, {optimal_range:.2f}]):")
    print(f"   {'Bins':<10} | {'Avg Hits/Bin/Graph':<20} | {'Sparsity Risk':<15} | {'Recommended'}")
    print("   " + "-"*70)
    
    test_bins = [50, 100, 150, 200, 250, 300, 400, 500]
    recommended_bins = 100  # default
    
    for b in test_bins:
        hist, _ = np.histogram(distances, bins=b, range=(0, optimal_range))
        avg_hits = np.sum(hist) / len(node_counts) / b
        
        if avg_hits >= 2.0:
            risk = "Very Low"
            rec = "✓ GOOD"
            recommended_bins = b
        elif avg_hits >= 1.0:
            risk = "Low"
            rec = "✓ OK"
            if recommended_bins < b:
                recommended_bins = b
        elif avg_hits >= 0.5:
            risk = "Medium"
            rec = ""
        else:
            risk = "HIGH"
            rec = ""
        
        print(f"   {b:<10} | {avg_hits:<20.4f} | {risk:<15} | {rec}")
    
    print(f"\n3. OPTIMAL PARAMETERS SELECTED:")
    print(f"   -> Range: {optimal_range:.4f} (99th percentile)")
    print(f"   -> Bins:  {recommended_bins}")
    
    # Save visualization
    if save_plot:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        viz_cutoff = np.percentile(distances, 99.5)
        viz_data = distances[distances <= viz_cutoff]
        
        ax.hist(viz_data, bins=150, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'95% ({p95:.2f})')
        ax.axvline(p99, color='red', linestyle='-', linewidth=2, label=f'99% = RANGE ({p99:.2f})')
        
        ax.set_title(f"REDDIT-MULTI-12K: {func_type.upper()} Spectral Distances (Sampled)", fontsize=14)
        ax.set_xlabel("Spectral Distance Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f'reddit_preanalysis_{func_type}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"   Plot saved: {save_path}")
    
    return OptimalParams(
        func_type=func_type,
        bins=recommended_bins,
        range_val=optimal_range,
        p99=p99,
        recommended_bins=recommended_bins
    )


def run_sampled_preanalysis(graphs: List[nx.Graph], sample_size: int = PREANALYSIS_SAMPLE_SIZE) -> Dict[str, OptimalParams]:
    """Run pre-analysis on a SAMPLE of graphs - sufficient for parameter tuning."""
    print("\n" + "="*80)
    print(f"RUNNING SAMPLED PRE-ANALYSIS (n={sample_size})")
    print("="*80)
    
    optimal_params = {}
    
    for func_type in ['harmonic', 'polynomial']:
        distances, node_counts = compute_spectral_distances_sampled(graphs, func_type, sample_size)
        params = determine_optimal_params(distances, node_counts, func_type)
        optimal_params[func_type] = params
        
        # Free memory
        del distances
        gc.collect()
    
    return optimal_params


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================
def evaluate_classifier(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray, 
    classifier_name: str, 
    clf
) -> Dict[str, Any]:
    """Evaluate a classifier and return metrics."""
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    start_time = time.time()
    y_pred = clf.predict(X_test)
    inference_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    try:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test)
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X_test)
            if len(y_score.shape) == 1:
                y_score = y_score.reshape(-1, 1)
        else:
            y_score = None

        if y_score is not None and y_test_bin.shape[1] > 1:
            auc = roc_auc_score(y_test_bin, y_score, average='weighted', multi_class='ovr')
        else:
            auc = None
    except:
        auc = None

    return {
        'classifier': classifier_name,
        'train_accuracy': train_accuracy,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'train_time': train_time,
        'inference_time': inference_time
    }


def run_experiment_with_optimal_params(
    optimal_params: Dict[str, OptimalParams],
    test_size: float = 0.15,
    random_state: int = 42
) -> List[Dict[str, Any]]:
    """Run classification experiments using pre-determined optimal parameters.
    
    OPTIMIZATION: Generate harmonic and polynomial embeddings ONCE,
    then concatenate them for hybrid instead of recomputing.
    """
    
    print("\n" + "="*80)
    print("RUNNING CLASSIFICATION EXPERIMENTS WITH OPTIMAL PARAMETERS")
    print("(Optimized: Reusing embeddings for hybrid)")
    print("="*80)
    
    ensure_dataset_ready()
    records = load_metadata(DATASET_DIR)
    all_labels = np.array([r.label for r in records])
    
    # Pre-calculate split indices
    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    
    results = []
    
    # Storage for reusable embeddings
    embeddings_cache = {}  # key: 'harmonic' or 'polynomial', value: (X_all, generation_time)
    
    # =================================================================
    # STEP 1: Generate base embeddings (harmonic and polynomial)
    # =================================================================
    base_configs = []
    if 'harmonic' in optimal_params:
        p = optimal_params['harmonic']
        base_configs.append({
            'func': 'harmonic',
            'bins': p.bins,
            'range': round(p.range_val, 2)
        })
    
    if 'polynomial' in optimal_params:
        p = optimal_params['polynomial']
        base_configs.append({
            'func': 'polynomial',
            'bins': p.bins,
            'range': round(p.range_val, 2)
        })
    
    print(f"\nStep 1: Generating base embeddings ({len(base_configs)} types)...")
    
    for config in base_configs:
        func = config['func']
        print(f"\n  Generating {func.upper()} embeddings (bins={config['bins']}, range={config['range']})...")
        
        start_time = time.time()
        model = FlexibleFGSD(
            hist_bins=config['bins'],
            hist_range=config['range'],
            func_type=func,
            seed=random_state
        )
        
        embeddings_list = []
        total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for graphs, labels, _ in tqdm(
            iter_graph_batches(DATASET_DIR, BATCH_SIZE),
            desc=f"  {func}",
            total=total_batches
        ):
            model.fit(graphs)
            batch_emb = model.get_embedding()
            embeddings_list.append(batch_emb)
            del graphs
            gc.collect()
        
        X_all = np.vstack(embeddings_list)
        generation_time = time.time() - start_time
        
        embeddings_cache[func] = (X_all, generation_time, config)
        print(f"    -> Shape: {X_all.shape}, Time: {generation_time:.2f}s")
        
        del embeddings_list
        gc.collect()
    
    # =================================================================
    # STEP 2: Create hybrid by concatenating (NO recomputation!)
    # =================================================================
    if 'harmonic' in embeddings_cache and 'polynomial' in embeddings_cache:
        print(f"\n  Creating HYBRID embeddings (concatenating harmonic + polynomial)...")
        
        X_harm, time_harm, config_harm = embeddings_cache['harmonic']
        X_poly, time_poly, config_poly = embeddings_cache['polynomial']
        
        X_hybrid = np.hstack([X_harm, X_poly])
        time_hybrid = time_harm + time_poly  # Combined time
        
        config_hybrid = {
            'func': 'hybrid',
            'harm_bins': config_harm['bins'],
            'harm_range': config_harm['range'],
            'pol_bins': config_poly['bins'],
            'pol_range': config_poly['range']
        }
        
        embeddings_cache['hybrid'] = (X_hybrid, time_hybrid, config_hybrid)
        print(f"    -> Shape: {X_hybrid.shape} (concatenated, no extra computation!)")
    
    # =================================================================
    # STEP 3: Evaluate all configurations
    # =================================================================
    print(f"\nStep 2: Evaluating classifiers on all embeddings...")
    
    y_all = all_labels
    
    for func_name, (X_all, generation_time, config) in embeddings_cache.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {func_name.upper()}")
        if func_name == 'hybrid':
            print(f"  Harmonic: bins={config['harm_bins']}, range={config['harm_range']}")
            print(f"  Polynomial: bins={config['pol_bins']}, range={config['pol_range']}")
        else:
            print(f"  bins={config['bins']}, range={config['range']}")
        print(f"{'='*80}")
        
        # Split
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        
        # Classifiers
        classifiers = {
            'SVM (RBF)': make_pipeline(
                StandardScaler(),
                SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=random_state)
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1
            ),
            'MLP': make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(512, 256, 128),
                    max_iter=1000,
                    early_stopping=True,
                    random_state=random_state
                )
            )
        }
        
        for clf_name, clf in classifiers.items():
            print(f"\n  Training {clf_name}...")
            res = evaluate_classifier(X_train, X_test, y_train, y_test, clf_name, clf)
            res.update(config)
            res['generation_time'] = generation_time
            results.append(res)
            print(f"    -> Train Acc: {res['train_accuracy']:.4f}, Test Acc: {res['accuracy']:.4f}, F1: {res['f1_score']:.4f}")
    
    # Cleanup
    del embeddings_cache
    gc.collect()
    
    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary table of results."""
    print("\n" + "="*130)
    print("SUMMARY OF RESULTS")
    print("="*130)
    print(f"{'Func':<12} {'Parameters':<40} {'Classifier':<18} {'Train Acc':<11} {'Test Acc':<10} {'F1':<10} {'GenTime':<8}")
    print("-" * 130)

    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    for r in sorted_results:
        if r['func'] == 'hybrid':
            params = f"h:{r['harm_bins']}/{r['harm_range']}, p:{r['pol_bins']}/{r['pol_range']}"
        else:
            params = f"bins={r['bins']}, range={r['range']}"

        print(f"{r['func']:<12} {params:<40} {r['classifier']:<18} "
              f"{r['train_accuracy']:<11.4f} {r['accuracy']:<10.4f} {r['f1_score']:<10.4f} {r['generation_time']:<8.1f}s")
    
    # Best results
    print("\n" + "-"*130)
    best = sorted_results[0]
    print(f"BEST: {best['func']} with {best['classifier']} -> Accuracy: {best['accuracy']:.4f}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("FGSD CLASSIFICATION ON REDDIT-MULTI-12K")
    print("Sampled Pre-Analysis + Optimal Parameter Selection")
    print("="*80)
    
    # Ensure dataset is ready
    ensure_dataset_ready()
    
    # Load all graphs for pre-analysis
    graphs, labels = load_all_graphs(DATASET_DIR)
    print(f"\nLoaded {len(graphs)} graphs with {len(np.unique(labels))} classes")
    
    # Show graph size distribution
    sizes = [g.number_of_nodes() for g in graphs]
    print(f"Graph sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")
    print(f"Graphs with <= {MAX_NODES_FOR_PREANALYSIS} nodes: {sum(1 for s in sizes if s <= MAX_NODES_FOR_PREANALYSIS)}")
    
    # Run SAMPLED pre-analysis to get optimal parameters (much faster!)
    optimal_params = run_sampled_preanalysis(graphs, sample_size=PREANALYSIS_SAMPLE_SIZE)
    
    # Free memory from full graph list (we'll reload in batches)
    del graphs
    gc.collect()
    
    # Print optimal parameters summary
    print("\n" + "="*80)
    print("OPTIMAL PARAMETERS SUMMARY (from sampled analysis)")
    print("="*80)
    for func_type, params in optimal_params.items():
        print(f"  {func_type.upper():12} -> bins={params.bins:3}, range={params.range_val:.2f}")
    
    # Run experiments with optimal parameters
    results = run_experiment_with_optimal_params(optimal_params)
    
    # Print summary
    print_summary(results)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_optimal_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Save optimal params
    params_path = os.path.join(RESULTS_DIR, 'reddit_optimal_params.txt')
    with open(params_path, 'w') as f:
        f.write(f"OPTIMAL PARAMETERS FROM SAMPLED ANALYSIS (n={PREANALYSIS_SAMPLE_SIZE})\n")
        f.write("="*50 + "\n")
        for func_type, params in optimal_params.items():
            f.write(f"{func_type}: bins={params.bins}, range={params.range_val:.4f}\n")
    print(f"Optimal parameters saved to: {params_path}")


if __name__ == "__main__":
    main()
