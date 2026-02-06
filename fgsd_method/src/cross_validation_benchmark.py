"""
Cross-Validation Benchmark for FGSD Embeddings.

Performs K-fold stratified cross-validation with multiple repetitions on:
- ENZYMES: 
  - Random Forest (n=500, max_depth=20) with polynomial (bins=150, range=4.0)
  - Random Forest (n=500, max_depth=20) with harmonic (bins=150, range=30.3)
- IMDB-MULTI: 
  - MLP (256,128,64, max_iter=1000, early_stopping) with polynomial (bins=150, range=3.1)
  - MLP (256,128,64, max_iter=1000, early_stopping) with harmonic (bins=100, range=3.1)
- REDDIT-MULTI-12K: 
  - Random Forest (n=500, max_depth=20) with harmonic (bins=500, range=14.61) - loaded from cache
  - Random Forest (n=500, max_depth=20) with polynomial (bins=200, range=3.48) - loaded from cache

All classifiers use RAW EMBEDDINGS (no StandardScaler preprocessing).

Usage:
    python -m cross_validation_benchmark
"""

import os
import sys
import gc
import time
import pickle
import tracemalloc
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy import stats

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fgsd import FlexibleFGSD

# Paths
RESULTS_DIR = os.path.join(current_dir, 'results')
CACHE_DIR = os.path.join(current_dir, 'cache')
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# CONFIGURATION - EASY TO CHANGE
# =============================================================================
N_SPLITS = 10          # Number of folds in cross-validation
N_REPETITIONS = 30     # Number of repetitions (total iterations = N_SPLITS * N_REPETITIONS)
RANDOM_STATE = 42      # Base random state


# =============================================================================
# DATA LOADING
# =============================================================================
def load_enzymes_data():
    """Load ENZYMES dataset."""
    from enzymes_ds.data_loader import ensure_dataset_ready, load_all_graphs
    from enzymes_ds.config import DATASET_DIR
    
    print("  Loading ENZYMES dataset...")
    ensure_dataset_ready()
    graphs, labels, _ = load_all_graphs(DATASET_DIR)
    print(f"  -> Loaded {len(graphs)} graphs, {len(set(labels))} classes")
    return graphs, np.array(labels)


def load_imdb_data():
    """Load IMDB-MULTI dataset."""
    from imbd_ds.data_loader import ensure_dataset_ready, load_all_graphs
    from imbd_ds.config import DATASET_DIR
    
    print("  Loading IMDB-MULTI dataset...")
    ensure_dataset_ready()
    graphs, labels = load_all_graphs(DATASET_DIR)
    print(f"  -> Loaded {len(graphs)} graphs, {len(set(labels))} classes")
    return graphs, np.array(labels)


def load_reddit_embedding_from_cache(func_type: str = 'harmonic', bins: int = 500, range_val: float = 14.61):
    """Load pre-computed Reddit embedding from cache."""
    cache_path = os.path.join(CACHE_DIR, 'embeddings', f'reddit_{func_type}_bins{bins}_range{range_val:.2f}.pkl')
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Reddit embedding not found at {cache_path}. "
            "Please run memory_benchmark first to generate it."
        )
    
    print(f"  Loading Reddit embedding from cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    X = data['embedding']
    print(f"  -> Loaded embedding: shape={X.shape}")
    return X


def load_reddit_labels():
    """Load Reddit labels."""
    from reddit_ds.data_loader import ensure_dataset_ready, load_metadata
    from reddit_ds.config import DATASET_DIR
    
    print("  Loading Reddit labels...")
    ensure_dataset_ready()
    records = load_metadata(DATASET_DIR)
    # GraphRecord is a dataclass, use attribute access not subscript
    labels = [r.label for r in records]
    print(f"  -> Loaded {len(labels)} labels, {len(set(labels))} classes")
    return np.array(labels)


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================
def generate_embedding(graphs, func_type: str, bins: int, range_val: float, seed: int = 42):
    """Generate FGSD embedding."""
    print(f"  Generating {func_type} embedding (bins={bins}, range={range_val})...")
    
    tracemalloc.start()
    start_time = time.time()
    
    model = FlexibleFGSD(
        hist_bins=bins,
        hist_range=range_val,
        func_type=func_type,
        seed=seed
    )
    model.fit(graphs)
    X = model.get_embedding()
    
    gen_time = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"  -> Shape: {X.shape}, Time: {gen_time:.2f}s, Peak Memory: {peak_mem/1024/1024:.2f} MB")
    return X, gen_time, peak_mem / 1024 / 1024


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================
def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h


def compute_detailed_statistics(values: List[float], metric_name: str = "metric") -> Dict[str, float]:
    """Compute detailed statistics for a list of values."""
    arr = np.array(values)
    
    ci_low, ci_high = compute_confidence_interval(arr)
    
    return {
        f'{metric_name}_mean': float(np.mean(arr)),
        f'{metric_name}_std': float(np.std(arr, ddof=1)),  # Sample std
        f'{metric_name}_min': float(np.min(arr)),
        f'{metric_name}_max': float(np.max(arr)),
        f'{metric_name}_median': float(np.median(arr)),
        f'{metric_name}_q25': float(np.percentile(arr, 25)),
        f'{metric_name}_q75': float(np.percentile(arr, 75)),
        f'{metric_name}_iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        f'{metric_name}_ci95_low': float(ci_low),
        f'{metric_name}_ci95_high': float(ci_high),
        f'{metric_name}_range': float(np.max(arr) - np.min(arr)),
        f'{metric_name}_cv': float(np.std(arr, ddof=1) / np.mean(arr)) if np.mean(arr) != 0 else 0,  # Coefficient of variation
    }


# =============================================================================
# CROSS-VALIDATION WITH REPETITIONS
# =============================================================================
def run_repeated_stratified_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    classifier_name: str,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPETITIONS,
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    Run repeated stratified K-fold cross-validation.
    
    Uses RAW embeddings (no preprocessing/scaling).
    
    Returns detailed results per fold/repetition and comprehensive aggregated metrics.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state
    )
    
    total_iterations = n_splits * n_repeats
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    
    # Per-repetition aggregates
    repetition_accuracies = {r: [] for r in range(n_repeats)}
    repetition_f1s = {r: [] for r in range(n_repeats)}
    
    print(f"\n  Running {n_splits}-fold Stratified CV x {n_repeats} repetitions = {total_iterations} iterations")
    print(f"  Classifier: {classifier_name} (RAW embeddings, no preprocessing)")
    print(f"  {'Rep':<5} {'Fold':<6} {'Train Acc':<12} {'Test Acc':<12} {'F1':<12} {'Time (s)':<10}")
    print(f"  {'-'*60}")
    
    iteration = 0
    for train_idx, test_idx in rskf.split(X, y):
        rep_idx = iteration // n_splits
        fold_idx = iteration % n_splits + 1
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone classifier for each fold (RAW - no scaler)
        clf = classifier.__class__(**classifier.get_params())
        
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_train_pred = clf.predict(X_train)
        y_pred = clf.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        fold_results.append({
            'repetition': rep_idx + 1,
            'fold': fold_idx,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'f1_score': f1,
            'train_time': train_time,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Track per-repetition
        repetition_accuracies[rep_idx].append(test_acc)
        repetition_f1s[rep_idx].append(f1)
        
        # Print every 10 iterations or at end of each repetition
        if fold_idx == n_splits or iteration < 10:
            print(f"  {rep_idx+1:<5} {fold_idx:<6} {train_acc:<12.4f} {test_acc:<12.4f} {f1:<12.4f} {train_time:<10.2f}")
        elif fold_idx == 1 and rep_idx > 0:
            print(f"  ... (Rep {rep_idx} completed)")
        
        iteration += 1
    
    # =================================================================
    # Compute comprehensive statistics
    # =================================================================
    test_accs = [r['test_accuracy'] for r in fold_results]
    f1_scores = [r['f1_score'] for r in fold_results]
    train_accs = [r['train_accuracy'] for r in fold_results]
    train_times = [r['train_time'] for r in fold_results]
    
    # Overall metrics (all predictions combined)
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
    
    # Per-repetition mean accuracy (for stability analysis)
    rep_mean_accs = [np.mean(repetition_accuracies[r]) for r in range(n_repeats)]
    rep_mean_f1s = [np.mean(repetition_f1s[r]) for r in range(n_repeats)]
    
    # Detailed statistics
    summary = {
        'n_splits': n_splits,
        'n_repetitions': n_repeats,
        'total_iterations': total_iterations,
        
        # Overall combined metrics
        'overall_accuracy': overall_acc,
        'overall_f1': overall_f1,
        
        # Per-fold statistics
        **compute_detailed_statistics(test_accs, 'test_accuracy'),
        **compute_detailed_statistics(f1_scores, 'f1_score'),
        **compute_detailed_statistics(train_accs, 'train_accuracy'),
        **compute_detailed_statistics(train_times, 'train_time'),
        
        # Per-repetition stability (how consistent are repetitions)
        'rep_accuracy_mean': float(np.mean(rep_mean_accs)),
        'rep_accuracy_std': float(np.std(rep_mean_accs, ddof=1)),
        'rep_f1_mean': float(np.mean(rep_mean_f1s)),
        'rep_f1_std': float(np.std(rep_mean_f1s, ddof=1)),
        
        # Total time
        'total_train_time': sum(train_times),
    }
    
    # Print summary
    print(f"  {'-'*60}")
    print(f"\n  === SUMMARY ({n_splits}-fold x {n_repeats} reps = {total_iterations} iterations) ===")
    print(f"  Test Accuracy:  {summary['test_accuracy_mean']:.4f} ± {summary['test_accuracy_std']:.4f}")
    print(f"                  [min={summary['test_accuracy_min']:.4f}, max={summary['test_accuracy_max']:.4f}]")
    print(f"                  95% CI: [{summary['test_accuracy_ci95_low']:.4f}, {summary['test_accuracy_ci95_high']:.4f}]")
    print(f"  F1 Score:       {summary['f1_score_mean']:.4f} ± {summary['f1_score_std']:.4f}")
    print(f"  Train Accuracy: {summary['train_accuracy_mean']:.4f} ± {summary['train_accuracy_std']:.4f}")
    print(f"  Per-Rep Stability: Acc std={summary['rep_accuracy_std']:.4f}")
    print(f"  Total Time:     {summary['total_train_time']:.2f}s")
    
    return {
        'fold_results': fold_results,
        'summary': summary,
        'repetition_means': {
            'accuracies': rep_mean_accs,
            'f1_scores': rep_mean_f1s
        }
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================
def run_enzymes_benchmark():
    """ENZYMES: RF (n=500, max_depth=20) + polynomial (bins=150, range=4.0)"""
    print("\n" + "="*80)
    print("ENZYMES BENCHMARK - POLYNOMIAL (RAW EMBEDDINGS)")
    print("Classifier: Random Forest (n_estimators=500, max_depth=20)")
    print("Embedding: polynomial (bins=150, range=4.0)")
    print(f"CV: {N_SPLITS}-fold x {N_REPETITIONS} repetitions = {N_SPLITS * N_REPETITIONS} iterations")
    print("="*80)
    
    # Load data
    graphs, labels = load_enzymes_data()
    
    # Generate embedding
    X, gen_time, gen_mem = generate_embedding(
        graphs, 
        func_type='polynomial',
        bins=150,
        range_val=4.0
    )
    
    # Create classifier (RAW - no scaler)
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Run CV
    results = run_repeated_stratified_kfold_cv(X, labels, clf, "Random Forest")
    
    # Add metadata
    results['dataset'] = 'ENZYMES'
    results['classifier'] = 'Random Forest (n=500, max_depth=20)'
    results['embedding'] = 'polynomial (bins=150, range=4.0)'
    results['embedding_dim'] = X.shape[1]
    results['generation_time'] = gen_time
    results['generation_memory_mb'] = gen_mem
    results['n_samples'] = len(labels)
    results['n_classes'] = len(set(labels))
    results['preprocessing'] = 'None (raw embeddings)'
    
    # Cleanup
    del graphs, X
    gc.collect()
    
    return results


def run_enzymes_harmonic_benchmark():
    """ENZYMES: RF (n=500, max_depth=20) + harmonic (bins=150, range=30.3)"""
    print("\n" + "="*80)
    print("ENZYMES BENCHMARK - HARMONIC (RAW EMBEDDINGS)")
    print("Classifier: Random Forest (n_estimators=500, max_depth=20)")
    print("Embedding: harmonic (bins=150, range=30.3)")
    print(f"CV: {N_SPLITS}-fold x {N_REPETITIONS} repetitions = {N_SPLITS * N_REPETITIONS} iterations")
    print("="*80)
    
    # Load data
    graphs, labels = load_enzymes_data()
    
    # Generate embedding
    X, gen_time, gen_mem = generate_embedding(
        graphs, 
        func_type='harmonic',
        bins=150,
        range_val=30.3
    )
    
    # Create classifier (RAW - no scaler)
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Run CV
    results = run_repeated_stratified_kfold_cv(X, labels, clf, "Random Forest")
    
    # Add metadata
    results['dataset'] = 'ENZYMES'
    results['classifier'] = 'Random Forest (n=500, max_depth=20)'
    results['embedding'] = 'harmonic (bins=150, range=30.3)'
    results['embedding_dim'] = X.shape[1]
    results['generation_time'] = gen_time
    results['generation_memory_mb'] = gen_mem
    results['n_samples'] = len(labels)
    results['n_classes'] = len(set(labels))
    results['preprocessing'] = 'None (raw embeddings)'
    
    # Cleanup
    del graphs, X
    gc.collect()
    
    return results


def run_imdb_benchmark():
    """IMDB-MULTI: MLP (256,128,64, max_iter=1000, early_stopping) + polynomial (bins=150, range=3.1)"""
    print("\n" + "="*80)
    print("IMDB-MULTI BENCHMARK - POLYNOMIAL (RAW EMBEDDINGS)")
    print("Classifier: MLP (256,128,64, max_iter=1000, early_stopping=True)")
    print("Embedding: polynomial (bins=150, range=3.1)")
    print(f"CV: {N_SPLITS}-fold x {N_REPETITIONS} repetitions = {N_SPLITS * N_REPETITIONS} iterations")
    print("="*80)
    
    # Load data
    graphs, labels = load_imdb_data()
    
    # Generate embedding
    X, gen_time, gen_mem = generate_embedding(
        graphs,
        func_type='polynomial',
        bins=150,
        range_val=3.1
    )
    
    # Create classifier (RAW - no scaler)
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=1000,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
    
    # Run CV
    results = run_repeated_stratified_kfold_cv(X, labels, clf, "MLP (256,128,64)")
    
    # Add metadata
    results['dataset'] = 'IMDB-MULTI'
    results['classifier'] = 'MLP (256,128,64, max_iter=1000, early_stopping)'
    results['embedding'] = 'polynomial (bins=150, range=3.1)'
    results['embedding_dim'] = X.shape[1]
    results['generation_time'] = gen_time
    results['generation_memory_mb'] = gen_mem
    results['n_samples'] = len(labels)
    results['n_classes'] = len(set(labels))
    results['preprocessing'] = 'None (raw embeddings)'
    
    # Cleanup
    del graphs, X
    gc.collect()
    
    return results


def run_imdb_harmonic_benchmark():
    """IMDB-MULTI: MLP (256,128,64, max_iter=1000, early_stopping) + harmonic (bins=100, range=3.1)"""
    print("\n" + "="*80)
    print("IMDB-MULTI BENCHMARK - HARMONIC (RAW EMBEDDINGS)")
    print("Classifier: MLP (256,128,64, max_iter=1000, early_stopping=True)")
    print("Embedding: harmonic (bins=100, range=3.1)")
    print(f"CV: {N_SPLITS}-fold x {N_REPETITIONS} repetitions = {N_SPLITS * N_REPETITIONS} iterations")
    print("="*80)
    
    # Load data
    graphs, labels = load_imdb_data()
    
    # Generate embedding
    X, gen_time, gen_mem = generate_embedding(
        graphs,
        func_type='harmonic',
        bins=100,
        range_val=3.1
    )
    
    # Create classifier (RAW - no scaler)
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=1000,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
    
    # Run CV
    results = run_repeated_stratified_kfold_cv(X, labels, clf, "MLP (256,128,64)")
    
    # Add metadata
    results['dataset'] = 'IMDB-MULTI'
    results['classifier'] = 'MLP (256,128,64, max_iter=1000, early_stopping)'
    results['embedding'] = 'harmonic (bins=100, range=3.1)'
    results['embedding_dim'] = X.shape[1]
    results['generation_time'] = gen_time
    results['generation_memory_mb'] = gen_mem
    results['n_samples'] = len(labels)
    results['n_classes'] = len(set(labels))
    results['preprocessing'] = 'None (raw embeddings)'
    
    # Cleanup
    del graphs, X
    gc.collect()
    
    return results


def run_reddit_benchmark():
    """REDDIT: RF (n=500, max_depth=20) + harmonic (bins=500, range=14.61) from cache"""
    print("\n" + "="*80)
    print("REDDIT-MULTI-12K BENCHMARK - HARMONIC (RAW EMBEDDINGS)")
    print("Classifier: Random Forest (n_estimators=500, max_depth=20)")
    print("Embedding: harmonic (bins=500, range=14.61) - LOADED FROM CACHE")
    print(f"CV: {N_SPLITS}-fold x {N_REPETITIONS} repetitions = {N_SPLITS * N_REPETITIONS} iterations")
    print("="*80)
    
    # Load embedding from cache
    X = load_reddit_embedding_from_cache(func_type='harmonic', bins=500, range_val=14.61)
    
    # Load labels
    labels = load_reddit_labels()
    
    # Verify sizes match
    if len(labels) != X.shape[0]:
        raise ValueError(f"Size mismatch: labels={len(labels)}, embedding={X.shape[0]}")
    
    # Create classifier (RAW - no scaler)
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Run CV
    results = run_reddit_benchmark()
    
    # Add metadata
    results['dataset'] = 'REDDIT-MULTI-12K'
    results['classifier'] = 'Random Forest (n=500, max_depth=20)'
    results['embedding'] = 'harmonic (bins=500, range=14.61) [cached]'
    results['embedding_dim'] = X.shape[1]
    results['generation_time'] = 0.0  # Loaded from cache
    results['generation_memory_mb'] = 0.0
    results['n_samples'] = len(labels)
    results['n_classes'] = len(set(labels))
    results['preprocessing'] = 'None (raw embeddings)'
    
    # Cleanup
    del X
    gc.collect()
    
    return results


def run_reddit_polynomial_benchmark():
    """REDDIT: RF (n=500, max_depth=20) + polynomial (bins=200, range=3.48) from cache"""
    print("\n" + "="*80)
    print("REDDIT-MULTI-12K BENCHMARK - POLYNOMIAL (RAW EMBEDDINGS)")
    print("Classifier: Random Forest (n_estimators=500, max_depth=20)")
    print("Embedding: polynomial (bins=200, range=3.48) - LOADED FROM CACHE")
    print(f"CV: {N_SPLITS}-fold x {N_REPETITIONS} repetitions = {N_SPLITS * N_REPETITIONS} iterations")
    print("="*80)
    
    # Load embedding from cache
    X = load_reddit_embedding_from_cache(func_type='polynomial', bins=200, range_val=3.48)
    
    # Load labels
    labels = load_reddit_labels()
    
    # Verify sizes match
    if len(labels) != X.shape[0]:
        raise ValueError(f"Size mismatch: labels={len(labels)}, embedding={X.shape[0]}")
    
    # Create classifier (RAW - no scaler)
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Run CV
    results = run_repeated_stratified_kfold_cv(X, labels, clf, "Random Forest")
    
    # Add metadata
    results['dataset'] = 'REDDIT-MULTI-12K'
    results['classifier'] = 'Random Forest (n=500, max_depth=20)'
    results['embedding'] = 'polynomial (bins=200, range=3.48) [cached]'
    results['embedding_dim'] = X.shape[1]
    results['generation_time'] = 0.0  # Loaded from cache
    results['generation_memory_mb'] = 0.0
    results['n_samples'] = len(labels)
    results['n_classes'] = len(set(labels))
    results['preprocessing'] = 'None (raw embeddings)'
    
    # Cleanup
    del X
    gc.collect()
    
    return results


def save_results(all_results: List[Dict], output_dir: str = RESULTS_DIR):
    """Save results to CSV files with comprehensive statistics."""
    
    # 1. Summary CSV (one row per dataset)
    summary_rows = []
    for res in all_results:
        s = res['summary']
        row = {
            'dataset': res['dataset'],
            'classifier': res['classifier'],
            'embedding': res['embedding'],
            'embedding_dim': res['embedding_dim'],
            'preprocessing': res['preprocessing'],
            'n_samples': res['n_samples'],
            'n_classes': res['n_classes'],
            'n_splits': s['n_splits'],
            'n_repetitions': s['n_repetitions'],
            'total_iterations': s['total_iterations'],
            
            # Main metrics
            'test_accuracy_mean': s['test_accuracy_mean'],
            'test_accuracy_std': s['test_accuracy_std'],
            'test_accuracy_min': s['test_accuracy_min'],
            'test_accuracy_max': s['test_accuracy_max'],
            'test_accuracy_median': s['test_accuracy_median'],
            'test_accuracy_ci95_low': s['test_accuracy_ci95_low'],
            'test_accuracy_ci95_high': s['test_accuracy_ci95_high'],
            'test_accuracy_iqr': s['test_accuracy_iqr'],
            'test_accuracy_cv': s['test_accuracy_cv'],
            
            'f1_score_mean': s['f1_score_mean'],
            'f1_score_std': s['f1_score_std'],
            'f1_score_min': s['f1_score_min'],
            'f1_score_max': s['f1_score_max'],
            
            'train_accuracy_mean': s['train_accuracy_mean'],
            'train_accuracy_std': s['train_accuracy_std'],
            
            # Overall combined
            'overall_accuracy': s['overall_accuracy'],
            'overall_f1': s['overall_f1'],
            
            # Repetition stability
            'rep_accuracy_mean': s['rep_accuracy_mean'],
            'rep_accuracy_std': s['rep_accuracy_std'],
            
            # Timing
            'train_time_mean': s['train_time_mean'],
            'train_time_std': s['train_time_std'],
            'total_train_time': s['total_train_time'],
            'generation_time': res['generation_time'],
            'generation_memory_mb': res['generation_memory_mb']
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'cv_benchmark_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved to: {summary_path}")
    
    # 2. Detailed fold results CSV
    fold_rows = []
    for res in all_results:
        for fold_res in res['fold_results']:
            row = {
                'dataset': res['dataset'],
                'classifier': res['classifier'],
                'embedding': res['embedding'],
                'preprocessing': res['preprocessing'],
                **fold_res
            }
            fold_rows.append(row)
    
    fold_df = pd.DataFrame(fold_rows)
    fold_path = os.path.join(output_dir, 'cv_benchmark_fold_details.csv')
    fold_df.to_csv(fold_path, index=False)
    print(f"✅ Fold details saved to: {fold_path}")
    
    # 3. Per-repetition summary CSV
    rep_rows = []
    for res in all_results:
        rep_means = res['repetition_means']
        for rep_idx, (acc, f1) in enumerate(zip(rep_means['accuracies'], rep_means['f1_scores'])):
            row = {
                'dataset': res['dataset'],
                'classifier': res['classifier'],
                'repetition': rep_idx + 1,
                'mean_accuracy': acc,
                'mean_f1': f1
            }
            rep_rows.append(row)
    
    rep_df = pd.DataFrame(rep_rows)
    rep_path = os.path.join(output_dir, 'cv_benchmark_per_repetition.csv')
    rep_df.to_csv(rep_path, index=False)
    print(f"✅ Per-repetition results saved to: {rep_path}")
    
    return summary_path, fold_path, rep_path


def print_final_summary(all_results: List[Dict]):
    """Print comprehensive final summary table."""
    print("\n" + "="*120)
    print(f"FINAL SUMMARY - {N_SPLITS}-Fold Stratified CV x {N_REPETITIONS} Repetitions (RAW EMBEDDINGS)")
    print("="*120)
    
    print(f"\n{'Dataset':<18} {'Mean Acc':<12} {'± Std':<10} {'95% CI':<22} {'Min-Max':<18} {'F1':<12}")
    print("-"*120)
    
    for res in all_results:
        s = res['summary']
        ci_str = f"[{s['test_accuracy_ci95_low']:.4f}, {s['test_accuracy_ci95_high']:.4f}]"
        range_str = f"[{s['test_accuracy_min']:.4f}, {s['test_accuracy_max']:.4f}]"
        
        print(f"{res['dataset']:<18} {s['test_accuracy_mean']:<12.4f} {s['test_accuracy_std']:<10.4f} "
              f"{ci_str:<22} {range_str:<18} {s['f1_score_mean']:<12.4f}")
    
    print("-"*120)
    
    # Additional insights
    print("\n📊 Additional Statistics:")
    for res in all_results:
        s = res['summary']
        print(f"\n  {res['dataset']}:")
        print(f"    • Coefficient of Variation: {s['test_accuracy_cv']*100:.2f}%")
        print(f"    • IQR: {s['test_accuracy_iqr']:.4f}")
        print(f"    • Repetition Stability (std of rep means): {s['rep_accuracy_std']:.4f}")
        print(f"    • Train vs Test gap: {s['train_accuracy_mean'] - s['test_accuracy_mean']:.4f}")
        print(f"    • Total training time: {s['total_train_time']:.1f}s")
    
    print("\n" + "="*120)
    
    # Best result
    best = max(all_results, key=lambda x: x['summary']['test_accuracy_mean'])
    s = best['summary']
    print(f"\n🏆 Best: {best['dataset']} with {best['classifier']}")
    print(f"   Mean Accuracy: {s['test_accuracy_mean']:.4f} ± {s['test_accuracy_std']:.4f}")
    print(f"   95% CI: [{s['test_accuracy_ci95_low']:.4f}, {s['test_accuracy_ci95_high']:.4f}]")


def main():
    print("="*80)
    print(f"{N_SPLITS}-FOLD STRATIFIED CV x {N_REPETITIONS} REPETITIONS BENCHMARK")
    print("(RAW EMBEDDINGS - NO PREPROCESSING)")
    print("="*80)
    print("\nConfigurations:")
    print("  1. ENZYMES: RF (n=500, max_depth=20) + polynomial (150, 4.0)")
    print("  2. ENZYMES: RF (n=500, max_depth=20) + harmonic (150, 30.3)")
    print("  3. IMDB-MULTI: MLP (256,128,64) + polynomial (150, 3.1)")
    print("  4. IMDB-MULTI: MLP (256,128,64) + harmonic (100, 3.1)")
    print("  5. REDDIT: RF (n=500, max_depth=20) + harmonic (500, 14.61) [cached]")
    print("  6. REDDIT: RF (n=500, max_depth=20) + polynomial (200, 3.48) [cached]")
    print(f"\nTotal iterations per config: {N_SPLITS * N_REPETITIONS}")
    print("="*80)
    
    all_results = []
    
    # 1. ENZYMES - Polynomial
    try:
        enzymes_results = run_enzymes_benchmark()
        all_results.append(enzymes_results)
    except Exception as e:
        print(f"\n❌ ENZYMES (polynomial) benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. ENZYMES - Harmonic
    try:
        enzymes_harm_results = run_enzymes_harmonic_benchmark()
        all_results.append(enzymes_harm_results)
    except Exception as e:
        print(f"\n❌ ENZYMES (harmonic) benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. IMDB-MULTI - Polynomial
    try:
        imdb_results = run_imdb_benchmark()
        all_results.append(imdb_results)
    except Exception as e:
        print(f"\n❌ IMDB-MULTI (polynomial) benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. IMDB-MULTI - Harmonic
    try:
        imdb_harm_results = run_imdb_harmonic_benchmark()
        all_results.append(imdb_harm_results)
    except Exception as e:
        print(f"\n❌ IMDB-MULTI (harmonic) benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. REDDIT - Harmonic
    try:
        reddit_results = run_reddit_benchmark()
        all_results.append(reddit_results)
    except Exception as e:
        print(f"\n❌ REDDIT (harmonic) benchmark failed: {e}")
        print("  Make sure to run memory_benchmark first to generate the cached embedding.")
    
    # 6. REDDIT - Polynomial
    try:
        reddit_poly_results = run_reddit_polynomial_benchmark()
        all_results.append(reddit_poly_results)
    except Exception as e:
        print(f"\n❌ REDDIT (polynomial) benchmark failed: {e}")
        print("  Make sure to run memory_benchmark first to generate the cached embedding.")
    
    if all_results:
        # Save results
        save_results(all_results)
        
        # Print final summary
        print_final_summary(all_results)
    else:
        print("\n❌ No benchmarks completed successfully.")


if __name__ == "__main__":
    main()
