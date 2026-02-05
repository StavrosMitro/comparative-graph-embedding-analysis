"""
Classification for FGSD experiments on IMDB-MULTI.
"""

import gc
import time
import tracemalloc
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from .config import DATASET_DIR, OptimalParams
from .data_loader import ensure_dataset_ready, load_all_graphs

# Results directory for raw embeddings
RAW_RESULTS_DIR = os.path.join(parent_dir, 'results', 'raw_embeddings')


def evaluate_classifier(X_train, X_test, y_train, y_test, classifier_name, clf):
    """Evaluate a classifier and return metrics."""
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    try:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test)
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
        'train_time': train_time
    }


def get_classifiers(random_state=42):
    return {
        'SVM (RBF)': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=random_state)),
        'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1),
        'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, early_stopping=True, random_state=random_state))
    }


def get_raw_classifiers(random_state=42):
    """Get all classifiers without any preprocessing (raw embeddings)."""
    return {
        'SVM (RBF) Raw': SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=random_state),
        'Random Forest Raw': RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1),
        'MLP Raw': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, early_stopping=True, random_state=random_state)
    }


def get_classifiers_tuned_or_default(X_train=None, y_train=None, use_tuned=False, random_state=42):
    """
    Get classifiers - either default or tuned via grid search.
    
    Args:
        X_train: Training features (required if use_tuned=True)
        y_train: Training labels (required if use_tuned=True)
        use_tuned: Whether to use hyperparameter tuning
        random_state: Random seed
    
    Returns:
        Dictionary of classifier name -> classifier instance
    """
    if not use_tuned:
        return get_classifiers(random_state)
    
    # Use tuned classifiers
    from .hyperparameter_search import get_classifiers_with_tuning
    
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train required for tuning")
    
    return get_classifiers_with_tuning(
        X_train, y_train,
        use_cache=True,
        fast_mode=True,
        random_state=random_state
    )


def generate_embeddings_with_tracking(graphs_train, graphs_test, func_type, bins, range_val, random_state=42):
    """Generate embeddings with time and memory tracking."""
    gc.collect()
    tracemalloc.start()
    start_time = time.time()
    
    model = FlexibleFGSD(hist_bins=bins, hist_range=range_val, func_type=func_type, seed=random_state)
    model.fit(graphs_train)
    X_train = model.get_embedding()
    X_test = model.infer(graphs_test)
    
    generation_time = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak_mem / 1024 / 1024
    
    return X_train, X_test, generation_time, memory_mb


def run_dimension_analysis(
    optimal_params: Dict[str, OptimalParams],
    bin_sizes: List[int],
    test_size: float = 0.15,
    random_state: int = 42,
    raw_mode: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[int, float]]]:
    """Run dimension analysis with bin sizes from preanalysis."""
    mode_str = "(Raw Embeddings)" if raw_mode else ""
    print("\n" + "="*80)
    print(f"DIMENSION ANALYSIS: Accuracy vs Compute Cost {mode_str}")
    print(f"Testing bin sizes: {bin_sizes}")
    print("="*80)
    
    ensure_dataset_ready()
    graphs, labels = load_all_graphs(DATASET_DIR)
    
    graphs_train, graphs_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    all_results = []
    best_config = {}
    classifiers = get_raw_classifiers(random_state) if raw_mode else get_classifiers(random_state)
    
    for func_type in ['harmonic', 'polynomial']:
        if func_type not in optimal_params:
            continue
        
        range_val = round(optimal_params[func_type].range_val, 2)
        print(f"\n{'='*60}")
        print(f"Function: {func_type.upper()} (range={range_val})")
        print(f"{'='*60}")
        
        best_acc, best_bins = 0, bin_sizes[0]
        
        for bins in bin_sizes:
            print(f"\n--- bins={bins} ---")
            
            X_train, X_test, gen_time, memory_mb = generate_embeddings_with_tracking(
                graphs_train, graphs_test, func_type, bins, range_val, random_state
            )
            
            print(f"  Shape: {X_train.shape}, Time: {gen_time:.2f}s, Memory: {memory_mb:.2f} MB")
            
            for clf_name, clf in classifiers.items():
                res = evaluate_classifier(X_train, X_test, y_train, y_test, clf_name, clf)
                
                result_entry = {
                    'func': func_type, 'bins': bins, 'range': range_val,
                    'embedding_dim': bins, 'generation_time': gen_time, 'memory_mb': memory_mb,
                    **res
                }
                all_results.append(result_entry)
                
                auc_str = f"{res['auc']:.4f}" if res['auc'] else "N/A"
                print(f"  {clf_name}: Acc={res['accuracy']:.4f}, F1={res['f1_score']:.4f}, AUC={auc_str}")
                
                if res['accuracy'] > best_acc:
                    best_acc, best_bins = res['accuracy'], bins
            
            del X_train, X_test
            gc.collect()
        
        best_config[func_type] = (best_bins, best_acc)
        print(f"\n✓ Best for {func_type}: bins={best_bins} (acc={best_acc:.4f})")
    
    return all_results, best_config


def run_final_classification(optimal_params, recommended_bins=None, test_size=0.15, random_state=42, raw_mode=False):
    """
    Run final classification testing ALL bin sizes.
    IMDB has no node labels, so only spectral features are used.
    """
    mode_str = "(Raw Embeddings)" if raw_mode else ""
    print(f"\n{'='*80}\nFINAL CLASSIFICATION {mode_str}\n{'='*80}")
    
    ensure_dataset_ready()
    graphs, labels = load_all_graphs(DATASET_DIR)
    
    print(f"IMDB-MULTI: No node labels available. Using spectral features only.")
    
    graphs_train, graphs_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=test_size, random_state=random_state, stratify=labels)
    
    results = []
    classifiers = get_raw_classifiers(random_state) if raw_mode else get_classifiers(random_state)
    
    # Determine bin sizes to test
    if recommended_bins is None:
        recommended_bins = {'harmonic': [50, 100, 150], 'polynomial': [50, 100, 150]}
    
    all_bin_sizes = set()
    for bins_list in recommended_bins.values():
        all_bin_sizes.update(bins_list)
    all_bin_sizes = sorted(all_bin_sizes)
    
    print(f"Testing bin sizes: {all_bin_sizes}")
    
    best_embeddings = {}  # Store best for hybrid creation
    
    # =================================================================
    # Test each function type with each bin size
    # =================================================================
    for func_type in ['harmonic', 'polynomial']:
        if func_type not in optimal_params:
            continue
        
        range_val = round(optimal_params[func_type].range_val, 2)
        bins_to_test = recommended_bins.get(func_type, all_bin_sizes)
        
        print(f"\n--- {func_type.upper()} (range={range_val}) ---")
        
        func_best_acc = 0
        func_best_bins = bins_to_test[0]
        
        for bins in bins_to_test:
            X_train, X_test, gen_time, memory_mb = generate_embeddings_with_tracking(
                graphs_train, graphs_test, func_type, bins, range_val, random_state)
            
            for clf_name, clf in classifiers.items():
                res = evaluate_classifier(X_train, X_test, y_train, y_test, clf_name, clf)
                result_entry = {
                    'func': func_type,
                    'bins': bins, 'range': range_val,
                    'generation_time': gen_time, 'memory_mb': memory_mb,
                    'embedding_dim': X_train.shape[1], **res
                }
                results.append(result_entry)
                
                if res['accuracy'] > func_best_acc:
                    func_best_acc = res['accuracy']
                    func_best_bins = bins
            
            print(f"  bins={bins}: best={max(r['accuracy'] for r in results if r['func']==func_type and r.get('bins')==bins):.4f}")
        
        # Store best embeddings for hybrid
        X_train_best, X_test_best, _, _ = generate_embeddings_with_tracking(
            graphs_train, graphs_test, func_type, func_best_bins, range_val, random_state)
        best_embeddings[func_type] = {
            'X_train': X_train_best, 'X_test': X_test_best,
            'bins': func_best_bins, 'range': range_val
        }
        print(f"  ✓ Best bins for {func_type}: {func_best_bins} (acc={func_best_acc:.4f})")
    
    # =================================================================
    # Create and test naive_hybrid
    # =================================================================
    if 'harmonic' in best_embeddings and 'polynomial' in best_embeddings:
        print(f"\n--- NAIVE_HYBRID ---")
        h, p = best_embeddings['harmonic'], best_embeddings['polynomial']
        
        X_train_hybrid = np.hstack([h['X_train'], p['X_train']])
        X_test_hybrid = np.hstack([h['X_test'], p['X_test']])
        
        print(f"  shape: {X_train_hybrid.shape}")
        
        for clf_name, clf in classifiers.items():
            res = evaluate_classifier(X_train_hybrid, X_test_hybrid, y_train, y_test, clf_name, clf)
            result_entry = {
                'func': 'naive_hybrid',
                'harm_bins': h['bins'], 'harm_range': h['range'],
                'pol_bins': p['bins'], 'pol_range': p['range'],
                'embedding_dim': X_train_hybrid.shape[1], **res
            }
            results.append(result_entry)
        
        print(f"  best={max(r['accuracy'] for r in results if r['func']=='naive_hybrid'):.4f}")
    
    # Print top results
    print(f"\n--- Top 10 Results ---")
    for r in sorted(results, key=lambda x: -x['accuracy'])[:10]:
        bins_info = f"bins={r.get('bins', 'hybrid')}"
        print(f"  {r['func']:<20} {bins_info:<15} {r['classifier']:<18} Acc={r['accuracy']:.4f}")
    
    return results


def print_dimension_analysis_summary(results):
    """Print dimension analysis summary."""
    print("\n" + "="*130)
    print("DIMENSION ANALYSIS SUMMARY")
    print("="*130)
    print(f"{'Func':<12} {'Bins':<8} {'Classifier':<18} {'Accuracy':<10} {'F1':<10} {'AUC':<10} {'GenTime':<12} {'Memory':<12}")
    print("-" * 130)
    
    for r in sorted(results, key=lambda x: (x['func'], x['bins'])):
        auc = f"{r['auc']:.4f}" if r['auc'] else "N/A"
        print(f"{r['func']:<12} {r['bins']:<8} {r['classifier']:<18} {r['accuracy']:<10.4f} "
              f"{r['f1_score']:<10.4f} {auc:<10} {r['generation_time']:<12.2f} {r['memory_mb']:<12.2f}")


def print_summary(results):
    """Print final summary."""
    print(f"\n{'='*140}\nFINAL RESULTS\n{'='*140}")
    print(f"{'Func':<20} {'Config':<30} {'Classifier':<18} {'TrainAcc':<10} {'TestAcc':<10} {'F1':<10} {'AUC':<10} {'Dim':<8}")
    print("-" * 140)
    
    for r in sorted(results, key=lambda x: -x['accuracy']):
        auc = f"{r['auc']:.4f}" if r.get('auc') else "N/A"
        
        if 'harm_bins' in r:
            config = f"h:{r['harm_bins']}/{r['harm_range']}, p:{r['pol_bins']}/{r['pol_range']}"
        else:
            config = f"bins={r.get('bins', 'N/A')}, range={r.get('range', 'N/A')}"
        
        print(f"{r['func']:<20} {config:<30} {r['classifier']:<18} "
              f"{r['train_accuracy']:<10.4f} {r['accuracy']:<10.4f} {r['f1_score']:<10.4f} "
              f"{auc:<10} {r.get('embedding_dim', 'N/A'):<8}")
    
    best = sorted(results, key=lambda x: -x['accuracy'])[0]
    print(f"\nBEST: {best['func']} with {best['classifier']} -> Accuracy: {best['accuracy']:.4f}")
