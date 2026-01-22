"""
Classification for FGSD experiments on REDDIT-MULTI-12K.
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
from tqdm import tqdm

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from .config import DATASET_DIR, BATCH_SIZE, OptimalParams
from .data_loader import ensure_dataset_ready, load_metadata, iter_graph_batches


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
        'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, early_stopping=True, random_state=random_state))
    }


def generate_all_embeddings(func_type, bins, range_val, random_state=42):
    """Generate embeddings for ALL graphs batch-wise."""
    ensure_dataset_ready()
    
    model = FlexibleFGSD(hist_bins=bins, hist_range=range_val, func_type=func_type, seed=random_state)
    
    embeddings_list = []
    total_batches = 0
    
    for graphs, labels, _ in tqdm(iter_graph_batches(DATASET_DIR, BATCH_SIZE), desc=f"  {func_type}"):
        model.fit(graphs)
        batch_emb = model.get_embedding()
        embeddings_list.append(batch_emb)
        total_batches += 1
        del graphs
        gc.collect()
    
    X_all = np.vstack(embeddings_list)
    return X_all


def run_dimension_analysis(optimal_params, bin_sizes=[100, 200, 500], test_size=0.15, random_state=42):
    """Run dimension analysis with bin sizes from preanalysis."""
    print(f"\n{'='*80}\nDIMENSION ANALYSIS (Spectral Only)\n{'='*80}")
    
    ensure_dataset_ready()
    records = load_metadata(DATASET_DIR)
    all_labels = np.array([r.label for r in records])
    
    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=all_labels)
    
    results = []
    best_config = {}
    classifiers = get_classifiers(random_state)
    
    for func_type in ['harmonic', 'polynomial']:
        if func_type not in optimal_params:
            continue
        
        range_val = round(optimal_params[func_type].range_val, 2)
        print(f"\n{'='*60}\nFunction: {func_type.upper()} (range={range_val})\n{'='*60}")
        
        best_acc, best_bins = 0, bin_sizes[0]
        
        for bins in bin_sizes:
            print(f"\n--- bins={bins} ---")
            
            start_time = time.time()
            X_all = generate_all_embeddings(func_type, bins, range_val, random_state)
            gen_time = time.time() - start_time
            
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = all_labels[train_idx], all_labels[test_idx]
            
            print(f"  Shape: {X_train.shape}, Time: {gen_time:.2f}s")
            
            for clf_name, clf in classifiers.items():
                res = evaluate_classifier(X_train, X_test, y_train, y_test, clf_name, clf)
                result_entry = {
                    'func': func_type, 'bins': bins, 'range': range_val,
                    'embedding_dim': bins, 'generation_time': gen_time, **res
                }
                results.append(result_entry)
                
                auc_str = f"{res['auc']:.4f}" if res['auc'] else "N/A"
                print(f"  {clf_name}: Acc={res['accuracy']:.4f}, F1={res['f1_score']:.4f}, AUC={auc_str}")
                
                if res['accuracy'] > best_acc:
                    best_acc, best_bins = res['accuracy'], bins
            
            del X_all
            gc.collect()
        
        best_config[func_type] = (best_bins, best_acc)
        print(f"\n✓ Best for {func_type}: bins={best_bins} (acc={best_acc:.4f})")
    
    return results, best_config


def run_final_classification(optimal_params, recommended_bins=None, test_size=0.15, random_state=42):
    """
    Run final classification testing ALL bin sizes.
    REDDIT has no node labels, so only spectral features are used.
    """
    print(f"\n{'='*80}\nFINAL CLASSIFICATION\n{'='*80}")
    
    ensure_dataset_ready()
    records = load_metadata(DATASET_DIR)
    all_labels = np.array([r.label for r in records])
    
    print(f"REDDIT-MULTI-12K: No node labels available. Using spectral features only.")
    
    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=all_labels)
    
    y_train, y_test = all_labels[train_idx], all_labels[test_idx]
    
    results = []
    classifiers = get_classifiers(random_state)
    
    if recommended_bins is None:
        recommended_bins = {'harmonic': [100, 200, 500], 'polynomial': [100, 200, 500]}
    
    all_bin_sizes = set()
    for bins_list in recommended_bins.values():
        all_bin_sizes.update(bins_list)
    all_bin_sizes = sorted(all_bin_sizes)
    
    print(f"Testing bin sizes: {all_bin_sizes}")
    
    best_embeddings = {}
    
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
        func_best_X = None
        
        for bins in bins_to_test:
            start_time = time.time()
            X_all = generate_all_embeddings(func_type, bins, range_val, random_state)
            gen_time = time.time() - start_time
            
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            
            for clf_name, clf in classifiers.items():
                res = evaluate_classifier(X_train, X_test, y_train, y_test, clf_name, clf)
                result_entry = {
                    'func': func_type,
                    'bins': bins, 'range': range_val,
                    'generation_time': gen_time,
                    'embedding_dim': X_train.shape[1], **res
                }
                results.append(result_entry)
                
                if res['accuracy'] > func_best_acc:
                    func_best_acc = res['accuracy']
                    func_best_bins = bins
                    func_best_X = X_all.copy()
            
            print(f"  bins={bins}: best={max(r['accuracy'] for r in results if r['func']==func_type and r.get('bins')==bins):.4f}")
            
            del X_all
            gc.collect()
        
        # Store best embeddings for hybrid
        if func_best_X is not None:
            best_embeddings[func_type] = {
                'X_all': func_best_X,
                'bins': func_best_bins, 'range': range_val
            }
        print(f"  ✓ Best bins for {func_type}: {func_best_bins} (acc={func_best_acc:.4f})")
    
    # =================================================================
    # Create and test naive_hybrid
    # =================================================================
    if 'harmonic' in best_embeddings and 'polynomial' in best_embeddings:
        print(f"\n--- NAIVE_HYBRID ---")
        h, p = best_embeddings['harmonic'], best_embeddings['polynomial']
        
        X_hybrid = np.hstack([h['X_all'], p['X_all']])
        X_train_hybrid, X_test_hybrid = X_hybrid[train_idx], X_hybrid[test_idx]
        
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
        
        del X_hybrid
        gc.collect()
    
    # Cleanup
    for k in best_embeddings:
        del best_embeddings[k]['X_all']
    gc.collect()
    
    # Print top results
    print(f"\n--- Top 10 Results ---")
    for r in sorted(results, key=lambda x: -x['accuracy'])[:10]:
        bins_info = f"bins={r.get('bins', 'hybrid')}"
        print(f"  {r['func']:<20} {bins_info:<15} {r['classifier']:<18} Acc={r['accuracy']:.4f}")
    
    return results


def print_dimension_analysis_summary(results):
    """Print dimension analysis summary."""
    print(f"\n{'='*130}\nDIMENSION ANALYSIS SUMMARY\n{'='*130}")
    print(f"{'Func':<12} {'Bins':<8} {'Classifier':<18} {'Accuracy':<10} {'F1':<10} {'AUC':<10} {'GenTime':<12}")
    print("-" * 130)
    
    for r in sorted(results, key=lambda x: (x['func'], x['bins'])):
        auc = f"{r['auc']:.4f}" if r['auc'] else "N/A"
        print(f"{r['func']:<12} {r['bins']:<8} {r['classifier']:<18} {r['accuracy']:<10.4f} "
              f"{r['f1_score']:<10.4f} {auc:<10} {r['generation_time']:<12.2f}")


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
