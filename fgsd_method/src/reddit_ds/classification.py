"""
Classification for FGSD experiments on REDDIT-MULTI-12K.
Core functions for embedding generation and evaluation.
"""

import gc
import time
from typing import List, Dict, Any

import numpy as np
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
from .config import DATASET_DIR, BATCH_SIZE
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
    """Return dictionary of classifiers to use."""
    return {
        'SVM (RBF)': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=random_state)),
        'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1),
        'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, early_stopping=True, random_state=random_state))
    }


def get_classifiers_tuned_or_default(X_train=None, y_train=None, use_tuned=False, random_state=42):
    """Get classifiers - either default or tuned via grid search."""
    if not use_tuned:
        return get_classifiers(random_state)
    
    from .hyperparameter_search import get_tuned_classifiers, load_tuned_params
    
    tuned_params = load_tuned_params()
    if tuned_params is not None:
        return get_tuned_classifiers(tuned_params, random_state)
    
    # No cached params, return defaults
    return get_classifiers(random_state)


def generate_all_embeddings(func_type, bins, range_val, random_state=42):
    """
    Generate embeddings for ALL graphs batch-wise.
    This is the core embedding generation function.
    """
    ensure_dataset_ready()
    
    model = FlexibleFGSD(hist_bins=bins, hist_range=range_val, func_type=func_type, seed=random_state)
    
    embeddings_list = []
    
    for graphs, labels, _ in tqdm(iter_graph_batches(DATASET_DIR, BATCH_SIZE), desc=f"  {func_type}"):
        model.fit(graphs)
        batch_emb = model.get_embedding()
        embeddings_list.append(batch_emb)
        del graphs
        gc.collect()
    
    X_all = np.vstack(embeddings_list)
    return X_all


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
        elif 'biharm_bins' in r:
            config = f"bh:{r['biharm_bins']}/{r['biharm_range']}, p:{r['pol_bins']}/{r['pol_range']}"
        else:
            config = f"bins={r.get('bins', 'N/A')}, range={r.get('range', 'N/A')}"
        
        print(f"{r['func']:<20} {config:<30} {r['classifier']:<18} "
              f"{r['train_accuracy']:<10.4f} {r['accuracy']:<10.4f} {r['f1_score']:<10.4f} "
              f"{auc:<10} {r.get('embedding_dim', 'N/A'):<8}")
    
    best = sorted(results, key=lambda x: -x['accuracy'])[0]
    print(f"\nBEST: {best['func']} with {best['classifier']} -> Accuracy: {best['accuracy']:.4f}")


# Legacy functions kept for backward compatibility but not used in main pipeline
def run_dimension_analysis(optimal_params, bin_sizes=[100, 200, 500], test_size=0.15, random_state=42):
    """Legacy function - prefer using run_dimension_analysis_with_cache in classification_main.py"""
    raise NotImplementedError("Use run_dimension_analysis_with_cache from classification_main.py instead")


def run_final_classification(best_embeddings, train_idx, test_idx, y_train, y_test, random_state=42):
    """Legacy function - prefer using run_final_classification_with_cache in classification_main.py"""
    raise NotImplementedError("Use run_final_classification_with_cache from classification_main.py instead")
