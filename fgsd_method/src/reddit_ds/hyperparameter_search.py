"""
Hyperparameter tuning for classifiers on REDDIT-MULTI-12K dataset.
Grid search for Random Forest and SVM parameters.
"""

import os
import gc
import time
import json
from typing import Dict, Any, Tuple, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .config import RESULTS_DIR, CACHE_DIR


# Parameter grids
RF_PARAM_GRID = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

SVM_PARAM_GRID = {
    'svc__C': [0.1, 1, 10, 100, 1000],
    'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'svc__kernel': ['rbf', 'poly', 'sigmoid']
}

# Reduced grids for faster search
RF_PARAM_GRID_FAST = {
    'n_estimators': [100, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

SVM_PARAM_GRID_FAST = {
    'svc__C': [1, 10, 100],
    'svc__gamma': ['scale', 0.01, 0.1],
    'svc__kernel': ['rbf']
}


def tune_random_forest(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    param_grid: Optional[Dict] = None,
    cv: int = 3,  # Reduced for large dataset
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1
) -> Tuple[RandomForestClassifier, Dict[str, Any], float]:
    """Tune Random Forest hyperparameters using GridSearchCV."""
    if param_grid is None:
        param_grid = RF_PARAM_GRID_FAST
    
    print("\n" + "="*60)
    print("TUNING RANDOM FOREST")
    print(f"Parameter grid: {param_grid}")
    print("="*60)
    
    rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\nGrid search completed in {search_time:.2f}s")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_svm(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    param_grid: Optional[Dict] = None,
    cv: int = 3,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1
) -> Tuple[Any, Dict[str, Any], float]:
    """Tune SVM hyperparameters using GridSearchCV."""
    if param_grid is None:
        param_grid = SVM_PARAM_GRID_FAST
    
    print("\n" + "="*60)
    print("TUNING SVM (with StandardScaler)")
    print(f"Parameter grid: {param_grid}")
    print("="*60)
    
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=random_state)
    )
    
    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\nGrid search completed in {search_time:.2f}s")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def run_classifier_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fast_mode: bool = True,
    cv: int = 3,
    random_state: int = 42,
    save_results: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Run hyperparameter tuning for all classifiers."""
    print("\n" + "="*80)
    print("CLASSIFIER HYPERPARAMETER TUNING (REDDIT)")
    print(f"Training data shape: {X_train.shape}")
    print(f"Fast mode: {fast_mode}")
    print(f"CV folds: {cv}")
    print("="*80)
    
    results = {}
    
    # Tune Random Forest
    rf_grid = RF_PARAM_GRID_FAST if fast_mode else RF_PARAM_GRID
    rf_model, rf_params, rf_score = tune_random_forest(
        X_train, y_train, rf_grid, cv, random_state=random_state
    )
    results['Random Forest'] = {
        'model': rf_model,
        'params': rf_params,
        'cv_score': rf_score
    }
    gc.collect()
    
    # Tune SVM
    svm_grid = SVM_PARAM_GRID_FAST if fast_mode else SVM_PARAM_GRID
    svm_model, svm_params, svm_score = tune_svm(
        X_train, y_train, svm_grid, cv, random_state=random_state
    )
    results['SVM (RBF)'] = {
        'model': svm_model,
        'params': svm_params,
        'cv_score': svm_score
    }
    gc.collect()
    
    # Print summary
    print("\n" + "="*80)
    print("TUNING SUMMARY")
    print("="*80)
    print(f"{'Classifier':<20} {'Best CV Score':<15} {'Best Parameters'}")
    print("-"*80)
    for clf_name, res in results.items():
        params_str = str(res['params'])[:50] + "..." if len(str(res['params'])) > 50 else str(res['params'])
        print(f"{clf_name:<20} {res['cv_score']:<15.4f} {params_str}")
    
    # Save to cache
    if save_results:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, 'reddit_tuned_params.json')
        
        params_to_save = {
            clf_name: {
                'params': res['params'],
                'cv_score': res['cv_score']
            }
            for clf_name, res in results.items()
        }
        
        with open(cache_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        print(f"\n✅ Tuned parameters saved to: {cache_path}")
    
    return results


def load_tuned_params() -> Optional[Dict[str, Dict[str, Any]]]:
    """Load previously tuned parameters from cache."""
    cache_path = os.path.join(CACHE_DIR, 'reddit_tuned_params.json')
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            params = json.load(f)
        print(f"✅ Loaded tuned parameters from: {cache_path}")
        return params
    return None


def get_tuned_classifiers(tuned_params: Dict[str, Dict[str, Any]], random_state: int = 42) -> Dict[str, Any]:
    """Create classifier instances with tuned parameters."""
    classifiers = {}
    
    if 'Random Forest' in tuned_params:
        rf_params = tuned_params['Random Forest']['params'].copy()
        rf_params['random_state'] = random_state
        rf_params['n_jobs'] = -1
        classifiers['Random Forest (Tuned)'] = RandomForestClassifier(**rf_params)
    
    if 'SVM (RBF)' in tuned_params:
        svm_params = tuned_params['SVM (RBF)']['params'].copy()
        svc_params = {k.replace('svc__', ''): v for k, v in svm_params.items() if k.startswith('svc__')}
        svc_params['probability'] = True
        svc_params['random_state'] = random_state
        classifiers['SVM (Tuned)'] = make_pipeline(StandardScaler(), SVC(**svc_params))
    
    return classifiers
