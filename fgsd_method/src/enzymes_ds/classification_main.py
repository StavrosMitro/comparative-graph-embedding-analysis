"""
FGSD Classification on ENZYMES

Usage:
    python -m enzymes_ds.classification_main                    # Full pipeline (auto-detects what needs to run)
    python -m enzymes_ds.classification_main --stability        # Include stability analysis
    python -m enzymes_ds.classification_main --stability-only   # Only stability (loads best config from results)
    python -m enzymes_ds.classification_main --force            # Force rerun everything
    python -m enzymes_ds.classification_main --skip-grid        # Skip grid search
    python -m enzymes_ds.classification_main --tune-classifiers # Run classifier hyperparameter tuning
"""

import os
import sys
import warnings
import gc
import argparse

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

warnings.filterwarnings('ignore')

from enzymes_ds.config import RESULTS_DIR, PREANALYSIS_SAMPLE_SIZE, OptimalParams, DATASET_DIR
from enzymes_ds.data_loader import ensure_dataset_ready, load_all_graphs, create_node_label_features
from enzymes_ds.preanalysis import run_sampled_preanalysis
from enzymes_ds.classification import (
    run_dimension_analysis, run_final_classification,
    print_dimension_analysis_summary, print_summary,
    generate_embeddings_with_tracking, evaluate_classifier, get_classifiers,
    get_classifiers_tuned_or_default
)


def check_cached_results():
    """Check which CACHED results exist (only preanalysis and grid)."""
    exists = {
        'preanalysis': os.path.exists(os.path.join(parent_dir, 'cache', 'enzymes_preanalysis_cache.json')),
        'grid': os.path.exists(os.path.join(RESULTS_DIR, 'fgsd_enzymes_grid_search.csv')),
    }
    return exists


def check_existing_results():
    """Check which results already exist."""
    exists = {
        'preanalysis': os.path.exists(os.path.join(parent_dir, 'cache', 'enzymes_preanalysis_cache.json')),
        'grid': os.path.exists(os.path.join(RESULTS_DIR, 'fgsd_enzymes_grid_search.csv')),
        'dimension': os.path.exists(os.path.join(RESULTS_DIR, 'fgsd_enzymes_dimension_analysis.csv')),
        'final': os.path.exists(os.path.join(RESULTS_DIR, 'fgsd_enzymes_final_results.csv')),
        'stability': os.path.exists(os.path.join(RESULTS_DIR, 'fgsd_enzymes_stability_results.csv')),
    }
    return exists


def load_best_config_from_results():
    """Load best configuration from existing results CSV."""
    final_path = os.path.join(RESULTS_DIR, 'fgsd_enzymes_final_results.csv')
    dim_path = os.path.join(RESULTS_DIR, 'fgsd_enzymes_dimension_analysis.csv')
    
    for path in [final_path, dim_path]:
        if os.path.exists(path):
            print(f"Loading best config from: {path}")
            df = pd.read_csv(path)
            
            optimal_params = {}
            
            for func_type in ['harmonic', 'polynomial']:
                func_df = df[df['func'] == func_type]
                if len(func_df) > 0:
                    best_row = func_df.loc[func_df['accuracy'].idxmax()]
                    bins = int(best_row['bins']) if pd.notna(best_row.get('bins')) else 100
                    range_val = float(best_row['range']) if pd.notna(best_row.get('range')) else 30.0
                    
                    optimal_params[func_type] = OptimalParams(
                        func_type=func_type,
                        bins=bins,
                        range_val=range_val,
                        p99=range_val,
                        recommended_bins=bins
                    )
                    print(f"  {func_type.upper()}: bins={bins}, range={range_val}")
            
            if optimal_params:
                return optimal_params
    
    raise FileNotFoundError(
        f"No results found in {RESULTS_DIR}. "
        "Run full pipeline first: python -m enzymes_ds.classification_main"
    )


def run_grid_search(optimal_params, test_size=0.15, random_state=42, use_node_labels=True):
    """
    Run grid search with binwidth h ∈ {0.05, 0.1, 0.2, 0.5, 1}.
    bins = range / h (as per FGSD paper methodology)
    Uses only Random Forest classifier.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    
    print("\n" + "="*80)
    print("GRID SEARCH: bins = range / binwidth")
    print("binwidth (h) values: [0.05, 0.1, 0.2, 0.5, 1]")
    print("Classifier: Random Forest only")
    print("="*80)
    
    h_values = [0.05, 0.1, 0.2, 0.5, 1]
    
    graphs, labels, node_labels_list = load_all_graphs(DATASET_DIR)
    X_node_labels = create_node_label_features(node_labels_list) if use_node_labels and node_labels_list else None
    
    if X_node_labels is not None:
        graphs_train, graphs_test, y_train, y_test, X_labels_train, X_labels_test = train_test_split(
            graphs, labels, X_node_labels, test_size=test_size, random_state=random_state, stratify=labels)
    else:
        graphs_train, graphs_test, y_train, y_test = train_test_split(
            graphs, labels, test_size=test_size, random_state=random_state, stratify=labels)
        X_labels_train = X_labels_test = None
    
    all_results = []
    
    for func_type in ['harmonic', 'polynomial']:
        if func_type not in optimal_params:
            continue
        
        range_val = round(optimal_params[func_type].range_val, 2)
        print(f"\n{'='*60}")
        print(f"Function: {func_type.upper()} (range={range_val})")
        print(f"{'='*60}")
        
        for h in h_values:
            bins = max(10, int(range_val / h))  # Minimum 10 bins
            
            print(f"\n--- binwidth h={h}, bins={bins} (range/h = {range_val}/{h}) ---")
            
            X_train, X_test, gen_time, memory_mb = generate_embeddings_with_tracking(
                graphs_train, graphs_test, func_type, bins, range_val, random_state)
            
            if X_labels_train is not None:
                X_train = np.hstack([X_train, X_labels_train])
                X_test = np.hstack([X_test, X_labels_test])
            
            print(f"  Shape: {X_train.shape}, Time: {gen_time:.2f}s")
            
            # Random Forest only
            clf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=random_state, n_jobs=-1)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            result_entry = {
                'func': func_type,
                'binwidth': h,
                'bins': bins,
                'range': range_val,
                'embedding_dim': X_train.shape[1],
                'generation_time': gen_time,
                'memory_mb': memory_mb,
                'classifier': 'Random Forest',
                'accuracy': acc,
                'f1_score': f1
            }
            all_results.append(result_entry)
            print(f"  Random Forest: Acc={acc:.4f}, F1={f1:.4f}")
            
            del X_train, X_test
            gc.collect()
    
    del graphs
    gc.collect()
    return all_results


def run_stability_only(use_node_labels: bool = True):
    """Run only stability analysis using best config from existing results."""
    from sklearn.model_selection import train_test_split
    from .stability import (print_stability_summary, DEFAULT_PERTURBATION_RATIOS,
                           generate_embeddings_for_graphs, perturb_graphs_batch,
                           compute_embedding_stability, compute_classification_stability)
    
    print("="*80)
    print("STABILITY-ONLY MODE: Loading best config from results")
    print("="*80)
    
    optimal_params = load_best_config_from_results()
    
    print("\n" + "="*80)
    print("Running Stability Analysis")
    print("="*80)
    
    ensure_dataset_ready()
    graphs, labels, node_labels_list = load_all_graphs(DATASET_DIR)
    X_node_labels = create_node_label_features(node_labels_list) if use_node_labels and node_labels_list else None
    
    stability_results = []
    all_stability_results = []
    
    configs_to_test = []
    for func_type in ['harmonic', 'polynomial']:
        if func_type in optimal_params:
            p = optimal_params[func_type]
            configs_to_test.append({
                'name': func_type, 'func': func_type,
                'bins': p.bins, 'range': round(p.range_val, 2)
            })
    
    if 'harmonic' in optimal_params and 'polynomial' in optimal_params:
        h, p = optimal_params['harmonic'], optimal_params['polynomial']
        configs_to_test.append({
            'name': 'naive_hybrid', 'func': 'naive_hybrid',
            'harm_bins': h.bins, 'harm_range': round(h.range_val, 2),
            'pol_bins': p.bins, 'pol_range': round(p.range_val, 2)
        })
    
    print(f"\nConfigs to test: {[c['name'] for c in configs_to_test]}")
    
    for config in configs_to_test:
        print(f"\n  Stability for {config['name']}...")
        X_spectral = generate_embeddings_for_graphs(graphs, config, seed=42)
        
        for test_with_labels in [False, True]:
            if test_with_labels and X_node_labels is None:
                continue
            
            suffix = "_with_labels" if test_with_labels else ""
            config_name = f"{config['name']}{suffix}"
            
            X_original = np.hstack([X_spectral, X_node_labels]) if test_with_labels else X_spectral
            
            print(f"    Testing: {config_name} (dim={X_original.shape[1]})")
            
            indices = np.arange(len(graphs))
            train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
            
            config_results = {'config': {**config, 'name': config_name}, 'perturbation_results': []}
            
            for ratio in DEFAULT_PERTURBATION_RATIOS:
                perturbed_graphs = perturb_graphs_batch(graphs, ratio, seed=42)
                X_pert_spectral = generate_embeddings_for_graphs(perturbed_graphs, config, seed=42)
                X_perturbed = np.hstack([X_pert_spectral, X_node_labels]) if test_with_labels else X_pert_spectral
                
                stability_cosine = compute_embedding_stability(X_original, X_perturbed, 'cosine')
                stability_euclidean = compute_embedding_stability(X_original, X_perturbed, 'euclidean')
                clf_stability = compute_classification_stability(
                    X_original[train_idx], X_original[test_idx],
                    X_perturbed[train_idx], X_perturbed[test_idx],
                    labels[train_idx], labels[test_idx], 42
                )
                
                result_entry = {
                    'ratio': ratio, 'func': config_name, 'with_node_labels': test_with_labels,
                    **stability_cosine, **stability_euclidean, **clf_stability
                }
                if config['func'] in ['naive_hybrid', 'hybrid']:
                    result_entry.update({'harm_bins': config['harm_bins'], 'harm_range': config['harm_range'],
                                        'pol_bins': config['pol_bins'], 'pol_range': config['pol_range']})
                else:
                    result_entry.update({'bins': config['bins'], 'range': config['range']})
                
                stability_results.append(result_entry)
                config_results['perturbation_results'].append(result_entry)
                del perturbed_graphs
                gc.collect()
            
            all_stability_results.append(config_results)
    
    print_stability_summary(all_stability_results)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_stab = pd.DataFrame(stability_results)
    stab_path = os.path.join(RESULTS_DIR, 'fgsd_enzymes_stability_results.csv')
    df_stab.to_csv(stab_path, index=False)
    print(f"\n✅ Stability saved to: {stab_path}")
    
    del graphs
    gc.collect()


def run_classifier_hyperparameter_search(optimal_params, test_size=0.15, random_state=42, use_node_labels=True, fast_mode=True):
    """
    Run hyperparameter tuning for RF and SVM classifiers.
    Uses the best embedding configuration from preanalysis.
    """
    from .hyperparameter_search import run_classifier_tuning
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*80)
    print("CLASSIFIER HYPERPARAMETER TUNING")
    print("="*80)
    
    graphs, labels, node_labels_list = load_all_graphs(DATASET_DIR)
    X_node_labels = create_node_label_features(node_labels_list) if use_node_labels and node_labels_list else None
    
    # Split data
    if X_node_labels is not None:
        graphs_train, graphs_test, y_train, y_test, X_labels_train, X_labels_test = train_test_split(
            graphs, labels, X_node_labels, test_size=test_size, random_state=random_state, stratify=labels)
    else:
        graphs_train, graphs_test, y_train, y_test = train_test_split(
            graphs, labels, test_size=test_size, random_state=random_state, stratify=labels)
        X_labels_train = X_labels_test = None
    
    # Generate embeddings using best config (polynomial usually best)
    func_type = 'polynomial' if 'polynomial' in optimal_params else list(optimal_params.keys())[0]
    params = optimal_params[func_type]
    
    print(f"\nGenerating embeddings with: {func_type}, bins={params.bins}, range={params.range_val:.2f}")
    
    X_train, X_test, gen_time, _ = generate_embeddings_with_tracking(
        graphs_train, graphs_test, func_type, params.bins, round(params.range_val, 2), random_state
    )
    
    # Add node labels if available
    if X_labels_train is not None:
        X_train = np.hstack([X_train, X_labels_train])
        X_test = np.hstack([X_test, X_labels_test])
    
    print(f"Embedding shape for tuning: {X_train.shape}")
    
    # Run tuning
    tuning_results = run_classifier_tuning(
        X_train, y_train,
        fast_mode=fast_mode,
        cv=5,
        random_state=random_state,
        save_results=True
    )
    
    # Evaluate tuned models on test set
    print("\n" + "="*80)
    print("EVALUATION OF TUNED CLASSIFIERS ON TEST SET")
    print("="*80)
    
    eval_results = []
    for clf_name, res in tuning_results.items():
        clf = res['model']
        
        # Re-fit on full training set (GridSearchCV uses CV, so we refit)
        clf.fit(X_train, y_train)
        
        eval_res = evaluate_classifier(X_train, X_test, y_train, y_test, f"{clf_name} (Tuned)", clf)
        eval_res['cv_score'] = res['cv_score']
        eval_res['best_params'] = str(res['params'])
        eval_results.append(eval_res)
        
        print(f"\n{clf_name} (Tuned):")
        print(f"  CV Score: {res['cv_score']:.4f}")
        print(f"  Test Accuracy: {eval_res['accuracy']:.4f}")
        print(f"  Test F1: {eval_res['f1_score']:.4f}")
        print(f"  Best Params: {res['params']}")
    
    # Save tuning results
    import pandas as pd
    df = pd.DataFrame(eval_results)
    tuning_path = os.path.join(RESULTS_DIR, 'fgsd_enzymes_classifier_tuning.csv')
    df.to_csv(tuning_path, index=False)
    print(f"\n✅ Tuning results saved to: {tuning_path}")
    
    del graphs
    gc.collect()
    
    return tuning_results


def main(run_stability: bool = False, force: bool = False, use_node_labels: bool = True, 
         skip_grid: bool = False, stability_only: bool = False, tune_classifiers: bool = False):
    if stability_only:
        run_stability_only(use_node_labels)
        return
    
    print("="*80)
    print("FGSD CLASSIFICATION ON ENZYMES")
    print("="*80)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ensure_dataset_ready()
    
    # Check what's cached (preanalysis and grid only)
    cached = check_cached_results()
    print("\nCached results (will reuse if exists):")
    for key, val in cached.items():
        status = "✅ Found" if val else "❌ Missing"
        print(f"  {key}: {status}")
    
    if force:
        print("\n⚠️  FORCE mode: Will recompute preanalysis and grid search")
        cached = {k: False for k in cached}
    
    # =================================================================
    # STEP 1: Pre-analysis (CACHED - only recompute if force or missing)
    # =================================================================
    print("\n" + "="*80)
    print("STEP 1: Pre-Analysis (cached)")
    print("="*80)
    
    try:
        optimal_params, recommended_bins = run_sampled_preanalysis(
            graphs=None, use_cache=True, force_recompute=force, dataset_name='enzymes'
        )
    except ValueError:
        print("Loading graphs for preanalysis...")
        graphs, _, _ = load_all_graphs(DATASET_DIR)
        optimal_params, recommended_bins = run_sampled_preanalysis(
            graphs, use_cache=True, force_recompute=force, dataset_name='enzymes'
        )
        del graphs
        gc.collect()
    
    print("\nPre-analysis results:")
    for func_type, params in optimal_params.items():
        bins_list = recommended_bins.get(func_type, [params.bins])
        print(f"  {func_type.upper()}: range={params.range_val:.2f}, bins={bins_list}")
    
    # =================================================================
    # STEP 1.5: Classifier Hyperparameter Tuning (OPTIONAL)
    # =================================================================
    if tune_classifiers:
        run_classifier_hyperparameter_search(
            optimal_params, 
            use_node_labels=use_node_labels,
            fast_mode=True
        )
        print("\n✅ Classifier tuning complete. Tuned params will be used in subsequent runs.")
    
    # =================================================================
    # STEP 2: Grid Search (CACHED - only recompute if force or missing)
    # =================================================================
    if not skip_grid:
        if not cached['grid']:
            grid_results = run_grid_search(optimal_params, use_node_labels=use_node_labels)
            
            df_grid = pd.DataFrame(grid_results)
            grid_path = os.path.join(RESULTS_DIR, 'fgsd_enzymes_grid_search.csv')
            df_grid.to_csv(grid_path, index=False)
            print(f"\n✅ Grid search saved to: {grid_path}")
            
            print("\nGRID SEARCH SUMMARY (bins = range / binwidth):")
            print(f"{'Func':<12} {'Binwidth':<10} {'Bins':<8} {'Accuracy':<10} {'F1':<10}")
            print("-"*60)
            for r in sorted(grid_results, key=lambda x: -x['accuracy']):
                print(f"{r['func']:<12} {r['binwidth']:<10} {r['bins']:<8} {r['accuracy']:<10.4f} {r['f1_score']:<10.4f}")
        else:
            print("\n✅ Grid search cached, skipping (use --force to recompute)")
    
    # =================================================================
    # STEP 3: Dimension Analysis (ALWAYS RUN)
    # =================================================================
    all_bins = set()
    for func_type in ['harmonic', 'polynomial']:
        if func_type in recommended_bins:
            all_bins.update(recommended_bins[func_type])
    bin_sizes = sorted(list(all_bins))
    
    print("\n" + "="*80)
    print("STEP 3: Dimension Analysis (always runs)")
    print("="*80)
    
    dim_results, best_config = run_dimension_analysis(optimal_params, bin_sizes=bin_sizes, use_node_labels=use_node_labels)
    print_dimension_analysis_summary(dim_results)
    
    df_dim = pd.DataFrame(dim_results)
    dim_path = os.path.join(RESULTS_DIR, 'fgsd_enzymes_dimension_analysis.csv')
    df_dim.to_csv(dim_path, index=False)
    print(f"\n✅ Dimension analysis saved to: {dim_path}")
    
    # Update optimal params with best bins
    for func_type, (best_bins, best_acc) in best_config.items():
        if func_type in optimal_params:
            old = optimal_params[func_type]
            optimal_params[func_type] = OptimalParams(old.func_type, best_bins, old.range_val, old.p99, best_bins)
    
    print("\nBest configuration:")
    for func_type, params in optimal_params.items():
        print(f"  {func_type.upper()}: bins={params.bins}, range={params.range_val:.2f}")
    
    # =================================================================
    # STEP 4: Final Classification (ALWAYS RUN)
    # =================================================================
    print("\n" + "="*80)
    print("STEP 4: Final Classification (always runs)")
    print("="*80)
    
    final_results = run_final_classification(
        optimal_params, 
        recommended_bins=recommended_bins,
        use_node_labels=use_node_labels
    )
    print_summary(final_results)
    
    df_final = pd.DataFrame(final_results)
    final_path = os.path.join(RESULTS_DIR, 'fgsd_enzymes_final_results.csv')
    df_final.to_csv(final_path, index=False)
    print(f"\n✅ Final results saved to: {final_path}")
    
    # Save optimal params
    params_path = os.path.join(RESULTS_DIR, 'enzymes_optimal_params.txt')
    with open(params_path, 'w') as f:
        f.write("OPTIMAL PARAMETERS FOR ENZYMES\n" + "="*50 + "\n")
        for func_type, params in optimal_params.items():
            f.write(f"{func_type}: bins={params.bins}, range={params.range_val:.4f}\n")
    
    # =================================================================
    # STEP 5: Stability Analysis (ALWAYS RUN if requested)
    # =================================================================
    if run_stability:
        print("\n" + "="*80)
        print("STEP 5: Stability Analysis (always runs)")
        print("="*80)
        
        from .stability import (run_stability_analysis, print_stability_summary, 
                               DEFAULT_PERTURBATION_RATIOS, generate_embeddings_for_graphs,
                               perturb_graphs_batch, compute_embedding_stability, 
                               compute_classification_stability)
        from sklearn.model_selection import train_test_split
        
        graphs, labels, node_labels_list = load_all_graphs(DATASET_DIR)
        X_node_labels = create_node_label_features(node_labels_list) if use_node_labels and node_labels_list else None
        
        stability_results = []
        all_stability_results = []
        
        configs_to_test = []
        for func_type in ['harmonic', 'polynomial']:
            if func_type in optimal_params:
                p = optimal_params[func_type]
                configs_to_test.append({
                    'name': func_type, 
                    'func': func_type, 
                    'bins': p.bins, 
                    'range': round(p.range_val, 2)
                })
        
        if 'harmonic' in optimal_params and 'polynomial' in optimal_params:
            h, p = optimal_params['harmonic'], optimal_params['polynomial']
            configs_to_test.append({
                'name': 'naive_hybrid',
                'func': 'naive_hybrid',
                'harm_bins': h.bins, 'harm_range': round(h.range_val, 2),
                'pol_bins': p.bins, 'pol_range': round(p.range_val, 2)
            })
        
        for config in configs_to_test:
            print(f"\n  Stability for {config['name']}...")
            X_spectral = generate_embeddings_for_graphs(graphs, config, seed=42)
            
            for test_with_labels in [False, True]:
                if test_with_labels and X_node_labels is None:
                    continue
                
                suffix = "_with_labels" if test_with_labels else ""
                config_name = f"{config['name']}{suffix}"
                
                if test_with_labels:
                    X_original = np.hstack([X_spectral, X_node_labels])
                else:
                    X_original = X_spectral
                
                print(f"    Testing: {config_name} (dim={X_original.shape[1]})")
                
                indices = np.arange(len(graphs))
                train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
                
                config_results = {'config': {**config, 'name': config_name}, 'perturbation_results': []}
                
                for ratio in DEFAULT_PERTURBATION_RATIOS:
                    perturbed_graphs = perturb_graphs_batch(graphs, ratio, seed=42)
                    X_pert_spectral = generate_embeddings_for_graphs(perturbed_graphs, config, seed=42)
                    
                    if test_with_labels:
                        X_perturbed = np.hstack([X_pert_spectral, X_node_labels])
                    else:
                        X_perturbed = X_pert_spectral
                    
                    stability_cosine = compute_embedding_stability(X_original, X_perturbed, 'cosine')
                    stability_euclidean = compute_embedding_stability(X_original, X_perturbed, 'euclidean')
                    clf_stability = compute_classification_stability(
                        X_original[train_idx], X_original[test_idx],
                        X_perturbed[train_idx], X_perturbed[test_idx],
                        labels[train_idx], labels[test_idx], 42
                    )
                    
                    result_entry = {
                        'ratio': ratio, 
                        'func': config_name, 
                        'with_node_labels': test_with_labels,
                        **stability_cosine, **stability_euclidean, **clf_stability
                    }
                    
                    if config['func'] in ['naive_hybrid', 'hybrid']:
                        result_entry['harm_bins'] = config['harm_bins']
                        result_entry['harm_range'] = config['harm_range']
                        result_entry['pol_bins'] = config['pol_bins']
                        result_entry['pol_range'] = config['pol_range']
                    else:
                        result_entry['bins'] = config['bins']
                        result_entry['range'] = config['range']
                    
                    stability_results.append(result_entry)
                    config_results['perturbation_results'].append(result_entry)
                    del perturbed_graphs
                    gc.collect()
                
                all_stability_results.append(config_results)
        
        print_stability_summary(all_stability_results)
        
        df_stab = pd.DataFrame(stability_results)
        stab_path = os.path.join(RESULTS_DIR, 'fgsd_enzymes_stability_results.csv')
        df_stab.to_csv(stab_path, index=False)
        print(f"\n✅ Stability saved to: {stab_path}")
        
        del graphs
        gc.collect()
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*80)
    print("✅ COMPLETE! Output files:")
    print("="*80)
    if not skip_grid:
        print(f"  📊 {os.path.join(RESULTS_DIR, 'fgsd_enzymes_grid_search.csv')} (cached)")
    print(f"  📊 {dim_path}")
    print(f"  📊 {final_path}")
    if tune_classifiers:
        print(f"  📊 {os.path.join(RESULTS_DIR, 'fgsd_enzymes_classifier_tuning.csv')}")
    if run_stability:
        print(f"  📊 {os.path.join(RESULTS_DIR, 'fgsd_enzymes_stability_results.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FGSD Classification on ENZYMES')
    parser.add_argument('--stability', action='store_true', help='Include stability analysis')
    parser.add_argument('--stability-only', action='store_true', help='Run only stability (loads config from results)')
    parser.add_argument('--force', action='store_true', help='Force rerun everything')
    parser.add_argument('--no-node-labels', action='store_true', help='Disable node labels')
    parser.add_argument('--skip-grid', action='store_true', help='Skip grid search')
    parser.add_argument('--tune-classifiers', action='store_true', help='Run classifier hyperparameter tuning (RF & SVM)')
    args = parser.parse_args()
    
    main(run_stability=args.stability, force=args.force, 
         use_node_labels=not args.no_node_labels, skip_grid=args.skip_grid,
         stability_only=args.stability_only, tune_classifiers=args.tune_classifiers)
