"""
FGSD Classification on IMDB-MULTI

Usage:
    python -m imbd_ds.classification_main                    # Full pipeline (auto-detects what needs to run)
    python -m imbd_ds.classification_main --stability        # Include stability analysis
    python -m imbd_ds.classification_main --force            # Force rerun everything
    python -m imbd_ds.classification_main --skip-grid        # Skip grid search
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

from imbd_ds.config import RESULTS_DIR, PREANALYSIS_SAMPLE_SIZE, OptimalParams, DATASET_DIR
from imbd_ds.data_loader import ensure_dataset_ready, load_all_graphs
from imbd_ds.preanalysis import run_sampled_preanalysis
from imbd_ds.classification import (
    run_dimension_analysis, run_final_classification,
    print_dimension_analysis_summary, print_summary,
    generate_embeddings_with_tracking, evaluate_classifier, get_classifiers
)


def check_cached_results():
    """Check which CACHED results exist (only preanalysis and grid)."""
    exists = {
        'preanalysis': os.path.exists(os.path.join(parent_dir, 'cache', 'imdb_preanalysis_cache.json')),
        'grid': os.path.exists(os.path.join(RESULTS_DIR, 'fgsd_imdb_grid_search.csv')),
    }
    return exists


def run_grid_search(optimal_params, test_size=0.15, random_state=42):
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
    
    graphs, labels = load_all_graphs(DATASET_DIR)
    graphs_train, graphs_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=test_size, random_state=random_state, stratify=labels)
    
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
                'embedding_dim': bins,
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


def main(run_stability: bool = False, force: bool = False, skip_grid: bool = False):
    print("="*80)
    print("FGSD CLASSIFICATION ON IMDB-MULTI")
    print("="*80)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ensure_dataset_ready()
    
    cached = check_cached_results()
    print("\nCached results (will reuse if exists):")
    for key, val in cached.items():
        status = "✅ Found" if val else "❌ Missing"
        print(f"  {key}: {status}")
    
    if force:
        print("\n⚠️  FORCE mode: Will recompute preanalysis and grid search")
        cached = {k: False for k in cached}
    
    # STEP 1: Pre-analysis (CACHED)
    print("\n" + "="*80)
    print("STEP 1: Pre-Analysis (cached)")
    print("="*80)
    
    try:
        optimal_params, recommended_bins = run_sampled_preanalysis(
            graphs=None, use_cache=True, force_recompute=force, dataset_name='imdb'
        )
    except ValueError:
        graphs, _ = load_all_graphs(DATASET_DIR)
        optimal_params, recommended_bins = run_sampled_preanalysis(
            graphs, use_cache=True, force_recompute=force, dataset_name='imdb'
        )
        del graphs
        gc.collect()
    
    print("\nPre-analysis results:")
    for func_type, params in optimal_params.items():
        print(f"  {func_type.upper()}: range={params.range_val:.2f}, bins={recommended_bins.get(func_type, [params.bins])}")
    
    # STEP 2: Grid Search (CACHED)
    if not skip_grid:
        if not cached['grid']:
            grid_results = run_grid_search(optimal_params)
            df_grid = pd.DataFrame(grid_results)
            grid_path = os.path.join(RESULTS_DIR, 'fgsd_imdb_grid_search.csv')
            df_grid.to_csv(grid_path, index=False)
            print(f"\n✅ Grid search saved to: {grid_path}")
            
            print("\nGRID SEARCH SUMMARY (bins = range / binwidth):")
            print(f"{'Func':<12} {'Binwidth':<10} {'Bins':<8} {'Accuracy':<10} {'F1':<10}")
            print("-"*60)
            for r in sorted(grid_results, key=lambda x: -x['accuracy']):
                print(f"{r['func']:<12} {r['binwidth']:<10} {r['bins']:<8} {r['accuracy']:<10.4f} {r['f1_score']:<10.4f}")
        else:
            print("\n✅ Grid search cached, skipping (use --force to recompute)")
    
    # STEP 3: Dimension Analysis (ALWAYS RUN)
    all_bins = set()
    for func_type in ['harmonic', 'polynomial']:
        if func_type in recommended_bins:
            all_bins.update(recommended_bins[func_type])
    bin_sizes = sorted(list(all_bins))
    
    print("\n" + "="*80)
    print("STEP 3: Dimension Analysis (always runs)")
    print("="*80)
    
    dim_results, best_config = run_dimension_analysis(optimal_params, bin_sizes=bin_sizes)
    print_dimension_analysis_summary(dim_results)
    
    df_dim = pd.DataFrame(dim_results)
    dim_path = os.path.join(RESULTS_DIR, 'fgsd_imdb_dimension_analysis.csv')
    df_dim.to_csv(dim_path, index=False)
    print(f"\n✅ Dimension analysis saved to: {dim_path}")
    
    for func_type, (best_bins, best_acc) in best_config.items():
        if func_type in optimal_params:
            old = optimal_params[func_type]
            optimal_params[func_type] = OptimalParams(old.func_type, best_bins, old.range_val, old.p99, best_bins)
    
    print("\nBest configuration:")
    for func_type, params in optimal_params.items():
        print(f"  {func_type.upper()}: bins={params.bins}, range={params.range_val:.2f}")
    
    # STEP 4: Final Classification (ALWAYS RUN)
    print("\n" + "="*80)
    print("STEP 4: Final Classification (always runs)")
    print("="*80)
    
    # Pass recommended_bins so final classification tests all bin sizes
    final_results = run_final_classification(optimal_params, recommended_bins=recommended_bins)
    print_summary(final_results)
    
    df_final = pd.DataFrame(final_results)
    final_path = os.path.join(RESULTS_DIR, 'fgsd_imdb_final_results.csv')
    df_final.to_csv(final_path, index=False)
    print(f"\n✅ Final results saved to: {final_path}")
    
    # STEP 5: Stability (ALWAYS RUN if requested)
    if run_stability:
        print("\n" + "="*80)
        print("STEP 5: Stability Analysis (always runs)")
        print("="*80)
        
        from .stability import (run_stability_analysis, print_stability_summary, DEFAULT_PERTURBATION_RATIOS,
                               generate_embeddings_for_graphs, perturb_graphs_batch,
                               compute_embedding_stability, compute_classification_stability)
        from sklearn.model_selection import train_test_split
        
        graphs, labels = load_all_graphs(DATASET_DIR)
        stability_results = []
        all_stability_results = []
        
        configs_to_test = []
        for func_type in ['harmonic', 'polynomial']:
            if func_type in optimal_params:
                p = optimal_params[func_type]
                configs_to_test.append({'name': func_type, 'func': func_type, 'bins': p.bins, 'range': round(p.range_val, 2)})
        
        if 'harmonic' in optimal_params and 'polynomial' in optimal_params:
            h, p = optimal_params['harmonic'], optimal_params['polynomial']
            configs_to_test.append({
                'name': 'hybrid', 'func': 'hybrid',
                'harm_bins': h.bins, 'harm_range': round(h.range_val, 2),
                'pol_bins': p.bins, 'pol_range': round(p.range_val, 2)
            })
        
        for config in configs_to_test:
            print(f"\n  Stability for {config['name']}...")
            X_original = generate_embeddings_for_graphs(graphs, config, seed=42)
            
            indices = np.arange(len(graphs))
            train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
            
            config_results = {'config': config, 'perturbation_results': []}
            
            for ratio in DEFAULT_PERTURBATION_RATIOS:
                perturbed_graphs = perturb_graphs_batch(graphs, ratio, seed=42)
                X_perturbed = generate_embeddings_for_graphs(perturbed_graphs, config, seed=42)
                
                stability_cosine = compute_embedding_stability(X_original, X_perturbed, 'cosine')
                stability_euclidean = compute_embedding_stability(X_original, X_perturbed, 'euclidean')
                clf_stability = compute_classification_stability(
                    X_original[train_idx], X_original[test_idx],
                    X_perturbed[train_idx], X_perturbed[test_idx],
                    labels[train_idx], labels[test_idx], 42
                )
                
                result_entry = {'ratio': ratio, 'func': config['func'], **stability_cosine, **stability_euclidean, **clf_stability}
                if config['func'] == 'hybrid':
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
        
        df_stab = pd.DataFrame(stability_results)
        stab_path = os.path.join(RESULTS_DIR, 'fgsd_imdb_stability_results.csv')
        df_stab.to_csv(stab_path, index=False)
        print(f"\n✅ Stability saved to: {stab_path}")
        
        del graphs
        gc.collect()
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FGSD Classification on IMDB-MULTI')
    parser.add_argument('--stability', action='store_true', help='Include stability analysis')
    parser.add_argument('--force', action='store_true', help='Force rerun everything')
    parser.add_argument('--skip-grid', action='store_true', help='Skip grid search')
    args = parser.parse_args()
    
    main(run_stability=args.stability, force=args.force, skip_grid=args.skip_grid)
