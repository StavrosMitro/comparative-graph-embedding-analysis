"""
FGSD Classification on IMDB-MULTI

Usage:
    python -m imbd_ds.classification_main                    # Full pipeline (auto-detects what needs to run)
    python -m imbd_ds.classification_main --stability        # Include stability analysis
    python -m imbd_ds.classification_main --stability-only   # Only stability (loads best config from results)
    python -m imbd_ds.classification_main --force            # Force rerun everything
    python -m imbd_ds.classification_main --skip-grid        # Skip grid search
    python -m imbd_ds.classification_main --tune-classifiers # Run classifier hyperparameter tuning
    python -m imbd_ds.classification_main --raw-embeddings   # Run with RF on raw embeddings (no preprocessing)
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
    generate_embeddings_with_tracking, evaluate_classifier, get_classifiers,
    get_classifiers_tuned_or_default, get_raw_classifiers, RAW_RESULTS_DIR
)


def get_results_dir(raw_mode=False):
    """Get the appropriate results directory based on mode."""
    if raw_mode:
        os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
        return RAW_RESULTS_DIR
    return RESULTS_DIR


def compare_raw_vs_preprocessed(raw_results_path: str, preprocessed_results_path: str):
    """
    Compare raw embedding results with preprocessed results.
    Shows side-by-side comparison for each function type.
    """
    if not os.path.exists(raw_results_path):
        print(f"\n⚠️  Raw results not found: {raw_results_path}")
        return
    
    if not os.path.exists(preprocessed_results_path):
        print(f"\n⚠️  Preprocessed results not found: {preprocessed_results_path}")
        return
    
    raw_df = pd.read_csv(raw_results_path)
    prep_df = pd.read_csv(preprocessed_results_path)
    
    print("\n" + "="*100)
    print("COMPARISON: RAW EMBEDDINGS vs PREPROCESSED (with StandardScaler)")
    print("="*100)
    
    # Get unique function types
    func_types = ['harmonic', 'polynomial', 'naive_hybrid']
    
    print(f"\n{'Function':<20} {'Metric':<12} {'Raw RF':<12} {'Prep RF':<12} {'Prep SVM':<12} {'Prep MLP':<12} {'Best Prep':<12} {'Diff (Raw-Best)':<15}")
    print("-"*100)
    
    for func in func_types:
        # Get raw results (only Random Forest)
        raw_func_df = raw_df[raw_df['func'] == func]
        
        if len(raw_func_df) == 0:
            continue
        
        raw_best = raw_func_df.loc[raw_func_df['accuracy'].idxmax()]
        raw_acc = raw_best['accuracy']
        raw_f1 = raw_best['f1_score']
        
        # Get preprocessed results
        prep_func_df = prep_df[prep_df['func'] == func]
        
        if len(prep_func_df) == 0:
            continue
        
        # Get best for each classifier
        prep_rf = prep_func_df[prep_func_df['classifier'] == 'Random Forest']
        prep_svm = prep_func_df[prep_func_df['classifier'] == 'SVM (RBF)']
        prep_mlp = prep_func_df[prep_func_df['classifier'] == 'MLP']
        
        prep_rf_acc = prep_rf['accuracy'].max() if len(prep_rf) > 0 else 0
        prep_svm_acc = prep_svm['accuracy'].max() if len(prep_svm) > 0 else 0
        prep_mlp_acc = prep_mlp['accuracy'].max() if len(prep_mlp) > 0 else 0
        
        prep_rf_f1 = prep_rf.loc[prep_rf['accuracy'].idxmax(), 'f1_score'] if len(prep_rf) > 0 else 0
        prep_svm_f1 = prep_svm.loc[prep_svm['accuracy'].idxmax(), 'f1_score'] if len(prep_svm) > 0 else 0
        prep_mlp_f1 = prep_mlp.loc[prep_mlp['accuracy'].idxmax(), 'f1_score'] if len(prep_mlp) > 0 else 0
        
        best_prep_acc = max(prep_rf_acc, prep_svm_acc, prep_mlp_acc)
        best_prep_f1 = max(prep_rf_f1, prep_svm_f1, prep_mlp_f1)
        
        diff_acc = raw_acc - best_prep_acc
        diff_f1 = raw_f1 - best_prep_f1
        
        diff_acc_str = f"{diff_acc:+.4f}" if diff_acc != 0 else "0.0000"
        diff_f1_str = f"{diff_f1:+.4f}" if diff_f1 != 0 else "0.0000"
        
        print(f"{func:<20} {'Accuracy':<12} {raw_acc:<12.4f} {prep_rf_acc:<12.4f} {prep_svm_acc:<12.4f} {prep_mlp_acc:<12.4f} {best_prep_acc:<12.4f} {diff_acc_str:<15}")
        print(f"{'':<20} {'F1-Score':<12} {raw_f1:<12.4f} {prep_rf_f1:<12.4f} {prep_svm_f1:<12.4f} {prep_mlp_f1:<12.4f} {best_prep_f1:<12.4f} {diff_f1_str:<15}")
        print()
    
    print("-"*100)
    print("\nSUMMARY:")
    print("  • Raw = Random Forest on raw embeddings (no preprocessing)")
    print("  • Prep RF/SVM/MLP = Classifiers with StandardScaler preprocessing")
    print("  • Diff > 0 means raw embeddings performed BETTER")
    print("  • Diff < 0 means preprocessed embeddings performed BETTER")


def check_cached_results():
    """Check which CACHED results exist (only preanalysis and grid)."""
    exists = {
        'preanalysis': os.path.exists(os.path.join(parent_dir, 'cache', 'imdb_preanalysis_cache.json')),
        'grid': os.path.exists(os.path.join(RESULTS_DIR, 'fgsd_imdb_grid_search.csv')),
    }
    return exists


def run_grid_search(optimal_params, test_size=0.15, random_state=42, raw_mode=False):
    """
    Run grid search with binwidth h ∈ {0.05, 0.1, 0.2, 0.5, 1}.
    Uses all classifiers (RF, SVM, MLP) - with or without preprocessing based on raw_mode.
    """
    from sklearn.model_selection import train_test_split
    
    mode_str = "(Raw Embeddings - No Preprocessing)" if raw_mode else ""
    print("\n" + "="*80)
    print(f"GRID SEARCH: bins = range / binwidth {mode_str}")
    print("binwidth (h) values: [0.05, 0.1, 0.2, 0.5, 1]")
    print("Classifiers: RF, SVM, MLP" + (" (no preprocessing)" if raw_mode else " (with StandardScaler)"))
    print("="*80)
    
    h_values = [0.05, 0.1, 0.2, 0.5, 1]
    
    graphs, labels = load_all_graphs(DATASET_DIR)
    graphs_train, graphs_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=test_size, random_state=random_state, stratify=labels)
    
    all_results = []
    classifiers = get_raw_classifiers(random_state) if raw_mode else get_classifiers(random_state)
    
    for func_type in ['harmonic', 'polynomial']:
        if func_type not in optimal_params:
            continue
        
        range_val = round(optimal_params[func_type].range_val, 2)
        print(f"\n{'='*60}")
        print(f"Function: {func_type.upper()} (range={range_val})")
        print(f"{'='*60}")
        
        for h in h_values:
            bins = max(10, int(range_val / h))
            
            print(f"\n--- binwidth h={h}, bins={bins} (range/h = {range_val}/{h}) ---")
            
            X_train, X_test, gen_time, memory_mb = generate_embeddings_with_tracking(
                graphs_train, graphs_test, func_type, bins, range_val, random_state)
            
            print(f"  Shape: {X_train.shape}, Time: {gen_time:.2f}s")
            
            for clf_name, clf in classifiers.items():
                res = evaluate_classifier(X_train, X_test, y_train, y_test, clf_name, clf)
                
                result_entry = {
                    'func': func_type,
                    'binwidth': h,
                    'bins': bins,
                    'range': range_val,
                    'embedding_dim': bins,
                    'generation_time': gen_time,
                    'memory_mb': memory_mb,
                    **res
                }
                all_results.append(result_entry)
                print(f"  {clf_name}: Acc={res['accuracy']:.4f}, F1={res['f1_score']:.4f}")
            
            del X_train, X_test
            gc.collect()
    
    del graphs
    gc.collect()
    return all_results


def load_best_config_from_results():
    """Load best configuration from existing results CSV."""
    final_path = os.path.join(RESULTS_DIR, 'fgsd_imdb_final_results.csv')
    dim_path = os.path.join(RESULTS_DIR, 'fgsd_imdb_dimension_analysis.csv')
    
    for path in [final_path, dim_path]:
        if os.path.exists(path):
            print(f"Loading best config from: {path}")
            df = pd.read_csv(path)
            
            optimal_params = {}
            
            for func_type in ['harmonic', 'polynomial']:
                func_df = df[df['func'] == func_type]
                if len(func_df) > 0:
                    best_row = func_df.loc[func_df['accuracy'].idxmax()]
                    bins = int(best_row['bins']) if pd.notna(best_row.get('bins')) else 70
                    range_val = float(best_row['range']) if pd.notna(best_row.get('range')) else 3.5
                    
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
        "Run full pipeline first: python -m imbd_ds.classification_main"
    )


def run_stability_only(raw_mode: bool = False):
    """Run only stability analysis using best config from existing results."""
    from .stability import (print_stability_summary, DEFAULT_PERTURBATION_RATIOS,
                           run_stability_analysis)
    
    mode_str = " (Raw Embeddings)" if raw_mode else ""
    print("="*80)
    print(f"STABILITY-ONLY MODE{mode_str}: Loading best config from results")
    print("="*80)
    
    optimal_params = load_best_config_from_results()
    
    print("\n" + "="*80)
    print("Running Stability Analysis")
    print("="*80)
    
    ensure_dataset_ready()
    graphs, labels = load_all_graphs(DATASET_DIR)
    
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
            'name': 'hybrid', 'func': 'hybrid',
            'harm_bins': h.bins, 'harm_range': round(h.range_val, 2),
            'pol_bins': p.bins, 'pol_range': round(p.range_val, 2)
        })
    
    print(f"\nConfigs to test: {[c['name'] for c in configs_to_test]}")
    print(f"Raw mode: {raw_mode} (classifiers {'without' if raw_mode else 'with'} preprocessing)")
    
    for config in configs_to_test:
        print(f"\n  Stability for {config['name']}...")
        
        # Run full stability analysis
        result, X_orig = run_stability_analysis(
            graphs=graphs,
            labels=labels,
            config=config,
            perturbation_ratios=DEFAULT_PERTURBATION_RATIOS,
            X_original=None,
            seed=42,
            test_size=0.15,
            compute_classification=True,
            use_raw_classifiers=raw_mode
        )
        
        # Add config params to each result entry
        for pr in result['perturbation_results']:
            pr['func'] = config['name']
            pr['raw_mode'] = raw_mode
            
            if config['func'] == 'hybrid':
                pr['harm_bins'] = config['harm_bins']
                pr['harm_range'] = config['harm_range']
                pr['pol_bins'] = config['pol_bins']
                pr['pol_range'] = config['pol_range']
            else:
                pr['bins'] = config['bins']
                pr['range'] = config['range']
            
            stability_results.append(pr)
        
        all_stability_results.append(result)
    
    print_stability_summary(all_stability_results)
    
    results_dir = get_results_dir(raw_mode)
    os.makedirs(results_dir, exist_ok=True)
    df_stab = pd.DataFrame(stability_results)
    stab_file = 'fgsd_imdb_raw_stability_results.csv' if raw_mode else 'fgsd_imdb_stability_results.csv'
    stab_path = os.path.join(results_dir, stab_file)
    df_stab.to_csv(stab_path, index=False)
    print(f"\n✅ Stability saved to: {stab_path}")
    
    del graphs
    gc.collect()


def run_classifier_hyperparameter_search(optimal_params, test_size=0.15, random_state=42, fast_mode=True):
    """
    Run hyperparameter tuning for RF and SVM classifiers.
    Uses the best embedding configuration from preanalysis.
    """
    from .hyperparameter_search import run_classifier_tuning
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*80)
    print("CLASSIFIER HYPERPARAMETER TUNING (IMDB)")
    print("="*80)
    
    graphs, labels = load_all_graphs(DATASET_DIR)
    
    # Split data
    graphs_train, graphs_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=test_size, random_state=random_state, stratify=labels)
    
    # Generate embeddings using best config (polynomial usually best)
    func_type = 'polynomial' if 'polynomial' in optimal_params else list(optimal_params.keys())[0]
    params = optimal_params[func_type]
    
    print(f"\nGenerating embeddings with: {func_type}, bins={params.bins}, range={params.range_val:.2f}")
    
    X_train, X_test, gen_time, _ = generate_embeddings_with_tracking(
        graphs_train, graphs_test, func_type, params.bins, round(params.range_val, 2), random_state
    )
    
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
    tuning_path = os.path.join(RESULTS_DIR, 'fgsd_imdb_classifier_tuning.csv')
    df.to_csv(tuning_path, index=False)
    print(f"\n✅ Tuning results saved to: {tuning_path}")
    
    del graphs
    gc.collect()
    
    return tuning_results


def main(run_stability: bool = False, force: bool = False, skip_grid: bool = False, 
         stability_only: bool = False, tune_classifiers: bool = False, raw_mode: bool = False):
    if stability_only:
        run_stability_only(raw_mode=raw_mode)
        return
    
    # Get appropriate results directory
    results_dir = get_results_dir(raw_mode)
    
    mode_str = " (RAW EMBEDDINGS - NO PREPROCESSING)" if raw_mode else ""
    print("="*80)
    print(f"FGSD CLASSIFICATION ON IMDB-MULTI{mode_str}")
    print("="*80)
    
    if raw_mode:
        print("\n⚠️  RAW MODE: Using all classifiers (RF, SVM, MLP) WITHOUT StandardScaler")
        print(f"   Results will be saved to: {results_dir}")
    
    os.makedirs(results_dir, exist_ok=True)
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
    
    # STEP 1.5: Classifier Hyperparameter Tuning (OPTIONAL)
    if tune_classifiers:
        if raw_mode:
            print("\n⚠️  Note: Tuning in raw mode will tune classifiers without StandardScaler")
        run_classifier_hyperparameter_search(
            optimal_params,
            fast_mode=True
        )
        print("\n✅ Classifier tuning complete.")
    
    # STEP 2: Grid Search (CACHED)
    grid_file = 'fgsd_imdb_raw_grid_search.csv' if raw_mode else 'fgsd_imdb_grid_search.csv'
    grid_path = os.path.join(results_dir, grid_file)
    
    if not skip_grid:
        grid_cached = os.path.exists(grid_path) if raw_mode else cached['grid']
        
        if not grid_cached or force:
            grid_results = run_grid_search(optimal_params, raw_mode=raw_mode)
            df_grid = pd.DataFrame(grid_results)
            df_grid.to_csv(grid_path, index=False)
            print(f"\n✅ Grid search saved to: {grid_path}")
            
            print("\nGRID SEARCH SUMMARY (bins = range / binwidth):")
            print(f"{'Func':<12} {'Binwidth':<10} {'Bins':<8} {'Classifier':<20} {'Accuracy':<10} {'F1':<10}")
            print("-"*80)
            for r in sorted(grid_results, key=lambda x: -x['accuracy'])[:15]:
                print(f"{r['func']:<12} {r['binwidth']:<10} {r['bins']:<8} {r['classifier']:<20} {r['accuracy']:<10.4f} {r['f1_score']:<10.4f}")
        else:
            print(f"\n✅ Grid search cached at {grid_path}, skipping (use --force to recompute)")
    
    # STEP 3: Dimension Analysis (ALWAYS RUN)
    all_bins = set()
    for func_type in ['harmonic', 'polynomial']:
        if func_type in recommended_bins:
            all_bins.update(recommended_bins[func_type])
    bin_sizes = sorted(list(all_bins))
    
    print("\n" + "="*80)
    print("STEP 3: Dimension Analysis (always runs)")
    print("="*80)
    
    dim_results, best_config = run_dimension_analysis(optimal_params, bin_sizes=bin_sizes, raw_mode=raw_mode)
    print_dimension_analysis_summary(dim_results)
    
    df_dim = pd.DataFrame(dim_results)
    dim_file = 'fgsd_imdb_raw_dimension_analysis.csv' if raw_mode else 'fgsd_imdb_dimension_analysis.csv'
    dim_path = os.path.join(results_dir, dim_file)
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
    
    final_results = run_final_classification(optimal_params, recommended_bins=recommended_bins, raw_mode=raw_mode)
    print_summary(final_results)
    
    df_final = pd.DataFrame(final_results)
    final_file = 'fgsd_imdb_raw_final_results.csv' if raw_mode else 'fgsd_imdb_final_results.csv'
    final_path = os.path.join(results_dir, final_file)
    df_final.to_csv(final_path, index=False)
    print(f"\n✅ Final results saved to: {final_path}")
    
    # STEP 5: Stability (NOW ENABLED IN RAW MODE)
    if run_stability:
        print("\n" + "="*80)
        print(f"STEP 5: Stability Analysis {'(Raw Mode)' if raw_mode else ''}")
        print("="*80)
        
        from .stability import (print_stability_summary, DEFAULT_PERTURBATION_RATIOS,
                               run_stability_analysis)
        
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
            
            # Run full stability analysis
            result, X_orig = run_stability_analysis(
                graphs=graphs,
                labels=labels,
                config=config,
                perturbation_ratios=DEFAULT_PERTURBATION_RATIOS,
                X_original=None,
                seed=42,
                test_size=0.15,
                compute_classification=True,
                use_raw_classifiers=raw_mode
            )
            
            # Add config params to each result entry
            for pr in result['perturbation_results']:
                pr['func'] = config['name']
                pr['raw_mode'] = raw_mode
                
                if config['func'] == 'hybrid':
                    pr['harm_bins'] = config['harm_bins']
                    pr['harm_range'] = config['harm_range']
                    pr['pol_bins'] = config['pol_bins']
                    pr['pol_range'] = config['pol_range']
                else:
                    pr['bins'] = config['bins']
                    pr['range'] = config['range']
            
            stability_results.append(pr)
        
        all_stability_results.append(result)
    
    print_stability_summary(all_stability_results)
    
    df_stab = pd.DataFrame(stability_results)
    stab_file = 'fgsd_imdb_raw_stability_results.csv' if raw_mode else 'fgsd_imdb_stability_results.csv'
    stab_path = os.path.join(results_dir, stab_file)
    df_stab.to_csv(stab_path, index=False)
    print(f"\n✅ Stability saved to: {stab_path}")
    
    del graphs
    gc.collect()
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print("Output files:")
    if not skip_grid:
        print(f"  📊 {grid_path}")
    print(f"  📊 {dim_path}")
    print(f"  📊 {final_path}")
    if tune_classifiers:
        print(f"  📊 {os.path.join(results_dir, 'fgsd_imdb_classifier_tuning.csv')}")
    if run_stability:
        stab_file = 'fgsd_imdb_raw_stability_results.csv' if raw_mode else 'fgsd_imdb_stability_results.csv'
        print(f"  📊 {os.path.join(results_dir, stab_file)}")
    
    # COMPARISON: Raw vs Preprocessed (only in raw mode)
    if raw_mode:
        preprocessed_final_path = os.path.join(RESULTS_DIR, 'fgsd_imdb_final_results.csv')
        compare_raw_vs_preprocessed(final_path, preprocessed_final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FGSD Classification on IMDB-MULTI')
    parser.add_argument('--stability', action='store_true', help='Include stability analysis')
    parser.add_argument('--stability-only', action='store_true', help='Run only stability (loads config from results)')
    parser.add_argument('--force', action='store_true', help='Force rerun everything')
    parser.add_argument('--skip-grid', action='store_true', help='Skip grid search')
    parser.add_argument('--tune-classifiers', action='store_true', help='Run classifier hyperparameter tuning (RF & SVM)')
    parser.add_argument('--raw-embeddings', action='store_true', help='Run with raw embeddings (no preprocessing)')
    args = parser.parse_args()
    
    main(run_stability=args.stability, force=args.force, skip_grid=args.skip_grid,
         stability_only=args.stability_only, tune_classifiers=args.tune_classifiers,
         raw_mode=args.raw_embeddings)
