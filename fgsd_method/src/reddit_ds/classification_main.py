"""
FGSD Classification on REDDIT-MULTI-12K

Usage:
    python -m reddit_ds.classification_main                    # Full pipeline
    python -m reddit_ds.classification_main --stability        # With stability analysis
    python -m reddit_ds.classification_main --stability-only   # Only stability (loads best config from results)
    python -m reddit_ds.classification_main --force            # Force rerun preanalysis
    python -m reddit_ds.classification_main --tune-classifiers # Run classifier hyperparameter tuning
"""

import os
import sys
import warnings
import gc
import argparse
import time

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

warnings.filterwarnings('ignore')

from reddit_ds.config import RESULTS_DIR, PREANALYSIS_SAMPLE_SIZE, OptimalParams, DATASET_DIR
from reddit_ds.data_loader import ensure_dataset_ready, load_all_graphs, load_metadata
from reddit_ds.preanalysis import run_sampled_preanalysis
from reddit_ds.classification import (
    evaluate_classifier, get_classifiers,
    print_dimension_analysis_summary, print_summary,
    generate_all_embeddings
)


# =============================================================================
# REDUCED BIN SIZES FOR FASTER EXECUTION
# =============================================================================
# Based on dimension analysis results (fgsd_reddit_dimension_analysis.csv):
#   - harmonic:    best=500 (0.457), second=100 (0.447)
#   - polynomial:  best=200 (0.441), second=500 (0.443)
#   - biharmonic:  best=200 (0.406), second=500 (0.407)
#
# Using [200, 500] for each = 6 total embeddings
# Estimated time: ~8 hours
#
# Original full analysis: [100, 200, 500] per function = 9 total (~12.5 hours)
REDUCED_BIN_SIZES = {
    'harmonic': [200, 500],      # Best: 500 (0.457), 200 (0.444)
    'polynomial': [200, 500],    # Best: 200 (0.441), 500 (0.443)
    'biharmonic': [200, 500],    # Best: 200 (0.406), 500 (0.407)
}

# Set to True to use reduced bin sizes (faster), False for full analysis
USE_REDUCED_BINS = True


class EmbeddingCache:
    """
    Cache for pre-computed embeddings.
    Stores embeddings for each (func_type, bins, range) combination.
    """
    def __init__(self):
        self.cache = {}  # key: (func_type, bins, range) -> X_all
        self.generation_times = {}
    
    def get_key(self, func_type, bins, range_val):
        return (func_type, bins, round(range_val, 2))
    
    def has(self, func_type, bins, range_val):
        return self.get_key(func_type, bins, range_val) in self.cache
    
    def get(self, func_type, bins, range_val):
        key = self.get_key(func_type, bins, range_val)
        return self.cache.get(key), self.generation_times.get(key, 0)
    
    def put(self, func_type, bins, range_val, X_all, gen_time):
        key = self.get_key(func_type, bins, range_val)
        self.cache[key] = X_all
        self.generation_times[key] = gen_time
        print(f"    Cached: {func_type} bins={bins} range={range_val:.2f} shape={X_all.shape}")
    
    def get_or_generate(self, func_type, bins, range_val, random_state=42):
        """Get from cache or generate if not present."""
        if self.has(func_type, bins, range_val):
            X_all, gen_time = self.get(func_type, bins, range_val)
            print(f"    Using cached: {func_type} bins={bins} (shape={X_all.shape})")
            return X_all, gen_time
        
        print(f"    Generating: {func_type} bins={bins} range={range_val:.2f}...")
        start_time = time.time()
        X_all = generate_all_embeddings(func_type, bins, range_val, random_state)
        gen_time = time.time() - start_time
        self.put(func_type, bins, range_val, X_all, gen_time)
        return X_all, gen_time
    
    def clear(self):
        """Clear cache and free memory."""
        for key in list(self.cache.keys()):
            del self.cache[key]
        self.cache.clear()
        self.generation_times.clear()
        gc.collect()
    
    def summary(self):
        """Print cache summary."""
        print("\n  Embedding Cache Summary:")
        total_size = 0
        for key, X in self.cache.items():
            size_mb = X.nbytes / 1024 / 1024
            total_size += size_mb
            print(f"    {key}: shape={X.shape}, size={size_mb:.1f}MB")
        print(f"    Total: {total_size:.1f}MB")


def check_cached_results():
    """Check which CACHED results exist."""
    exists = {
        'preanalysis': os.path.exists(os.path.join(parent_dir, 'cache', 'reddit_preanalysis_cache.json')),
    }
    return exists


def load_best_config_from_results():
    """Load best configuration from existing results CSV."""
    final_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_final_results.csv')
    dim_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_dimension_analysis.csv')
    
    for path in [final_path, dim_path]:
        if os.path.exists(path):
            print(f"Loading best config from: {path}")
            df = pd.read_csv(path)
            
            optimal_params = {}
            recommended_bins = {}
            
            for func_type in ['harmonic', 'polynomial', 'biharmonic']:
                func_df = df[df['func'] == func_type]
                if len(func_df) > 0:
                    best_row = func_df.loc[func_df['accuracy'].idxmax()]
                    bins = int(best_row['bins']) if pd.notna(best_row.get('bins')) else 200
                    range_val = float(best_row['range']) if pd.notna(best_row.get('range')) else 15.0
                    
                    optimal_params[func_type] = OptimalParams(
                        func_type=func_type,
                        bins=bins,
                        range_val=range_val,
                        p99=range_val,
                        recommended_bins=bins
                    )
                    recommended_bins[func_type] = [bins]
                    print(f"  {func_type.upper()}: bins={bins}, range={range_val}")
            
            if optimal_params:
                return optimal_params, recommended_bins
    
    raise FileNotFoundError(
        f"No results found in {RESULTS_DIR}. "
        "Run full pipeline first: python -m reddit_ds.classification_main"
    )


def generate_all_required_embeddings(optimal_params, recommended_bins, embedding_cache, random_state=42):
    """
    Pre-generate ALL embeddings that will be needed for the entire pipeline.
    This is called ONCE at the start and embeddings are reused everywhere.
    """
    print("\n" + "="*80)
    print("PRE-GENERATING ALL EMBEDDINGS (will be reused throughout pipeline)")
    print("="*80)
    
    # Use reduced bins if enabled
    if USE_REDUCED_BINS:
        print("\n⚠️  USING REDUCED BIN SIZES FOR FASTER EXECUTION")
        print(f"   Original bins would be: {recommended_bins}")
        print(f"   Using reduced bins: {REDUCED_BIN_SIZES}")
        bins_to_use = REDUCED_BIN_SIZES
    else:
        bins_to_use = recommended_bins
    
    for func_type in ['harmonic', 'polynomial', 'biharmonic']:
        if func_type not in optimal_params:
            continue
        
        range_val = round(optimal_params[func_type].range_val, 2)
        # bins_to_generate = recommended_bins.get(func_type, [optimal_params[func_type].bins])  # Original
        bins_to_generate = bins_to_use.get(func_type, [optimal_params[func_type].bins])
        
        print(f"\n  {func_type.upper()} (range={range_val}):")
        print(f"    Bins to generate: {bins_to_generate}")
        
        for bins in bins_to_generate:
            embedding_cache.get_or_generate(func_type, bins, range_val, random_state)
    
    embedding_cache.summary()
    print("\n✅ All embeddings pre-generated and cached!")


def run_classifier_hyperparameter_search_with_cache(
    embedding_cache, optimal_params, train_idx, test_idx, all_labels,
    test_size=0.15, random_state=42, fast_mode=True
):
    """Run hyperparameter tuning using cached embeddings."""
    from .hyperparameter_search import run_classifier_tuning
    
    print("\n" + "="*80)
    print("CLASSIFIER HYPERPARAMETER TUNING (using cached embeddings)")
    print("="*80)
    
    # Use polynomial (usually best) for tuning
    func_type = 'polynomial' if 'polynomial' in optimal_params else list(optimal_params.keys())[0]
    params = optimal_params[func_type]
    
    # Get from cache - no regeneration!
    X_all, gen_time = embedding_cache.get_or_generate(
        func_type, params.bins, round(params.range_val, 2), random_state
    )
    
    X_train = X_all[train_idx]
    y_train = all_labels[train_idx]
    X_test = X_all[test_idx]
    y_test = all_labels[test_idx]
    
    print(f"\nUsing cached {func_type} embeddings: shape={X_train.shape}")
    
    # Run tuning
    tuning_results = run_classifier_tuning(
        X_train, y_train,
        fast_mode=fast_mode,
        cv=3,
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
    df = pd.DataFrame(eval_results)
    tuning_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_classifier_tuning.csv')
    df.to_csv(tuning_path, index=False)
    print(f"\n✅ Tuning results saved to: {tuning_path}")
    
    return tuning_results


def run_dimension_analysis_with_cache(
    embedding_cache, optimal_params, recommended_bins,
    train_idx, test_idx, all_labels, random_state=42
):
    """
    Run dimension analysis using cached embeddings - NO regeneration.
    Returns results AND identifies best bins for each function type.
    """
    print(f"\n{'='*80}\nDIMENSION ANALYSIS (using cached embeddings)\n{'='*80}")
    
    y_train, y_test = all_labels[train_idx], all_labels[test_idx]
    
    results = []
    best_config = {}
    classifiers = get_classifiers(random_state)
    
    # Use reduced bins if enabled
    if USE_REDUCED_BINS:
        bins_to_use = REDUCED_BIN_SIZES
    else:
        bins_to_use = recommended_bins
    
    for func_type in ['harmonic', 'polynomial', 'biharmonic']:
        if func_type not in optimal_params:
            continue
        
        range_val = round(optimal_params[func_type].range_val, 2)
        # bin_sizes = recommended_bins.get(func_type, [optimal_params[func_type].bins])  # Original
        bin_sizes = bins_to_use.get(func_type, [optimal_params[func_type].bins])
        
        print(f"\n{'='*60}\nFunction: {func_type.upper()} (range={range_val})\n{'='*60}")
        
        best_acc, best_bins = 0, bin_sizes[0]
        
        for bins in bin_sizes:
            print(f"\n--- bins={bins} ---")
            
            # Get from cache - NO regeneration!
            X_all, gen_time = embedding_cache.get_or_generate(func_type, bins, range_val, random_state)
            
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            
            print(f"  Shape: {X_train.shape}, Time: {gen_time:.2f}s (cached)")
            
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
                    best_acc = res['accuracy']
                    best_bins = bins
        
        best_config[func_type] = (best_bins, best_acc)
        print(f"\n✓ Best for {func_type}: bins={best_bins} (acc={best_acc:.4f})")
    
    return results, best_config


def run_final_classification_with_cache(
    embedding_cache, optimal_params, train_idx, test_idx, all_labels, random_state=42
):
    """
    Run final classification (hybrids) using cached embeddings.
    Creates hybrids by concatenating cached embeddings - NO regeneration!
    """
    print(f"\n{'='*80}\nFINAL CLASSIFICATION (Hybrids - using cached embeddings)\n{'='*80}")
    
    y_train, y_test = all_labels[train_idx], all_labels[test_idx]
    
    results = []
    classifiers = get_classifiers(random_state)
    
    # =================================================================
    # Create naive_hybrid (harmonic + polynomial) by concatenation
    # =================================================================
    if 'harmonic' in optimal_params and 'polynomial' in optimal_params:
        print(f"\n--- NAIVE_HYBRID (harmonic + polynomial) ---")
        
        h_params = optimal_params['harmonic']
        p_params = optimal_params['polynomial']
        
        # Get from cache
        X_harm, _ = embedding_cache.get_or_generate(
            'harmonic', h_params.bins, round(h_params.range_val, 2), random_state
        )
        X_poly, _ = embedding_cache.get_or_generate(
            'polynomial', p_params.bins, round(p_params.range_val, 2), random_state
        )
        
        # Concatenate - NO recomputation!
        X_hybrid = np.hstack([X_harm, X_poly])
        X_train_hybrid = X_hybrid[train_idx]
        X_test_hybrid = X_hybrid[test_idx]
        
        print(f"  shape: {X_train_hybrid.shape} (concatenated from cached embeddings)")
        
        for clf_name, clf in classifiers.items():
            res = evaluate_classifier(X_train_hybrid, X_test_hybrid, y_train, y_test, clf_name, clf)
            result_entry = {
                'func': 'naive_hybrid',
                'harm_bins': h_params.bins, 'harm_range': round(h_params.range_val, 2),
                'pol_bins': p_params.bins, 'pol_range': round(p_params.range_val, 2),
                'embedding_dim': X_train_hybrid.shape[1], **res
            }
            results.append(result_entry)
            print(f"  {clf_name}: Acc={res['accuracy']:.4f}")
        
        del X_hybrid
        gc.collect()
    
    # =================================================================
    # Create biharmonic_hybrid (biharmonic + polynomial) by concatenation
    # =================================================================
    if 'biharmonic' in optimal_params and 'polynomial' in optimal_params:
        print(f"\n--- BIHARMONIC_HYBRID (biharmonic + polynomial) ---")
        
        bh_params = optimal_params['biharmonic']
        p_params = optimal_params['polynomial']
        
        # Get from cache
        X_biharm, _ = embedding_cache.get_or_generate(
            'biharmonic', bh_params.bins, round(bh_params.range_val, 2), random_state
        )
        X_poly, _ = embedding_cache.get_or_generate(
            'polynomial', p_params.bins, round(p_params.range_val, 2), random_state
        )
        
        # Concatenate - NO recomputation!
        X_hybrid = np.hstack([X_biharm, X_poly])
        X_train_hybrid = X_hybrid[train_idx]
        X_test_hybrid = X_hybrid[test_idx]
        
        print(f"  shape: {X_train_hybrid.shape} (concatenated from cached embeddings)")
        
        for clf_name, clf in classifiers.items():
            res = evaluate_classifier(X_train_hybrid, X_test_hybrid, y_train, y_test, clf_name, clf)
            result_entry = {
                'func': 'biharmonic_hybrid',
                'biharm_bins': bh_params.bins, 'biharm_range': round(bh_params.range_val, 2),
                'pol_bins': p_params.bins, 'pol_range': round(p_params.range_val, 2),
                'embedding_dim': X_train_hybrid.shape[1], **res
            }
            results.append(result_entry)
            print(f"  {clf_name}: Acc={res['accuracy']:.4f}")
        
        del X_hybrid
        gc.collect()
    
    return results


def run_stability_only():
    """Run only stability analysis using best config from existing results."""
    from .stability import (
        run_stability_analysis, print_stability_summary,
        load_best_configs_from_csv
    )
    from .data_loader import load_metadata
    
    print("="*80)
    print("STABILITY-ONLY MODE (Batch-wise, RAM-efficient)")
    print("Loading best harmonic & polynomial from results")
    print("="*80)
    
    # Load best configs from CSV
    final_results_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_final_results.csv')
    print(f"\nLoading best configs from: {final_results_path}")
    configs = load_best_configs_from_csv(final_results_path)
    
    if not configs:
        raise ValueError("No valid configs found in results CSV")
    
    # Load only labels (not graphs - they're loaded batch-wise)
    print("\n" + "="*80)
    print("Loading metadata...")
    print("="*80)
    
    ensure_dataset_ready()
    records = load_metadata(DATASET_DIR)
    labels = np.array([r.label for r in records])
    print(f"  Total samples: {len(labels)}")
    
    # Run stability analysis (graphs loaded batch-wise internally)
    results, control_embeddings = run_stability_analysis(
        graphs=None,  # Not needed - loaded batch-wise
        labels=labels,
        configs=configs,
        seed=42,
        test_size=0.15
    )
    
    # Print summary
    print_stability_summary(results)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_stab = pd.DataFrame(results)
    stab_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_stability_results.csv')
    df_stab.to_csv(stab_path, index=False)
    print(f"\n✅ Stability results saved to: {stab_path}")
    
    # Report cache status
    print(f"\n✅ Control embeddings cached in: {os.path.join(parent_dir, 'cache')}")
    for key in control_embeddings:
        print(f"    - {key}: shape={control_embeddings[key].shape}")
    
    gc.collect()


def main(run_stability: bool = False, force: bool = False, stability_only: bool = False, tune_classifiers: bool = False):
    if stability_only:
        run_stability_only()
        return
    
    print("="*80)
    print("FGSD CLASSIFICATION ON REDDIT-MULTI-12K")
    print("(OPTIMIZED: All embeddings generated once and reused)")
    print("="*80)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ensure_dataset_ready()
    
    cached = check_cached_results()
    print("\nCached results (will reuse if exists):")
    for key, val in cached.items():
        status = "✅ Found" if val else "❌ Missing"
        print(f"  {key}: {status}")
    
    if force:
        print("\n⚠️  FORCE mode: Will recompute preanalysis")
        cached = {k: False for k in cached}
    
    # =================================================================
    # STEP 1: Pre-analysis (CACHED)
    # =================================================================
    print("\n" + "="*80)
    print("STEP 1: Pre-Analysis (cached)")
    print("="*80)
    
    try:
        optimal_params, recommended_bins = run_sampled_preanalysis(
            graphs=None, use_cache=True, force_recompute=force, dataset_name='reddit'
        )
    except ValueError:
        print("Loading graphs for preanalysis...")
        graphs, _ = load_all_graphs(DATASET_DIR)
        optimal_params, recommended_bins = run_sampled_preanalysis(
            graphs, use_cache=True, force_recompute=force, dataset_name='reddit'
        )
        del graphs
        gc.collect()
    
    print("\nPre-analysis results:")
    for func_type, params in optimal_params.items():
        print(f"  {func_type.upper()}: range={params.range_val:.2f}, bins={recommended_bins.get(func_type, [params.bins])}")
    
    # =================================================================
    # STEP 2: Load metadata and create train/test split
    # =================================================================
    print("\n" + "="*80)
    print("STEP 2: Loading metadata and creating train/test split")
    print("="*80)
    
    records = load_metadata(DATASET_DIR)
    all_labels = np.array([r.label for r in records])
    indices = np.arange(len(records))
    
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=all_labels)
    
    print(f"  Total samples: {len(records)}")
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Test samples: {len(test_idx)}")
    
    # =================================================================
    # STEP 3: PRE-GENERATE ALL EMBEDDINGS (DONE ONCE!)
    # =================================================================
    embedding_cache = EmbeddingCache()
    generate_all_required_embeddings(optimal_params, recommended_bins, embedding_cache, random_state=42)
    
    # =================================================================
    # STEP 4: Classifier Hyperparameter Tuning (OPTIONAL - uses cached embeddings)
    # =================================================================
    if tune_classifiers:
        run_classifier_hyperparameter_search_with_cache(
            embedding_cache, optimal_params, train_idx, test_idx, all_labels,
            fast_mode=True
        )
        print("\n✅ Classifier tuning complete.")
    
    # =================================================================
    # STEP 5: Dimension Analysis (uses cached embeddings)
    # =================================================================
    print("\n" + "="*80)
    print("STEP 5: Dimension Analysis (using cached embeddings)")
    print("="*80)
    
    dim_results, best_config = run_dimension_analysis_with_cache(
        embedding_cache, optimal_params, recommended_bins,
        train_idx, test_idx, all_labels
    )
    print_dimension_analysis_summary(dim_results)
    
    df_dim = pd.DataFrame(dim_results)
    dim_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_dimension_analysis.csv')
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
    # STEP 6: Final Classification (hybrids - uses cached embeddings)
    # =================================================================
    print("\n" + "="*80)
    print("STEP 6: Final Classification (Hybrids - using cached embeddings)")
    print("="*80)
    
    hybrid_results = run_final_classification_with_cache(
        embedding_cache, optimal_params, train_idx, test_idx, all_labels
    )
    
    # Combine all results
    all_results = dim_results + hybrid_results
    print_summary(all_results)
    
    df_final = pd.DataFrame(all_results)
    final_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_final_results.csv')
    df_final.to_csv(final_path, index=False)
    print(f"\n✅ Final results saved to: {final_path}")
    
    # =================================================================
    # STEP 7: Stability Analysis (if requested)
    # =================================================================
    if run_stability:
        from .stability import (
            run_stability_analysis, print_stability_summary,
            load_best_configs_from_csv
        )
        
        print("\n" + "="*80)
        print("STEP 7: Stability Analysis")
        print("="*80)
        
        # Load best configs from the final results we just saved
        final_results_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_final_results.csv')
        configs = load_best_configs_from_csv(final_results_path)
        
        graphs, labels = load_all_graphs(DATASET_DIR)
        
        # Run stability analysis
        stability_results, control_embeddings = run_stability_analysis(
            graphs, labels, configs, seed=42, test_size=0.15
        )
        
        print_stability_summary(stability_results)
        
        df_stab = pd.DataFrame(stability_results)
        stab_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_stability_results.csv')
        df_stab.to_csv(stab_path, index=False)
        print(f"\n✅ Stability saved to: {stab_path}")
        
        del graphs
        gc.collect()
    
    # =================================================================
    # Cleanup and Summary
    # =================================================================
    print("\n" + "="*80)
    print("Cleaning up embedding cache...")
    embedding_cache.clear()
    
    print("\n" + "="*80)
    print("✅ COMPLETE! Output files:")
    print("="*80)
    print(f"  📊 {dim_path}")
    print(f"  📊 {final_path}")
    if tune_classifiers:
        print(f"  📊 {os.path.join(RESULTS_DIR, 'fgsd_reddit_classifier_tuning.csv')}")
    if run_stability:
        print(f"  📊 {os.path.join(RESULTS_DIR, 'fgsd_reddit_stability_results.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FGSD Classification on REDDIT-MULTI-12K')
    parser.add_argument('--stability', action='store_true', help='Include stability analysis')
    parser.add_argument('--stability-only', action='store_true', help='Run only stability analysis (loads best config from results)')
    parser.add_argument('--force', action='store_true', help='Force rerun preanalysis')
    parser.add_argument('--tune-classifiers', action='store_true', help='Run classifier hyperparameter tuning (RF & SVM)')
    args = parser.parse_args()
    
    main(run_stability=args.stability, force=args.force, stability_only=args.stability_only, tune_classifiers=args.tune_classifiers)
