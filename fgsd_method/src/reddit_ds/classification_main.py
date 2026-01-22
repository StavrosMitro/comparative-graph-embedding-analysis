"""
FGSD Classification on REDDIT-MULTI-12K

Usage:
    python -m reddit_ds.classification_main                    # Full pipeline
    python -m reddit_ds.classification_main --stability        # With stability analysis
    python -m reddit_ds.classification_main --force            # Force rerun preanalysis
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

from reddit_ds.config import RESULTS_DIR, PREANALYSIS_SAMPLE_SIZE, OptimalParams, DATASET_DIR
from reddit_ds.data_loader import ensure_dataset_ready, load_all_graphs
from reddit_ds.preanalysis import run_sampled_preanalysis
from reddit_ds.classification import (
    run_dimension_analysis, run_final_classification,
    print_dimension_analysis_summary, print_summary
)


def check_cached_results():
    """Check which CACHED results exist (only preanalysis)."""
    exists = {
        'preanalysis': os.path.exists(os.path.join(parent_dir, 'cache', 'reddit_preanalysis_cache.json')),
    }
    return exists


def main(run_stability: bool = False, force: bool = False):
    print("="*80)
    print("FGSD CLASSIFICATION ON REDDIT-MULTI-12K")
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
    
    # Get combined bin sizes to test
    all_bins = set()
    for func_type in ['harmonic', 'polynomial']:
        if func_type in recommended_bins:
            all_bins.update(recommended_bins[func_type])
    bin_sizes = sorted(list(all_bins))
    
    # =================================================================
    # STEP 2: Dimension Analysis (ALWAYS RUN)
    # =================================================================
    print("\n" + "="*80)
    print("STEP 2: Dimension Analysis (always runs)")
    print("="*80)
    
    dim_results, best_config = run_dimension_analysis(optimal_params, bin_sizes=bin_sizes)
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
    # STEP 3: Final Classification (ALWAYS RUN)
    # =================================================================
    print("\n" + "="*80)
    print("STEP 3: Final Classification (always runs)")
    print("="*80)
    
    final_results = run_final_classification(optimal_params, recommended_bins=recommended_bins)
    print_summary(final_results)
    
    df_final = pd.DataFrame(final_results)
    final_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_final_results.csv')
    df_final.to_csv(final_path, index=False)
    print(f"\n✅ Final results saved to: {final_path}")
    
    # Save optimal params
    params_path = os.path.join(RESULTS_DIR, 'reddit_optimal_params.txt')
    with open(params_path, 'w') as f:
        f.write("OPTIMAL PARAMETERS FOR REDDIT-MULTI-12K\n" + "="*50 + "\n")
        for func_type, params in optimal_params.items():
            f.write(f"{func_type}: bins={params.bins}, range={params.range_val:.4f}\n")
    
    # =================================================================
    # STEP 4: Stability Analysis (if requested)
    # =================================================================
    if run_stability:
        print("\n" + "="*80)
        print("STEP 4: Stability Analysis (always runs)")
        print("="*80)
        
        from .stability import (print_stability_summary, DEFAULT_PERTURBATION_RATIOS,
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
                'name': 'naive_hybrid', 'func': 'naive_hybrid',
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
                if config['func'] == 'naive_hybrid':
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
        stab_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_stability_results.csv')
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
    print(f"  📊 {dim_path}")
    print(f"  📊 {final_path}")
    if run_stability:
        print(f"  📊 {os.path.join(RESULTS_DIR, 'fgsd_reddit_stability_results.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FGSD Classification on REDDIT-MULTI-12K')
    parser.add_argument('--stability', action='store_true', help='Include stability analysis')
    parser.add_argument('--force', action='store_true', help='Force rerun preanalysis')
    args = parser.parse_args()
    
    main(run_stability=args.stability, force=args.force)
