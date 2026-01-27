"""
Pre-analysis functions for determining optimal bins and range parameters.
Includes caching to avoid recomputation.
"""

import os
import gc
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import (
    RESULTS_DIR, CACHE_DIR, PREANALYSIS_SAMPLE_SIZE, 
    MAX_NODES_FOR_PREANALYSIS, OptimalParams
)


# =============================================================================
# CACHING FUNCTIONS
# =============================================================================
def get_cache_path(dataset_name: str = 'reddit') -> Path:
    """Get path to cache file for preanalysis results."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return Path(CACHE_DIR) / f'{dataset_name}_preanalysis_cache.json'


def save_preanalysis_to_cache(
    optimal_params: Dict[str, OptimalParams],
    recommended_bins: Dict[str, List[int]],
    dataset_name: str = 'reddit',
    sample_size: int = PREANALYSIS_SAMPLE_SIZE,
    max_nodes: int = MAX_NODES_FOR_PREANALYSIS
):
    """Save preanalysis results including recommended bin sizes to cache."""
    cache_path = get_cache_path(dataset_name)
    
    cache_data = {
        'metadata': {
            'dataset_name': dataset_name,
            'sample_size': sample_size,
            'max_nodes': max_nodes,
        },
        'params': {},
        'recommended_bins': recommended_bins  # {func_type: [bin1, bin2, bin3]}
    }
    
    for func_type, params in optimal_params.items():
        cache_data['params'][func_type] = {
            'func_type': params.func_type,
            'bins': params.bins,
            'range_val': params.range_val,
            'p99': params.p99,
            'recommended_bins': params.recommended_bins
        }
    
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"   Cache saved: {cache_path}")


def load_preanalysis_from_cache(
    dataset_name: str = 'reddit',
    sample_size: int = PREANALYSIS_SAMPLE_SIZE,
    max_nodes: int = MAX_NODES_FOR_PREANALYSIS
) -> Optional[Tuple[Dict[str, OptimalParams], Dict[str, List[int]]]]:
    """
    Load preanalysis results from cache.
    
    Returns:
        Tuple of (optimal_params, recommended_bins) or None if cache invalid
    """
    cache_path = get_cache_path(dataset_name)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        meta = cache_data.get('metadata', {})
        if (meta.get('sample_size') != sample_size or 
            meta.get('max_nodes') != max_nodes):
            print(f"   Cache exists but settings differ. Will recompute.")
            return None
        
        # Reconstruct OptimalParams
        optimal_params = {}
        for func_type, params_dict in cache_data['params'].items():
            optimal_params[func_type] = OptimalParams(
                func_type=params_dict['func_type'],
                bins=params_dict['bins'],
                range_val=params_dict['range_val'],
                p99=params_dict['p99'],
                recommended_bins=params_dict['recommended_bins']
            )
        
        # Get recommended bins
        recommended_bins = cache_data.get('recommended_bins', {})
        
        return optimal_params, recommended_bins
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"   Cache file corrupted: {e}. Will recompute.")
        return None


def load_default_params() -> Tuple[Dict[str, OptimalParams], Dict[str, List[int]]]:
    """Load hardcoded default parameters if no cache exists."""
    print("   Using hardcoded default parameters")
    
    optimal_params = {
        'harmonic': OptimalParams(
            func_type='harmonic',
            bins=500,
            range_val=14.61,
            p99=14.61,
            recommended_bins=500
        ),
        'polynomial': OptimalParams(
            func_type='polynomial',
            bins=500,
            range_val=3.48,
            p99=3.48,
            recommended_bins=500
        ),
        'biharmonic': OptimalParams(
            func_type='biharmonic',
            bins=200,
            range_val=500.0,  # Biharmonic has much larger range (1/λ²)
            p99=500.0,
            recommended_bins=200
        )
    }
    
    # Default bin sizes to test
    recommended_bins = {
        'harmonic': [100, 200, 500],
        'polynomial': [100, 200, 500],
        'biharmonic': [100, 200, 500]
    }
    
    return optimal_params, recommended_bins


# =============================================================================
# COMPUTATION FUNCTIONS
# =============================================================================
def compute_spectral_distances_sampled(
    graphs: List[nx.Graph], 
    func_type: str,
    sample_size: int = PREANALYSIS_SAMPLE_SIZE,
    max_nodes: int = MAX_NODES_FOR_PREANALYSIS
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spectral distances for a SAMPLE of graphs."""
    # Filter out very large graphs first
    valid_indices = [
        i for i, g in enumerate(graphs) 
        if g.number_of_nodes() <= max_nodes and g.number_of_nodes() >= 2
    ]
    
    # Sample from valid graphs
    np.random.seed(42)
    if len(valid_indices) > sample_size:
        sample_indices = np.random.choice(valid_indices, sample_size, replace=False)
    else:
        sample_indices = valid_indices
    
    print(f"\nComputing spectral distances for {len(sample_indices)} graphs "
          f"(max {max_nodes} nodes) using func='{func_type}'...")
    
    all_distances = []
    node_counts = []
    skipped = 0
    
    for idx in tqdm(sample_indices, desc=f"Processing {func_type}"):
        G = graphs[idx]
        n_nodes = G.number_of_nodes()
        
        if n_nodes < 2:
            skipped += 1
            continue
        
        node_counts.append(n_nodes)
        
        try:
            L = np.asarray(nx.normalized_laplacian_matrix(G).todense())
            w, v = np.linalg.eigh(L)
            
            if func_type == 'harmonic':
                with np.errstate(divide='ignore', invalid='ignore'):
                    func_w = np.where(w > 1e-9, 1.0 / w, 0)
            elif func_type == 'polynomial':
                func_w = w ** 2
            elif func_type == 'biharmonic':
                with np.errstate(divide='ignore', invalid='ignore'):
                    func_w = np.where(w > 1e-9, 1.0 / (w**2), 0)
            else:
                func_w = w
            
            fL = v @ np.diag(func_w) @ v.T
            ones = np.ones(L.shape[0])
            S = np.outer(np.diag(fL), ones) + np.outer(ones, np.diag(fL)) - 2 * fL
            S = np.asarray(S)
            
            triu_indices = np.triu_indices_from(S, k=1)
            distances = S[triu_indices]
            all_distances.extend(distances.flatten().tolist())
            
            del L, w, v, fL, S, distances
            
        except Exception:
            skipped += 1
        
        if len(node_counts) % 100 == 0:
            gc.collect()
    
    if skipped > 0:
        print(f"  Skipped {skipped} graphs")
    
    gc.collect()
    return np.array(all_distances), np.array(node_counts)


def determine_optimal_params_and_bins(
    distances: np.ndarray, 
    node_counts: np.ndarray, 
    func_type: str,
    save_plot: bool = True,
    top_n_bins: int = 3
) -> Tuple[OptimalParams, List[int]]:
    """
    Analyze distribution and determine:
    1. Optimal range (99th percentile)
    2. Top N recommended bin sizes based on sparsity analysis
    
    Returns:
        Tuple of (OptimalParams, list of top N bin sizes)
    """
    print(f"\n{'='*60}")
    print(f"PRE-ANALYSIS: {func_type.upper()}")
    print(f"{'='*60}")
    
    min_val = np.min(distances)
    max_val = np.max(distances)
    p95 = np.percentile(distances, 95)
    p99 = np.percentile(distances, 99)
    
    print(f"Total distance values: {len(distances):,}")
    print(f"Graphs in sample: {len(node_counts):,}")
    print(f"\n1. SPECTRAL DISTANCE STATISTICS:")
    print(f"   Min:  {min_val:.4f}")
    print(f"   Max:  {max_val:.4f}")
    print(f"   95th Percentile: {p95:.4f}")
    print(f"   99th Percentile: {p99:.4f} <- OPTIMAL RANGE")
    
    optimal_range = p99
    
    # Analyze different bin sizes
    print(f"\n2. BIN SIZE ANALYSIS (Range = [0, {optimal_range:.2f}]):")
    print(f"   {'Bins':<10} | {'Avg Hits/Bin/Graph':<20} | {'Sparsity':<15} | {'Score':<10}")
    print("   " + "-"*70)
    
    test_bins = [50, 100, 150, 200, 250, 300, 400, 500]
    bin_scores = []
    
    for b in test_bins:
        hist, _ = np.histogram(distances, bins=b, range=(0, optimal_range))
        avg_hits = np.sum(hist) / len(node_counts) / b
        
        # Score: balance between resolution (more bins) and coverage (more hits)
        # Higher avg_hits = less sparse = better
        # More bins = more resolution = better (but diminishing returns)
        if avg_hits >= 2.0:
            sparsity = "Very Low"
            score = avg_hits * np.log(b)  # Reward more bins when not sparse
        elif avg_hits >= 1.0:
            sparsity = "Low"
            score = avg_hits * np.log(b) * 0.8
        elif avg_hits >= 0.5:
            sparsity = "Medium"
            score = avg_hits * np.log(b) * 0.5
        else:
            sparsity = "HIGH"
            score = avg_hits * np.log(b) * 0.2
        
        bin_scores.append((b, avg_hits, sparsity, score))
        print(f"   {b:<10} | {avg_hits:<20.4f} | {sparsity:<15} | {score:<10.4f}")
    
    # Sort by score and get top N
    sorted_bins = sorted(bin_scores, key=lambda x: x[3], reverse=True)
    top_bins = [b[0] for b in sorted_bins[:top_n_bins]]
    best_bins = top_bins[0]
    
    print(f"\n3. RECOMMENDED BIN SIZES (Top {top_n_bins}):")
    for i, (bins, hits, sparsity, score) in enumerate(sorted_bins[:top_n_bins]):
        marker = "★ BEST" if i == 0 else ""
        print(f"   {i+1}. bins={bins} (score={score:.4f}) {marker}")
    
    # Save plot
    if save_plot:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        viz_cutoff = np.percentile(distances, 99.5)
        viz_data = distances[distances <= viz_cutoff]
        
        ax.hist(viz_data, bins=150, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'95% ({p95:.2f})')
        ax.axvline(p99, color='red', linestyle='-', linewidth=2, label=f'99% = RANGE ({p99:.2f})')
        
        ax.set_title(f"REDDIT-MULTI-12K: {func_type.upper()} Spectral Distances\n"
                    f"Recommended bins: {top_bins}", fontsize=14)
        ax.set_xlabel("Spectral Distance Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f'reddit_preanalysis_{func_type}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"   Plot saved: {save_path}")
    
    optimal_params = OptimalParams(
        func_type=func_type,
        bins=best_bins,
        range_val=optimal_range,
        p99=p99,
        recommended_bins=best_bins
    )
    
    return optimal_params, top_bins


def run_sampled_preanalysis(
    graphs: List[nx.Graph] = None, 
    sample_size: int = PREANALYSIS_SAMPLE_SIZE,
    use_cache: bool = True,
    force_recompute: bool = False,
    dataset_name: str = 'reddit',
    top_n_bins: int = 3
) -> Tuple[Dict[str, OptimalParams], Dict[str, List[int]]]:
    """
    Run pre-analysis to determine optimal range and bin sizes.
    
    Args:
        graphs: List of graphs (can be None if using cache)
        sample_size: Number of graphs to sample
        use_cache: Whether to use cached results
        force_recompute: Force recomputation even if cache exists
        dataset_name: Name for cache file
        top_n_bins: Number of top bin sizes to recommend
    
    Returns:
        Tuple of (optimal_params_dict, recommended_bins_dict)
        - optimal_params_dict: {func_type: OptimalParams}
        - recommended_bins_dict: {func_type: [bin1, bin2, bin3]}
    """
    print("\n" + "="*80)
    print(f"PRE-ANALYSIS FOR {dataset_name.upper()}")
    print("="*80)
    
    # Try cache first
    if use_cache and not force_recompute:
        print("Checking for cached pre-analysis results...")
        
        cached = load_preanalysis_from_cache(dataset_name, sample_size, MAX_NODES_FOR_PREANALYSIS)
        if cached is not None:
            optimal_params, recommended_bins = cached
            print("✅ Loaded from cache!")
            for func_type, params in optimal_params.items():
                bins_list = recommended_bins.get(func_type, [params.bins])
                print(f"   {func_type.upper():12} -> range={params.range_val:.2f}, bins={bins_list}")
            return optimal_params, recommended_bins
        
        # No cache, try defaults if no graphs
        if graphs is None:
            print("No graphs provided and no cache. Loading defaults...")
            return load_default_params()
    
    # Need to compute
    if graphs is None:
        raise ValueError("Graphs required for preanalysis computation.")
    
    print(f"Computing pre-analysis (sample_size={sample_size})...")
    
    optimal_params = {}
    recommended_bins = {}
    
    # Include biharmonic in the analysis
    for func_type in ['harmonic', 'polynomial', 'biharmonic']:
        distances, node_counts = compute_spectral_distances_sampled(
            graphs, func_type, sample_size
        )
        params, top_bins = determine_optimal_params_and_bins(
            distances, node_counts, func_type, top_n_bins=top_n_bins
        )
        optimal_params[func_type] = params
        recommended_bins[func_type] = top_bins
        
        del distances
        gc.collect()
    
    # Save to cache
    save_preanalysis_to_cache(optimal_params, recommended_bins, dataset_name, sample_size, MAX_NODES_FOR_PREANALYSIS)
    
    return optimal_params, recommended_bins
