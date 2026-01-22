"""
Pre-analysis functions for ENZYMES.
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


def get_cache_path(dataset_name: str = 'enzymes') -> Path:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return Path(CACHE_DIR) / f'{dataset_name}_preanalysis_cache.json'


def save_preanalysis_to_cache(
    optimal_params: Dict[str, OptimalParams],
    recommended_bins: Dict[str, List[int]],
    dataset_name: str = 'enzymes',
    sample_size: int = PREANALYSIS_SAMPLE_SIZE,
    max_nodes: int = MAX_NODES_FOR_PREANALYSIS
):
    cache_path = get_cache_path(dataset_name)
    
    cache_data = {
        'metadata': {'dataset_name': dataset_name, 'sample_size': sample_size, 'max_nodes': max_nodes},
        'params': {},
        'recommended_bins': recommended_bins
    }
    
    for func_type, params in optimal_params.items():
        cache_data['params'][func_type] = {
            'func_type': params.func_type, 'bins': params.bins,
            'range_val': params.range_val, 'p99': params.p99,
            'recommended_bins': params.recommended_bins
        }
    
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"   Cache saved: {cache_path}")


def load_preanalysis_from_cache(
    dataset_name: str = 'enzymes',
    sample_size: int = PREANALYSIS_SAMPLE_SIZE,
    max_nodes: int = MAX_NODES_FOR_PREANALYSIS
) -> Optional[Tuple[Dict[str, OptimalParams], Dict[str, List[int]]]]:
    cache_path = get_cache_path(dataset_name)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        meta = cache_data.get('metadata', {})
        if meta.get('sample_size') != sample_size or meta.get('max_nodes') != max_nodes:
            return None
        
        optimal_params = {}
        for func_type, params_dict in cache_data['params'].items():
            optimal_params[func_type] = OptimalParams(
                func_type=params_dict['func_type'], bins=params_dict['bins'],
                range_val=params_dict['range_val'], p99=params_dict['p99'],
                recommended_bins=params_dict['recommended_bins']
            )
        
        return optimal_params, cache_data.get('recommended_bins', {})
    except:
        return None


def load_default_params() -> Tuple[Dict[str, OptimalParams], Dict[str, List[int]]]:
    print("   Using hardcoded default parameters (best known)")
    optimal_params = {
        'harmonic': OptimalParams('harmonic', 100, 33.0, 33.0, 100),  # Updated to 33.0
        'polynomial': OptimalParams('polynomial', 100, 4.1, 4.1, 100)  # Updated to 4.1
    }
    recommended_bins = {'harmonic': [50, 100, 150], 'polynomial': [50, 100, 150]}
    return optimal_params, recommended_bins


def compute_spectral_distances_sampled(graphs, func_type, sample_size=PREANALYSIS_SAMPLE_SIZE, max_nodes=MAX_NODES_FOR_PREANALYSIS):
    valid_indices = [i for i, g in enumerate(graphs) if 2 <= g.number_of_nodes() <= max_nodes]
    
    np.random.seed(42)
    sample_indices = np.random.choice(valid_indices, min(sample_size, len(valid_indices)), replace=False) if len(valid_indices) > sample_size else valid_indices
    
    print(f"\nComputing spectral distances for {len(sample_indices)} graphs using '{func_type}'...")
    
    all_distances, node_counts = [], []
    
    for idx in tqdm(sample_indices, desc=f"Processing {func_type}"):
        G = graphs[idx]
        if G.number_of_nodes() < 2:
            continue
        node_counts.append(G.number_of_nodes())
        
        try:
            L = np.asarray(nx.normalized_laplacian_matrix(G).todense())
            w, v = np.linalg.eigh(L)
            
            if func_type == 'harmonic':
                with np.errstate(divide='ignore', invalid='ignore'):
                    func_w = np.where(w > 1e-9, 1.0 / w, 0)
            elif func_type == 'polynomial':
                func_w = w ** 2
            else:
                func_w = w
            
            fL = v @ np.diag(func_w) @ v.T
            ones = np.ones(L.shape[0])
            S = np.outer(np.diag(fL), ones) + np.outer(ones, np.diag(fL)) - 2 * fL
            
            triu_indices = np.triu_indices_from(S, k=1)
            all_distances.extend(S[triu_indices].flatten().tolist())
        except:
            pass
    
    gc.collect()
    return np.array(all_distances), np.array(node_counts)


def determine_optimal_params_and_bins(distances, node_counts, func_type, save_plot=True, top_n_bins=3):
    print(f"\n{'='*60}\nPRE-ANALYSIS: {func_type.upper()}\n{'='*60}")
    
    p95, p99 = np.percentile(distances, 95), np.percentile(distances, 99)
    optimal_range = p99
    
    print(f"99th Percentile (RANGE): {p99:.4f}")
    print(f"\nBIN SIZE ANALYSIS:")
    
    test_bins = [50, 100, 150, 200, 250, 300]
    bin_scores = []
    
    for b in test_bins:
        hist, _ = np.histogram(distances, bins=b, range=(0, optimal_range))
        avg_hits = np.sum(hist) / len(node_counts) / b
        score = avg_hits * np.log(b) * (1.0 if avg_hits >= 2.0 else 0.8 if avg_hits >= 1.0 else 0.5 if avg_hits >= 0.5 else 0.2)
        bin_scores.append((b, avg_hits, score))
        print(f"   bins={b}: avg_hits={avg_hits:.4f}, score={score:.4f}")
    
    sorted_bins = sorted(bin_scores, key=lambda x: x[2], reverse=True)
    top_bins = [b[0] for b in sorted_bins[:top_n_bins]]
    
    print(f"\nTop {top_n_bins} bins: {top_bins}")
    
    if save_plot:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        viz_data = distances[distances <= np.percentile(distances, 99.5)]
        ax.hist(viz_data, bins=150, color='salmon', edgecolor='black', alpha=0.7)
        ax.axvline(p99, color='red', linestyle='-', linewidth=2, label=f'99% = RANGE ({p99:.2f})')
        ax.set_title(f"ENZYMES: {func_type.upper()} - Recommended bins: {top_bins}")
        ax.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f'enzymes_preanalysis_{func_type}.png'), dpi=150)
        plt.close()
    
    return OptimalParams(func_type, top_bins[0], optimal_range, p99, top_bins[0]), top_bins


def run_sampled_preanalysis(graphs=None, sample_size=PREANALYSIS_SAMPLE_SIZE, use_cache=True, force_recompute=False, dataset_name='enzymes', top_n_bins=3):
    print(f"\n{'='*80}\nPRE-ANALYSIS FOR {dataset_name.upper()}\n{'='*80}")
    
    if use_cache and not force_recompute:
        cached = load_preanalysis_from_cache(dataset_name, sample_size, MAX_NODES_FOR_PREANALYSIS)
        if cached:
            print("✅ Loaded from cache!")
            for ft, p in cached[0].items():
                print(f"   {ft.upper()}: range={p.range_val:.2f}, bins={cached[1].get(ft, [p.bins])}")
            return cached
        if graphs is None:
            return load_default_params()
    
    if graphs is None:
        raise ValueError("Graphs required")
    
    optimal_params, recommended_bins = {}, {}
    
    for func_type in ['harmonic', 'polynomial']:
        distances, node_counts = compute_spectral_distances_sampled(graphs, func_type, sample_size)
        params, top_bins = determine_optimal_params_and_bins(distances, node_counts, func_type, top_n_bins=top_n_bins)
        optimal_params[func_type] = params
        recommended_bins[func_type] = top_bins
        del distances
        gc.collect()
    
    save_preanalysis_to_cache(optimal_params, recommended_bins, dataset_name, sample_size, MAX_NODES_FOR_PREANALYSIS)
    return optimal_params, recommended_bins
