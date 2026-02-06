"""
Generate and Store Stability Embeddings for REDDIT-MULTI-12K.

Creates embeddings for:
- Edge Add: 5%, 15%
- Edge Remove: 5%, 15%

Control embeddings already exist in cache/embeddings/ (from memory_benchmark.py).

Configurations:
- harmonic: bins=500, range=14.61
- polynomial: bins=200, range=3.48

All embeddings are generated batch-wise to minimize RAM usage.
For each batch: load graphs -> generate ALL perturbation embeddings -> free graphs.

Usage:
    python -m reddit_ds.generate_stability_embeddings
    python -m reddit_ds.generate_stability_embeddings --batch-size 200
"""

import os
import sys
import gc
import time
import copy
import pickle
import argparse
from typing import List, Dict, Any

import numpy as np
import networkx as nx
from tqdm import tqdm

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from reddit_ds.config import DATASET_DIR, CACHE_DIR
from reddit_ds.data_loader import ensure_dataset_ready, load_metadata, iter_graph_batches


# =============================================================================
# CONFIGURATION - EASY TO CHANGE
# =============================================================================
BATCH_SIZE = 500  # Batch size for processing graphs

# Embedding configurations (must match what's in cache/embeddings/)
EMBEDDING_CONFIGS = [
    {'name': 'harmonic_500_14.61', 'func': 'harmonic', 'bins': 500, 'range': 14.61},
    {'name': 'polynomial_200_3.48', 'func': 'polynomial', 'bins': 200, 'range': 3.48},
]

# Perturbation ratios
ADD_RATIOS = [0.05, 0.15]      # 5%, 15%
REMOVE_RATIOS = [0.05, 0.15]   # 5%, 15%

# Directories
EMBEDDINGS_CACHE_DIR = os.path.join(CACHE_DIR, 'embeddings')
STABILITY_EMBEDDINGS_DIR = os.path.join(CACHE_DIR, 'stability_embeddings')


# =============================================================================
# PERTURBATION FUNCTIONS
# =============================================================================
def perturb_graph_add_edges(graph: nx.Graph, ratio: float, seed: int = 42) -> nx.Graph:
    """Perturb a graph by adding edges."""
    np.random.seed(seed)
    G = copy.deepcopy(graph)
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_nodes < 2:
        return G
    
    n_add = max(1, int(n_edges * ratio))
    nodes = list(G.nodes())
    added = 0
    max_attempts = n_add * 20
    attempts = 0
    
    while added < n_add and attempts < max_attempts:
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            added += 1
        attempts += 1
    
    return G


def perturb_graph_remove_edges(graph: nx.Graph, ratio: float, seed: int = 42) -> nx.Graph:
    """Perturb a graph by removing edges."""
    np.random.seed(seed)
    G = copy.deepcopy(graph)
    
    n_edges = G.number_of_edges()
    if n_edges < 1:
        return G
    
    n_remove = max(1, int(n_edges * ratio))
    edges = list(G.edges())
    
    if n_remove > 0 and len(edges) > 0:
        remove_indices = np.random.choice(len(edges), min(n_remove, len(edges)), replace=False)
        for idx in remove_indices:
            G.remove_edge(*edges[idx])
    
    return G


def perturb_batch(graphs: List[nx.Graph], mode: str, ratio: float, base_seed: int = 42) -> List[nx.Graph]:
    """Perturb a batch of graphs."""
    perturbed = []
    for i, g in enumerate(graphs):
        if mode == 'add':
            perturbed.append(perturb_graph_add_edges(g, ratio, seed=base_seed + i))
        elif mode == 'remove':
            perturbed.append(perturb_graph_remove_edges(g, ratio, seed=base_seed + i))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return perturbed


# =============================================================================
# FILE I/O
# =============================================================================
def get_output_path(config_name: str, perturbation_key: str) -> str:
    """Get output file path for an embedding."""
    os.makedirs(STABILITY_EMBEDDINGS_DIR, exist_ok=True)
    return os.path.join(STABILITY_EMBEDDINGS_DIR, f'reddit_{config_name}_{perturbation_key}.pkl')


def save_embedding(X: np.ndarray, config: Dict, perturbation_key: str, metadata: Dict = None) -> str:
    """Save embedding to disk."""
    path = get_output_path(config['name'], perturbation_key)
    
    data = {
        'embedding': X,
        'config': config,
        'perturbation': perturbation_key,
        'shape': X.shape,
        **(metadata or {})
    }
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"      Saved: {perturbation_key} -> {path} ({size_mb:.1f} MB)")
    return path


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================
def generate_all_perturbation_embeddings_batchwise_multi(
    configs: List[Dict],
    batch_size: int = BATCH_SIZE,
    seed: int = 42
) -> Dict[str, Dict[str, str]]:
    """
    Generate ALL perturbation embeddings for MULTIPLE configs in ONE pass.

    - Load batch of graphs ONCE
    - Generate ALL perturbation embeddings for ALL configs in that batch
    - Free batch, keep only embeddings
    - Repeat for all batches

    This minimizes disk I/O and ensures graphs are only loaded once.

    Returns:
        Dict mapping config_name -> (perturbation_key -> saved file path)
    """
    phase_start = time.time()

    print(f"\n{'='*80}")
    print("GENERATING PERTURBATION EMBEDDINGS (MULTI-CONFIG)")
    print(f"Configs: {[c['name'] for c in configs]}")
    print(f"batch_size={batch_size}")
    print(f"{'='*80}")

    # Build list of all perturbations to generate
    perturbations = []
    for ratio in ADD_RATIOS:
        perturbations.append({'mode': 'add', 'ratio': ratio, 'key': f'add_{int(ratio*100)}'})
    for ratio in REMOVE_RATIOS:
        perturbations.append({'mode': 'remove', 'ratio': ratio, 'key': f'remove_{int(ratio*100)}'})

    print(f"\n  Perturbations to generate: {len(perturbations)}")
    for p in perturbations:
        print(f"    - {p['key']} ({p['mode']} {p['ratio']*100:.0f}%)")

    # Initialize collectors per config and perturbation
    embeddings_collectors = {
        cfg['name']: {p['key']: [] for p in perturbations}
        for cfg in configs
    }

    # Create FGSD models (reused for all embeddings)
    models = {
        cfg['name']: FlexibleFGSD(
            hist_bins=cfg['bins'],
            hist_range=cfg['range'],
            func_type=cfg['func'],
            seed=seed
        )
        for cfg in configs
    }

    # Get dataset info
    records = load_metadata(DATASET_DIR)
    total_graphs = len(records)
    total_batches = (total_graphs + batch_size - 1) // batch_size

    print(f"\n  Dataset: {total_graphs} graphs, {total_batches} batches")
    print(f"  Embeddings per batch: {len(perturbations)} perturbation types x {len(configs)} configs")
    print(
        f"  Total embeddings to generate: {len(perturbations)} x {len(configs)} x {total_graphs} = "
        f"{len(perturbations) * len(configs) * total_graphs}"
    )

    # Total inner work items per batch: perturbations × configs
    n_work_per_batch = len(perturbations) * len(configs)
    config_short = '+'.join(c['func'] for c in configs)

    # Process all batches
    print(f"\n  Processing batches...")
    batch_start_time = time.time()
    batch_idx = 0

    # Outer bar: tracks graphs processed
    pbar = tqdm(
        total=total_graphs,
        desc=f"  📊 Graphs [{config_short}]",
        unit="g",
        bar_format="{l_bar}{bar:30}{r_bar}",
        dynamic_ncols=True,
    )

    for graphs, labels, gids in iter_graph_batches(DATASET_DIR, batch_size):
        actual_batch_size = len(graphs)
        base_seed = seed + batch_idx * 1000

        # Inner bar: tracks perturbation×config work within the batch
        inner_bar = tqdm(
            total=n_work_per_batch,
            desc=f"    batch {batch_idx+1}/{total_batches}",
            unit="emb",
            leave=False,
            bar_format="{l_bar}{bar:20}{r_bar}",
            dynamic_ncols=True,
        )

        # Generate ALL perturbation embeddings for this batch, for ALL configs
        for pert in perturbations:
            perturbed_batch = perturb_batch(graphs, pert['mode'], pert['ratio'], base_seed)

            for cfg in configs:
                cfg_name = cfg['name']
                model = models[cfg_name]
                model.fit(perturbed_batch)
                embeddings_collectors[cfg_name][pert['key']].append(model.get_embedding())
                inner_bar.set_postfix_str(f"{cfg['func']}:{pert['key']}")
                inner_bar.update(1)

            # Free perturbed batch immediately
            del perturbed_batch

        inner_bar.close()

        # Free original batch
        del graphs
        batch_idx += 1

        # Update outer progress bar
        pbar.update(actual_batch_size)
        elapsed = time.time() - batch_start_time
        graphs_done = min(pbar.n, total_graphs)
        rate = graphs_done / elapsed if elapsed > 0 else 0
        pbar.set_postfix({
            'rate': f'{rate:.1f} g/s',
            'batch': f'{batch_idx}/{total_batches}',
        })

        # Periodic garbage collection
        if batch_idx % 5 == 0:
            gc.collect()

    pbar.close()
    gc.collect()
    batch_time = time.time() - batch_start_time

    print(f"\n  ✓ Batch processing complete in {batch_time:.1f}s ({batch_time/60:.1f} min)")
    print(f"    Rate: {total_graphs / batch_time:.1f} graphs/s")

    # Stack and save all embeddings
    print(f"\n  Stacking and saving embeddings...")
    all_saved_paths: Dict[str, Dict[str, str]] = {}

    for cfg in configs:
        cfg_name = cfg['name']
        print(f"\n  Saving config: {cfg_name}")
        saved_paths = {}

        for pert in perturbations:
            key = pert['key']
            X_all = np.vstack(embeddings_collectors[cfg_name][key])

            metadata = {
                'generation_time': time.time() - phase_start,
                'n_graphs': total_graphs,
                'batch_size': batch_size,
                'mode': pert['mode'],
                'ratio': pert['ratio'],
            }

            path = save_embedding(X_all, cfg, key, metadata)
            saved_paths[key] = path

            # Free after saving
            del X_all
            del embeddings_collectors[cfg_name][key]

        all_saved_paths[cfg_name] = saved_paths

    # Final cleanup
    del embeddings_collectors
    gc.collect()

    total_time = time.time() - phase_start
    print(f"\n  ✓ All configs complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    return all_saved_paths


def run_generation(batch_size: int = BATCH_SIZE, configs: List[Dict] = None):
    """Run embedding generation for all configurations in a single pass."""
    if configs is None:
        configs = EMBEDDING_CONFIGS

    total_start = time.time()

    # Calculate total work
    n_perturbations = len(ADD_RATIOS) + len(REMOVE_RATIOS)

    print("="*80)
    print("REDDIT STABILITY EMBEDDINGS GENERATOR")
    print("="*80)

    print(f"\n📋 GENERATION PLAN:")
    print(f"   Configs to process: {len(configs)}")
    print(f"   Perturbations per config: {n_perturbations}")
    print(f"   Total embedding sets: {len(configs) * n_perturbations}")

    print(f"\n📁 CONFIGURATIONS:")
    for i, cfg in enumerate(configs, 1):
        print(f"   [{i}/{len(configs)}] {cfg['name']}: func={cfg['func']}, bins={cfg['bins']}, range={cfg['range']}")

    print(f"\n🔀 PERTURBATION TYPES:")
    print(f"   Add:    {[f'{r*100:.0f}%' for r in ADD_RATIOS]}")
    print(f"   Remove: {[f'{r*100:.0f}%' for r in REMOVE_RATIOS]}")

    print(f"\n⚙️  SETTINGS:")
    print(f"   Batch size: {batch_size}")
    print(f"   Control embeddings: {EMBEDDINGS_CACHE_DIR} (already exist)")
    print(f"   Output directory: {STABILITY_EMBEDDINGS_DIR}")

    print(f"\n{'='*80}")
    print(f"STARTING GENERATION...")
    print(f"{'='*80}")

    # Ensure dataset ready
    ensure_dataset_ready()

    all_saved_paths = generate_all_perturbation_embeddings_batchwise_multi(configs, batch_size)

    total_time = time.time() - total_start

    # Summary
    print("\n" + "="*80)
    print("✅ GENERATION COMPLETE")
    print("="*80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    print(f"\n📦 SAVED EMBEDDINGS:")
    total_size = 0
    for config_name, paths in all_saved_paths.items():
        print(f"\n  {config_name}:")
        for key, path in paths.items():
            size_mb = os.path.getsize(path) / 1024 / 1024
            total_size += size_mb
            print(f"    - {key}: {size_mb:.1f} MB")

    print(f"\n💾 Total storage: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"📂 Output directory: {STABILITY_EMBEDDINGS_DIR}")

    return all_saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Reddit Stability Embeddings')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for processing (default: {BATCH_SIZE})')
    args = parser.parse_args()
    
    run_generation(batch_size=args.batch_size)
