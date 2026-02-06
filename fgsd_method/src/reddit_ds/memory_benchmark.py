"""
Memory Benchmark for FGSD Embedding Generation on REDDIT-MULTI-12K.

Creates harmonic (bins=500) and polynomial (bins=200) embeddings batch-wise,
measuring memory usage throughout the process using tracemalloc.

Usage:
    python -m reddit_ds.memory_benchmark
"""

import os
import sys
import gc
import time
import pickle
import tracemalloc
import linecache
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import psutil
from tqdm import tqdm

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from reddit_ds.config import DATASET_DIR, BATCH_SIZE, RESULTS_DIR, CACHE_DIR
from reddit_ds.data_loader import ensure_dataset_ready, load_metadata, iter_graph_batches
from reddit_ds.preanalysis import run_sampled_preanalysis


# =============================================================================
# TRACEMALLOC UTILITIES
# =============================================================================
def display_top_allocators(snapshot, key_type='lineno', limit=10):
    """Display top memory allocators from tracemalloc snapshot."""
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)
    
    print(f"\n  Top {limit} memory allocators:")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(f"    #{index}: {frame.filename}:{frame.lineno}: {stat.size / 1024 / 1024:.2f} MB")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"         {line[:80]}")
    
    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print(f"    ... {len(other)} other: {size / 1024 / 1024:.2f} MB")
    
    total = sum(stat.size for stat in top_stats)
    print(f"    Total allocated: {total / 1024 / 1024:.2f} MB")
    return total


def compare_snapshots(snapshot1, snapshot2, key_type='lineno', limit=10):
    """Compare two tracemalloc snapshots to find memory differences."""
    top_stats = snapshot2.compare_to(snapshot1, key_type)
    
    print(f"\n  Memory diff (top {limit} changes):")
    for index, stat in enumerate(top_stats[:limit], 1):
        size_diff = stat.size_diff / 1024 / 1024
        if abs(size_diff) > 0.1:  # Only show > 0.1 MB changes
            frame = stat.traceback[0]
            print(f"    #{index}: {frame.filename}:{frame.lineno}: {size_diff:+.2f} MB (total: {stat.size / 1024 / 1024:.2f} MB)")


# =============================================================================
# MEMORY TRACKING WITH TRACEMALLOC
# =============================================================================
@dataclass
class MemorySnapshot:
    """Single memory measurement with tracemalloc."""
    timestamp: float
    rss_mb: float               # Resident Set Size (actual RAM used)
    vms_mb: float               # Virtual Memory Size
    tracemalloc_current_mb: float  # Current Python allocations
    tracemalloc_peak_mb: float     # Peak Python allocations since start
    batch_idx: int
    phase: str
    snapshot: Any = field(default=None, repr=False)  # tracemalloc snapshot object


class MemoryTracker:
    """Track memory usage over time using tracemalloc."""
    
    def __init__(self, nframe: int = 25):
        """
        Initialize tracker.
        
        Args:
            nframe: Number of frames to trace (more = detailed but slower)
        """
        self.nframe = nframe
        self.snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process()
        self.start_time = None
        self.initial_snapshot = None
    
    def start(self):
        """Start tracking with tracemalloc."""
        gc.collect()
        tracemalloc.start(self.nframe)
        self.start_time = time.time()
        self.initial_snapshot = tracemalloc.take_snapshot()
        self._take_snapshot(batch_idx=-1, phase='init')
    
    def _take_snapshot(self, batch_idx: int, phase: str, take_tracemalloc_snapshot: bool = True) -> MemorySnapshot:
        """Internal snapshot taking."""
        mem_info = self.process.memory_info()
        current, peak = tracemalloc.get_traced_memory()
        
        snap = MemorySnapshot(
            timestamp=time.time() - (self.start_time or time.time()),
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            tracemalloc_current_mb=current / 1024 / 1024,
            tracemalloc_peak_mb=peak / 1024 / 1024,
            batch_idx=batch_idx,
            phase=phase,
            snapshot=tracemalloc.take_snapshot() if take_tracemalloc_snapshot else None
        )
        self.snapshots.append(snap)
        return snap
    
    def snapshot(self, batch_idx: int = -1, phase: str = 'unknown', detailed: bool = False) -> MemorySnapshot:
        """Take a memory snapshot."""
        return self._take_snapshot(batch_idx, phase, take_tracemalloc_snapshot=detailed)
    
    def get_peak_rss(self) -> float:
        """Get peak RSS memory usage in MB."""
        if not self.snapshots:
            return 0.0
        return max(s.rss_mb for s in self.snapshots)
    
    def get_peak_tracemalloc(self) -> float:
        """Get peak tracemalloc memory in MB."""
        _, peak = tracemalloc.get_traced_memory()
        return peak / 1024 / 1024
    
    def get_current_tracemalloc(self) -> float:
        """Get current tracemalloc memory in MB."""
        current, _ = tracemalloc.get_traced_memory()
        return current / 1024 / 1024
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.snapshots:
            return {}
        
        rss_values = [s.rss_mb for s in self.snapshots]
        tracemalloc_values = [s.tracemalloc_current_mb for s in self.snapshots]
        
        current, peak = tracemalloc.get_traced_memory()
        
        return {
            'peak_rss_mb': max(rss_values),
            'min_rss_mb': min(rss_values),
            'avg_rss_mb': np.mean(rss_values),
            'final_rss_mb': rss_values[-1] if rss_values else 0,
            'peak_tracemalloc_mb': peak / 1024 / 1024,
            'current_tracemalloc_mb': current / 1024 / 1024,
            'max_snapshot_tracemalloc_mb': max(tracemalloc_values) if tracemalloc_values else 0,
            'total_snapshots': len(self.snapshots),
            'duration_sec': self.snapshots[-1].timestamp if self.snapshots else 0
        }
    
    def stop(self):
        """Stop tracking."""
        self._take_snapshot(batch_idx=-1, phase='final', take_tracemalloc_snapshot=True)
    
    def print_report(self, title: str = "Memory Report", show_top_allocators: bool = True):
        """Print detailed memory report using tracemalloc."""
        summary = self.get_summary()
        
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
        
        print(f"\n  === RSS Memory (Process) ===")
        print(f"  Peak RSS:              {summary['peak_rss_mb']:.2f} MB")
        print(f"  Min RSS:               {summary['min_rss_mb']:.2f} MB")
        print(f"  Avg RSS:               {summary['avg_rss_mb']:.2f} MB")
        print(f"  Final RSS:             {summary['final_rss_mb']:.2f} MB")
        
        print(f"\n  === Tracemalloc (Python Allocations) ===")
        print(f"  Peak Traced:           {summary['peak_tracemalloc_mb']:.2f} MB")
        print(f"  Current Traced:        {summary['current_tracemalloc_mb']:.2f} MB")
        print(f"  Max Snapshot Traced:   {summary['max_snapshot_tracemalloc_mb']:.2f} MB")
        
        print(f"\n  === Timing ===")
        print(f"  Duration:              {summary['duration_sec']:.2f} sec")
        print(f"  Snapshots:             {summary['total_snapshots']}")
        
        # Per-phase breakdown
        phases = {}
        for s in self.snapshots:
            if s.phase not in phases:
                phases[s.phase] = {'rss': [], 'tracemalloc': []}
            phases[s.phase]['rss'].append(s.rss_mb)
            phases[s.phase]['tracemalloc'].append(s.tracemalloc_current_mb)
        
        print(f"\n  === Per-Phase Peak Memory ===")
        print(f"  {'Phase':<20} {'Peak RSS (MB)':<15} {'Peak Traced (MB)':<15}")
        print(f"  {'-'*50}")
        for phase, values in phases.items():
            print(f"  {phase:<20} {max(values['rss']):<15.2f} {max(values['tracemalloc']):<15.2f}")
        
        # Show top allocators from final snapshot
        if show_top_allocators and self.snapshots:
            final_snap = self.snapshots[-1]
            if final_snap.snapshot:
                print(f"\n  === Top Memory Allocators (Final State) ===")
                display_top_allocators(final_snap.snapshot, limit=10)
            
            # Compare with initial
            if self.initial_snapshot and final_snap.snapshot:
                print(f"\n  === Memory Growth (Final vs Initial) ===")
                compare_snapshots(self.initial_snapshot, final_snap.snapshot, limit=5)


# =============================================================================
# EMBEDDING GENERATION WITH TRACEMALLOC
# =============================================================================
def generate_embedding_batchwise_with_memory(
    func_type: str,
    bins: int,
    range_val: float,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
    detailed_snapshots: bool = False
) -> Tuple[np.ndarray, MemoryTracker, float]:
    """
    Generate embeddings batch-wise with tracemalloc tracking.
    
    Args:
        detailed_snapshots: If True, take full tracemalloc snapshots per batch (slower)
    
    Returns:
        Tuple of (embedding_matrix, memory_tracker, generation_time)
    """
    print(f"\n{'='*70}")
    print(f"Generating: {func_type.upper()} (bins={bins}, range={range_val:.2f})")
    print(f"Batch size: {batch_size}")
    print(f"Tracemalloc: ENABLED (detailed={detailed_snapshots})")
    print(f"{'='*70}")
    
    # Initialize memory tracker
    tracker = MemoryTracker(nframe=25)
    tracker.start()
    
    # Create model
    model = FlexibleFGSD(
        hist_bins=bins,
        hist_range=range_val,
        func_type=func_type,
        seed=seed
    )
    
    tracker.snapshot(-1, 'model_created')
    
    # Get total batches
    records = load_metadata(DATASET_DIR)
    total_batches = (len(records) + batch_size - 1) // batch_size
    total_graphs = len(records)
    
    print(f"Total graphs: {total_graphs}, Batches: {total_batches}")
    
    # Collect embeddings
    embeddings_list = []
    start_time = time.time()
    
    for batch_idx, (graphs, labels, gids) in enumerate(tqdm(
        iter_graph_batches(DATASET_DIR, batch_size),
        total=total_batches,
        desc=f"  {func_type}"
    )):
        # Take snapshot before processing (lightweight unless detailed)
        tracker.snapshot(batch_idx, 'before_batch', detailed=detailed_snapshots and batch_idx % 5 == 0)
        
        # Generate embeddings for batch
        model.fit(graphs)
        batch_emb = model.get_embedding()
        embeddings_list.append(batch_emb)
        
        # Take snapshot after processing
        tracker.snapshot(batch_idx, 'after_batch', detailed=detailed_snapshots and batch_idx % 5 == 0)
        
        # Free batch graphs
        del graphs
        
        # Periodic GC and status
        if batch_idx % 5 == 0:
            gc.collect()
            current_mb = tracker.get_current_tracemalloc()
            peak_mb = tracker.get_peak_tracemalloc()
            # tqdm.write(f"    Batch {batch_idx}: current={current_mb:.1f}MB, peak={peak_mb:.1f}MB")
    
    generation_time = time.time() - start_time
    
    # Stack all embeddings
    tracker.snapshot(-1, 'before_stack', detailed=True)
    X_all = np.vstack(embeddings_list)
    tracker.snapshot(-1, 'after_stack', detailed=True)
    
    # Free list
    del embeddings_list
    gc.collect()
    
    tracker.snapshot(-1, 'after_cleanup', detailed=True)
    tracker.stop()
    
    print(f"\n  -> Shape: {X_all.shape}")
    print(f"  -> Time: {generation_time:.2f}s")
    print(f"  -> Embedding size: {X_all.nbytes / 1024 / 1024:.2f} MB")
    print(f"  -> Peak tracemalloc: {tracker.get_peak_tracemalloc():.2f} MB")
    
    return X_all, tracker, generation_time


def save_embedding(X: np.ndarray, func_type: str, bins: int, range_val: float, output_dir: str = None):
    """Save embedding to disk."""
    if output_dir is None:
        output_dir = os.path.join(CACHE_DIR, 'embeddings')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"reddit_{func_type}_bins{bins}_range{range_val:.2f}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    print(f"  Saving to: {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump({
            'embedding': X,
            'func_type': func_type,
            'bins': bins,
            'range': range_val,
            'shape': X.shape
        }, f)
    
    file_size = os.path.getsize(filepath) / 1024 / 1024
    print(f"  File size: {file_size:.2f} MB")
    
    return filepath


# =============================================================================
# MAIN BENCHMARK
# =============================================================================
def run_memory_benchmark(detailed: bool = False):
    """
    Run memory benchmark for embedding generation using tracemalloc.
    
    Args:
        detailed: If True, take detailed tracemalloc snapshots (slower but more info)
    
    Creates:
    - harmonic: bins=500, range from preanalysis
    - polynomial: bins=200, range from preanalysis
    """
    print("="*80)
    print("MEMORY BENCHMARK: FGSD Embedding Generation (TRACEMALLOC)")
    print("Reddit-Multi-12K Dataset")
    print("="*80)
    
    # Ensure dataset is ready
    ensure_dataset_ready()
    
    # Get optimal ranges from preanalysis (cached)
    print("\nLoading preanalysis parameters...")
    optimal_params, _ = run_sampled_preanalysis(
        graphs=None, use_cache=True, dataset_name='reddit'
    )
    
    # Configurations to benchmark
    configs = [
        {
            'name': 'harmonic_500',
            'func_type': 'harmonic',
            'bins': 500,
            'range': round(optimal_params['harmonic'].range_val, 2)
        },
        {
            'name': 'polynomial_200',
            'func_type': 'polynomial',
            'bins': 200,
            'range': round(optimal_params['polynomial'].range_val, 2)
        }
    ]
    
    print("\nConfigurations to benchmark:")
    for cfg in configs:
        print(f"  - {cfg['name']}: bins={cfg['bins']}, range={cfg['range']}")
    
    # Run benchmarks
    results = []
    saved_files = []
    
    for cfg in configs:
        print(f"\n{'#'*80}")
        print(f"# BENCHMARK: {cfg['name']}")
        print(f"{'#'*80}")
        
        # Initial memory state
        gc.collect()
        initial_mem = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"\nInitial memory: {initial_mem:.2f} MB")
        
        # Generate with memory tracking
        X, tracker, gen_time = generate_embedding_batchwise_with_memory(
            func_type=cfg['func_type'],
            bins=cfg['bins'],
            range_val=cfg['range'],
            batch_size=BATCH_SIZE,
            detailed_snapshots=detailed
        )
        
        # Print memory report with tracemalloc details
        tracker.print_report(f"Memory Report: {cfg['name']}", show_top_allocators=True)
        
        # Save embedding
        filepath = save_embedding(X, cfg['func_type'], cfg['bins'], cfg['range'])
        saved_files.append(filepath)
        
        # Store results
        summary = tracker.get_summary()
        result = {
            'name': cfg['name'],
            'func_type': cfg['func_type'],
            'bins': cfg['bins'],
            'range': cfg['range'],
            'shape': X.shape,
            'embedding_size_mb': X.nbytes / 1024 / 1024,
            'generation_time_sec': gen_time,
            **summary
        }
        results.append(result)
        
        # Free embedding to prepare for next
        del X
        gc.collect()
        
        # Stop tracemalloc between configs
        tracemalloc.stop()
        
        final_mem = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"\nFinal memory after cleanup: {final_mem:.2f} MB")
    
    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print("\n" + "="*100)
    print("FINAL SUMMARY (TRACEMALLOC)")
    print("="*100)
    
    print(f"\n{'Config':<20} {'Shape':<15} {'Embed MB':<12} {'Peak RSS MB':<14} {'Peak Traced MB':<16} {'Time (s)':<10}")
    print("-"*100)
    
    for r in results:
        print(f"{r['name']:<20} {str(r['shape']):<15} {r['embedding_size_mb']:<12.2f} "
              f"{r['peak_rss_mb']:<14.2f} {r['peak_tracemalloc_mb']:<16.2f} {r['generation_time_sec']:<10.2f}")
    
    # Save summary to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    summary_path = os.path.join(RESULTS_DIR, 'reddit_memory_benchmark_tracemalloc.csv')
    df.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved to: {summary_path}")
    
    print(f"\n✅ Embeddings saved to:")
    for fp in saved_files:
        print(f"   - {fp}")
    
    # Combined hybrid creation test
    print("\n" + "="*80)
    print("HYBRID CREATION TEST (concatenation) - TRACEMALLOC")
    print("="*80)
    
    # Load saved embeddings
    print("\nLoading saved embeddings...")
    embeddings = {}
    for fp in saved_files:
        with open(fp, 'rb') as f:
            data = pickle.load(f)
            embeddings[data['func_type']] = data['embedding']
            print(f"  Loaded {data['func_type']}: {data['embedding'].shape}")
    
    # Track hybrid creation with tracemalloc
    gc.collect()
    tracemalloc.start(25)
    snapshot_before = tracemalloc.take_snapshot()
    
    X_harmonic = embeddings['harmonic']
    X_polynomial = embeddings['polynomial']
    
    X_hybrid = np.hstack([X_harmonic, X_polynomial])
    
    snapshot_after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nHybrid shape: {X_hybrid.shape}")
    print(f"Hybrid size: {X_hybrid.nbytes / 1024 / 1024:.2f} MB")
    print(f"Peak tracemalloc during concat: {peak / 1024 / 1024:.2f} MB")
    
    print("\n  Memory growth during concatenation:")
    compare_snapshots(snapshot_before, snapshot_after, limit=5)
    
    # Save hybrid
    hybrid_path = save_embedding(X_hybrid, 'hybrid', 700, 0.0)
    print(f"\n✅ Hybrid saved to: {hybrid_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Memory Benchmark with Tracemalloc')
    parser.add_argument('--detailed', action='store_true', help='Take detailed tracemalloc snapshots per batch')
    args = parser.parse_args()
    
    run_memory_benchmark(detailed=args.detailed)
