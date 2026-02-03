import numpy as np
import time
import sys
import os
from tqdm import tqdm

# --- Import Loader (with fallback) ---
try:
    from fast_loader import ensure_dataset_ready, graph_generator
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from fast_loader import ensure_dataset_ready, graph_generator

# Import canonical Chebyshev implementations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chebyshev_fgsd import ChebyshevFGSD, HybridFGSD

# ==========================================
# Main Benchmark Script
# ==========================================
if __name__ == "__main__":
    print("="*100)
    print("COMPLETE BENCHMARK: Polynomial vs Harmonic vs Biharmonic")
    print("="*100)
    
    ensure_dataset_ready()
    
    # Initialize Models
    models = {
        'poly': (ChebyshevFGSD(func_type='polynomial', order=50), HybridFGSD(func_type='polynomial')),
        'harm': (ChebyshevFGSD(func_type='harmonic', order=50), HybridFGSD(func_type='harmonic')),
        'biharm': (ChebyshevFGSD(func_type='biharmonic', order=50), HybridFGSD(func_type='biharmonic'))
    }
    
    # Results Storage
    results = {
        'poly': {'0-400': [], '400-800': [], '800-1500': [], '1500+': []},
        'harm': {'0-400': [], '400-800': [], '800-1500': [], '1500+': []},
        'biharm': {'0-400': [], '400-800': [], '800-1500': [], '1500+': []}
    }
    
    # Safety Threshold for Exact Method
    MAX_EXACT_NODES = 1000 
    
    print("\nStarting Single-Pass processing...")
    print(f"NOTE: Exact method skipped for graphs > {MAX_EXACT_NODES} nodes.")
    
    gen = graph_generator()
    total_graphs = 11929
    
    # Process Loop
    for graph, _, gid in tqdm(gen, total=total_graphs):
        N = graph.number_of_nodes()
        
        # Determine Bucket
        if N < 400: bucket = '0-400'
        elif N < 800: bucket = '400-800'
        elif N < 1500: bucket = '800-1500'
        else: bucket = '1500+'
        
        # Run for all 3 types
        for m_key in ['poly', 'harm', 'biharm']:
            cheb_model, exact_model = models[m_key]
            
            # 1. Chebyshev
            try:
                t0 = time.time()
                cheb_model.calculate(graph)
                dt_cheb = time.time() - t0
            except: dt_cheb = np.nan
            
            # 2. Exact (Conditional)
            dt_exact = np.nan
            if N <= MAX_EXACT_NODES:
                try:
                    t0 = time.time()
                    exact_model.calculate(graph)
                    dt_exact = time.time() - t0
                except: pass
                
            results[m_key][bucket].append((dt_cheb, dt_exact))

    # ==========================================
    # REPORT GENERATION
    # ==========================================
    
    def print_table(title, results_dict):
        print("\n" + "-"*90)
        print(f"RESULTS FOR: {title}")
        print("-" * 90)
        print(f"{'Graph Size':<15} | {'Count':<8} | {'Avg Cheb (s)':<15} | {'Avg Exact (s)':<15} | {'Speedup':<10}")
        print("-" * 90)
        
        for key, times in results_dict.items():
            if not times: continue
            
            valid_pairs = [t for t in times if not np.isnan(t[1])]
            count = len(times)
            
            # Avg Chebyshev (Over ALL graphs)
            all_cheb_times = [t[0] for t in times if not np.isnan(t[0])]
            avg_cheb_all = np.mean(all_cheb_times) if all_cheb_times else 0
            
            if valid_pairs:
                avg_cheb_comp = np.mean([t[0] for t in valid_pairs])
                avg_exact_comp = np.mean([t[1] for t in valid_pairs])
                speedup = avg_exact_comp / avg_cheb_comp
                speedup_str = f"{speedup:.2f}x"
                exact_str = f"{avg_exact_comp:.4f}"
            else:
                exact_str = "Skipped"
                speedup_str = "Inf"
            
            print(f"{key:<15} | {count:<8} | {avg_cheb_all:.4f}          | {exact_str:<15} | {speedup_str:<10}")

    print_table("POLYNOMIAL (x^2)", results['poly'])
    print_table("HARMONIC (1/x)", results['harm'])
    print_table("BIHARMONIC (1/x^2)", results['biharm'])
    
    print("\nDone.")