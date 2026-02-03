import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys
import os

# Import with fallback
try:
    from data_loader import load_all_graphs, ensure_dataset_ready
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_all_graphs, ensure_dataset_ready

# Import canonical Chebyshev implementations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chebyshev_fgsd import ChebyshevFGSD, HybridFGSD

# ===========================================================
# Main Execution
# ===========================================================
if __name__ == "__main__":
    print("="*80)
    print("FULL DATASET COMPARISON: REDDIT-MULTI-12K")
    print("Objective: Prove Chebyshev efficiency on large dataset.")
    print("="*80)

    print("\n[Step 0] Loading Data...")
    ensure_dataset_ready()
    graphs, y = load_all_graphs()
    print(f"Loaded {len(graphs)} graphs.")
    
    # -------------------------------------------------------
    # TEST 1: HARMONIC (Most computationally demanding)
    # -------------------------------------------------------
    print("\n" + "-"*60)
    print("TEST: HARMONIC FUNCTION (1/x)")
    print("-"*60)
    
    # --- 1. CHEBYSHEV (Run First) ---
    print("\n>>> Running CHEBYSHEV (Approx)...")
    start_time = time.time()
    
    cheb_model = ChebyshevFGSD(func_type='harmonic', order=50) 
    cheb_model.fit(graphs)
    emb_approx = cheb_model.get_embedding()
    
    cheb_time = time.time() - start_time
    print(f"DONE. Chebyshev Time: {cheb_time:.4f} seconds")

    # --- 2. EXACT / HYBRID (Run Second) ---
    print("\n>>> Running EXACT (O(N^3))...")
    start_time = time.time()
    
    exact_model = HybridFGSD(func_type='harmonic')
    exact_model.fit(graphs)
    emb_exact = exact_model.get_embedding()
    
    exact_time = time.time() - start_time
    print(f"DONE. Exact Time:     {exact_time:.4f} seconds")

    # --- 3. COMPARISON RESULTS ---
    print("\n" + "="*40)
    print("       FINAL RESULTS       ")
    print("="*40)
    print(f"Dataset Size:     {len(graphs)} graphs")
    print(f"Chebyshev Time:   {cheb_time:.2f} s")
    print(f"Exact Time:       {exact_time:.2f} s")
    print(f"Speedup Factor:   {exact_time / cheb_time:.2f}x FASTER")
    
    # Calculate similarity only for graphs where Exact didn't fail
    valid_indices = [i for i in range(len(graphs)) if np.any(emb_exact[i])]
    if valid_indices:
        sims = [cosine_similarity(emb_exact[i].reshape(1,-1), emb_approx[i].reshape(1,-1))[0][0] for i in valid_indices]
        print(f"Mean Cosine Sim:  {np.mean(sims):.5f}")
        print(f"Min Cosine Sim:   {np.min(sims):.5f}")
    else:
        print("Could not compute similarity (Exact method failed on all graphs?)")

    print("\nDone.")