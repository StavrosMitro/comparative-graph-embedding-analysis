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
# Main Execution for IMDB-MULTI
# ===========================================================
if __name__ == "__main__":
    print("="*80)
    print("Compare HybridFGSD vs ChebyshevFGSD Embeddings (IMDB-MULTI)")
    print("="*80)

    print("Ensuring dataset is ready...")
    ensure_dataset_ready()
    
    print("Loading IMDB-MULTI graphs...")
    graphs, y = load_all_graphs()
    print(f"Loaded {len(graphs)} graphs successfully.\n")

    # --- TEST 1: POLYNOMIAL ---
    print("[TEST 1] POLYNOMIAL (x^2)")
    
    # Exact
    t0 = time.time()
    hybrid_poly = HybridFGSD(func_type='polynomial')
    hybrid_poly.fit(graphs)
    emb_exact = hybrid_poly.get_embedding()
    print(f"  Exact Time:  {time.time()-t0:.2f}s | Shape: {emb_exact.shape}")

    # Approx (Order 20 is enough)
    t0 = time.time()
    cheb_poly = ChebyshevFGSD(func_type='polynomial', order=20)
    cheb_poly.fit(graphs)
    emb_approx = cheb_poly.get_embedding()
    print(f"  Approx Time: {time.time()-t0:.2f}s | Shape: {emb_approx.shape}")

    # Similarity
    sims = [cosine_similarity(emb_exact[i].reshape(1,-1), emb_approx[i].reshape(1,-1))[0][0] for i in range(len(graphs))]
    print(f"  >>> Mean Cosine Sim: {np.mean(sims):.5f}")

    # --- TEST 2: HARMONIC ---
    print("\n[TEST 2] HARMONIC (1/x Approx using x/(x^2+1e-5))")
    
    # Exact
    t0 = time.time()
    hybrid_harm = HybridFGSD(func_type='harmonic')
    hybrid_harm.fit(graphs)
    emb_exact = hybrid_harm.get_embedding()
    print(f"  Exact Time:  {time.time()-t0:.2f}s")

    # Approx (Order 100 for precision near 0)
    t0 = time.time()
    cheb_harm = ChebyshevFGSD(func_type='harmonic', order=50)
    cheb_harm.fit(graphs)
    emb_approx = cheb_harm.get_embedding()
    print(f"  Approx Time: {time.time()-t0:.2f}s")

    # Similarity
    sims = [cosine_similarity(emb_exact[i].reshape(1,-1), emb_approx[i].reshape(1,-1))[0][0] for i in range(len(graphs))]
    print(f"  >>> Mean Cosine Sim: {np.mean(sims):.5f}")
    
    print("\nDone.")