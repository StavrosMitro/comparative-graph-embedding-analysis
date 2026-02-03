"""
Canonical ChebyshevFGSD and HybridFGSD implementations.
All comparison scripts should import from this module.
"""
import numpy as np
import networkx as nx
from karateclub.estimator import Estimator


class ChebyshevFGSD(Estimator):
    """Fast FGSD using Chebyshev polynomial approximation."""
    
    def __init__(self, hist_bins=200, hist_range=20, order=50, func_type='harmonic', seed=42):
        self.hist_bins = hist_bins
        self.hist_range = (0, hist_range)
        self.order = order
        self.func_type = func_type
        self.seed = seed

    def _compute_chebyshev_coeffs(self, lambda_max):
        k = self.order
        coeffs = np.zeros(k + 1)
        theta = np.pi * (np.arange(k + 1) + 0.5) / (k + 1)
        y_nodes = np.cos(theta)
        
        # Map [-1, 1] to [0, lambda_max]
        x_vals = (y_nodes + 1) * lambda_max / 2.0
        
        if self.func_type == 'harmonic':
            # TUNED: Epsilon 1e-5 and formula x/(x^2+e)
            # This ensures f(0)=0 and f(x)~1/x for x>0
            eps = 1e-5  
            f_vals = x_vals / (x_vals**2 + eps)
            
        elif self.func_type == 'biharmonic':
            # Target: 1/x^2 (Pseudoinverse)
            eps = 1e-5
            f_vals = x_vals**2 / (x_vals**4 + eps)
            
        elif self.func_type == 'polynomial':
            f_vals = x_vals ** 2
            
        else:
            raise ValueError(f"Unknown func_type: {self.func_type}")
        
        for j in range(k + 1):
            term = f_vals * np.cos(j * theta)
            coeffs[j] = (2.0 / (k + 1)) * np.sum(term)
        return coeffs

    def _apply_chebyshev_filter(self, L, X, coeffs, lambda_max):
        a = 2.0 / lambda_max
        t0 = X
        y = 0.5 * coeffs[0] * t0
        t1 = a * (L @ X) - X
        y += coeffs[1] * t1
        
        for k in range(2, len(coeffs)):
            L_t1 = L @ t1
            L_tilde_t1 = a * L_t1 - t1
            t2 = 2.0 * L_tilde_t1 - t0
            y += coeffs[k] * t2
            t0 = t1
            t1 = t2
        return y

    def _calculate_fgsd(self, graph):
        # Sparse Normalized Laplacian
        L_sparse = nx.normalized_laplacian_matrix(graph).astype(float)
        N = L_sparse.shape[0]

        # Fixed Lambda Max (Safe for Normalized Laplacian)
        lambda_max = 2.0
            
        coeffs = self._compute_chebyshev_coeffs(lambda_max)
        I = np.eye(N)
        
        # Dense Calculation
        fL = self._apply_chebyshev_filter(L_sparse, I, coeffs, lambda_max)
        
        # Distance & Histogram
        diag_fL = np.diag(fL)
        S = diag_fL[:, None] + diag_fL[None, :] - 2 * fL
        
        hist, _ = np.histogram(S.flatten(), bins=self.hist_bins, range=self.hist_range)
        return hist

    def calculate(self, graph):
        """For backward compatibility with benchmark code."""
        return self._calculate_fgsd(graph)

    def fit(self, graphs):
        self._set_seed()
        self._embedding = [self._calculate_fgsd(graph) for graph in graphs]

    def get_embedding(self):
        return np.array(self._embedding)


class HybridFGSD(Estimator):
    """Exact FGSD using eigendecomposition (O(N^3))."""
    
    def __init__(self, hist_bins=200, hist_range=20, func_type='harmonic', seed=42):
        self.hist_bins = hist_bins
        self.hist_range = (0, hist_range)
        self.func_type = func_type
        self.seed = seed

    def _calculate_fgsd(self, graph):
        # Dense computation - expensive!
        L = np.asarray(nx.normalized_laplacian_matrix(graph).todense(), dtype=float)
        w, v = np.linalg.eigh(L)  # O(N^3) bottleneck
        
        if self.func_type == 'harmonic':
            # Pseudoinverse: 0 where w=0, 1/w elsewhere
            func_w = np.where(w > 1e-9, 1.0 / w, 0)
        elif self.func_type == 'biharmonic':
            # Pseudoinverse 1/x^2
            func_w = np.where(w > 1e-9, 1.0 / (w**2), 0)
        elif self.func_type == 'polynomial':
            func_w = w ** 2
        else:
            raise ValueError(f"Unknown func_type: {self.func_type}")
            
        fL = v @ np.diag(func_w) @ v.T
        ones = np.ones(L.shape[0])
        S = np.outer(np.diag(fL), ones) + np.outer(ones, np.diag(fL)) - 2 * fL
        
        hist, _ = np.histogram(S.flatten(), bins=self.hist_bins, range=self.hist_range)
        return hist

    def calculate(self, graph):
        """For backward compatibility with benchmark code."""
        return self._calculate_fgsd(graph)

    def fit(self, graphs):
        self._set_seed()
        self._embedding = []
        for i, graph in enumerate(graphs):
            try:
                self._embedding.append(self._calculate_fgsd(graph))
            except Exception as e:
                print(f"\n[Warning] Exact method failed on graph {i} (Size: {len(graph)} nodes). Error: {e}")
                self._embedding.append(np.zeros(self.hist_bins))

    def get_embedding(self):
        return np.array(self._embedding)
