import numpy as np
import networkx as nx
from typing import List
from karateclub.estimator import Estimator

class FlexibleFGSD(Estimator):
    """
    Μια ευέλικτη υλοποίηση της FGSD που επιτρέπει αλλαγή της συνάρτησης f(λ).
    
    Args:
        hist_bins (int): Αριθμός κάδων.
        hist_range (int/float): Το εύρος του ιστογράμματος (0, range).
        func_type (str): 'harmonic' (1/λ), 'biharmonic' (1/λ^2), 'polynomial' (λ^2).
        seed (int): Random seed.
    """
    def __init__(self, hist_bins=200, hist_range=20, func_type='harmonic', seed=42):
        self.hist_bins = hist_bins
        self.hist_range = (0, hist_range)
        self.func_type = func_type
        self.seed = seed

    def _calculate_fgsd(self, graph):
        """Calculate FGSD embedding for a single graph."""
        # 1. Υπολογισμός Normalized Laplacian
        # Χρησιμοποιούμε asarray για να αποφύγουμε προβλήματα με matrix types
        L = np.asarray(nx.normalized_laplacian_matrix(graph).todense())
        
        # w: ιδιοτιμές, v: ιδιοδιανύσματα
        w, v = np.linalg.eigh(L)
        
        if self.func_type == 'harmonic':
            # f(λ) = 1/λ (Global Structure)
            with np.errstate(divide='ignore', invalid='ignore'):
                func_w = np.where(w > 1e-9, 1.0 / w, 0)
            
        elif self.func_type == 'biharmonic':
            # f(λ) = 1/λ² (Global Structure - Stronger)
            with np.errstate(divide='ignore', invalid='ignore'):
                func_w = np.where(w > 1e-9, 1.0 / (w**2), 0)
            
        elif self.func_type == 'polynomial':
            # f(λ) = λ² (Local Structure)
            func_w = w ** 2
            
        else:
            raise ValueError(f"Unknown function type: {self.func_type}. Supported: harmonic, polynomial, biharmonic")
            
        # 4. Ανακατασκευή του πίνακα f(L)
        # f(L) = V * diag(f(λ)) * V^T
        fL = v @ np.diag(func_w) @ v.T
        
        # 5. Υπολογισμός Αποστάσεων (ίδιος τύπος για όλα)
        # S_xy = fL_xx + fL_yy - 2*fL_xy
        ones = np.ones(L.shape[0])
        S = np.outer(np.diag(fL), ones) + np.outer(ones, np.diag(fL)) - 2 * fL
        
        # 6. Ιστόγραμμα
        hist, _ = np.histogram(S.flatten(), bins=self.hist_bins, range=self.hist_range)
        return hist

    def fit(self, graphs):
        self._set_seed()
        # Σιγουρευόμαστε ότι οι γράφοι είναι σωστοί
        self._embedding = [self._calculate_fgsd(graph) for graph in graphs]

    def get_embedding(self):
        return np.array(self._embedding)

    def infer(self, graphs):
        self._set_seed()
        return np.array([self._calculate_fgsd(graph) for graph in graphs])