import numpy as np
import networkx as nx
from typing import List
from karateclub.estimator import Estimator

class HybridFGSD(Estimator):
    """
    Μια ευέλικτη υλοποίηση της FGSD που επιτρέπει αλλαγή της συνάρτησης f(λ).
    
    Args:
        harm_bins (int): Αριθμός κάδων για το αρμονικό μέρος.
        harm_range (int/float): Εύρος για το αρμονικό μέρος.
        pol_bins (int): Αριθμός κάδων για το πολυωνυμικό μέρος.
        pol_range (int/float): Εύρος για το πολυωνυμικό μέρος.
        func_type (str): 'harmonic', 'biharmonic', 'polynomial', 'hybrid'.
        seed (int): Random seed.
    """
    def __init__(self, harm_bins=200, harm_range=20, pol_bins=200, pol_range=20, func_type='harmonic', seed=42):
        # Parameters for hybrid mode
        self.harm_bins = harm_bins
        self.harm_range = (0, harm_range)
        self.pol_bins = pol_bins
        self.pol_range = (0, pol_range)

        # General parameters for non-hybrid modes (defaults to harmonic)
        self.hist_bins = harm_bins
        self.hist_range = (0, harm_range)
        
        self.func_type = func_type
        self.seed = seed

    def _calculate_fgsd(self, graph):
        # 1. Υπολογισμός Normalized Laplacian
        # Χρησιμοποιούμε asarray για να αποφύγουμε προβλήματα με matrix types
        L = np.asarray(nx.normalized_laplacian_matrix(graph).todense())
        
        # w: ιδιοτιμές, v: ιδιοδιανύσματα
        w, v = np.linalg.eigh(L)
        
        if self.func_type == 'hybrid':
            # Hybrid: [Hist_Harmonic || Hist_Polynomial]
            # Calculate Harmonic part
            with np.errstate(divide='ignore', invalid='ignore'):
                func_w_harmonic = np.where(w > 1e-9, 1.0 / w, 0)
            fL_harmonic = v @ np.diag(func_w_harmonic) @ v.T
            ones = np.ones(L.shape[0])
            S_harmonic = np.outer(np.diag(fL_harmonic), ones) + np.outer(ones, np.diag(fL_harmonic)) - 2 * fL_harmonic
            hist_harmonic, _ = np.histogram(S_harmonic.flatten(), bins=self.harm_bins, range=self.harm_range)

            # Calculate Polynomial part
            func_w_poly = w ** 2
            fL_poly = v @ np.diag(func_w_poly) @ v.T
            S_poly = np.outer(np.diag(fL_poly), ones) + np.outer(ones, np.diag(fL_poly)) - 2 * fL_poly
            hist_poly, _ = np.histogram(S_poly.flatten(), bins=self.pol_bins, range=self.pol_range)
            
            return np.concatenate([hist_harmonic, hist_poly])

        if self.func_type == 'harmonic':
            # f(λ) = 1/λ (Global Structure)
            # Αν λ είναι πολύ κοντά στο 0 (λόγω float precision), το αγνοούμε ή βάζουμε 0
            func_w = np.where(w > 1e-9, 1.0 / w, 0)
            
        elif self.func_type == 'biharmonic':
            # f(λ) = 1/λ^2 (Global Structure - More Unique)
            func_w = np.where(w > 1e-9, 1.0 / (w**2), 0)
            
        elif self.func_type == 'polynomial':
            # f(λ) = λ^2 (Local Structure - Fine Grained)
            func_w = w ** 2
        elif self.func_type == 'mixed':
            # This mixes eigenvalues, not histograms. Correcting the variable name.
            func_w = w ** 2 + np.where(w > 1e-9, 1.0 / w, 0)
            
        else:
            raise ValueError(f"Unknown function type: {self.func_type}")
            
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