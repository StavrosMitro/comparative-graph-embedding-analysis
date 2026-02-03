"""
Clustering analysis functions for FGSD experiments on ENZYMES.
Enhanced with grid search for clustering hyperparameters.
"""

import os
import gc
import time
from typing import List, Dict, Any, Tuple, Optional
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from optimized_method import HybridFGSD
from .config import RESULTS_DIR
from .data_loader import create_node_label_features


# Grid search parameters (lightweight)
CLUSTERING_GRID = {
    'normalization': ['l2', 'standard', 'minmax'],  # Normalization methods
    'pca_variance': [0.90, 0.95, 0.99],  # PCA variance thresholds
    'kmeans_n_init': [10, 50],  # K-Means n_init
    'spectral_n_neighbors': [5, 10, 15, 20],  # Spectral n_neighbors
    'spectral_affinity': ['nearest_neighbors', 'rbf'],  # Spectral affinity
}


def apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to feature matrix."""
    if method == 'l2':
        return normalize(X, norm='l2')
    elif method == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    elif method == 'none':
        return X
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def generate_embeddings(graphs: List, config: Dict[str, Any], node_labels_list=None) -> np.ndarray:
    """Generate FGSD embeddings for graphs based on configuration."""
    func = config['func']
    
    if func == 'hybrid':
        model = HybridFGSD(
            harm_bins=config['harm_bins'],
            harm_range=config['harm_range'],
            pol_bins=config['pol_bins'],
            pol_range=config['pol_range'],
            func_type='hybrid',
            seed=42
        )
    else:
        model = FlexibleFGSD(
            hist_bins=config['bins'],
            hist_range=config['range'],
            func_type=func,
            seed=42
        )
    
    model.fit(graphs)
    X_spectral = model.get_embedding()
    
    # Add node labels if available
    if node_labels_list is not None:
        X_labels = create_node_label_features(node_labels_list)
        scaler_spec = StandardScaler()
        X_spectral_norm = scaler_spec.fit_transform(X_spectral)
        scaler_lbl = StandardScaler()
        X_labels_norm = scaler_lbl.fit_transform(X_labels)
        X_final = np.hstack([X_spectral_norm, X_labels_norm])
    else:
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X_spectral)
    
    return X_final


def perform_clustering_with_params(
    X: np.ndarray,
    y_true: np.ndarray,
    norm_method: str = 'l2',
    pca_variance: float = 0.95,
    kmeans_n_init: int = 50,
    spectral_n_neighbors: int = 10,
    spectral_affinity: str = 'nearest_neighbors'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Perform clustering with specific hyperparameters.
    
    Returns:
        X_reduced, y_kmeans, y_spectral, params_used
    """
    # Apply normalization
    X_norm = apply_normalization(X, norm_method)
    
    # PCA reduction
    pca = PCA(n_components=pca_variance, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    
    n_classes = len(np.unique(y_true))
    
    # K-Means
    kmeans = KMeans(
        n_clusters=n_classes,
        random_state=42,
        n_init=kmeans_n_init,
        max_iter=300
    )
    y_kmeans = kmeans.fit_predict(X_pca)
    
    # Spectral Clustering
    try:
        spectral = SpectralClustering(
            n_clusters=n_classes,
            affinity=spectral_affinity,
            n_neighbors=spectral_n_neighbors if spectral_affinity == 'nearest_neighbors' else None,
            assign_labels='discretize',
            random_state=42,
            n_jobs=-1
        )
        y_spectral = spectral.fit_predict(X_pca)
    except Exception as e:
        print(f"  Warning: Spectral clustering failed with {spectral_affinity}: {e}")
        y_spectral = np.zeros(len(y_true), dtype=int)
    
    params_used = {
        'normalization': norm_method,
        'pca_variance': pca_variance,
        'pca_components': X_pca.shape[1],
        'kmeans_n_init': kmeans_n_init,
        'spectral_n_neighbors': spectral_n_neighbors,
        'spectral_affinity': spectral_affinity
    }
    
    return X_pca, y_kmeans, y_spectral, params_used


def run_clustering_grid_search(
    X: np.ndarray,
    y_true: np.ndarray,
    config_name: str,
    lightweight: bool = True
) -> List[Dict[str, Any]]:
    """
    Run grid search over clustering hyperparameters.
    
    Args:
        X: Feature matrix
        y_true: Ground truth labels
        config_name: Name of the embedding configuration
        lightweight: Use reduced grid for faster search
    
    Returns:
        List of results for each parameter combination
    """
    if lightweight:
        grid = {
            'norm_method': ['l2', 'standard'],
            'pca_variance': [0.95, 0.99],
            'kmeans_n_init': [10, 50],
            'spectral_n_neighbors': [10, 20],
            'spectral_affinity': ['nearest_neighbors'],
        }
    else:
        grid = {
            'norm_method': ['l2', 'standard', 'minmax'],
            'pca_variance': [0.90, 0.95, 0.99],
            'kmeans_n_init': [10, 50],
            'spectral_n_neighbors': [5, 10, 15, 20],
            'spectral_affinity': ['nearest_neighbors', 'rbf'],
        }
    
    results = []
    
    # Generate all combinations
    keys = list(grid.keys())
    combinations = list(product(*[grid[k] for k in keys]))
    
    print(f"  Running grid search with {len(combinations)} combinations...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        try:
            X_pca, y_kmeans, y_spectral, params_used = perform_clustering_with_params(
                X, y_true, **params
            )
            
            # Evaluate K-Means
            km_ari = adjusted_rand_score(y_true, y_kmeans)
            km_sil = silhouette_score(X_pca, y_kmeans) if len(np.unique(y_kmeans)) > 1 else -1
            
            # Evaluate Spectral
            sp_ari = adjusted_rand_score(y_true, y_spectral)
            sp_sil = silhouette_score(X_pca, y_spectral) if len(np.unique(y_spectral)) > 1 else -1
            
            results.append({
                'config_name': config_name,
                **params_used,
                'method': 'K-Means',
                'ari': km_ari,
                'silhouette': km_sil
            })
            
            results.append({
                'config_name': config_name,
                **params_used,
                'method': 'Spectral',
                'ari': sp_ari,
                'silhouette': sp_sil
            })
            
        except Exception as e:
            print(f"  Warning: Failed for params {params}: {e}")
            continue
    
    return results


def evaluate_clustering(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    X: np.ndarray,
    method_name: str
) -> Dict[str, float]:
    """Evaluate clustering results."""
    ari = adjusted_rand_score(y_true, y_pred)
    try:
        silhouette = silhouette_score(X, y_pred) if len(np.unique(y_pred)) > 1 else -1
    except:
        silhouette = -1
    
    return {
        'method': method_name,
        'ari': ari,
        'silhouette': silhouette
    }


def visualize_clusters(
    X_scaled: np.ndarray,
    y_true: np.ndarray,
    y_kmeans: np.ndarray,
    y_spectral: np.ndarray,
    config_name: str,
    params_info: str = "",
    save_dir: str = None
):
    """Visualize clustering results using t-SNE and optionally UMAP."""
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    labels_list = [y_true, y_kmeans, y_spectral]
    
    if HAS_UMAP:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        umap_row = axes[0]
        tsne_row = axes[1]
        
        print("  -> Running UMAP...")
        reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_umap = reducer_umap.fit_transform(X_scaled)
        
        titles_umap = ['GT (UMAP)', 'KMeans (UMAP)', 'Spectral (UMAP)']
        for ax, labels, title in zip(umap_row, labels_list, titles_umap):
            ax.scatter(embedding_umap[:, 0], embedding_umap[:, 1], 
                      c=labels, cmap='tab10', s=15, alpha=0.7)
            ax.set_title(title)
            ax.axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        tsne_row = axes
    
    print("  -> Running t-SNE...")
    reducer_tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding_tsne = reducer_tsne.fit_transform(X_scaled)
    
    titles_tsne = ['GT (t-SNE)', 'KMeans (t-SNE)', 'Spectral (t-SNE)']
    for ax, labels, title in zip(tsne_row, labels_list, titles_tsne):
        ax.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], 
                  c=labels, cmap='tab10', s=15, alpha=0.7)
        ax.set_title(title)
        ax.axis('off')
    
    if params_info:
        fig.suptitle(f"Best Config: {config_name}\n{params_info}", fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'clustering_enzymes_{config_name}_best.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Plot saved to: {save_path}")


def run_clustering_experiment(
    graphs: List,
    labels: np.ndarray,
    configs: List[Dict[str, Any]],
    node_labels_list=None,
    neighbor_values: List[int] = [10, 20],
    visualize: bool = True,
    run_grid_search: bool = True
) -> List[Dict[str, Any]]:
    """
    Run full clustering experiment on multiple configurations.
    
    Args:
        graphs: List of NetworkX graphs
        labels: Ground truth labels
        configs: List of embedding configurations
        node_labels_list: Optional node labels for ENZYMES
        neighbor_values: List of n_neighbors (used if grid_search=False)
        visualize: Whether to generate visualizations (only for best)
        run_grid_search: Whether to run hyperparameter grid search
    
    Returns:
        List of result dictionaries
    """
    all_results = []
    best_result = {'ari': -1}
    best_X = None
    best_y_kmeans = None
    best_y_spectral = None
    best_config_name = None
    best_params = None
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Processing Configuration: {config['name']}")
        print(f"{'='*80}")
        
        print("Generating embeddings...")
        start_time = time.time()
        X = generate_embeddings(graphs, config, node_labels_list)
        embed_time = time.time() - start_time
        print(f"  -> Embedding shape: {X.shape}, Time: {embed_time:.2f}s")
        
        if run_grid_search:
            # Run grid search
            grid_results = run_clustering_grid_search(X, labels, config['name'], lightweight=True)
            
            for res in grid_results:
                res['embed_time'] = embed_time
                res['func'] = config['func']
                if 'bins' in config:
                    res['bins'] = config['bins']
                    res['range'] = config['range']
                elif 'harm_bins' in config:
                    res['harm_bins'] = config['harm_bins']
                    res['harm_range'] = config['harm_range']
                    res['pol_bins'] = config['pol_bins']
                    res['pol_range'] = config['pol_range']
                
                all_results.append(res)
                
                # Track best result
                if res['ari'] > best_result['ari']:
                    best_result = res
                    best_config_name = config['name']
                    # Keep 'normalization' key for printing, but call perform_clustering_with_params
                    # with the correct parameter names (norm_method...). This avoids KeyError.
                    best_params = {
                        'normalization': res['normalization'],
                        'pca_variance': res['pca_variance'],
                        'kmeans_n_init': res['kmeans_n_init'],
                        'spectral_n_neighbors': res['spectral_n_neighbors'],
                        'spectral_affinity': res['spectral_affinity']
                    }
                    # Store for visualization (map normalization -> norm_method)
                    X_pca, y_km, y_sp, _ = perform_clustering_with_params(
                        X, labels,
                        norm_method=best_params['normalization'],
                        pca_variance=best_params['pca_variance'],
                        kmeans_n_init=best_params['kmeans_n_init'],
                        spectral_n_neighbors=best_params['spectral_n_neighbors'],
                        spectral_affinity=best_params['spectral_affinity']
                    )
                    best_X = X_pca
                    best_y_kmeans = y_km
                    best_y_spectral = y_sp
        else:
            # Original behavior without grid search
            for n_neighbors in neighbor_values:
                print(f"\n--- n_neighbors={n_neighbors} ---")
                
                X_pca, y_kmeans, y_spectral, params_used = perform_clustering_with_params(
                    X, labels, spectral_n_neighbors=n_neighbors
                )
                
                km_results = evaluate_clustering(labels, y_kmeans, X_pca, 'K-Means')
                sp_results = evaluate_clustering(labels, y_spectral, X_pca, 'Spectral')
                
                print(f"K-Means  -> ARI: {km_results['ari']:.4f} | Silhouette: {km_results['silhouette']:.4f}")
                print(f"Spectral -> ARI: {sp_results['ari']:.4f} | Silhouette: {sp_results['silhouette']:.4f}")
                
                for res in [km_results, sp_results]:
                    result_entry = {
                        'name': config['name'],
                        'func': config['func'],
                        'bins': config.get('bins'),
                        'range': config.get('range'),
                        'n_neighbors': n_neighbors,
                        'embed_time': embed_time,
                        **res
                    }
                    if 'harm_bins' in config:
                        result_entry['harm_bins'] = config['harm_bins']
                        result_entry['harm_range'] = config['harm_range']
                        result_entry['pol_bins'] = config['pol_bins']
                        result_entry['pol_range'] = config['pol_range']
                    all_results.append(result_entry)
                    
                    if res['ari'] > best_result.get('ari', -1):
                        best_result = result_entry
                        best_X = X_pca
                        best_y_kmeans = y_kmeans
                        best_y_spectral = y_spectral
                        best_config_name = config['name']
        
        del X
        gc.collect()
    
    # Visualize only the best result
    if visualize and best_X is not None:
        print(f"\n{'='*80}")
        print(f"Generating visualization for BEST result: {best_config_name}")
        print(f"Best ARI: {best_result['ari']:.4f}")
        print(f"{'='*80}")
        
        params_str = ""
        if best_params:
            params_str = f"norm={best_params['normalization']}, pca={best_params['pca_variance']}, n_neighbors={best_params['spectral_n_neighbors']}"
        
        visualize_clusters(
            best_X, labels, best_y_kmeans, best_y_spectral,
            best_config_name, params_str
        )
    
    return all_results


def print_clustering_summary(results: List[Dict[str, Any]]):
    """Print summary of clustering results."""
    print("\n" + "="*100)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*100)
    
    sorted_results = sorted(results, key=lambda x: x['ari'], reverse=True)
    
    # Print top 10
    print(f"\nTOP 10 RESULTS:")
    print(f"{'Config':<25} {'Method':<10} {'Norm':<10} {'PCA':<6} {'ARI':<8} {'Silhouette':<10}")
    print("-"*80)
    
    for r in sorted_results[:10]:
        norm = r.get('normalization', 'l2')[:8]
        pca = r.get('pca_variance', 0.95)
        print(f"{r.get('config_name', r.get('name', 'N/A')):<25} {r['method']:<10} {norm:<10} {pca:<6.2f} "
              f"{r['ari']:<8.4f} {r['silhouette']:<10.4f}")
    
    print("\n" + "-"*100)
    best = sorted_results[0]
    print(f"BEST: {best.get('config_name', best.get('name', 'N/A'))} with {best['method']}")
    print(f"  ARI: {best['ari']:.4f}, Silhouette: {best['silhouette']:.4f}")
    if 'normalization' in best:
        print(f"  Params: norm={best['normalization']}, pca={best['pca_variance']}, "
              f"n_neighbors={best.get('spectral_n_neighbors', 'N/A')}")


# Backward compatibility alias
def perform_clustering_analysis(
    X: np.ndarray, 
    y_true: np.ndarray, 
    n_neighbors: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward compatible function."""
    X_pca, y_kmeans, y_spectral, _ = perform_clustering_with_params(
        X, y_true, spectral_n_neighbors=n_neighbors
    )
    return X_pca, y_kmeans, y_spectral
