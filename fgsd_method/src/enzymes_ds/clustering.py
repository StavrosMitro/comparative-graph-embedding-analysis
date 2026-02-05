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
    print("Warning: 'umap-learn' not installed. UMAP features disabled.")

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from optimized_method import HybridFGSD
from .config import RESULTS_DIR
from .data_loader import create_node_label_features


# Grid search parameters - NOW INCLUDES 'none' for raw embeddings
CLUSTERING_GRID = {
    'normalization': ['l2', 'standard', 'minmax', 'none'],  # Added 'none' for raw
    'pca_variance': [0.80, 0.90, 0.95, 0.99],
    'kmeans_n_init': [10, 50],
    'spectral_n_neighbors': [5, 10, 15, 20],
    'spectral_affinity': ['nearest_neighbors', 'rbf'],
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
        return X  # No normalization - raw embeddings
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_umap_embedding(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, 
                           random_state: int = 42) -> Optional[np.ndarray]:
    """Compute UMAP 2D embedding for visualization and analysis."""
    if not HAS_UMAP:
        return None
    
    try:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                           n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(X)
        return embedding
    except Exception as e:
        print(f"  Warning: UMAP computation failed: {e}")
        return None


def compute_tsne_embedding(X: np.ndarray, perplexity: int = 30, 
                           random_state: int = 42) -> np.ndarray:
    """Compute t-SNE 2D embedding."""
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    return reducer.fit_transform(X)


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
    lightweight: bool = True,
    compute_umap: bool = True
) -> List[Dict[str, Any]]:
    """
    Run grid search over clustering hyperparameters.
    
    Args:
        X: Feature matrix
        y_true: Ground truth labels
        config_name: Name of the embedding configuration
        lightweight: Use reduced grid for faster search
        compute_umap: Whether to compute and store UMAP embeddings
    
    Returns:
        List of results for each parameter combination
    """
    if lightweight:
        grid = {
            'norm_method': ['l2', 'standard', 'none'],  # Added 'none' for raw
            'pca_variance': [0.95, 0.99],
            'kmeans_n_init': [10, 50],
            'spectral_n_neighbors': [10, 20],
            'spectral_affinity': ['nearest_neighbors'],
        }
    else:
        grid = {
            'norm_method': ['l2', 'standard', 'minmax', 'none'],  # Added 'none'
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
    print(f"  Normalization options: {grid['norm_method']} (includes 'none' for raw embeddings)")
    
    # Compute UMAP once per normalization (expensive)
    umap_embeddings = {}
    if compute_umap and HAS_UMAP:
        print("  Pre-computing UMAP embeddings for each normalization...")
        for norm in grid['norm_method']:
            X_norm = apply_normalization(X, norm)
            umap_emb = compute_umap_embedding(X_norm)
            if umap_emb is not None:
                umap_embeddings[norm] = umap_emb
                # Compute UMAP-based clustering metrics
                km_umap = KMeans(n_clusters=len(np.unique(y_true)), random_state=42, n_init=10)
                y_km_umap = km_umap.fit_predict(umap_emb)
                umap_ari = adjusted_rand_score(y_true, y_km_umap)
                umap_sil = silhouette_score(umap_emb, y_km_umap) if len(np.unique(y_km_umap)) > 1 else -1
                print(f"    {norm}: UMAP K-Means ARI={umap_ari:.4f}, Sil={umap_sil:.4f}")
    
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
            
            # Base result for K-Means
            km_result = {
                'config_name': config_name,
                **params_used,
                'method': 'K-Means',
                'ari': km_ari,
                'silhouette': km_sil
            }
            
            # Add UMAP metrics if available
            if params['norm_method'] in umap_embeddings:
                umap_emb = umap_embeddings[params['norm_method']]
                km_umap = KMeans(n_clusters=len(np.unique(y_true)), random_state=42, n_init=params['kmeans_n_init'])
                y_km_umap = km_umap.fit_predict(umap_emb)
                km_result['umap_ari'] = adjusted_rand_score(y_true, y_km_umap)
                km_result['umap_silhouette'] = silhouette_score(umap_emb, y_km_umap) if len(np.unique(y_km_umap)) > 1 else -1
            
            results.append(km_result)
            
            # Spectral result
            sp_result = {
                'config_name': config_name,
                **params_used,
                'method': 'Spectral',
                'ari': sp_ari,
                'silhouette': sp_sil
            }
            
            # Add UMAP metrics for spectral (same UMAP embedding, different clustering)
            if params['norm_method'] in umap_embeddings:
                umap_emb = umap_embeddings[params['norm_method']]
                try:
                    sp_umap = SpectralClustering(
                        n_clusters=len(np.unique(y_true)),
                        affinity='nearest_neighbors',
                        n_neighbors=params['spectral_n_neighbors'],
                        random_state=42
                    )
                    y_sp_umap = sp_umap.fit_predict(umap_emb)
                    sp_result['umap_ari'] = adjusted_rand_score(y_true, y_sp_umap)
                    sp_result['umap_silhouette'] = silhouette_score(umap_emb, y_sp_umap) if len(np.unique(y_sp_umap)) > 1 else -1
                except:
                    pass
            
            results.append(sp_result)
            
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
) -> Dict[str, np.ndarray]:
    """
    Visualize clustering results using t-SNE and optionally UMAP.
    
    Returns:
        Dictionary with 'umap' and 'tsne' embeddings if computed
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    labels_list = [y_true, y_kmeans, y_spectral]
    embeddings = {}
    
    if HAS_UMAP:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        umap_row = axes[0]
        tsne_row = axes[1]
        
        print("  -> Running UMAP...")
        embedding_umap = compute_umap_embedding(X_scaled)
        if embedding_umap is not None:
            embeddings['umap'] = embedding_umap
            
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
    embedding_tsne = compute_tsne_embedding(X_scaled)
    embeddings['tsne'] = embedding_tsne
    
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
    
    return embeddings


def run_clustering_experiment(
    graphs: List,
    labels: np.ndarray,
    configs: List[Dict[str, Any]],
    node_labels_list=None,
    neighbor_values: List[int] = [10, 20],
    visualize: bool = True,
    run_grid_search: bool = True,
    save_umap_coords: bool = True
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
        save_umap_coords: Whether to save UMAP coordinates to file
    
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
    
    # Store UMAP coordinates for best configurations
    umap_coordinates = {}
    
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
            # Run grid search with UMAP computation
            grid_results = run_clustering_grid_search(
                X, labels, config['name'], 
                lightweight=True, 
                compute_umap=HAS_UMAP
            )
            
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
                    best_params = {
                        'normalization': res['normalization'],
                        'pca_variance': res['pca_variance'],
                        'kmeans_n_init': res['kmeans_n_init'],
                        'spectral_n_neighbors': res['spectral_n_neighbors'],
                        'spectral_affinity': res['spectral_affinity']
                    }
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
            
            # Compute and save UMAP coordinates for this config
            if save_umap_coords and HAS_UMAP:
                X_norm = apply_normalization(X, 'standard')  # Use standard for UMAP
                umap_emb = compute_umap_embedding(X_norm)
                if umap_emb is not None:
                    umap_coordinates[config['name']] = {
                        'coords': umap_emb,
                        'labels': labels
                    }
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
        
        embeddings = visualize_clusters(
            best_X, labels, best_y_kmeans, best_y_spectral,
            best_config_name, params_str
        )
        
        # Save UMAP coordinates to CSV
        if save_umap_coords and 'umap' in embeddings:
            import pandas as pd
            umap_df = pd.DataFrame({
                'umap_x': embeddings['umap'][:, 0],
                'umap_y': embeddings['umap'][:, 1],
                'true_label': labels,
                'kmeans_label': best_y_kmeans,
                'spectral_label': best_y_spectral
            })
            umap_path = os.path.join(RESULTS_DIR, f'umap_coords_enzymes_{best_config_name}.csv')
            umap_df.to_csv(umap_path, index=False)
            print(f"  -> UMAP coordinates saved to: {umap_path}")
    
    return all_results


def print_clustering_summary(results: List[Dict[str, Any]]):
    """Print summary of clustering results including UMAP metrics if available."""
    print("\n" + "="*120)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*120)
    
    sorted_results = sorted(results, key=lambda x: x['ari'], reverse=True)
    
    # Check if UMAP metrics are available
    has_umap = any('umap_ari' in r for r in results)
    
    # Print top 10
    print(f"\nTOP 10 RESULTS (ALL METHODS):")
    if has_umap:
        print(f"{'Config':<25} {'Method':<10} {'Norm':<10} {'PCA':<6} {'ARI':<8} {'Sil':<8} {'UMAP_ARI':<10} {'UMAP_Sil':<10}")
    else:
        print(f"{'Config':<25} {'Method':<10} {'Norm':<10} {'PCA':<6} {'ARI':<8} {'Silhouette':<10}")
    print("-"*100)
    
    for r in sorted_results[:10]:
        norm = r.get('normalization', 'l2')[:8]
        pca = r.get('pca_variance', 0.95)
        
        if has_umap:
            umap_ari = r.get('umap_ari', 'N/A')
            umap_sil = r.get('umap_silhouette', 'N/A')
            umap_ari_str = f"{umap_ari:.4f}" if isinstance(umap_ari, float) else umap_ari
            umap_sil_str = f"{umap_sil:.4f}" if isinstance(umap_sil, float) else umap_sil
            print(f"{r.get('config_name', r.get('name', 'N/A')):<25} {r['method']:<10} {norm:<10} {pca:<6.2f} "
                  f"{r['ari']:<8.4f} {r['silhouette']:<8.4f} {umap_ari_str:<10} {umap_sil_str:<10}")
        else:
            print(f"{r.get('config_name', r.get('name', 'N/A')):<25} {r['method']:<10} {norm:<10} {pca:<6.2f} "
                  f"{r['ari']:<8.4f} {r['silhouette']:<10.4f}")
    
    # === BEST PER METHOD ===
    print("\n" + "-"*100)
    print("BEST RESULT PER CLUSTERING METHOD:")
    for method in ['K-Means', 'Spectral']:
        method_results = [r for r in results if r.get('method') == method]
        if method_results:
            best = max(method_results, key=lambda x: x['ari'])
            config_name = best.get('config_name', best.get('name', 'N/A'))
            norm = best.get('normalization', 'l2')
            pca = best.get('pca_variance', 0.95)
            print(f"  {method:<12}: ARI = {best['ari']:.4f}, Sil = {best['silhouette']:.4f}")
            print(f"               Config: {config_name}, norm={norm}, pca={pca}")
    
    # Summary of normalization methods
    print("\n" + "-"*100)
    print("NORMALIZATION COMPARISON (Best per norm, any method):")
    norm_methods = set(r.get('normalization', 'unknown') for r in results)
    for norm in sorted(norm_methods):
        norm_results = [r for r in results if r.get('normalization') == norm]
        if norm_results:
            best = max(norm_results, key=lambda x: x['ari'])
            print(f"  {norm:<12}: Best ARI = {best['ari']:.4f} ({best['method']}, {best.get('config_name', 'N/A')}, PCA={best.get('pca_variance', 'N/A')})")
    
    # === K-MEANS vs SPECTRAL COMPARISON ===
    print("\n" + "-"*100)
    print("K-MEANS vs SPECTRAL COMPARISON (per config):")
    print(f"{'Config':<30} {'KM ARI':<10} {'SP ARI':<10} {'Winner':<12}")
    print("-"*70)
    
    config_names = set(r.get('config_name', r.get('name', 'N/A')) for r in results)
    for config_name in sorted(config_names):
        config_results = [r for r in results if r.get('config_name', r.get('name', 'N/A')) == config_name]
        
        km_results = [r for r in config_results if r['method'] == 'K-Means']
        sp_results = [r for r in config_results if r['method'] == 'Spectral']
        
        km_best = max(km_results, key=lambda x: x['ari'])['ari'] if km_results else 0
        sp_best = max(sp_results, key=lambda x: x['ari'])['ari'] if sp_results else 0
        
        winner = "K-Means" if km_best > sp_best else ("Spectral" if sp_best > km_best else "Tie")
        
        print(f"{config_name:<30} {km_best:<10.4f} {sp_best:<10.4f} {winner:<12}")
    
    print("\n" + "-"*100)
    best = sorted_results[0]
    print(f"OVERALL BEST: {best.get('config_name', best.get('name', 'N/A'))} with {best['method']}")
    print(f"  ARI: {best['ari']:.4f}, Silhouette: {best['silhouette']:.4f}")
    if 'normalization' in best:
        print(f"  Params: norm={best['normalization']}, pca={best['pca_variance']}, "
              f"n_neighbors={best.get('spectral_n_neighbors', 'N/A')}")
    if has_umap and 'umap_ari' in best:
        print(f"  UMAP: ARI={best['umap_ari']:.4f}, Silhouette={best['umap_silhouette']:.4f}")


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
