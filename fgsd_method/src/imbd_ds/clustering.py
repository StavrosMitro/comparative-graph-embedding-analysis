"""
Clustering analysis functions for FGSD experiments on IMDB-MULTI.
Enhanced with grid search for clustering hyperparameters.
(Spectral-only: no node-label features)
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
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: 'umap-learn' not installed. Install with: pip install umap-learn")

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from optimized_method import HybridFGSD
from .config import RESULTS_DIR


# Grid search parameters - NOW INCLUDES 'none' for raw embeddings
CLUSTERING_GRID = {
    'normalization': ['l2', 'standard', 'minmax', 'none'],
    'pca_variance': [0.90, 0.95, 0.99],
    'kmeans_n_init': [10, 50],
    'spectral_n_neighbors': [5, 15],
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


def generate_embeddings(graphs: List, config: Dict[str, Any]) -> np.ndarray:
    """Generate FGSD embeddings for graphs (spectral-only)."""
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

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_spectral)
    return X_normalized


def perform_clustering_with_params(
    X: np.ndarray,
    y_true: np.ndarray,
    norm_method: str = 'l2',
    pca_variance: float = 0.95,
    kmeans_n_init: int = 50,
    spectral_n_neighbors: int = 15,
    spectral_affinity: str = 'nearest_neighbors'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Perform clustering with specific hyperparameters."""
    X_norm = apply_normalization(X, norm_method)
    pca = PCA(n_components=pca_variance, random_state=42)
    X_pca = pca.fit_transform(X_norm)

    n_classes = len(np.unique(y_true))

    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=kmeans_n_init, max_iter=300)
    y_kmeans = kmeans.fit_predict(X_pca)

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
        print(f"  Warning: Spectral clustering failed: {e}")
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
    compute_umap: bool = True,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run grid search with detailed output."""
    if lightweight:
        grid = {
            'norm_method': ['l2', 'standard', 'none'],
            'pca_variance': [0.95, 0.99],
            'kmeans_n_init': [10, 50],
            'spectral_n_neighbors': [10, 15, 20],
            'spectral_affinity': ['nearest_neighbors'],
        }
    else:
        grid = {
            'norm_method': ['l2', 'standard', 'minmax', 'none'],
            'pca_variance': [0.90, 0.95, 0.99],
            'kmeans_n_init': [10, 50],
            'spectral_n_neighbors': [5, 10, 15, 20],
            'spectral_affinity': ['nearest_neighbors', 'rbf'],
        }

    results = []
    keys = list(grid.keys())
    combinations = list(product(*[grid[k] for k in keys]))
    print(f"  Running grid search with {len(combinations)} combinations...")

    umap_embeddings = {}
    if compute_umap and HAS_UMAP:
        print("\n  Pre-computing UMAP embeddings...")
        for norm in grid['norm_method']:
            X_norm = apply_normalization(X, norm)
            umap_emb = compute_umap_embedding(X_norm)
            if umap_emb is not None:
                umap_embeddings[norm] = umap_emb
                km_umap = KMeans(n_clusters=len(np.unique(y_true)), random_state=42, n_init=10)
                y_km_umap = km_umap.fit_predict(umap_emb)
                umap_ari = adjusted_rand_score(y_true, y_km_umap)
                umap_sil = silhouette_score(umap_emb, y_km_umap) if len(np.unique(y_km_umap)) > 1 else -1
                print(f"    {norm}: UMAP K-Means ARI={umap_ari:.4f}, Sil={umap_sil:.4f}")

    if verbose:
        print(f"\n  {'Norm':<10} {'PCA':<6} {'KM_init':<8} {'SP_n':<6} {'Method':<10} {'ARI':<8} {'Sil':<8} {'UMAP_ARI':<10} {'UMAP_Sil':<10}")
        print("  " + "-"*90)

    for combo in combinations:
        params = dict(zip(keys, combo))
        try:
            X_pca, y_kmeans, y_spectral, params_used = perform_clustering_with_params(X, y_true, **params)

            km_ari = adjusted_rand_score(y_true, y_kmeans)
            km_sil = silhouette_score(X_pca, y_kmeans) if len(np.unique(y_kmeans)) > 1 else -1
            sp_ari = adjusted_rand_score(y_true, y_spectral)
            sp_sil = silhouette_score(X_pca, y_spectral) if len(np.unique(y_spectral)) > 1 else -1

            km_result = {'config_name': config_name, **params_used, 'method': 'K-Means', 'ari': km_ari, 'silhouette': km_sil}
            
            if params['norm_method'] in umap_embeddings:
                umap_emb = umap_embeddings[params['norm_method']]
                km_umap = KMeans(n_clusters=len(np.unique(y_true)), random_state=42, n_init=params['kmeans_n_init'])
                y_km_umap = km_umap.fit_predict(umap_emb)
                km_result['umap_ari'] = adjusted_rand_score(y_true, y_km_umap)
                km_result['umap_silhouette'] = silhouette_score(umap_emb, y_km_umap) if len(np.unique(y_km_umap)) > 1 else -1
            
            results.append(km_result)
            
            if verbose:
                umap_ari_str = f"{km_result.get('umap_ari', 0):.4f}" if 'umap_ari' in km_result else "N/A"
                umap_sil_str = f"{km_result.get('umap_silhouette', 0):.4f}" if 'umap_silhouette' in km_result else "N/A"
                print(f"  {params['norm_method']:<10} {params['pca_variance']:<6.2f} {params['kmeans_n_init']:<8} "
                      f"{params['spectral_n_neighbors']:<6} {'K-Means':<10} {km_ari:<8.4f} {km_sil:<8.4f} {umap_ari_str:<10} {umap_sil_str:<10}")

            sp_result = {'config_name': config_name, **params_used, 'method': 'Spectral', 'ari': sp_ari, 'silhouette': sp_sil}
            
            if params['norm_method'] in umap_embeddings:
                umap_emb = umap_embeddings[params['norm_method']]
                try:
                    sp_umap = SpectralClustering(n_clusters=len(np.unique(y_true)), affinity='nearest_neighbors',
                                                  n_neighbors=params['spectral_n_neighbors'], random_state=42)
                    y_sp_umap = sp_umap.fit_predict(umap_emb)
                    sp_result['umap_ari'] = adjusted_rand_score(y_true, y_sp_umap)
                    sp_result['umap_silhouette'] = silhouette_score(umap_emb, y_sp_umap) if len(np.unique(y_sp_umap)) > 1 else -1
                except:
                    pass
            
            results.append(sp_result)
            
            if verbose:
                umap_ari_str = f"{sp_result.get('umap_ari', 0):.4f}" if 'umap_ari' in sp_result else "N/A"
                umap_sil_str = f"{sp_result.get('umap_silhouette', 0):.4f}" if 'umap_silhouette' in sp_result else "N/A"
                print(f"  {params['norm_method']:<10} {params['pca_variance']:<6.2f} {params['kmeans_n_init']:<8} "
                      f"{params['spectral_n_neighbors']:<6} {'Spectral':<10} {sp_ari:<8.4f} {sp_sil:<8.4f} {umap_ari_str:<10} {umap_sil_str:<10}")
        except Exception as e:
            print(f"  Warning: Failed for params {params}: {e}")

    return results


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray, method_name: str) -> Dict[str, float]:
    """Evaluate clustering results."""
    ari = adjusted_rand_score(y_true, y_pred)
    try:
        silhouette = silhouette_score(X, y_pred) if len(np.unique(y_pred)) > 1 else -1
    except:
        silhouette = -1
    return {'method': method_name, 'ari': ari, 'silhouette': silhouette}


def visualize_clusters(
    X_scaled: np.ndarray,
    y_true: np.ndarray,
    y_kmeans: np.ndarray,
    y_spectral: np.ndarray,
    config_name: str,
    params_info: str = "",
    save_dir: str = None,
    func_type: str = None
) -> Dict[str, np.ndarray]:
    """Visualize clustering results using t-SNE and optionally UMAP."""
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    labels_list = [y_true, y_kmeans, y_spectral]
    embeddings = {}
    metrics = {}

    if HAS_UMAP:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        umap_row = axes[0]
        tsne_row = axes[1]

        print("  -> Running UMAP...")
        embedding_umap = compute_umap_embedding(X_scaled)
        if embedding_umap is not None:
            embeddings['umap'] = embedding_umap
            
            umap_km_ari = adjusted_rand_score(y_true, y_kmeans)
            umap_sp_ari = adjusted_rand_score(y_true, y_spectral)
            umap_km_sil = silhouette_score(embedding_umap, y_kmeans) if len(np.unique(y_kmeans)) > 1 else -1
            umap_sp_sil = silhouette_score(embedding_umap, y_spectral) if len(np.unique(y_spectral)) > 1 else -1
            
            metrics['umap'] = {
                'kmeans_ari': umap_km_ari, 'spectral_ari': umap_sp_ari,
                'kmeans_sil': umap_km_sil, 'spectral_sil': umap_sp_sil
            }

            titles_umap = [
                f'GT (UMAP)',
                f'KMeans (UMAP)\nARI={umap_km_ari:.4f}, Sil={umap_km_sil:.4f}',
                f'Spectral (UMAP)\nARI={umap_sp_ari:.4f}, Sil={umap_sp_sil:.4f}'
            ]
            for ax, labels, title in zip(umap_row, labels_list, titles_umap):
                ax.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        tsne_row = axes

    print("  -> Running t-SNE...")
    embedding_tsne = compute_tsne_embedding(X_scaled)
    embeddings['tsne'] = embedding_tsne
    
    tsne_km_ari = adjusted_rand_score(y_true, y_kmeans)
    tsne_sp_ari = adjusted_rand_score(y_true, y_spectral)
    tsne_km_sil = silhouette_score(embedding_tsne, y_kmeans) if len(np.unique(y_kmeans)) > 1 else -1
    tsne_sp_sil = silhouette_score(embedding_tsne, y_spectral) if len(np.unique(y_spectral)) > 1 else -1
    
    metrics['tsne'] = {
        'kmeans_ari': tsne_km_ari, 'spectral_ari': tsne_sp_ari,
        'kmeans_sil': tsne_km_sil, 'spectral_sil': tsne_sp_sil
    }

    titles_tsne = [
        f'GT (t-SNE)',
        f'KMeans (t-SNE)\nARI={tsne_km_ari:.4f}, Sil={tsne_km_sil:.4f}',
        f'Spectral (t-SNE)\nARI={tsne_sp_ari:.4f}, Sil={tsne_sp_sil:.4f}'
    ]
    for ax, labels, title in zip(tsne_row, labels_list, titles_tsne):
        ax.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    title_str = f"Config: {config_name}"
    if func_type:
        title_str = f"Function: {func_type.upper()} | {title_str}"
    if params_info:
        title_str += f"\n{params_info}"
    fig.suptitle(title_str, fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    if func_type:
        save_path = os.path.join(save_dir, f'clustering_imdb_{func_type}_{config_name}.png')
    else:
        save_path = os.path.join(save_dir, f'clustering_imdb_{config_name}_best.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved to: {save_path}")

    print(f"\n  === VISUALIZATION METRICS for {config_name} ===")
    if 'umap' in metrics:
        print(f"  UMAP:  KMeans ARI={metrics['umap']['kmeans_ari']:.4f}, Sil={metrics['umap']['kmeans_sil']:.4f}")
        print(f"         Spectral ARI={metrics['umap']['spectral_ari']:.4f}, Sil={metrics['umap']['spectral_sil']:.4f}")
    print(f"  t-SNE: KMeans ARI={metrics['tsne']['kmeans_ari']:.4f}, Sil={metrics['tsne']['kmeans_sil']:.4f}")
    print(f"         Spectral ARI={metrics['tsne']['spectral_ari']:.4f}, Sil={metrics['tsne']['spectral_sil']:.4f}")

    return embeddings


def run_clustering_experiment(
    graphs: List,
    labels: np.ndarray,
    configs: List[Dict[str, Any]],
    neighbor_values: List[int] = [10, 15, 20],
    visualize: bool = True,
    run_grid_search: bool = True,
    save_umap_coords: bool = True
) -> List[Dict[str, Any]]:
    """Run full clustering experiment with plots for each function type."""
    all_results = []
    best_per_func = {}

    for config in configs:
        print(f"\n{'='*80}")
        print(f"Processing Configuration: {config['name']}")
        print(f"Function: {config['func'].upper()}")
        print(f"{'='*80}")

        print("Generating embeddings...")
        start_time = time.time()
        X = generate_embeddings(graphs, config)
        embed_time = time.time() - start_time
        print(f"  -> Embedding shape: {X.shape}, Time: {embed_time:.2f}s")

        if run_grid_search:
            grid_results = run_clustering_grid_search(X, labels, config['name'], lightweight=True, 
                                                       compute_umap=HAS_UMAP, verbose=True)

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

                func_type = config['func']
                if func_type not in best_per_func or res['ari'] > best_per_func[func_type]['result']['ari']:
                    X_pca, y_km, y_sp, _ = perform_clustering_with_params(
                        X, labels,
                        norm_method=res['normalization'],
                        pca_variance=res['pca_variance'],
                        kmeans_n_init=res['kmeans_n_init'],
                        spectral_n_neighbors=res['spectral_n_neighbors'],
                        spectral_affinity=res['spectral_affinity']
                    )
                    best_per_func[func_type] = {
                        'result': res, 'X': X_pca.copy(), 'y_kmeans': y_km.copy(),
                        'y_spectral': y_sp.copy(), 'config': config, 'labels': labels
                    }

            config_results = [r for r in all_results if r.get('config_name') == config['name']]
            if config_results:
                best_km = max([r for r in config_results if r['method'] == 'K-Means'], key=lambda x: x['ari'])
                best_sp = max([r for r in config_results if r['method'] == 'Spectral'], key=lambda x: x['ari'])
                print(f"\n  === CONFIG SUMMARY: {config['name']} ===")
                print(f"  Best K-Means:  ARI={best_km['ari']:.4f}, Sil={best_km['silhouette']:.4f}, norm={best_km['normalization']}")
                print(f"  Best Spectral: ARI={best_sp['ari']:.4f}, Sil={best_sp['silhouette']:.4f}, norm={best_sp['normalization']}")
        else:
            for n_neighbors in neighbor_values:
                print(f"\n--- n_neighbors={n_neighbors} ---")
                X_pca, y_kmeans, y_spectral, params_used = perform_clustering_with_params(X, labels, spectral_n_neighbors=n_neighbors)
                km_results = evaluate_clustering(labels, y_kmeans, X_pca, 'K-Means')
                sp_results = evaluate_clustering(labels, y_spectral, X_pca, 'Spectral')
                print(f"K-Means  -> ARI: {km_results['ari']:.4f} | Silhouette: {km_results['silhouette']:.4f}")
                print(f"Spectral -> ARI: {sp_results['ari']:.4f} | Silhouette: {sp_results['silhouette']:.4f}")

        del X
        gc.collect()

    if visualize and best_per_func:
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS FOR BEST OF EACH FUNCTION TYPE")
        print(f"{'='*80}")

        for func_type, data in best_per_func.items():
            print(f"\n  Visualizing best {func_type.upper()}...")
            result = data['result']
            params_str = f"norm={result['normalization']}, pca={result['pca_variance']}"
            params_str += f"\nBest ARI={result['ari']:.4f}, Sil={result['silhouette']:.4f}"

            embeddings = visualize_clusters(
                data['X'], data['labels'], data['y_kmeans'], data['y_spectral'],
                result['config_name'], params_str, func_type=func_type
            )

            if save_umap_coords and 'umap' in embeddings:
                import pandas as pd
                umap_df = pd.DataFrame({
                    'umap_x': embeddings['umap'][:, 0], 'umap_y': embeddings['umap'][:, 1],
                    'true_label': data['labels'], 'kmeans_label': data['y_kmeans'],
                    'spectral_label': data['y_spectral']
                })
                umap_path = os.path.join(RESULTS_DIR, f'umap_coords_imdb_{func_type}_{result["config_name"]}.csv')
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
        print(f"  Params: norm={best['normalization']}, pca={best['pca_variance']}, n_neighbors={best.get('spectral_n_neighbors', 'N/A')}")
    if has_umap and 'umap_ari' in best:
        print(f"  UMAP: ARI={best['umap_ari']:.4f}, Silhouette={best['umap_silhouette']:.4f}")
