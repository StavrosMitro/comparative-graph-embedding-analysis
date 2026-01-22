"""
Clustering analysis functions for FGSD experiments on ENZYMES.
"""

import os
import gc
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
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


def perform_clustering_analysis(
    X: np.ndarray, 
    y_true: np.ndarray, 
    n_neighbors: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform K-Means and Spectral Clustering on embeddings."""
    X_norm = normalize(X, norm='l2')
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    
    n_classes = len(np.unique(y_true))
    
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=50)
    y_kmeans = kmeans.fit_predict(X_pca)
    
    spectral = SpectralClustering(
        n_clusters=n_classes,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        assign_labels='discretize',
        random_state=42,
        n_jobs=-1
    )
    y_spectral = spectral.fit_predict(X_pca)
    
    return X_pca, y_kmeans, y_spectral


def evaluate_clustering(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    X: np.ndarray,
    method_name: str
) -> Dict[str, float]:
    """Evaluate clustering results."""
    ari = adjusted_rand_score(y_true, y_pred)
    silhouette = silhouette_score(X, y_pred)
    
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
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'clustering_enzymes_{config_name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Plot saved to: {save_path}")


def run_clustering_experiment(
    graphs: List,
    labels: np.ndarray,
    configs: List[Dict[str, Any]],
    node_labels_list=None,
    neighbor_values: List[int] = [10, 20],
    visualize: bool = True
) -> List[Dict[str, Any]]:
    """Run full clustering experiment on multiple configurations."""
    all_results = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Processing Configuration: {config['name']}")
        print(f"{'='*80}")
        
        print("Generating embeddings...")
        start_time = time.time()
        X = generate_embeddings(graphs, config, node_labels_list)
        embed_time = time.time() - start_time
        print(f"  -> Embedding shape: {X.shape}, Time: {embed_time:.2f}s")
        
        for n_neighbors in neighbor_values:
            print(f"\n--- n_neighbors={n_neighbors} ---")
            
            X_scaled, y_kmeans, y_spectral = perform_clustering_analysis(
                X, labels, n_neighbors=n_neighbors
            )
            
            km_results = evaluate_clustering(labels, y_kmeans, X_scaled, 'K-Means')
            sp_results = evaluate_clustering(labels, y_spectral, X_scaled, 'Spectral')
            
            print(f"K-Means  -> ARI: {km_results['ari']:.4f} | Silhouette: {km_results['silhouette']:.4f}")
            print(f"Spectral -> ARI: {sp_results['ari']:.4f} | Silhouette: {sp_results['silhouette']:.4f}")
            
            for res in [km_results, sp_results]:
                result_entry = {
                    **config,
                    'n_neighbors': n_neighbors,
                    'embed_time': embed_time,
                    **res
                }
                all_results.append(result_entry)
            
            if visualize and n_neighbors == neighbor_values[0]:
                print("Generating visualization...")
                visualize_clusters(X_scaled, labels, y_kmeans, y_spectral, config['name'])
        
        del X
        gc.collect()
    
    return all_results


def print_clustering_summary(results: List[Dict[str, Any]]):
    """Print summary of clustering results."""
    print("\n" + "="*100)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*100)
    print(f"{'Config':<30} {'Method':<12} {'n_neighbors':<12} {'ARI':<10} {'Silhouette':<12}")
    print("-"*100)
    
    sorted_results = sorted(results, key=lambda x: x['ari'], reverse=True)
    
    for r in sorted_results:
        print(f"{r['name']:<30} {r['method']:<12} {r['n_neighbors']:<12} "
              f"{r['ari']:<10.4f} {r['silhouette']:<12.4f}")
    
    print("\n" + "-"*100)
    best = sorted_results[0]
    print(f"BEST: {best['name']} with {best['method']} (n_neighbors={best['n_neighbors']}) -> ARI: {best['ari']:.4f}")
