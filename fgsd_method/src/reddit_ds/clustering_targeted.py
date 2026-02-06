"""
Targeted FGSD Clustering on REDDIT-MULTI-12K
- Only 2 embeddings: polynomial_100_3.81 and harmonic_100_30.3
- Only 2 param sets: norm=l2 and norm=none, both with pca=0.99, n_neighbors=20
- Both K-Means and Spectral clustering
- Both UMAP and t-SNE visualizations
- Embeddings saved to disk

Usage:
    python -m reddit_ds.clustering_targeted
"""

import os
import sys
import gc
import time
import pickle
import warnings
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# UMAP import
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: 'umap-learn' not installed. Install with: pip install umap-learn")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fgsd import FlexibleFGSD
from reddit_ds.config import RESULTS_DIR, DATASET_DIR, BATCH_SIZE
from reddit_ds.data_loader import ensure_dataset_ready, load_metadata, iter_graph_batches

# =============================================================================
# CONFIGURATIONS
# =============================================================================
EMBEDDING_CONFIGS = [
    {'name': 'polynomial_100_3.48', 'func': 'polynomial', 'bins': 100, 'range': 3.48},
    {'name': 'harmonic_100_14.61', 'func': 'harmonic', 'bins': 100, 'range': 14.61},
    # Hybrid will be created by concatenating the above two
]

# Hybrid config (created from cached embeddings)
HYBRID_CONFIG = {
    'name': 'hybrid_harm100_pol100',
    'func': 'hybrid',
    'harm_bins': 100, 'harm_range': 14.61,
    'pol_bins': 100, 'pol_range': 3.48
}

CLUSTERING_PARAMS = [
    {'norm': 'l2', 'pca': 0.99, 'n_neighbors': 20},
    {'norm': 'none', 'pca': 0.99, 'n_neighbors': 20},
]

# Cache directory for embeddings
EMBEDDINGS_CACHE_DIR = os.path.join(parent_dir, 'cache', 'reddit_clustering_embeddings')


def get_embedding_cache_path(config: Dict) -> str:
    """Get cache file path for embeddings."""
    os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
    return os.path.join(EMBEDDINGS_CACHE_DIR, f"reddit_{config['name']}_embedding.pkl")


def save_embedding(X: np.ndarray, config: Dict):
    """Save embedding to disk."""
    path = get_embedding_cache_path(config)
    print(f"  Saving embedding to: {path}")
    with open(path, 'wb') as f:
        pickle.dump({'embedding': X, 'config': config}, f)
    print(f"  Saved: {X.shape}, {X.nbytes / 1024 / 1024:.1f} MB")


def load_embedding(config: Dict) -> Optional[np.ndarray]:
    """Load embedding from disk if exists."""
    path = get_embedding_cache_path(config)
    if os.path.exists(path):
        print(f"  Loading cached embedding: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['embedding']
    return None


def generate_embeddings_batchwise(config: Dict, batch_size: int = BATCH_SIZE, seed: int = 42) -> np.ndarray:
    """
    Generate embeddings batch-wise (like stability.py).
    Process graphs in batches to minimize RAM usage.
    """
    print(f"\n  Generating {config['name']} embeddings batch-wise...")
    print(f"  func={config['func']}, bins={config['bins']}, range={config['range']}")
    
    # Check cache first
    cached = load_embedding(config)
    if cached is not None:
        print(f"  Using cached embedding: shape={cached.shape}")
        return cached
    
    # Create model
    model = FlexibleFGSD(
        hist_bins=config['bins'],
        hist_range=config['range'],
        func_type=config['func'],
        seed=seed
    )
    
    # Get total batches
    records = load_metadata(DATASET_DIR)
    total_batches = (len(records) + batch_size - 1) // batch_size
    
    print(f"  Total graphs: {len(records)}, Batches: {total_batches}")
    
    embeddings_list = []
    start_time = time.time()
    
    for graphs, labels, gids in tqdm(iter_graph_batches(DATASET_DIR, batch_size),
                                      total=total_batches,
                                      desc=f"  {config['func']}"):
        # Generate embedding for this batch
        model.fit(graphs)
        batch_emb = model.get_embedding()
        embeddings_list.append(batch_emb)
        
        # Free batch
        del graphs
        gc.collect()
    
    # Stack all embeddings
    X_all = np.vstack(embeddings_list)
    generation_time = time.time() - start_time
    
    print(f"  Generated: shape={X_all.shape}, time={generation_time:.1f}s")
    
    # Save to disk
    save_embedding(X_all, config)
    
    del embeddings_list
    gc.collect()
    
    return X_all


def apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization."""
    if method == 'l2':
        return normalize(X, norm='l2')
    elif method == 'none':
        return X
    elif method == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    else:
        raise ValueError(f"Unknown normalization: {method}")


def compute_umap_embedding(X: np.ndarray, n_neighbors: int = 15, random_state: int = 42) -> Optional[np.ndarray]:
    """Compute UMAP 2D embedding."""
    if not HAS_UMAP:
        return None
    try:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=random_state)
        return reducer.fit_transform(X)
    except Exception as e:
        print(f"  Warning: UMAP failed: {e}")
        return None


def compute_tsne_embedding(X: np.ndarray, perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """Compute t-SNE 2D embedding."""
    # For large datasets, subsample for t-SNE
    n_samples = X.shape[0]
    if n_samples > 5000:
        print(f"  t-SNE: Subsampling from {n_samples} to 5000 for speed...")
        indices = np.random.choice(n_samples, 5000, replace=False)
        X_sub = X[indices]
    else:
        X_sub = X
        indices = np.arange(n_samples)
    
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    emb = reducer.fit_transform(X_sub)
    
    return emb, indices


def perform_clustering(
    X: np.ndarray, 
    y_true: np.ndarray,
    norm_method: str,
    pca_variance: float,
    n_neighbors: int,
    n_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Perform clustering with given params."""
    # Normalize
    X_norm = apply_normalization(X, norm_method)
    
    # PCA
    pca = PCA(n_components=pca_variance, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    print(f"    PCA: {X.shape[1]} -> {X_pca.shape[1]} components")
    
    # K-Means
    print(f"    Running K-Means (k={n_classes})...")
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=50, max_iter=300)
    y_kmeans = kmeans.fit_predict(X_pca)
    
    # Spectral
    print(f"    Running Spectral (n_neighbors={n_neighbors})...")
    try:
        spectral = SpectralClustering(
            n_clusters=n_classes,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            assign_labels='discretize',
            random_state=42,
            n_jobs=-1
        )
        y_spectral = spectral.fit_predict(X_pca)
    except Exception as e:
        print(f"    Warning: Spectral failed: {e}")
        y_spectral = np.zeros(len(y_true), dtype=int)
    
    return X_pca, y_kmeans, y_spectral, {'pca_components': X_pca.shape[1]}


def create_visualization(
    X_pca: np.ndarray,
    y_true: np.ndarray,
    y_kmeans: np.ndarray,
    y_spectral: np.ndarray,
    config_name: str,
    norm: str,
    save_dir: str
) -> Dict[str, np.ndarray]:
    """Create UMAP and t-SNE visualizations with clustering results."""
    
    os.makedirs(save_dir, exist_ok=True)
    embeddings = {}
    metrics = {}
    
    # Compute metrics on PCA space
    km_ari = adjusted_rand_score(y_true, y_kmeans)
    sp_ari = adjusted_rand_score(y_true, y_spectral)
    km_sil = silhouette_score(X_pca, y_kmeans) if len(np.unique(y_kmeans)) > 1 else -1
    sp_sil = silhouette_score(X_pca, y_spectral) if len(np.unique(y_spectral)) > 1 else -1
    
    metrics['pca'] = {'km_ari': km_ari, 'sp_ari': sp_ari, 'km_sil': km_sil, 'sp_sil': sp_sil}
    
    print(f"\n    === Metrics (PCA space) ===")
    print(f"    K-Means:  ARI={km_ari:.4f}, Sil={km_sil:.4f}")
    print(f"    Spectral: ARI={sp_ari:.4f}, Sil={sp_sil:.4f}")
    
    # Create figure with UMAP and t-SNE rows
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    labels_list = [y_true, y_kmeans, y_spectral]
    
    # === UMAP Row ===
    if HAS_UMAP:
        print(f"    Computing UMAP...")
        umap_emb = compute_umap_embedding(X_pca)
        if umap_emb is not None:
            embeddings['umap'] = umap_emb
            
            # UMAP metrics
            umap_km_sil = silhouette_score(umap_emb, y_kmeans) if len(np.unique(y_kmeans)) > 1 else -1
            umap_sp_sil = silhouette_score(umap_emb, y_spectral) if len(np.unique(y_spectral)) > 1 else -1
            metrics['umap'] = {'km_sil': umap_km_sil, 'sp_sil': umap_sp_sil}
            
            print(f"    UMAP Silhouettes: K-Means={umap_km_sil:.4f}, Spectral={umap_sp_sil:.4f}")
            
            titles_umap = [
                f'GT (UMAP)',
                f'K-Means (UMAP)\nARI={km_ari:.4f}, Sil={umap_km_sil:.4f}',
                f'Spectral (UMAP)\nARI={sp_ari:.4f}, Sil={umap_sp_sil:.4f}'
            ]
            for ax, labels, title in zip(axes[0], labels_list, titles_umap):
                scatter = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], c=labels, cmap='tab10', s=5, alpha=0.5)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
    else:
        for ax in axes[0]:
            ax.text(0.5, 0.5, 'UMAP not available', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # === t-SNE Row ===
    print(f"    Computing t-SNE (subsampled)...")
    tsne_emb, tsne_indices = compute_tsne_embedding(X_pca)
    embeddings['tsne'] = tsne_emb
    embeddings['tsne_indices'] = tsne_indices
    
    # Get labels for subsampled points
    y_true_sub = y_true[tsne_indices]
    y_km_sub = y_kmeans[tsne_indices]
    y_sp_sub = y_spectral[tsne_indices]
    
    tsne_km_sil = silhouette_score(tsne_emb, y_km_sub) if len(np.unique(y_km_sub)) > 1 else -1
    tsne_sp_sil = silhouette_score(tsne_emb, y_sp_sub) if len(np.unique(y_sp_sub)) > 1 else -1
    metrics['tsne'] = {'km_sil': tsne_km_sil, 'sp_sil': tsne_sp_sil}
    
    print(f"    t-SNE Silhouettes: K-Means={tsne_km_sil:.4f}, Spectral={tsne_sp_sil:.4f}")
    
    titles_tsne = [
        f'GT (t-SNE, n={len(tsne_indices)})',
        f'K-Means (t-SNE)\nARI={km_ari:.4f}, Sil={tsne_km_sil:.4f}',
        f'Spectral (t-SNE)\nARI={sp_ari:.4f}, Sil={tsne_sp_sil:.4f}'
    ]
    labels_sub = [y_true_sub, y_km_sub, y_sp_sub]
    for ax, labels, title in zip(axes[1], labels_sub, titles_tsne):
        ax.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=labels, cmap='tab10', s=5, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Title
    fig.suptitle(f"REDDIT Clustering: {config_name}\nnorm={norm}, pca=0.99, n_neighbors=20", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'clustering_reddit_{config_name}_{norm}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved: {plot_path}")
    
    return embeddings, metrics


def run_targeted_clustering():
    """Run targeted clustering with specific configurations."""
    
    print("="*80)
    print("REDDIT TARGETED CLUSTERING")
    print("Configs: polynomial_100_3.48, harmonic_100_14.61, hybrid (concatenated)")
    print("Params: norm=l2 and norm=none, pca=0.99, n_neighbors=20")
    print("="*80)
    
    # Ensure dataset ready
    ensure_dataset_ready()
    
    # Load labels
    records = load_metadata(DATASET_DIR)
    labels = np.array([r.label for r in records])
    n_classes = len(np.unique(labels))
    print(f"\nTotal samples: {len(labels)}, Classes: {n_classes}")
    
    all_results = []
    cached_embeddings = {}  # Store for hybrid creation
    
    # =================================================================
    # STEP 1: Generate base embeddings (polynomial and harmonic)
    # =================================================================
    for emb_config in EMBEDDING_CONFIGS:
        print(f"\n{'='*80}")
        print(f"EMBEDDING: {emb_config['name']}")
        print(f"{'='*80}")
        
        # Generate embedding (batch-wise, saved to disk)
        X = generate_embeddings_batchwise(emb_config)
        
        # Cache for hybrid
        cached_embeddings[emb_config['func']] = {
            'X': X,
            'config': emb_config
        }
        
        # Run clustering for each param set
        for params in CLUSTERING_PARAMS:
            print(f"\n  --- Params: norm={params['norm']}, pca={params['pca']}, n_neighbors={params['n_neighbors']} ---")
            
            # Clustering
            X_pca, y_kmeans, y_spectral, info = perform_clustering(
                X, labels, 
                norm_method=params['norm'],
                pca_variance=params['pca'],
                n_neighbors=params['n_neighbors'],
                n_classes=n_classes
            )
            
            # Visualization
            embeddings, metrics = create_visualization(
                X_pca, labels, y_kmeans, y_spectral,
                emb_config['name'], params['norm'],
                RESULTS_DIR
            )
            
            # Store results
            result = {
                'config_name': emb_config['name'],
                'func': emb_config['func'],
                'bins': emb_config['bins'],
                'range': emb_config['range'],
                'normalization': params['norm'],
                'pca_variance': params['pca'],
                'pca_components': info['pca_components'],
                'spectral_n_neighbors': params['n_neighbors'],
                'kmeans_ari': metrics['pca']['km_ari'],
                'kmeans_silhouette': metrics['pca']['km_sil'],
                'spectral_ari': metrics['pca']['sp_ari'],
                'spectral_silhouette': metrics['pca']['sp_sil'],
            }
            
            if 'umap' in metrics:
                result['umap_kmeans_sil'] = metrics['umap']['km_sil']
                result['umap_spectral_sil'] = metrics['umap']['sp_sil']
            if 'tsne' in metrics:
                result['tsne_kmeans_sil'] = metrics['tsne']['km_sil']
                result['tsne_spectral_sil'] = metrics['tsne']['sp_sil']
            
            all_results.append(result)
            
            # Save UMAP coordinates
            if 'umap' in embeddings:
                umap_df = pd.DataFrame({
                    'umap_x': embeddings['umap'][:, 0],
                    'umap_y': embeddings['umap'][:, 1],
                    'true_label': labels,
                    'kmeans_label': y_kmeans,
                    'spectral_label': y_spectral
                })
                umap_path = os.path.join(RESULTS_DIR, f'umap_coords_reddit_{emb_config["name"]}_{params["norm"]}.csv')
                umap_df.to_csv(umap_path, index=False)
                print(f"    UMAP coords saved: {umap_path}")
    
    # =================================================================
    # STEP 2: Create and test HYBRID embedding (concatenation)
    # =================================================================
    if 'harmonic' in cached_embeddings and 'polynomial' in cached_embeddings:
        print(f"\n{'='*80}")
        print(f"HYBRID EMBEDDING: Concatenating harmonic + polynomial")
        print(f"{'='*80}")
        
        X_harm = cached_embeddings['harmonic']['X']
        X_poly = cached_embeddings['polynomial']['X']
        
        # Concatenate
        X_hybrid = np.hstack([X_harm, X_poly])
        print(f"  Hybrid shape: {X_hybrid.shape} (harmonic:{X_harm.shape[1]} + polynomial:{X_poly.shape[1]})")
        
        # Save hybrid embedding to disk
        hybrid_cache_path = os.path.join(EMBEDDINGS_CACHE_DIR, f"reddit_{HYBRID_CONFIG['name']}_embedding.pkl")
        print(f"  Saving hybrid embedding to: {hybrid_cache_path}")
        with open(hybrid_cache_path, 'wb') as f:
            pickle.dump({'embedding': X_hybrid, 'config': HYBRID_CONFIG}, f)
        
        # Run clustering for each param set
        for params in CLUSTERING_PARAMS:
            print(f"\n  --- Params: norm={params['norm']}, pca={params['pca']}, n_neighbors={params['n_neighbors']} ---")
            
            # Clustering
            X_pca, y_kmeans, y_spectral, info = perform_clustering(
                X_hybrid, labels, 
                norm_method=params['norm'],
                pca_variance=params['pca'],
                n_neighbors=params['n_neighbors'],
                n_classes=n_classes
            )
            
            # Visualization
            embeddings, metrics = create_visualization(
                X_pca, labels, y_kmeans, y_spectral,
                HYBRID_CONFIG['name'], params['norm'],
                RESULTS_DIR
            )
            
            # Store results
            result = {
                'config_name': HYBRID_CONFIG['name'],
                'func': 'hybrid',
                'harm_bins': HYBRID_CONFIG['harm_bins'],
                'harm_range': HYBRID_CONFIG['harm_range'],
                'pol_bins': HYBRID_CONFIG['pol_bins'],
                'pol_range': HYBRID_CONFIG['pol_range'],
                'normalization': params['norm'],
                'pca_variance': params['pca'],
                'pca_components': info['pca_components'],
                'spectral_n_neighbors': params['n_neighbors'],
                'kmeans_ari': metrics['pca']['km_ari'],
                'kmeans_silhouette': metrics['pca']['km_sil'],
                'spectral_ari': metrics['pca']['sp_ari'],
                'spectral_silhouette': metrics['pca']['sp_sil'],
            }
            
            if 'umap' in metrics:
                result['umap_kmeans_sil'] = metrics['umap']['km_sil']
                result['umap_spectral_sil'] = metrics['umap']['sp_sil']
            if 'tsne' in metrics:
                result['tsne_kmeans_sil'] = metrics['tsne']['km_sil']
                result['tsne_spectral_sil'] = metrics['tsne']['sp_sil']
            
            all_results.append(result)
            
            # Save UMAP coordinates for hybrid
            if 'umap' in embeddings:
                umap_df = pd.DataFrame({
                    'umap_x': embeddings['umap'][:, 0],
                    'umap_y': embeddings['umap'][:, 1],
                    'true_label': labels,
                    'kmeans_label': y_kmeans,
                    'spectral_label': y_spectral
                })
                umap_path = os.path.join(RESULTS_DIR, f'umap_coords_reddit_{HYBRID_CONFIG["name"]}_{params["norm"]}.csv')
                umap_df.to_csv(umap_path, index=False)
                print(f"    UMAP coords saved: {umap_path}")
        
        # Free hybrid memory
        del X_hybrid
    
    # Free base embeddings
    del cached_embeddings
    gc.collect()
    
    # Save results summary
    df = pd.DataFrame(all_results)
    results_path = os.path.join(RESULTS_DIR, 'fgsd_reddit_clustering_targeted.csv')
    df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved: {results_path}")
    
    # Print summary
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"{'Config':<30} {'Norm':<8} {'KM_ARI':<10} {'KM_Sil':<10} {'SP_ARI':<10} {'SP_Sil':<10}")
    print("-"*100)
    for r in all_results:
        print(f"{r['config_name']:<30} {r['normalization']:<8} "
              f"{r['kmeans_ari']:<10.4f} {r['kmeans_silhouette']:<10.4f} "
              f"{r['spectral_ari']:<10.4f} {r['spectral_silhouette']:<10.4f}")
    
    # Best results
    print("\n" + "-"*100)
    best_km = max(all_results, key=lambda x: x['kmeans_ari'])
    best_sp = max(all_results, key=lambda x: x['spectral_ari'])
    print(f"Best K-Means:  {best_km['config_name']} (norm={best_km['normalization']}) -> ARI={best_km['kmeans_ari']:.4f}")
    print(f"Best Spectral: {best_sp['config_name']} (norm={best_sp['normalization']}) -> ARI={best_sp['spectral_ari']:.4f}")
    
    print(f"\n✅ Embeddings cached in: {EMBEDDINGS_CACHE_DIR}")
    
    return all_results


if __name__ == "__main__":
    run_targeted_clustering()
