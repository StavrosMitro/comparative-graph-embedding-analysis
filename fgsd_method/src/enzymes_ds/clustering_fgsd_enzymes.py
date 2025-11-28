import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import warnings
import urllib.request
import zipfile
from sklearn.decomposition import PCA

from sklearn.preprocessing import normalize 

# --- UMAP SAFE IMPORT ---
try:
    import umap
    HAS_UMAP = True
except ImportError:
    print("Warning: 'umap-learn' not found. Visualization will only use t-SNE.")
    HAS_UMAP = False

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

# --- SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the HYBRID class
from optimized_method import HybridFGSD 
from fgsd import FlexibleFGSD 

warnings.filterwarnings('ignore')

# --- 1. DATA LOADING (SAFE & ROBUST) ---
def download_and_load_enzymes():
    """Φορτώνει το ENZYMES dataset. Αν δεν υπάρχει, το κατεβάζει."""
    data_dir = '/tmp/ENZYMES'
    zip_path = os.path.join(data_dir, 'ENZYMES.zip')
    dataset_path = os.path.join(data_dir, 'ENZYMES')
    
    # Check & Download
    if not os.path.exists(dataset_path):
        print(f"Dataset not found in {dataset_path}. Downloading...")
        os.makedirs(data_dir, exist_ok=True)
        base_url = 'https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip'
        try:
            urllib.request.urlretrieve(base_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading: {e}")
            sys.exit(1)
    else:
        print(f"Dataset found in {dataset_path}.")

    # Load Files
    try:
        graph_indicator = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_graph_indicator.txt'), dtype=int)
        edges = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_A.txt'), dtype=int, delimiter=',')
        graph_labels = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_graph_labels.txt'), dtype=int)
        node_labels_raw = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_node_labels.txt'), dtype=int)
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)
        
    num_graphs = len(graph_labels)
    graphs = [nx.Graph() for _ in range(num_graphs)]
    node_labels_list = []
    
    # Nodes
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graphs[graph_id - 1].add_node(node_id)
    
    # Node Labels per Graph
    current_idx = 0
    for i in range(1, num_graphs + 1):
        count = np.sum(graph_indicator == i)
        labels_of_graph = node_labels_raw[current_idx : current_idx + count]
        node_labels_list.append(labels_of_graph)
        current_idx += count

    # Edges
    for edge in edges:
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]
        graphs[graph_id - 1].add_edge(node1, node2)
    
    # Convert to 0-indexed
    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]
    labels = graph_labels - 1
    
    return graphs, labels, node_labels_list

# --- 2. FEATURE GENERATION (USING HYBRID CLASS) ---
def create_node_label_features(node_labels_list):
    """Bag-of-Words για τα Node Labels"""
    all_labels = np.concatenate(node_labels_list)
    unique_labels = np.unique(all_labels)
    n_unique = len(unique_labels)
    min_lbl, max_lbl = min(unique_labels), max(unique_labels)
    
    features = []
    for labels in node_labels_list:
        hist, _ = np.histogram(labels, bins=n_unique, range=(min_lbl, max_lbl + 1))
        features.append(hist)
    return np.array(features)

def generate_embeddings(graphs, node_labels_list, config):
    """
    Generate embeddings based on configuration.
    
    Args:
        graphs: List of NetworkX graphs
        node_labels_list: List of node labels per graph
        config: Dictionary with 'func' and parameters
    
    Returns:
        Combined embedding matrix
    """
    func = config['func']
    
    if func == 'hybrid':
        print(f"Generating Hybrid Embeddings (harm_bins={config['harm_bins']}, harm_range={config['harm_range']}, "
              f"pol_bins={config['pol_bins']}, pol_range={config['pol_range']})...")
        model = HybridFGSD(
            harm_bins=config['harm_bins'], 
            harm_range=config['harm_range'],
            pol_bins=config['pol_bins'], 
            pol_range=config['pol_range'],
            func_type='hybrid', 
            seed=42
        )
    else:
        print(f"Generating {func.capitalize()} Embeddings (bins={config['bins']}, range={config['range']})...")
        model = FlexibleFGSD(
            hist_bins=config['bins'], 
            hist_range=config['range'], 
            func_type=func, 
            seed=42
        )
    
    model.fit(graphs)
    X_spectral = model.get_embedding()
    
    # Node Labels
    X_labels = create_node_label_features(node_labels_list)
    
    # Fusion
    X_final = np.hstack([X_spectral, X_labels])
    print(f"  -> Spectral Shape: {X_spectral.shape}")
    print(f"  -> Node Labels Shape: {X_labels.shape}")
    print(f"  -> Final Embedding Shape: {X_final.shape}")
    return X_final

# --- 3. CLUSTERING & EVALUATION ---
def perform_clustering_analysis(X, y_true):
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS RESULTS (OPTIMIZED)")
    print("="*60)
    
    # 1. Scaling (Βάζει τα features στην ίδια κλίμακα)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # 2. L2 Normalization (Κρίσιμο για embeddings - Cosine Similarity proxy)
    X_norm = normalize(X_std, norm='l2')
    
    # 3. PCA (Μείωση διαστάσεων για να βοηθήσουμε τον K-Means)
    # Κρατάμε το 95% της διακύμανσης (variance)
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    
    print(f"  -> Original Shape: {X.shape}")
    print(f"  -> Shape after PCA (95% variance): {X_pca.shape}")
    
    n_classes = len(np.unique(y_true)) # 6 classes
    
    # --- K-Means ---
    # n_init=50: Τρέχει 50 φορές για να βρει το βέλτιστο κέντρο (βοηθάει στη σταθερότητα)
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=50)
    y_kmeans = kmeans.fit_predict(X_pca)
    
    ari_kmeans = adjusted_rand_score(y_true, y_kmeans)
    sil_kmeans = silhouette_score(X_pca, y_kmeans) # Silhouette στο PCA space
    
    print(f"K-Means (k={n_classes}):")
    print(f"  -> ARI: {ari_kmeans:.4f}")
    print(f"  -> Silhouette: {sil_kmeans:.4f}")
    
    # --- Spectral Clustering ---
    # Το Spectral δουλεύει καλύτερα αν ορίσουμε affinity='nearest_neighbors'
    spectral = SpectralClustering(n_clusters=n_classes, 
                                  affinity='nearest_neighbors', 
                                  n_neighbors=10,
                                  random_state=42, 
                                  n_jobs=-1)
    y_spectral = spectral.fit_predict(X_pca)
    
    ari_spectral = adjusted_rand_score(y_true, y_spectral)
    sil_spectral = silhouette_score(X_pca, y_spectral)
    
    print(f"\nSpectral Clustering (k={n_classes}):")
    print(f"  -> ARI: {ari_spectral:.4f}")
    print(f"  -> Silhouette: {sil_spectral:.4f}")
    
    # Επιστρέφουμε το X_pca για να το χρησιμοποιήσεις στο visualization
    return X_pca, y_kmeans, y_spectral


# --- 4. VISUALIZATION (ROBUST) ---
def visualize_clusters(X_scaled, y_true, y_kmeans, y_spectral, config_name):
    print("\nGenerating visualizations...")
    
    # Decide Layout based on UMAP availability
    if HAS_UMAP:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        umap_row = axes[0]
        tsne_row = axes[1]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        tsne_row = axes
        umap_row = None

    labels_list = [y_true, y_kmeans, y_spectral]
    
    # --- UMAP ---
    if HAS_UMAP:
        print("  -> Running UMAP...")
        reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_umap = reducer_umap.fit_transform(X_scaled)
        
        titles_umap = ['Ground Truth (UMAP)', 'K-Means (UMAP)', 'Spectral (UMAP)']
        for ax, labels, title in zip(umap_row, labels_list, titles_umap):
            ax.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
            ax.set_title(title)
            ax.axis('off')

    # --- t-SNE ---
    print("  -> Running t-SNE...")
    reducer_tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding_tsne = reducer_tsne.fit_transform(X_scaled)
    
    titles_tsne = ['Ground Truth (t-SNE)', 'K-Means (t-SNE)', 'Spectral (t-SNE)']
    for ax, labels, title in zip(tsne_row, labels_list, titles_tsne):
        ax.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    output_file = f'clustering_visualization_{config_name}.png'
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to '{output_file}'")

# --- MAIN ---
if __name__ == "__main__":
    configurations = [
        # Hybrid configuration (best from classification)
        {
            'name': 'hybrid_100_33_100_4.1',
            'func': 'hybrid', 
            'harm_bins': 100, 
            'harm_range': 33, 
            'pol_bins': 100, 
            'pol_range': 4.1
        },
        # Polynomial configuration
        {
            'name': 'polynomial_100_4.1',
            'func': 'polynomial', 
            'bins': 100, 
            'range': 4.1
        },
        {
            'name': 'harmonic_300_33',
            'func': 'harmonic', 
            'bins': 300, 
            'range': 33
        },
    ]
    
    print("Loading ENZYMES dataset...")
    graphs, labels, node_labels_list = download_and_load_enzymes()
    print(f"Loaded {len(graphs)} graphs with {len(np.unique(labels))} classes")
    
    # Test each configuration
    for config in configurations:
        print("\n" + "="*70)
        print(f"Testing Configuration: {config['name']}")
        print("="*70)
        
        X = generate_embeddings(graphs, node_labels_list, config)
        X_scaled, y_kmeans, y_spectral = perform_clustering_analysis(X, labels)
        visualize_clusters(X_scaled, labels, y_kmeans, y_spectral, config['name'])