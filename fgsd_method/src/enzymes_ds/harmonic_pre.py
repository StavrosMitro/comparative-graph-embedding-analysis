import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import urllib.request
import zipfile
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
# --- 1. DATA LOADING (Ο Κώδικάς σου) ---

def download_and_load_enzymes():
    """Download and load ENZYMES dataset from TU Dortmund."""
    data_dir = '/tmp/ENZYMES'
    os.makedirs(data_dir, exist_ok=True)
    base_url = 'https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip'
    zip_path = os.path.join(data_dir, 'ENZYMES.zip')

    if not os.path.exists(os.path.join(data_dir, 'ENZYMES')):
        print("Downloading ENZYMES dataset...")
        urllib.request.urlretrieve(base_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Download complete.")

    dataset_path = os.path.join(data_dir, 'ENZYMES')
    graph_indicator = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_graph_indicator.txt'), dtype=int)
    edges = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_A.txt'), dtype=int, delimiter=',')
    graph_labels = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_graph_labels.txt'), dtype=int)
    num_graphs = len(graph_labels)
    graphs = [nx.Graph() for _ in range(num_graphs)]

    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graphs[graph_id - 1].add_node(node_id)

    for edge in edges:
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]
        graphs[graph_id - 1].add_edge(node1, node2)

    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]
    labels = graph_labels - 1
    return graphs, labels


def compute_raw_spectral_distances(graphs):
    """
    Υπολογίζει όλες τις φασματικές αποστάσεις για όλους τους γράφους
    χωρίς να φτιάξει ιστογράμματα.
    """
    print(f"Computing spectral distances for {len(graphs)} graphs...")
    all_distances = []
    node_counts = []

    for i, G in enumerate(graphs):
        if G.number_of_nodes() < 2:
            continue
        node_counts.append(G.number_of_nodes())
        try:
            # Υπολογισμός Normalized Laplacian
            # Το np.asarray εξασφαλίζει ότι δεν θα είναι 'matrix' object
            L = np.asarray(nx.normalized_laplacian_matrix(G).todense())

            # Υπολογισμός Harmonic Distance (pinv = 1/λ)
            fL = np.linalg.pinv(L)
            ones = np.ones(L.shape[0])

            # Τύπος: S(x,y) = fL_xx + fL_yy - 2*fL_xy
            S = np.outer(np.diag(fL), ones) + np.outer(ones, np.diag(fL)) - 2 * fL

            # Σιγουρευόμαστε ξανά ότι το S είναι απλό array
            S = np.asarray(S)

            # Παίρνουμε μόνο το άνω τρίγωνο
            triu_indices = np.triu_indices_from(S, k=1)
            distances = S[triu_indices]

            # Flatten για να σιγουρευτούμε ότι είναι 1D και tolist για καθαρά Python floats
            all_distances.extend(distances.flatten().tolist())
        except Exception as e:
            print(f"Error in graph {i}: {e}")
            import traceback
            traceback.print_exc()

    return np.array(all_distances), np.array(node_counts)


def analyze_and_visualize(distances, node_counts, save_path='fgsd_analysis.png'):
    """
    Εκτυπώνει στατιστικά και φτιάχνει διαγράμματα.
    """
    print("\n" + "="*60)
    print("PRE-ANALYSIS RESULTS FOR ENZYMES")
    print("="*60)

    # --- A. GRAPH SIZES ---
    print(f"Graph Size Stats (Nodes): Mean={np.mean(node_counts):.1f}, Max={np.max(node_counts)}")

    # --- B. RANGE ANALYSIS ---
    min_val = np.min(distances)
    max_val = np.max(distances)
    mean_val = np.mean(distances)
    p95 = np.percentile(distances, 95)
    p99 = np.percentile(distances, 99)

    print("\n1. SPECTRAL DISTANCE VALUES (For choosing 'hist_range')")
    print(f" Min: {min_val:.4f}")
    print(f" Mean: {mean_val:.4f}")
    print(f" Max: {max_val:.4f} <-- Absolute maximum")
    print(f" 95th Percentile: {p95:.4f}")
    print(f" 99th Percentile: {p99:.4f} <-- RECOMMENDED Range Cutoff")

    # --- C. SPARSITY ANALYSIS ---
    print("\n2. SPARSITY CHECK (For choosing 'hist_bins')")
    print(" (Simulating binning on the entire dataset distribution)")

    # Χρησιμοποιούμε το 99th percentile ως υποθετικό range
    sim_range = p99
    test_bins = [20, 50, 100, 200, 500]
    print(f" Assume Range = [0, {sim_range:.2f}]")
    print(f" {'Bins':<10} | {'Avg Value/Bin':<15} | {'Sparsity Risk'}")
    print(" " + "-"*45)

    for b in test_bins:
        # Φτιάχνουμε ένα ιστόγραμμα με όλα τα data
        hist, _ = np.histogram(distances, bins=b, range=(0, sim_range))
        # Πόσα δεδομένα πέφτουν κατά μέσο όρο σε κάθε bin;
        # Αν είναι πολύ λίγα, το vector θα είναι αραιό ανά γράφο.
        # Προσοχή: Εδώ διαιρούμε με το πλήθος των γράφων για να δούμε
        # πόσα hits ανά γράφο θα έχουμε στο vector.
        avg_hits_per_graph = np.sum(hist) / len(node_counts) / b
        risk = "Low"
        if avg_hits_per_graph < 1.0: risk = "Medium"
        if avg_hits_per_graph < 0.1: risk = "HIGH (Too Sparse)"
        print(f" {b:<10} | {avg_hits_per_graph:.4f}{'':<10} | {risk}")

    # --- D. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Graph Sizes
    ax1.hist(node_counts, bins=30, color='skyblue', edgecolor='black')
    ax1.set_title("Distribution of Graph Sizes (Nodes)")
    ax1.set_xlabel("Number of Nodes")
    ax1.set_ylabel("Count")

    # Plot 2: Spectral Distances
    # Κόβουμε τα outliers για να φαίνεται καλά το γράφημα
    viz_data = distances[distances <= np.percentile(distances, 99.5)]
    ax2.hist(viz_data, bins=100, color='salmon', edgecolor='black', alpha=0.7)
    ax2.axvline(p95, color='red', linestyle='--', label=f'95th % ({p95:.1f})')
    ax2.axvline(p99, color='darkred', linestyle='-', label=f'99th % ({p99:.1f})')
    ax2.set_title("Distribution of Spectral Distances (Harmonic)")
    ax2.set_xlabel("Spectral Distance Value")
    ax2.set_ylabel("Frequency (All pairs)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nAnalysis plot saved to: {save_path}")
    print("="*60)


if __name__ == "__main__":
    # 1. Load
    graphs, labels = download_and_load_enzymes()
    # 2. Compute Raw Stats
    distances, node_counts = compute_raw_spectral_distances(graphs)
    # 3. Visualize & Guide
    analyze_and_visualize(distances, node_counts, save_path='enzymes_preanalysis.png')