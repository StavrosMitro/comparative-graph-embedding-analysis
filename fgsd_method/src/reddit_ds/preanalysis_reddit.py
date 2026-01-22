import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import urllib.request
import zipfile
import warnings

warnings.filterwarnings('ignore')

# --- 1. DATA LOADING ---
def download_and_load_reddit():
    """Download and load REDDIT-MULTI-12K dataset from TU Dortmund."""
    data_dir = '/tmp/REDDIT-MULTI-12K'
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = 'https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-MULTI-12K.zip'
    zip_path = os.path.join(data_dir, 'REDDIT-MULTI-12K.zip')
    
    # Download if not exists
    if not os.path.exists(os.path.join(data_dir, 'REDDIT-MULTI-12K')):
        print("Downloading REDDIT-MULTI-12K dataset...")
        urllib.request.urlretrieve(base_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Download complete.")
    
    # Parse dataset files
    dataset_path = os.path.join(data_dir, 'REDDIT-MULTI-12K')
    
    # Read graph indicator
    graph_indicator = np.loadtxt(os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_indicator.txt'), dtype=int)
    
    # Read edges
    edges = np.loadtxt(os.path.join(dataset_path, 'REDDIT-MULTI-12K_A.txt'), dtype=int, delimiter=',')
    
    # Read graph labels
    graph_labels = np.loadtxt(os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_labels.txt'), dtype=int)
    
    # Build NetworkX graphs
    num_graphs = len(graph_labels)
    graphs = [nx.Graph() for _ in range(num_graphs)]
    
    # Add nodes
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graphs[graph_id - 1].add_node(node_id)
    
    # Add edges
    for edge in edges:
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]
        graphs[graph_id - 1].add_edge(node1, node2)
    
    # Relabel nodes to be contiguous starting from 0
    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]
    
    # Convert labels to 0-indexed
    labels = graph_labels - 1
    
    return graphs, labels

# --- 2. CALCULATION CORE ---
def compute_spectral_distances_for_func(graphs, func_type='harmonic'):
    """
    Computes spectral distances based on function f(lambda).
    """
    print(f"Computing spectral distances using func='{func_type}'...")
    all_distances = []
    node_counts = []
    
    for i, G in enumerate(graphs):
        if G.number_of_nodes() < 2:
            continue
        
        node_counts.append(G.number_of_nodes())
        
        try:
            # 1. Normalized Laplacian
            L = np.asarray(nx.normalized_laplacian_matrix(G).todense())
            
            # 2. Eigen-decomposition
            w, v = np.linalg.eigh(L)
            
            # 3. Apply function f(lambda)
            if func_type == 'harmonic':
                # f(lambda) = 1/lambda
                with np.errstate(divide='ignore', invalid='ignore'):
                    func_w = np.where(w > 1e-9, 1.0 / w, 0)
                    
            elif func_type == 'polynomial':
                # f(lambda) = lambda^2
                func_w = w ** 2
                
            elif func_type == 'biharmonic':
                # f(lambda) = 1/lambda^2
                with np.errstate(divide='ignore', invalid='ignore'):
                    func_w = np.where(w > 1e-9, 1.0 / (w**2), 0)
            else:
                raise ValueError("Unknown function type")

            # 4. Reconstruct f(L)
            fL = v @ np.diag(func_w) @ v.T
            
            # 5. Compute Distances
            ones = np.ones(L.shape[0])
            S = np.outer(np.diag(fL), ones) + np.outer(ones, np.diag(fL)) - 2 * fL
            S = np.asarray(S)
            
            # Extract upper triangle distances
            triu_indices = np.triu_indices_from(S, k=1)
            distances = S[triu_indices]
            
            all_distances.extend(distances.flatten().tolist())
            
        except Exception as e:
            # print(f"Error in graph {i}: {e}")
            pass

    return np.array(all_distances), np.array(node_counts)

# --- 3. ANALYSIS & VISUALIZATION ---
def analyze_and_visualize(distances, node_counts, func_type):
    """
    Prints statistics and saves the plot.
    """
    save_path = f'reddit_analysis_{func_type}.png'
    
    print("\n" + "="*60)
    print(f"PRE-ANALYSIS RESULTS FOR: {func_type.upper()}")
    print("="*60)
    
    # --- RANGE ANALYSIS ---
    min_val = np.min(distances)
    max_val = np.max(distances)
    p95 = np.percentile(distances, 95)
    p99 = np.percentile(distances, 99)
    
    print(f"1. SPECTRAL DISTANCE VALUES (hist_range)")
    print(f"   Min:  {min_val:.4f}")
    print(f"   Max:  {max_val:.4f}")
    print(f"   95th Percentile: {p95:.4f}")
    print(f"   99th Percentile: {p99:.4f}  <-- RECOMMENDED Range")
    
    # --- SPARSITY ANALYSIS ---
    print("\n2. SPARSITY CHECK (hist_bins)")
    
    sim_range = p99
    test_bins = [50, 100, 200, 300, 500]
    
    print(f"   Assume Range = [0, {sim_range:.2f}]")
    print(f"   {'Bins':<10} | {'Avg Value/Bin':<15} | {'Sparsity Risk'}")
    print("   " + "-"*45)
    
    for b in test_bins:
        hist, _ = np.histogram(distances, bins=b, range=(0, sim_range))
        avg_hits_per_graph = np.sum(hist) / len(node_counts) / b
        
        risk = "Low"
        if avg_hits_per_graph < 1.0: risk = "Medium"
        if avg_hits_per_graph < 0.1: risk = "HIGH (Too Sparse)"
        
        print(f"   {b:<10} | {avg_hits_per_graph:.4f}{'':<10} | {risk}")

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    viz_cutoff = np.percentile(distances, 99.5)
    viz_data = distances[distances <= viz_cutoff]
    
    ax.hist(viz_data, bins=100, color='orange', edgecolor='black', alpha=0.7)
    ax.axvline(p95, color='blue', linestyle='--', label=f'95% ({p95:.1f})')
    ax.axvline(p99, color='darkred', linestyle='-', label=f'99% ({p99:.1f})')
    
    ax.set_title(f"REDDIT-MULTI-12K Distribution: {func_type} distances")
    ax.set_xlabel("Spectral Distance Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")

if __name__ == "__main__":
    # 1. Load Data
    graphs, labels = download_and_load_reddit()
    print(f"Loaded {len(graphs)} graphs.")
    
    # 2. Define functions to test
    functions_to_test = ['harmonic', 'polynomial', 'biharmonic']
    
    # 3. Run Loop
    for func in functions_to_test:
        # Sampling for speed if dataset is too large for full analysis
        sample_size = 1000
        if len(graphs) > sample_size:
            print(f"Sampling {sample_size} graphs for analysis...")
            indices = np.random.choice(len(graphs), sample_size, replace=False)
            graphs_sample = [graphs[i] for i in indices]
        else:
            graphs_sample = graphs
            
        distances, node_counts = compute_spectral_distances_for_func(graphs_sample, func_type=func)
        analyze_and_visualize(distances, node_counts, func_type=func)
