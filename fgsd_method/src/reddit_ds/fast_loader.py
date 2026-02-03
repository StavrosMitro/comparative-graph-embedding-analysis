"""
Optimized Batched Data loader for REDDIT-MULTI-12K.
"""

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

try:
    from .config import DATASET_DIR
except ImportError:
    try:
        from config import DATASET_DIR
    except ImportError:
        import os
        DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def ensure_dataset_ready() -> str:
    os.makedirs(DATASET_DIR, exist_ok=True)
    base_url = 'https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-MULTI-12K.zip'
    zip_path = os.path.join(DATASET_DIR, 'REDDIT-MULTI-12K.zip')
    dataset_path = os.path.join(DATASET_DIR, 'REDDIT-MULTI-12K')
    
    if not os.path.exists(dataset_path):
        print("Downloading REDDIT-MULTI-12K dataset...")
        urllib.request.urlretrieve(base_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("Download complete.")
    return dataset_path

def load_raw_data(dataset_dir: str = DATASET_DIR):
    """Loads raw CSV data efficiently."""
    dataset_path = os.path.join(dataset_dir, 'REDDIT-MULTI-12K')
    indicator_path = os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_indicator.txt')
    edges_path = os.path.join(dataset_path, 'REDDIT-MULTI-12K_A.txt')
    labels_path = os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_labels.txt')
    
    graph_indicator = pd.read_csv(indicator_path, header=None)[0].values
    edges = pd.read_csv(edges_path, header=None, names=['src', 'dst']).values
    graph_labels = pd.read_csv(labels_path, header=None)[0].values
    
    return graph_indicator, edges, graph_labels

def graph_batch_generator(dataset_dir: str = DATASET_DIR, batch_size: int = 1000):
    """
    Yields LISTS of graphs (Batches).
    Uses ~500MB RAM for batch_size=1000.
    """
    # 1. Load Raw Data
    graph_indicator, edges, graph_labels = load_raw_data(dataset_dir)
    num_graphs = len(graph_labels)
    
    # 2. Pre-process indices (Vectorized)
    node_counts = np.bincount(graph_indicator)
    node_offsets = np.cumsum(node_counts)
    graph_start_indices = np.roll(node_offsets, 1)
    graph_start_indices[1] = 0
    
    edge_graph_ids = graph_indicator[edges[:, 0] - 1]
    sort_idx = np.argsort(edge_graph_ids)
    edges = edges[sort_idx]
    edge_graph_ids = edge_graph_ids[sort_idx]
    
    unique_gids = np.arange(1, num_graphs + 1)
    edge_boundaries = np.searchsorted(edge_graph_ids, unique_gids, side='right')
    edge_starts = np.concatenate(([0], edge_boundaries[:-1]))
    edge_ends = edge_boundaries
    
    print(f"Starting Batch Generator ({batch_size} graphs per batch)...")
    
    # 3. Batch Loop
    for i in range(0, num_graphs, batch_size):
        # Determine batch range
        end_i = min(i + batch_size, num_graphs)
        batch_graphs = []
        batch_labels = []
        batch_gids = []
        
        # Build graphs for this batch
        for j in range(i, end_i):
            gid = j + 1
            n_nodes = node_counts[gid]
            node_offset = graph_start_indices[gid]
            
            idx_start = edge_starts[j]
            idx_end = edge_ends[j]
            local_edges = edges[idx_start:idx_end]
            
            G = nx.Graph()
            if n_nodes > 0:
                G.add_nodes_from(range(n_nodes))
                if len(local_edges) > 0:
                    normalized_edges = local_edges - (node_offset + 1)
                    G.add_edges_from(normalized_edges)
            
            batch_graphs.append(G)
            batch_labels.append(graph_labels[j] - 1)
            batch_gids.append(gid)
        
        # Yield the full batch (consumes memory temporarily)
        yield batch_graphs, np.array(batch_labels), np.array(batch_gids)
        
        # Explicit cleanup hints (optional, Python handles ref counting)
        del batch_graphs
        del batch_labels