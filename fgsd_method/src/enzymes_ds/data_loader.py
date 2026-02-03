"""
Data loading utilities for ENZYMES dataset.
"""

import os
import urllib.request
import zipfile
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx

try:
    # Προσπάθεια relative import (αν τρέχει ως πακέτο)
    from .config import DATASET_DIR
except ImportError:
    # Fallback σε absolute import (αν τρέχει ως script)
    try:
        from config import DATASET_DIR
    except ImportError:
        # Αν δεν υπάρχει καν το config.py, ορίζουμε εδώ έναν φάκελο data
        import os
        DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def ensure_dataset_ready() -> str:
    """Ensure the dataset is downloaded and extracted."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    base_url = 'https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip'
    zip_path = os.path.join(DATASET_DIR, 'ENZYMES.zip')
    dataset_path = os.path.join(DATASET_DIR, 'ENZYMES')
    
    if not os.path.exists(dataset_path):
        print("Downloading ENZYMES dataset...")
        urllib.request.urlretrieve(base_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("Download complete.")
    return dataset_path


def load_all_graphs(dataset_dir: str = DATASET_DIR) -> Tuple[List[nx.Graph], np.ndarray, Optional[List[np.ndarray]]]:
    """
    Load all graphs into memory, including node labels if available.
    
    Returns:
        Tuple of (graphs, labels, node_labels_list)
        node_labels_list is a list of arrays, one per graph
    """
    print("Loading ENZYMES graphs...")
    dataset_path = os.path.join(dataset_dir, 'ENZYMES')
    
    # Read graph indicator
    graph_indicator = np.loadtxt(
        os.path.join(dataset_path, 'ENZYMES_graph_indicator.txt'), 
        dtype=int
    )
    
    # Read edges
    edges = np.loadtxt(
        os.path.join(dataset_path, 'ENZYMES_A.txt'), 
        dtype=int, 
        delimiter=','
    )
    
    # Read graph labels
    graph_labels = np.loadtxt(
        os.path.join(dataset_path, 'ENZYMES_graph_labels.txt'), 
        dtype=int
    )
    
    # Read node labels (ENZYMES has these)
    node_labels_path = os.path.join(dataset_path, 'ENZYMES_node_labels.txt')
    if os.path.exists(node_labels_path):
        node_labels_raw = np.loadtxt(node_labels_path, dtype=int)
    else:
        node_labels_raw = None
    
    # Build NetworkX graphs
    num_graphs = len(graph_labels)
    graphs = [nx.Graph() for _ in range(num_graphs)]
    node_labels_list = []
    
    # Add nodes
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graphs[graph_id - 1].add_node(node_id)
    
    # Split node labels by graph
    if node_labels_raw is not None:
        current_idx = 0
        for i in range(1, num_graphs + 1):
            count = np.sum(graph_indicator == i)
            labels_of_graph = node_labels_raw[current_idx:current_idx + count]
            node_labels_list.append(labels_of_graph)
            current_idx += count
    else:
        node_labels_list = None
    
    # Add edges
    for edge in edges:
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]
        graphs[graph_id - 1].add_edge(node1, node2)
    
    # Relabel nodes to be contiguous starting from 0
    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]
    
    # Convert labels to 0-indexed
    labels = graph_labels - 1
    
    print(f"Loaded {len(graphs)} graphs with {len(np.unique(labels))} classes")
    return graphs, labels, node_labels_list


def create_node_label_features(node_labels_list: List[np.ndarray]) -> np.ndarray:
    """Creates a histogram of node labels for each graph."""
    # ...creates bag-of-words style features from node chemical types
    if node_labels_list is None:
        return None
    
    all_labels = np.concatenate(node_labels_list)
    unique_labels = np.unique(all_labels)
    n_unique = len(unique_labels)
    min_lbl, max_lbl = min(unique_labels), max(unique_labels)
    
    features = []
    for labels in node_labels_list:
        hist, _ = np.histogram(labels, bins=n_unique, range=(min_lbl, max_lbl + 1))
        features.append(hist)
    
    return np.array(features)
