"""
Data loading utilities for IMDB-MULTI dataset.
"""

import os
import urllib.request
import zipfile
from typing import List, Tuple

import numpy as np
import networkx as nx

from .config import DATASET_DIR


def ensure_dataset_ready() -> str:
    """Ensure the dataset is downloaded and extracted."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    base_url = 'https://www.chrsmrrs.com/graphkerneldatasets/IMDB-MULTI.zip'
    zip_path = os.path.join(DATASET_DIR, 'IMDB-MULTI.zip')
    dataset_path = os.path.join(DATASET_DIR, 'IMDB-MULTI')
    
    if not os.path.exists(dataset_path):
        print("Downloading IMDB-MULTI dataset...")
        urllib.request.urlretrieve(base_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("Download complete.")
    return dataset_path


def load_all_graphs(dataset_dir: str = DATASET_DIR) -> Tuple[List[nx.Graph], np.ndarray]:
    """Load all graphs into memory."""
    print("Loading IMDB-MULTI graphs...")
    dataset_path = os.path.join(dataset_dir, 'IMDB-MULTI')
    
    # Read graph indicator
    graph_indicator = np.loadtxt(
        os.path.join(dataset_path, 'IMDB-MULTI_graph_indicator.txt'), 
        dtype=int
    )
    
    # Read edges
    edges = np.loadtxt(
        os.path.join(dataset_path, 'IMDB-MULTI_A.txt'), 
        dtype=int, 
        delimiter=','
    )
    
    # Read graph labels
    graph_labels = np.loadtxt(
        os.path.join(dataset_path, 'IMDB-MULTI_graph_labels.txt'), 
        dtype=int
    )
    
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
    
    print(f"Loaded {len(graphs)} graphs with {len(np.unique(labels))} classes")
    return graphs, labels
