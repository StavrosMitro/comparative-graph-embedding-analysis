"""
Data loading utilities for REDDIT-MULTI-12K dataset.
"""

import os
import urllib.request
import zipfile
from typing import List, Tuple, Iterator, Optional

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from .config import DATASET_DIR, BATCH_SIZE, GraphRecord


def ensure_dataset_ready() -> str:
    """Ensure the dataset is downloaded and extracted."""
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


def load_metadata(dataset_dir: str = DATASET_DIR) -> List[GraphRecord]:
    """Load graph metadata without loading full graphs into memory."""
    dataset_path = os.path.join(dataset_dir, 'REDDIT-MULTI-12K')
    
    graph_indicator = pd.read_csv(
        os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_indicator.txt'), 
        header=None
    )[0].values
    graph_labels = pd.read_csv(
        os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_labels.txt'), 
        header=None
    )[0].values
    
    records = []
    num_graphs = len(graph_labels)
    
    for gid in range(1, num_graphs + 1):
        mask = graph_indicator == gid
        indices = np.where(mask)[0]
        if len(indices) > 0:
            node_start = indices[0]
            node_end = indices[-1] + 1
        else:
            node_start = node_end = 0
        
        records.append(GraphRecord(
            graph_id=gid,
            label=graph_labels[gid - 1] - 1,
            node_start=node_start,
            node_end=node_end
        ))
    
    return records


def iter_graph_batches(
    dataset_dir: str = DATASET_DIR, 
    batch_size: int = BATCH_SIZE, 
    records: Optional[List[GraphRecord]] = None
) -> Iterator[Tuple[List[nx.Graph], np.ndarray, List[int]]]:
    """Iterate over graphs in batches."""
    dataset_path = os.path.join(dataset_dir, 'REDDIT-MULTI-12K')
    
    if records is None:
        records = load_metadata(dataset_dir)
    
    edges_df = pd.read_csv(
        os.path.join(dataset_path, 'REDDIT-MULTI-12K_A.txt'),
        header=None, names=['src', 'dst']
    )
    graph_indicator = pd.read_csv(
        os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_indicator.txt'),
        header=None
    )[0].values
    
    for i in range(0, len(records), batch_size):
        batch_records = records[i:i + batch_size]
        graphs = []
        labels = []
        gids = []
        
        for rec in batch_records:
            G = nx.Graph()
            node_ids = np.where(graph_indicator == rec.graph_id)[0] + 1
            for nid in node_ids:
                G.add_node(nid)
            
            mask = graph_indicator[edges_df['src'].values - 1] == rec.graph_id
            graph_edges = edges_df[mask][['src', 'dst']].values
            G.add_edges_from(graph_edges)
            G = nx.convert_node_labels_to_integers(G)
            
            graphs.append(G)
            labels.append(rec.label)
            gids.append(rec.graph_id)
        
        yield graphs, np.array(labels), gids


def load_all_graphs(dataset_dir: str = DATASET_DIR) -> Tuple[List[nx.Graph], np.ndarray]:
    """Load all graphs into memory."""
    print("Loading all graphs into memory...")
    all_graphs = []
    all_labels = []
    
    for graphs, labels, _ in tqdm(iter_graph_batches(dataset_dir, BATCH_SIZE), 
                                   desc="Loading graphs"):
        all_graphs.extend(graphs)
        all_labels.extend(labels)
    
    return all_graphs, np.array(all_labels)
