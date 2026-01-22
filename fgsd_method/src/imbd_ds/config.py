"""
Configuration constants and data structures for IMDB-MULTI experiments.
"""

import os
from dataclasses import dataclass

# =============================================================================
# PATHS
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
DATASET_DIR = '/tmp/IMDB-MULTI'
RESULTS_DIR = os.path.join(PARENT_DIR, 'results')
CACHE_DIR = os.path.join(PARENT_DIR, 'cache')

# =============================================================================
# PROCESSING CONSTANTS
# =============================================================================
BATCH_SIZE = 500
PREANALYSIS_SAMPLE_SIZE = 500  # IMDB is smaller, can use more
MAX_NODES_FOR_PREANALYSIS = 300

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class GraphRecord:
    """Metadata for a single graph."""
    graph_id: int
    label: int
    node_start: int
    node_end: int

@dataclass
class OptimalParams:
    """Optimal parameters determined from pre-analysis."""
    func_type: str
    bins: int
    range_val: float
    p99: float
    recommended_bins: int
