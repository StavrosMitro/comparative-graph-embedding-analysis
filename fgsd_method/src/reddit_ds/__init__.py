"""
REDDIT-MULTI-12K FGSD Package
"""

from .config import (
    DATASET_DIR, RESULTS_DIR, BATCH_SIZE, CACHE_DIR,
    PREANALYSIS_SAMPLE_SIZE, MAX_NODES_FOR_PREANALYSIS,
    GraphRecord, OptimalParams
)
from .data_loader import (
    ensure_dataset_ready, load_metadata, iter_graph_batches, load_all_graphs
)
from .preanalysis import (
    compute_spectral_distances_sampled,
    determine_optimal_params_and_bins,
    run_sampled_preanalysis,
    load_preanalysis_from_cache,
    save_preanalysis_to_cache,
    load_default_params
)
from .classification import (
    evaluate_classifier, get_classifiers, get_classifiers_tuned_or_default,
    print_dimension_analysis_summary, print_summary,
    generate_all_embeddings
)
from .stability import (
    perturb_graph_edges,
    perturb_graphs_batch,
    compute_embedding_stability,
    generate_embeddings_for_graphs,
    compute_classification_stability,
    print_stability_summary,
    DEFAULT_PERTURBATION_RATIOS
)
from .hyperparameter_search import (
    run_classifier_tuning,
    load_tuned_params,
    get_tuned_classifiers
)

__all__ = [
    # Config
    'DATASET_DIR', 'RESULTS_DIR', 'BATCH_SIZE', 'CACHE_DIR',
    'PREANALYSIS_SAMPLE_SIZE', 'MAX_NODES_FOR_PREANALYSIS',
    'GraphRecord', 'OptimalParams',
    # Data loading
    'ensure_dataset_ready', 'load_metadata', 'iter_graph_batches', 'load_all_graphs',
    # Pre-analysis
    'compute_spectral_distances_sampled',
    'determine_optimal_params_and_bins', 'run_sampled_preanalysis',
    'load_preanalysis_from_cache', 'save_preanalysis_to_cache',
    'load_default_params',
    # Classification
    'evaluate_classifier', 'get_classifiers', 'get_classifiers_tuned_or_default',
    'print_dimension_analysis_summary', 'print_summary',
    'generate_all_embeddings',
    # Stability
    'perturb_graph_edges', 'perturb_graphs_batch',
    'compute_embedding_stability', 'generate_embeddings_for_graphs',
    'compute_classification_stability', 'print_stability_summary',
    'DEFAULT_PERTURBATION_RATIOS',
    # Hyperparameter tuning
    'run_classifier_tuning', 'load_tuned_params', 'get_tuned_classifiers',
]
