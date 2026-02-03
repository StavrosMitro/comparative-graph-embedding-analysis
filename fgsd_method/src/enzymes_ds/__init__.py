"""
ENZYMES FGSD Package
"""

from .config import (
    DATASET_DIR, RESULTS_DIR, BATCH_SIZE, CACHE_DIR,
    PREANALYSIS_SAMPLE_SIZE, MAX_NODES_FOR_PREANALYSIS,
    GraphRecord, OptimalParams
)
from .data_loader import (
    ensure_dataset_ready, load_all_graphs, create_node_label_features
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
    evaluate_classifier, get_classifiers,
    run_dimension_analysis, run_final_classification,
    print_dimension_analysis_summary, print_summary
)
from .clustering import (
    run_clustering_experiment,
    print_clustering_summary,
    generate_embeddings,
    perform_clustering_with_params,
    evaluate_clustering,
    visualize_clusters,
    run_clustering_grid_search,
    apply_normalization
)
from .stability import (
    perturb_graph_edges,
    perturb_graphs_batch,
    compute_embedding_stability,
    generate_embeddings_for_graphs,
    run_stability_analysis,
    compute_classification_stability,
    print_stability_summary,
    DEFAULT_PERTURBATION_RATIOS
)

__all__ = [
    # Config
    'DATASET_DIR', 'RESULTS_DIR', 'BATCH_SIZE', 'CACHE_DIR',
    'PREANALYSIS_SAMPLE_SIZE', 'MAX_NODES_FOR_PREANALYSIS',
    'GraphRecord', 'OptimalParams',
    # Data loading
    'ensure_dataset_ready', 'load_all_graphs', 'create_node_label_features',
    # Pre-analysis
    'compute_spectral_distances_sampled',
    'determine_optimal_params_and_bins', 'run_sampled_preanalysis',
    'load_preanalysis_from_cache', 'save_preanalysis_to_cache',
    'load_default_params',
    # Classification
    'evaluate_classifier', 'get_classifiers',
    'run_dimension_analysis', 'run_final_classification',
    'print_dimension_analysis_summary', 'print_summary',
    # Clustering
    'generate_embeddings', 'perform_clustering_with_params',
    'evaluate_clustering', 'visualize_clusters',
    'run_clustering_experiment', 'print_clustering_summary',
    # Stability
    'perturb_graph_edges', 'perturb_graphs_batch',
    'compute_embedding_stability', 'generate_embeddings_for_graphs',
    'run_stability_analysis', 'compute_classification_stability',
    'print_stability_summary',
    'DEFAULT_PERTURBATION_RATIOS',
]
