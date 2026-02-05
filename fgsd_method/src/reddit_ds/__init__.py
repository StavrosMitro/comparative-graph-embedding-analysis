"""
REDDIT-MULTI-12K FGSD Package
"""

from .config import (
    DATASET_DIR, RESULTS_DIR, BATCH_SIZE, CACHE_DIR,
    PREANALYSIS_SAMPLE_SIZE, MAX_NODES_FOR_PREANALYSIS,
    GraphRecord, OptimalParams
)
from .data_loader import (
    ensure_dataset_ready, load_all_graphs, load_metadata
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
    print_dimension_analysis_summary, print_summary,
    generate_all_embeddings
)
from .stability import (
    perturb_graph_edges,
    perturb_graphs_batch,
    compute_embedding_stability,
    compute_classification_accuracy,
    generate_all_embeddings_batchwise,
    run_stability_analysis,
    print_stability_summary,
    load_control_embedding,
    save_control_embedding,
    load_best_configs_from_csv,
    PERTURBATION_RATIOS,
    PERTURBATION_MODES,
)

__all__ = [
    # Config
    'DATASET_DIR', 'RESULTS_DIR', 'BATCH_SIZE', 'CACHE_DIR',
    'PREANALYSIS_SAMPLE_SIZE', 'MAX_NODES_FOR_PREANALYSIS',
    'GraphRecord', 'OptimalParams',
    # Data loading
    'ensure_dataset_ready', 'load_all_graphs', 'load_metadata',
    # Pre-analysis
    'compute_spectral_distances_sampled',
    'determine_optimal_params_and_bins', 'run_sampled_preanalysis',
    'load_preanalysis_from_cache', 'save_preanalysis_to_cache',
    'load_default_params',
    # Classification
    'evaluate_classifier', 'get_classifiers',
    'print_dimension_analysis_summary', 'print_summary',
    'generate_all_embeddings',
    # Stability
    'perturb_graph_edges', 'perturb_graphs_batch',
    'compute_embedding_stability', 'compute_classification_accuracy',
    'generate_all_embeddings_batchwise',
    'run_stability_analysis', 'print_stability_summary',
    'load_control_embedding', 'save_control_embedding',
    'load_best_configs_from_csv',
    'PERTURBATION_RATIOS', 'PERTURBATION_MODES',
]
