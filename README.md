# Graph Embedding Evaluation Benchmark

A comprehensive evaluation and comparison of **three graph embedding methods** across **three benchmark datasets**, covering classification, clustering, and stability analysis.

## Methods

| Method | Type | Description |
|--------|------|-------------|
| **FGSD** (Flexible Graph Spectral Descriptor) | Unsupervised | Spectral graph embedding using eigendecomposition of the Normalized Laplacian with configurable kernel functions (`harmonic`, `polynomial`, `biharmonic`) |
| **Graph2Vec** (g2v) | Unsupervised | Learns graph-level embeddings via Weisfeiler-Lehman subtree patterns + PV-DBOW (Doc2Vec-style) training |
| **GIN** (Graph Isomorphism Network) | Supervised | Graph neural network that learns node/graph representations through message passing with learnable aggregation |

## Datasets

| Dataset | Graphs | Classes | Node Attributes | Source |
|---------|--------|---------|-----------------|--------|
| **ENZYMES** | 600 | 6 | Yes (18 features) | TU Dortmund |
| **IMDB-MULTI** | 1,500 | 3 | No | TU Dortmund |
| **REDDIT-MULTI-12K** | 11,929 | 11 | No | TU Dortmund |

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Project Structure](#2-project-structure)
3. [Exercise Tasks Overview](#3-exercise-tasks-overview)
4. [Task (a): Classification](#4-task-a-classification)
5. [Task (b): Clustering](#5-task-b-clustering)
6. [Task (c): Stability Analysis](#6-task-c-stability-analysis)
7. [Additional Analyses](#7-additional-analyses)
8. [Plotting & Visualization](#8-plotting--visualization)
9. [Full Run Commands Reference](#9-full-run-commands-reference)

---

## 1. Environment Setup

This project requires **two separate conda environments** — one for FGSD and one for Graph2Vec — due to conflicting dependency versions. The GIN notebooks require a GPU-enabled PyTorch environment.

### 1.1 FGSD Environment (Python 3.9, CPU)

Used for all scripts under `fgsd_method/`.

```bash
# Option A: Create from the conda environment file (recommended)
conda env create -f env_fgsd.yml
conda activate graphs

# Option B: Create manually with pip
conda create -n graphs python=3.9 -y
conda activate graphs
conda install pytorch cpuonly -c pytorch -y
pip install numpy>=1.21.0 networkx>=2.6.0 scikit-learn>=1.0.0 karateclub>=1.3.0 pandas>=1.3.0
pip install matplotlib umap-learn scipy memory_profiler
```

**Key packages:** `numpy 1.22`, `scikit-learn`, `networkx`, `karateclub` (provides the `Estimator` base class), `pandas`, `matplotlib`, `umap-learn`

### 1.2 Graph2Vec Environment (Python 3.x, CUDA 11.8)

Used for all scripts under `g2v/`.

```bash
# Create a separate environment
conda create -n g2v python=3.10 -y
conda activate g2v

# Install from the requirements file
pip install -r g2v/requirements.txt
```

Or install key packages manually:

```bash
pip install gensim==4.4.0 numpy==2.3.5 networkx==3.5 scikit-learn==1.8.0
pip install torch==2.7.1+cu118 torch-geometric==2.7.0 --extra-index-url https://download.pytorch.org/whl/cu118
pip install matplotlib umap-learn tqdm
```

**Key packages:** `gensim 4.4.0` (PV-DBOW), `torch + torch-geometric` (TUDataset loading), `CUDA 11.8`

### 1.3 GIN Environment (Python 3.x, GPU Required)

Used for the Jupyter notebooks under `GIN_model/`. Run just the notebook, it sets up its own enviroment.

**Note:** The GIN notebooks require a **GPU with CUDA support** for practical training times. They can run on CPU but will be significantly slower.

### Quick Environment Reference

| Task | Environment | Activate Command |
|------|-------------|------------------|
| FGSD (all datasets) | `graphs` | `conda activate graphs` |
| Graph2Vec (all datasets) | `g2v` | `conda activate g2v` |
| GIN notebooks | `graphs` (or `gin`) | `conda activate graphs` |

---

## 2. Project Structure

```
emb3/
├── README.md                          # This file
├── environment.yml                    # Conda env for GIN / general
├── env_fgsd.yml                       # Conda env for FGSD
├── requirements.txt                   # Pip requirements for FGSD
│
├── fgsd_method/                       # ══════ FGSD METHOD ══════
│   ├── src/
│   │   ├── fgsd.py                    # Core: FlexibleFGSD class (harmonic/biharmonic/polynomial)
│   │   ├── chebyshev_fgsd.py          # Fast Chebyshev polynomial approximation
│   │   ├── optimized_method.py        # Hybrid FGSD (concatenated embeddings)
│   │   ├── cross_validation_benchmark.py  # 10-fold × 30-repeat CV benchmark
│   │   │
│   │   ├── enzymes_ds/                # ── ENZYMES Dataset ──
│   │   │   ├── config.py              # Dataset constants & paths
│   │   │   ├── data_loader.py         # Graph loading with node labels
│   │   │   ├── classification_main.py # ★ Classification entry point
│   │   │   ├── classification.py      # SVM, RF, MLP classifiers
│   │   │   ├── clustering_main.py     # ★ Clustering entry point
│   │   │   ├── clustering.py          # K-Means, Spectral, t-SNE/UMAP
│   │   │   ├── preanalysis.py         # Optimal histogram range detection
│   │   │   ├── hyperparameter_search.py # Grid search for classifiers
│   │   │   ├── stability.py           # Perturbation & stability metrics
│   │   │   └── compare_hybrid_chebyshev.py # Exact vs Chebyshev timing
│   │   │
│   │   ├── imbd_ds/                   # ── IMDB-MULTI Dataset ──
│   │   │   ├── config.py              # Dataset constants
│   │   │   ├── data_loader.py         # Graph loading (no node labels)
│   │   │   ├── classification_main.py # ★ Classification entry point
│   │   │   ├── classification.py      # Classifiers (smaller MLP)
│   │   │   ├── clustering_main.py     # ★ Clustering entry point
│   │   │   ├── clustering.py          # Spectral-only clustering
│   │   │   ├── preanalysis.py         # Histogram range detection
│   │   │   ├── preanalysis_imdb.py    # Standalone preanalysis
│   │   │   ├── hyperparameter_search.py
│   │   │   ├── stability.py           # Edge perturbation stability
│   │   │   ├── evaluate_fgsd_imdb.py  # Standalone FGSD evaluation
│   │   │   └── chebyshev_imdb.py      # Chebyshev comparison
│   │   │
│   │   ├── reddit_ds/                 # ── REDDIT-MULTI-12K Dataset ──
│   │   │   ├── config.py              # Dataset constants
│   │   │   ├── data_loader.py         # Batch-wise graph loading
│   │   │   ├── fast_loader.py         # Optimized vectorized loader
│   │   │   ├── classification_main.py # ★ Classification entry point
│   │   │   ├── classification.py      # Batch-wise classification
│   │   │   ├── classification_reddit_full.py # Full standalone classification
│   │   │   ├── clustering_main.py     # ★ Clustering entry point
│   │   │   ├── clustering.py          # Clustering with biharmonic support
│   │   │   ├── clustering_targeted.py # Targeted 2-config clustering
│   │   │   ├── preanalysis.py         # Cached histogram ranges
│   │   │   ├── preanalysis_reddit.py  # Standalone preanalysis
│   │   │   ├── hyperparameter_search.py
│   │   │   ├── stability.py           # Batch-wise stability analysis
│   │   │   ├── generate_stability_embeddings.py # Pre-generate perturbed embeddings
│   │   │   ├── compare_chebyshev.py   # Chebyshev vs Exact comparison
│   │   │   ├── fast_compare.py        # Multi-kernel benchmark
│   │   │   └── memory_benchmark.py    # RAM profiling with tracemalloc
│   │   │
│   │   ├── cache/                     # Cached parameters & embeddings
│   │   └── results/                   # Output CSVs & analysis files
│   │
│   ├── plots/                         # Generated plots (organized by analysis type)
│   ├── plot_computational_analysis.py # Computational analysis plots
│   ├── plot_dimension_analysis.py     # Dimension vs accuracy plots
│   ├── plot_grid_search_analysis.py   # Grid search heatmaps
│   └── plot_stability_analysis.py     # Stability analysis plots
│
├── g2v/                               # ══════ GRAPH2VEC METHOD ══════
│   ├── requirements.txt               # G2V-specific dependencies
│   ├── enzymes_first.py               # ★ Train G2V on ENZYMES
│   ├── enzymes_second.py              # ★ Evaluate ENZYMES embeddings
│   ├── imdb_first.py                  # ★ Train G2V on IMDB-MULTI
│   ├── imdb_second.py                 # ★ Evaluate IMDB embeddings
│   ├── reddit_first.py                # ★ Train G2V on REDDIT-MULTI-12K
│   └── reddit_second.py               # ★ Evaluate REDDIT embeddings
│
└── GIN_model/                         # ══════ GIN METHOD ══════
    ├── GIN_enzymes_full_pipeline.ipynb # ★ Full GIN pipeline for ENZYMES
    ├── GIN_IMDB_full_pipeline.ipynb    # ★ Full GIN pipeline for IMDB
    └── GIN_REDDIT_full_pipeline.ipynb  # ★ Full GIN pipeline for REDDIT
```

---

## 3. Exercise Tasks Overview

The project implements the following evaluation tasks across all three methods (FGSD, Graph2Vec, GIN) and all three datasets (ENZYMES, IMDB-MULTI, REDDIT-MULTI-12K):

| Task | What is Evaluated | Key Metrics |
|------|-------------------|-------------|
| **(a) Classification** | Embedding quality for supervised downstream tasks | Accuracy, F1-score, AUC, training time, memory |
| **(b) Clustering** | Unsupervised structure discovery in embeddings | ARI (Adjusted Rand Index), t-SNE/UMAP plots |
| **(c) Stability** | Robustness of embeddings under graph perturbations | Cosine similarity, L2 distance, accuracy drop |

---

## 4. Task (a): Classification

> *Use embeddings as input features to train classifiers (SVM, MLP, etc.). Report accuracy, F1-score, AUC. Record training time, generation time, memory use. Vary embedding dimensions.*

### 4.1 FGSD Classification

All commands are run from `fgsd_method/src/` with the `graphs` environment activated.

```bash
conda activate graphs
cd fgsd_method/src
```

#### ENZYMES

```bash
# Basic classification (preanalysis → grid search → dimension analysis → classification)
python -m enzymes_ds.classification_main

# Full pipeline with stability analysis and classifier tuning
python -m enzymes_ds.classification_main --stability --tune-classifiers

# Skip grid search (use cached parameters)
python -m enzymes_ds.classification_main --skip-grid

# Use raw embeddings without preprocessing
python -m enzymes_ds.classification_main --raw-embeddings

# Force rerun everything (ignore caches)
python -m enzymes_ds.classification_main --force

# Disable node label features
python -m enzymes_ds.classification_main --no-node-labels

# Run stability analysis only (requires previous classification run)
python -m enzymes_ds.classification_main --stability-only
```

**CLI Flags:**

| Flag | Description |
|------|-------------|
| `--stability` | Include stability analysis in the pipeline |
| `--stability-only` | Run only stability (loads best config from previous results) |
| `--force` | Force rerun everything, ignoring cached parameters |
| `--no-node-labels` | Disable node label features |
| `--skip-grid` | Skip grid search, use cached/default parameters |
| `--tune-classifiers` | Run hyperparameter tuning for RF & SVM classifiers |
| `--raw-embeddings` | Use raw embeddings without StandardScaler preprocessing |

**Classifiers used:** SVM (C=100), Random Forest (1000 trees), MLP (1024→512→256→128)

**Output:** CSVs in `fgsd_method/src/results/`, plots in `fgsd_method/src/enzymes_ds/analysis_plots/`

#### IMDB-MULTI

```bash
# Basic classification
python -m imbd_ds.classification_main

# Full pipeline with stability and classifier tuning
python -m imbd_ds.classification_main --stability --tune-classifiers

# Raw embeddings mode
python -m imbd_ds.classification_main --raw-embeddings

# Run only stability analysis
python -m imbd_ds.classification_main --stability-only
```

**CLI Flags:** Same as ENZYMES except no `--no-node-labels` flag (IMDB has no node labels).

**Classifiers used:** SVM, Random Forest (500 trees), MLP (256→128→64)

#### REDDIT-MULTI-12K

```bash
# Basic classification
python -m reddit_ds.classification_main

# Full pipeline with stability and classifier tuning
python -m reddit_ds.classification_main --stability --tune-classifiers

# Force rerun preanalysis
python -m reddit_ds.classification_main --force

# Run only stability analysis
python -m reddit_ds.classification_main --stability-only
```

**CLI Flags:**

| Flag | Description |
|------|-------------|
| `--stability` | Include stability analysis |
| `--stability-only` | Run only stability analysis (loads best config from results) |
| `--force` | Force rerun preanalysis |
| `--tune-classifiers` | Run classifier hyperparameter tuning |

**Note:** Reddit also has a standalone full-dataset classification script:

```bash
cd fgsd_method/src/reddit_ds
python classification_reddit_full.py
```

**Classifiers used:** MLP (512→256→128), SVM, Random Forest

#### FGSD Cross-Validation Benchmark (All Datasets)

Runs a formal 10-fold stratified cross-validation × 30 repetitions across all three datasets using raw embeddings:

```bash
cd fgsd_method/src
python cross_validation_benchmark.py
```

This runs 6 benchmarks sequentially: ENZYMES (harmonic + polynomial), IMDB (harmonic + polynomial), REDDIT (harmonic + polynomial). Reports mean accuracy, standard deviation, 95% CI, IQR, and coefficient of variation.

---

### 4.2 Graph2Vec Classification

All commands are run from `g2v/` with the `g2v` environment activated.

Graph2Vec operates in **two phases**: first generate embeddings, then evaluate them.

```bash
conda activate g2v
cd g2v
```

#### ENZYMES

```bash
# Phase 1: Generate embeddings (WL subtree → PV-DBOW training)
python enzymes_first.py

# Phase 2: Evaluate (10-fold CV, 7 classifiers, clustering, stability, dimension sweep)
python enzymes_second.py
```

**Phase 1** performs: Feature-to-Label mapping → WL subtree extraction → PV-DBOW Graph2Vec training → saves embeddings to `g2v_embeddings_selected_epochs_named/`.

**Phase 2** evaluates with **7 classifiers:** Logistic Regression, Ridge Classifier, Linear SVC, SVM (RBF), KNN, Random Forest, MLP — each with hyperparameter grid search. Also runs clustering, dimension sweep, and stability analysis (if perturbed embeddings exist).

**Output:** `embedding_eval_results_merged/` with JSONL logs, CSV summaries, JSON results, PNG plots.

#### IMDB-MULTI

```bash
# Phase 1: Generate embeddings
python imdb_first.py

# Phase 2: Evaluate
python imdb_second.py
```

**Output:** `imdb_multi_embedding_eval_results/`

#### REDDIT-MULTI-12K

```bash
# Phase 1: Generate embeddings (RAM-safe, reads from disk export)
python reddit_first.py

# Phase 2: Evaluate
python reddit_second.py
```

**Note:** Reddit Phase 1 requires a pre-exported dataset in `./reddit_multi_12k_disk/` (with `metadata.tsv` + `graphs/*.edgelist`). The script handles this automatically via TUDataset.

**Output:** `reddit_multi12k_embedding_eval_results/`

---

### 4.3 GIN Classification

GIN uses Jupyter notebooks. **GPU with CUDA support is required.**

```bash
conda activate graphs   # or your GPU-enabled environment
```

Open the notebooks in VS Code or Jupyter Lab and execute all cells:

| Dataset | Notebook |
|---------|----------|
| ENZYMES | `GIN_model/GIN_enzymes_full_pipeline.ipynb` |
| IMDB-MULTI | `GIN_model/GIN_IMDB_full_pipeline.ipynb` |
| REDDIT-MULTI-12K | `GIN_model/GIN_REDDIT_full_pipeline.ipynb` |

Each notebook contains a `run_pipeline()` function that orchestrates the full experiment:

```python
# Inside the notebook — run all cells, then call:
run_pipeline(
    run_dataset_analysis=True,    # Dataset statistics & plots
    run_main_experiment=True,     # 10-fold CV with classification, clustering, stability
    enable_stability=True         # Include stability analysis
)
```

**Architecture:** 5-layer GIN, hidden_channels=64, dropout=0.5, JK='cat', train_eps=True, 350 epochs, Adam + CosineAnnealingLR.

**External classifiers** (applied to GIN embeddings): SVM (Linear), SVM (RBF), Random Forest, MLP.

**Output:** `classification_results.csv`, `classification_results.json`, per-fold embeddings (`embeddings_fold_*.npz`), trained models (`model_fold_*.pth`), learning curve plots.

---

### 4.4 Dimension Analysis (Accuracy vs. Compute Cost)

FGSD varies the number of histogram bins to change embedding dimensionality:

```bash
cd fgsd_method/src

# Dimension analysis is included in the classification pipeline by default:
python -m enzymes_ds.classification_main
python -m imbd_ds.classification_main
python -m reddit_ds.classification_main

# Graph2Vec also performs dimension sweep during evaluation:
cd ../g2v
python enzymes_second.py   # Includes dimension analysis section
python imdb_second.py
python reddit_second.py
```

**GIN:** Dimension sweep is available via the hyperparameter tuning section in each notebook (embedding dimension parameter).

---

## 5. Task (b): Clustering

> *Apply K-Means and/or Spectral Clustering to embeddings. Report ARI. Provide t-SNE/UMAP visualizations. Identify which embeddings yield the clearest cluster separation.*

### 5.1 FGSD Clustering

```bash
conda activate graphs
cd fgsd_method/src
```

#### ENZYMES

```bash
# Full clustering (K-Means + Spectral, t-SNE/UMAP, tests harmonic/polynomial/hybrid)
python -m enzymes_ds.clustering_main

# Minimal run (no grid search)
python -m enzymes_ds.clustering_main --no-grid-search

# Without node labels
python -m enzymes_ds.clustering_main --no-node-labels --no-grid-search
```

**CLI Flags:**

| Flag | Description |
|------|-------------|
| `--no-node-labels` | Disable node label features |
| `--no-grid-search` | Skip clustering grid search optimization |

**Clustering methods:** K-Means, Spectral Clustering  
**Visualizations:** PCA, t-SNE, UMAP projections  
**Configurations tested:** harmonic, polynomial, hybrid (concatenation of harmonic + polynomial)

#### IMDB-MULTI

```bash
# Full clustering
python -m imbd_ds.clustering_main

# Skip grid search
python -m imbd_ds.clustering_main --no-grid-search
```

**CLI Flags:**

| Flag | Description |
|------|-------------|
| `--no-grid-search` | Disable clustering grid search |

**Configurations tested:** harmonic (range=3.52), polynomial (range=3.13), hybrid

#### REDDIT-MULTI-12K

```bash
# Full clustering (11 configs: harmonic/polynomial/biharmonic + hybrid variants)
python -m reddit_ds.clustering_main

# Targeted clustering (2 best configs, with disk caching)
python -m reddit_ds.clustering_targeted
```

No CLI flags for Reddit clustering.

**Configurations tested:** harmonic, polynomial, biharmonic, biharmonic_hybrid, and various bin/range combinations.

---

### 5.2 Graph2Vec Clustering

Clustering is integrated into the Phase 2 evaluation scripts:

```bash
conda activate g2v
cd g2v

python enzymes_second.py    # Includes K-Means + Spectral clustering + t-SNE/UMAP
python imdb_second.py
python reddit_second.py
```

Clustering is automatically run as part of the evaluation pipeline. Results and visualizations are saved in the respective output directories.

---

### 5.3 GIN Clustering

Clustering is integrated into the `run_pipeline()` notebooks:

```python
# In each GIN notebook:
run_pipeline(run_main_experiment=True)  # Includes clustering in 10-fold CV
```

K-Means and Spectral Clustering are applied to the GIN embeddings from each fold.

---

## 6. Task (c): Stability Analysis

> *Introduce random perturbations (add/remove % of edges, shuffle node attributes). Recompute embeddings. Report embedding stability score and change in classification accuracy.*

### 6.1 FGSD Stability

```bash
conda activate graphs
cd fgsd_method/src
```

#### ENZYMES

```bash
# Run classification with stability analysis
python -m enzymes_ds.classification_main --stability

# Run only stability (after a classification run exists)
python -m enzymes_ds.classification_main --stability-only
```

**Perturbation types:** `edge_add`, `edge_remove`, `node_attribute_shuffle`  
**Perturbation ratios:** 1%, 5%, 10%, 20%  
**Metrics:** Cosine similarity (original vs perturbed), L2 distance, classification accuracy drop

#### IMDB-MULTI

```bash
# Run with stability
python -m imbd_ds.classification_main --stability

# Stability only
python -m imbd_ds.classification_main --stability-only
```

**Perturbation types:** `edge_add`, `edge_remove` (no node attribute shuffle — IMDB has no node attributes)  
**Perturbation ratios:** 1%, 5%, 10%, 20%

#### REDDIT-MULTI-12K

```bash
# Option 1: Inline stability during classification
python -m reddit_ds.classification_main --stability

# Option 2: Pre-generate perturbed embeddings (recommended for large dataset)
python -m reddit_ds.generate_stability_embeddings --batch-size 500

# Then run stability-only analysis
python -m reddit_ds.classification_main --stability-only
```

**CLI Flags for `generate_stability_embeddings`:**

| Flag | Description |
|------|-------------|
| `--batch-size N` | Batch size for processing (default: dataset-dependent) |

**Perturbation types:** `edge_add`, `edge_remove`  
**Perturbation ratios:** 5%, 10% (reduced for scalability on 12K graphs)  
**Note:** Pre-generates and caches control embeddings as pickle files for efficiency.

---

### 6.2 Graph2Vec Stability

Stability is integrated into the Phase 2 evaluation scripts (uses Procrustes alignment):

```bash
conda activate g2v
cd g2v

python enzymes_second.py   # Includes stability section (if perturbed embeddings exist)
python imdb_second.py
python reddit_second.py
```

For IMDB, the Phase 1 script supports optional perturbation during embedding generation:

```bash
# Generate perturbed embeddings for IMDB (edge removal)
python imdb_first.py       # Check perturbation config at top of file
```

---

### 6.3 GIN Stability

Stability is integrated into the GIN notebooks:

```python
# In each GIN notebook:
run_pipeline(
    run_main_experiment=True,
    enable_stability=True       # Enable stability analysis
)
```

**Perturbation ratios:** 5%, 10%, 15%, 20% (edge add/remove)  
**Metrics:** Embedding similarity, classification accuracy drop

---

## 7. Additional Analyses

### 7.1 Computational Analysis (FGSD)

Compare exact eigendecomposition vs. Chebyshev polynomial approximation:

```bash
conda activate graphs

# ENZYMES: Exact vs Chebyshev comparison
cd fgsd_method/src/enzymes_ds
python compare_hybrid_chebyshev.py

# IMDB: Exact vs Chebyshev comparison
cd ../imbd_ds
python chebyshev_imdb.py

# REDDIT: Exact vs Chebyshev (harmonic)
cd ../reddit_ds
python compare_chebyshev.py

# REDDIT: Multi-kernel benchmark (polynomial/harmonic/biharmonic, bucketed by graph size)
python fast_compare.py
```

### 7.2 Memory Benchmarking (REDDIT)

Profile RAM usage during embedding generation:

```bash
cd fgsd_method/src
python -m reddit_ds.memory_benchmark             # Basic profiling
python -m reddit_ds.memory_benchmark --detailed   # Detailed per-batch snapshots
```

Uses `tracemalloc` to measure memory for harmonic (bins=500) and polynomial (bins=200) embeddings.

### 7.3 Preanalysis (Histogram Range Detection)

Determine optimal histogram ranges by analyzing spectral distance distributions:

```bash
cd fgsd_method/src

# IMDB-MULTI: Test harmonic, polynomial, biharmonic
cd imbd_ds
python preanalysis_imdb.py

# REDDIT-MULTI-12K: Test harmonic, polynomial, biharmonic (samples 1000 graphs)
cd ../reddit_ds
python preanalysis_reddit.py
```

Results are cached in `fgsd_method/src/cache/` as JSON files.

### 7.4 Standalone IMDB Evaluation

A self-contained cross-validation script for FGSD on IMDB:

```bash
cd fgsd_method/src/imbd_ds
python evaluate_fgsd_imdb.py
```

Tests embedding dimensions: 50, 100, 200, 400. Saves results to `fgsd_imdb_results.csv`.

---

## 8. Plotting & Visualization

Generate publication-quality plots from saved CSV results:

```bash
conda activate graphs
cd fgsd_method
```

| Script | Input Data | Output |
|--------|-----------|--------|
| `python plot_computational_analysis.py` | `plots/computational_analysis/computational_summary.csv` | Time/memory vs dimension, Pareto fronts, cross-dataset comparison |
| `python plot_dimension_analysis.py` | `plots/dimension_analysis/dimension_analysis_summary.csv` | Accuracy vs embedding dimension, accuracy vs generation time |
| `python plot_grid_search_analysis.py` | `plots/grid_search_analysis/grid_search_summary.csv` | Accuracy/F1 vs bin width, heatmaps |
| `python plot_stability_analysis.py` | `plots/stability_analysis/stability_analysis_summary.csv` | Cosine similarity vs perturbation ratio, accuracy drop curves |

**Note:** These scripts read from pre-generated CSV files in the `plots/` subdirectories. Run the corresponding analysis scripts first to generate the data.

---

## 9. Full Run Commands Reference

### Complete FGSD Pipeline (All Datasets)

```bash
conda activate graphs
cd fgsd_method/src

# ── ENZYMES ──
python -m enzymes_ds.classification_main --stability --tune-classifiers   # Classification + Stability
python -m enzymes_ds.clustering_main                                       # Clustering

# ── IMDB-MULTI ──
python -m imbd_ds.classification_main --stability --tune-classifiers       # Classification + Stability
python -m imbd_ds.clustering_main                                          # Clustering

# ── REDDIT-MULTI-12K ──
python -m reddit_ds.classification_main --stability --tune-classifiers     # Classification + Stability
python -m reddit_ds.clustering_main                                        # Clustering
python -m reddit_ds.clustering_targeted                                    # Targeted Clustering

# ── Cross-Validation Benchmark (all datasets) ──
python cross_validation_benchmark.py

# ── Computational Analysis ──
cd enzymes_ds && python compare_hybrid_chebyshev.py && cd ..
cd imbd_ds && python chebyshev_imdb.py && cd ..
cd reddit_ds && python compare_chebyshev.py && python fast_compare.py && cd ..

# ── Memory Benchmark (Reddit) ──
python -m reddit_ds.memory_benchmark --detailed

# ── Preanalysis ──
cd imbd_ds && python preanalysis_imdb.py && cd ..
cd reddit_ds && python preanalysis_reddit.py && cd ..

# ── Plots ──
cd ../../
python plot_computational_analysis.py
python plot_dimension_analysis.py
python plot_grid_search_analysis.py
python plot_stability_analysis.py
```

### Complete Graph2Vec Pipeline (All Datasets)

```bash
conda activate g2v
cd g2v

# ── ENZYMES ──
python enzymes_first.py      # Generate embeddings
python enzymes_second.py     # Evaluate (classification, clustering, stability, dimension sweep)

# ── IMDB-MULTI ──
python imdb_first.py         # Generate embeddings
python imdb_second.py        # Evaluate

# ── REDDIT-MULTI-12K ──
python reddit_first.py       # Generate embeddings
python reddit_second.py      # Evaluate
```

### Complete GIN Pipeline (All Datasets)

Open each notebook in VS Code or Jupyter and **run all cells**:

```
GIN_model/GIN_enzymes_full_pipeline.ipynb    # ENZYMES
GIN_model/GIN_IMDB_full_pipeline.ipynb       # IMDB-MULTI
GIN_model/GIN_REDDIT_full_pipeline.ipynb     # REDDIT-MULTI-12K
```

Or run from the command line:

```bash
conda activate graphs
cd GIN_model
jupyter nbconvert --to notebook --execute GIN_enzymes_full_pipeline.ipynb
jupyter nbconvert --to notebook --execute GIN_IMDB_full_pipeline.ipynb
jupyter nbconvert --to notebook --execute GIN_REDDIT_full_pipeline.ipynb
```

---

### Method × Task Summary Matrix

| | Classification | Clustering | Stability | Dim. Analysis | Computational |
|---|---|---|---|---|---|
| **FGSD × ENZYMES** | `classification_main.py` | `clustering_main.py` | `--stability` flag | Included in classification | `compare_hybrid_chebyshev.py` |
| **FGSD × IMDB** | `classification_main.py` | `clustering_main.py` | `--stability` flag | Included in classification | `chebyshev_imdb.py` |
| **FGSD × REDDIT** | `classification_main.py` | `clustering_main.py` | `--stability` / `generate_stability_embeddings.py` | Included in classification | `compare_chebyshev.py`, `fast_compare.py`, `memory_benchmark.py` |
| **G2V × ENZYMES** | `enzymes_second.py` | `enzymes_second.py` | `enzymes_second.py` | `enzymes_second.py` | — |
| **G2V × IMDB** | `imdb_second.py` | `imdb_second.py` | `imdb_second.py` | `imdb_second.py` | — |
| **G2V × REDDIT** | `reddit_second.py` | `reddit_second.py` | `reddit_second.py` | `reddit_second.py` | — |
| **GIN × ENZYMES** | Notebook | Notebook | Notebook | Notebook | — |
| **GIN × IMDB** | Notebook | Notebook | Notebook | Notebook | — |
| **GIN × REDDIT** | Notebook | Notebook | Notebook | Notebook | — |

---

## License

See [LICENSE](LICENSE) for details.
