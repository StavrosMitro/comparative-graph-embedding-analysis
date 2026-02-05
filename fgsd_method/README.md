# FGSD Method: Graph Classification & Clustering

This repository implements the **Flexible Graph Spectral Distance (FGSD)** method for graph classification and clustering on three benchmark datasets: **ENZYMES**, **IMDB-MULTI**, and **REDDIT-MULTI-12K**.

## Project Structure

```
fgsd_method/
├── src/
│   ├── fgsd.py                 # Core FlexibleFGSD implementation
│   ├── optimized_method.py     # HybridFGSD implementation
│   ├── enzymes_ds/             # ENZYMES dataset module
│   │   ├── classification_main.py
│   │   ├── clustering_main.py
│   │   └── ...
│   ├── imbd_ds/                # IMDB-MULTI dataset module
│   │   ├── classification_main.py
│   │   ├── clustering_main.py
│   │   └── ...
│   └── reddit_ds/              # REDDIT-MULTI-12K dataset module
│       ├── classification_main.py
│       ├── clustering_main.py
│       └── ...
├── results/                    # Output CSV files and plots
├── cache/                      # Cached preanalysis results
├── fgsd_env.yml               # Conda environment specification
└── README.md
```

## Installation

### Prerequisites
- Conda (Miniconda or Anaconda)
- Linux (tested on Ubuntu)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fgsd_method
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f fgsd_env.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate graphs
   ```

## Datasets

All datasets are automatically downloaded on first run:

| Dataset | Graphs | Classes | Node Labels | Download Location |
|---------|--------|---------|-------------|-------------------|
| ENZYMES | 600 | 6 | ✅ Yes (3 features) | `/tmp/ENZYMES` |
| IMDB-MULTI | 1,500 | 3 | ❌ No | `/tmp/IMDB-MULTI` |
| REDDIT-MULTI-12K | 11,929 | 11 | ❌ No | `/tmp/REDDIT-MULTI-12K` |

## Usage

All commands should be run from the `src/` directory:

```bash
cd src
```

---

### ENZYMES Dataset

**Classification:**
```bash
# Full pipeline (preanalysis → grid search → dimension analysis → final classification)
python -m enzymes_ds.classification_main

# Include stability analysis
python -m enzymes_ds.classification_main --stability

# Run only stability analysis (requires previous results)
python -m enzymes_ds.classification_main --stability-only

# Force recompute cached results
python -m enzymes_ds.classification_main --force

# Disable node label features (spectral only)
python -m enzymes_ds.classification_main --no-node-labels

# Run classifier hyperparameter tuning (RF & SVM)
python -m enzymes_ds.classification_main --tune-classifiers

# Run with raw embeddings (no StandardScaler preprocessing)
python -m enzymes_ds.classification_main --raw-embeddings

# Raw embeddings with stability analysis
python -m enzymes_ds.classification_main --raw-embeddings --stability
```

**Clustering:**
```bash
# With node labels (default)
python -m enzymes_ds.clustering_main

# Without node labels (spectral only)
python -m enzymes_ds.clustering_main --no-node-labels
```

---

### IMDB-MULTI Dataset

**Classification:**
```bash
# Full pipeline
python -m imbd_ds.classification_main

# Include stability analysis
python -m imbd_ds.classification_main --stability

# Run only stability analysis
python -m imbd_ds.classification_main --stability-only

# Force recompute
python -m imbd_ds.classification_main --force

# Run classifier hyperparameter tuning
python -m imbd_ds.classification_main --tune-classifiers

# Run with raw embeddings (no StandardScaler preprocessing)
python -m imbd_ds.classification_main --raw-embeddings

# Raw embeddings with stability analysis
python -m imbd_ds.classification_main --raw-embeddings --stability
```

**Clustering:**
```bash
python -m imbd_ds.clustering_main
```

---

### REDDIT-MULTI-12K Dataset

> **Note:** This is a large dataset (~12K graphs). Full pipeline takes several hours.

**Classification:**
```bash
# Full pipeline
python -m reddit_ds.classification_main

# Include stability analysis
python -m reddit_ds.classification_main --stability

# Run only stability analysis
python -m reddit_ds.classification_main --stability-only

# Force recompute
python -m reddit_ds.classification_main --force

# Run classifier hyperparameter tuning
python -m reddit_ds.classification_main --tune-classifiers
```

**Clustering:**
```bash
python -m reddit_ds.clustering_main
```

---

## Output Files

Results are saved in the `results/` directory:

| File | Description |
|------|-------------|
| `fgsd_<dataset>_grid_search.csv` | Grid search over binwidth values |
| `fgsd_<dataset>_dimension_analysis.csv` | Analysis of embedding dimensions |
| `fgsd_<dataset>_final_results.csv` | Final classification results |
| `fgsd_<dataset>_stability_results.csv` | Stability analysis under edge perturbations |
| `fgsd_<dataset>_classifier_tuning.csv` | Hyperparameter tuning results |
| `clustering_<dataset>_*.png` | Clustering visualization plots |

## Method Overview

The FGSD method computes graph embeddings using spectral distances:

- **Harmonic (1/λ):** Captures global graph structure
- **Polynomial (λ²):** Captures local graph structure  
- **Biharmonic (1/λ²):** Alternative global measure
- **Hybrid:** Concatenation of harmonic + polynomial embeddings

### Pipeline Stages

1. **Preanalysis:** Determines optimal histogram range from spectral distance distribution
2. **Grid Search:** Tests different binwidth values (h ∈ {0.05, 0.1, 0.2, 0.5, 1})
3. **Dimension Analysis:** Evaluates different embedding dimensions
4. **Classification:** Random Forest, SVM, and MLP classifiers
5. **Stability Analysis:** Tests robustness under edge perturbations (1%, 5%, 10%)

## Classifiers

- **Random Forest:** 500 trees, max_depth=20
- **SVM (RBF):** With StandardScaler preprocessing
- **MLP:** 4-layer network (1024→512→256→128)

## Notes

- Preanalysis results are cached in `cache/` directory to avoid recomputation
- Use `--force` flag to recompute cached results
- REDDIT dataset uses reduced bin sizes by default for faster execution
- Visualizations require `umap-learn` (optional, falls back to t-SNE only)
