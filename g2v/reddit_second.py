#!/usr/bin/env python3
# reddit_multi12k_evaluate_saved_embeddings_all_metrics.py
#
# Evaluates saved REDDIT-MULTI-12K embeddings:
#   - FAST tuning split (cheaty) on full data
#   - final repeated stratified CV across models
#   - clustering (kmeans + optional spectral)
#   - dim summary (best clean per D)
#   - stability (if perturbed embeddings exist)
#   - t-SNE + optional UMAP plots

import os
import glob
import json
import math
import time
import random
import itertools
import inspect
import csv
import re
import shutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from sklearn.cluster import SpectralClustering  # type: ignore
    HAS_SPECTRAL = True
except Exception:
    HAS_SPECTRAL = False

try:
    from sklearn.model_selection import RepeatedStratifiedKFold  # type: ignore
    HAS_REPEATED_SKF = True
except Exception:
    RepeatedStratifiedKFold = None  # type: ignore
    HAS_REPEATED_SKF = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# =============================================================================
# USER CONFIG
# =============================================================================
DATASET_ROOTS: List[str] = ["./g2v_reddit_multi12k_embeddings_selected_epochs_named/REDDIT-MULTI-12K"]
EPOCHS_TO_EVAL: List[int] = []  # [] => all discovered epochs

FAST_TUNE_TRAIN_FRAC = 0.8
FAST_TUNE_SEED = 123
FAST_SELECT_METRIC = "f1_macro"
HYPERPARAM_MAX_COMBOS_PER_MODEL: Optional[int] = 80

CV_N_SPLITS = 10
CV_N_REPEATS = 3
CV_SEED = 123

MAX_RF_JOBS = 1

DO_KMEANS = True
DO_SPECTRAL = True
KMEANS_N_INIT = 10
CLUSTER_SEED = 7
MAX_SIL_SAMPLES = 5000

BUILD_DIM_SWEEP_SUMMARY = True

COMPUTE_STABILITY = True
ASSUME_MISSING_P_IS_CLEAN = True
CLEAN_P_VALUE = 0.0

COMPUTE_FIXED_SVC_FOR_STABILITY = True
FIXED_SVC_PARAMS = {
    "clf__C": 1.0,
    "clf__gamma": "scale",
    "clf__class_weight": None,
}

MAKE_PLOTS = True
TSNE_PERPLEXITY = 30.0
TSNE_N_ITER = 1000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
PLOT_MAX_POINTS = 4000
PLOT_SEED = 777
PLOT_ONLY_BEST_CLEAN_PER_RUN = True

OUT_DIR = "./reddit_multi12k_embedding_eval_results"
SAVE_JSONL = True
SAVE_CSV = True
SAVE_RESULTS_JSON = True
OVERWRITE_OUTPUT = True

GLOBAL_SEED = 999


# =============================================================================
# Helpers
# =============================================================================
def set_all_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def wipe_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except Exception:
        return "NA"
    if math.isnan(xf) or math.isinf(xf):
        return "NA"
    return f"{xf:.{nd}f}"

def _min_class_count(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min()) if counts.size else 0

def _stratify_possible(y: np.ndarray, n_splits: int = 2) -> bool:
    return _min_class_count(y) >= int(n_splits)

def load_embeddings(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        x = np.load(path, allow_pickle=False)
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            raise ValueError(f"Bad embedding shape in {path}: {getattr(x, 'shape', None)}")
        return x.astype(np.float32, copy=False)

    if path.endswith(".pt"):
        if not HAS_TORCH:
            raise RuntimeError("Found .pt embeddings but torch is not available.")
        t = torch.load(path, map_location="cpu")
        if isinstance(t, torch.Tensor):
            x = t.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported .pt content type: {type(t)} in {path}")
        if x.ndim != 2:
            raise ValueError(f"Bad embedding shape in {path}: {x.shape}")
        return x.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported embedding file extension: {path}")

_EPOCH_RE = re.compile(r"(?:^|[^0-9])epoch[_\-]?(\d{1,6})(?:[^0-9]|$)", re.IGNORECASE)
_D_RE = re.compile(r"(?:^|[^0-9])D(\d{1,5})(?:[^0-9]|$)")
_P_FLOAT_RE_LIST = [
    re.compile(r"(?:^|[^a-z0-9])p(?:ert)?[_\-]?([01](?:\.\d+)?)(?:[^0-9]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[^a-z0-9])edge[_\-]?drop[_\-]?([01](?:\.\d+)?)(?:[^0-9]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[^a-z0-9])remove[_\-]?p[_\-]?([01](?:\.\d+)?)(?:[^0-9]|$)", re.IGNORECASE),
]
_P_INT_RE_LIST = [
    re.compile(r"(?:^|[^a-z0-9])p(?:ert)?[_\-]?([0-9]{1,2})(?:[^0-9]|$)", re.IGNORECASE),
]

def infer_epoch_from_name(fname: str) -> Optional[int]:
    base = os.path.basename(fname)
    m = _EPOCH_RE.search(base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def infer_dim_from_name(fname: str) -> Optional[int]:
    base = os.path.basename(fname)
    m = _D_RE.search(base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def infer_perturbation_p(path: str) -> Optional[float]:
    text = path.replace("\\", "/").lower()

    for rgx in _P_FLOAT_RE_LIST:
        m = rgx.search(text)
        if m:
            try:
                p = float(m.group(1))
                if 0.0 <= p <= 1.0:
                    return p
            except Exception:
                pass

    for rgx in _P_INT_RE_LIST:
        m = rgx.search(text)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return float(v) / 100.0
            except Exception:
                pass
    return None

def discover_runs(dataset_root: str) -> List[str]:
    if not os.path.isdir(dataset_root):
        return []

    runs: List[str] = []
    patterns = [
        os.path.join(dataset_root, "*"),
        os.path.join(dataset_root, "*", "*"),
        os.path.join(dataset_root, "*", "*", "*"),
    ]
    seen = set()
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            if not os.path.isdir(d) or d in seen:
                continue
            seen.add(d)

            labels_p = os.path.join(d, "labels.npy")
            gids_p = os.path.join(d, "graph_ids.npy")
            emb_dir = os.path.join(d, "embeddings")
            if not (os.path.exists(labels_p) and os.path.exists(gids_p) and os.path.isdir(emb_dir)):
                continue

            emb_files = glob.glob(os.path.join(emb_dir, "*.npy")) + glob.glob(os.path.join(emb_dir, "*.pt"))
            emb_files = [p for p in emb_files if infer_epoch_from_name(p) is not None]
            if emb_files:
                runs.append(d)

    return sorted(runs)

def discover_embedding_files(run_dir: str) -> List[str]:
    emb_dir = os.path.join(run_dir, "embeddings")
    if not os.path.isdir(emb_dir):
        return []
    files = sorted(glob.glob(os.path.join(emb_dir, "*.npy"))) + sorted(glob.glob(os.path.join(emb_dir, "*.pt")))
    return [p for p in files if infer_epoch_from_name(p) is not None]

def run_display_name(dataset_root: str, run_dir: str) -> str:
    return os.path.relpath(run_dir, start=dataset_root).replace("\\", "/")


# =============================================================================
# FAST split
# =============================================================================
def stratified_train_val_split(
    X: np.ndarray, y: np.ndarray, train_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_frac = float(train_frac)
    seed = int(seed)

    do_stratify = _stratify_possible(y, n_splits=2)
    try:
        if do_stratify:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
            (tr, va) = next(sss.split(X, y))
            return X[tr], X[va], y[tr], y[va]

        idx = np.arange(y.shape[0])
        tr, va = train_test_split(idx, train_size=train_frac, random_state=seed, stratify=None)
        return X[tr], X[va], y[tr], y[va]
    except ValueError as e:
        print(f"[WARN] fast split failed ({e}); fallback non-stratified.")
        idx = np.arange(y.shape[0])
        tr, va = train_test_split(idx, train_size=train_frac, random_state=seed, stratify=None)
        return X[tr], X[va], y[tr], y[va]


# =============================================================================
# AUC helpers
# =============================================================================
def _n_classes(y: np.ndarray) -> int:
    return int(np.unique(y).size)

def _scores_for_auc(est, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(est, "decision_function"):
        try:
            return np.asarray(est.decision_function(X))
        except Exception:
            pass
    if hasattr(est, "predict_proba"):
        try:
            return np.asarray(est.predict_proba(X))
        except Exception:
            pass
    return None

def auc_binary_or_multiclass_ovr(est, X: np.ndarray, y: np.ndarray) -> Optional[float]:
    try:
        k = _n_classes(y)
        scores = _scores_for_auc(est, X)
        if scores is None:
            return None

        if k == 2:
            if scores.ndim == 1:
                s1 = scores
            elif scores.ndim == 2 and scores.shape[1] == 2:
                s1 = scores[:, 1]
            else:
                s1 = scores.reshape(-1)
            return float(roc_auc_score(y, s1))

        if scores.ndim == 1:
            return None
        return float(roc_auc_score(y, scores, multi_class="ovr", average="macro"))
    except Exception:
        return None


# =============================================================================
# Models + tune
# =============================================================================
def _filter_kwargs_for_init(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)
        allowed = set(params.keys())
        allowed.discard("self")
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)

def _safe_instantiate(cls: Any, candidate_kwargs: List[Dict[str, Any]]) -> Any:
    last_err: Optional[Exception] = None
    for kw in candidate_kwargs:
        kw_f = _filter_kwargs_for_init(cls, kw)
        try:
            return cls(**kw_f)
        except (TypeError, ValueError) as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Could not instantiate {cls}.")

def _grid_to_combos(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    out: List[Dict[str, Any]] = []
    for prod in itertools.product(*vals):
        out.append({k: v for k, v in zip(keys, prod)})
    return out

def _maybe_cap_combos(combos: List[Dict[str, Any]], cap: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if cap is None or cap <= 0 or cap >= len(combos):
        return combos
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(len(combos), size=int(cap), replace=False)
    return [combos[i] for i in idx]

def _metric_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro"))
    raise ValueError(f"Unknown metric: {metric}")

@dataclass
class ModelSpec:
    key: str
    name: str
    grid: Dict[str, List[Any]]

LOGREG_GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}
RIDGE_GRID  = {"clf__alpha": [0.1, 1.0, 10.0, 100.0], "clf__class_weight": [None, "balanced"]}
LINEARSVC_GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}
SVC_RBF_GRID = {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto", 0.01, 0.1], "clf__class_weight": [None, "balanced"]}
KNN_GRID = {"clf__n_neighbors": [3, 5, 7, 11, 15], "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}
RF_GRID = {
    "clf__n_estimators": [200],
    "clf__max_depth": [None, 20],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2],
    "clf__max_features": ["sqrt"],
    "clf__bootstrap": [True],
}
MLP_GRID = {
    "clf__hidden_layer_sizes": [(64,), (128,), (128, 64)],
    "clf__alpha": [1e-4, 1e-3, 1e-2],
    "clf__learning_rate_init": [1e-3, 3e-4],
}

MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("logreg", "LogisticRegression", LOGREG_GRID),
    ModelSpec("ridge", "RidgeClassifier", RIDGE_GRID),
    ModelSpec("linearsvc", "LinearSVC", LINEARSVC_GRID),
    ModelSpec("svc_rbf", "SVC(RBF)", SVC_RBF_GRID),
    ModelSpec("knn", "KNN", KNN_GRID),
    ModelSpec("rf", "RandomForest", RF_GRID),
    ModelSpec("mlp", "MLP", MLP_GRID),
]

def build_pipeline(model_key: str, *, seed: int) -> Pipeline:
    if model_key == "logreg":
        clf = _safe_instantiate(
            LogisticRegression,
            [
                {"solver": "lbfgs", "max_iter": 5000, "random_state": seed},
                {"solver": "liblinear", "max_iter": 5000, "random_state": seed},
                {"max_iter": 5000, "random_state": seed},
                {"random_state": seed},
                {},
            ],
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_key == "ridge":
        clf = _safe_instantiate(RidgeClassifier, [{"random_state": seed}, {}])
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_key == "linearsvc":
        clf = _safe_instantiate(LinearSVC, [{"max_iter": 10000, "random_state": seed}, {"max_iter": 10000}, {}])
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_key == "svc_rbf":
        clf = _safe_instantiate(SVC, [{"kernel": "rbf", "probability": False, "random_state": seed}, {"kernel": "rbf"}])
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_key == "knn":
        return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])

    if model_key == "rf":
        clf = _safe_instantiate(
            RandomForestClassifier,
            [
                {"random_state": seed, "n_jobs": int(MAX_RF_JOBS)},
                {"random_state": seed},
                {},
            ],
        )
        return Pipeline([("clf", clf)])

    if model_key == "mlp":
        clf = _safe_instantiate(
            MLPClassifier,
            [
                {"max_iter": 2000, "early_stopping": True, "n_iter_no_change": 25, "random_state": seed},
                {"max_iter": 2000, "random_state": seed},
                {"max_iter": 2000},
                {},
            ],
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    raise KeyError(f"Unknown model_key: {model_key}")

def _filter_grid_by_supported_params(pipe: Pipeline, grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    supported = set(pipe.get_params(deep=True).keys())
    return {k: v for k, v in grid.items() if k in supported}

@dataclass
class FastTuneResult:
    model_key: str
    model_name: str
    best_val_score: float
    best_params: Dict[str, Any]
    n_tried: int
    n_failed: int
    elapsed_sec: float

def fast_tune_single_split(
    spec: ModelSpec,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    metric: str,
    cap_combos: Optional[int],
    seed: int,
) -> FastTuneResult:
    base = build_pipeline(spec.key, seed=seed)
    grid = _filter_grid_by_supported_params(base, spec.grid)
    combos = _grid_to_combos(grid)
    combos = _maybe_cap_combos(combos, cap_combos, seed=seed + 1337)

    best_score = -1.0
    best_params: Dict[str, Any] = {}
    n_failed = 0

    t0 = time.time()
    for params in combos:
        model = build_pipeline(spec.key, seed=seed)
        try:
            model.set_params(**params)
        except Exception:
            n_failed += 1
            continue

        try:
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            score = _metric_score(yva, pred, metric=metric)
        except Exception:
            n_failed += 1
            continue

        if score > best_score:
            best_score = float(score)
            best_params = dict(params)

    dt = time.time() - t0
    return FastTuneResult(
        model_key=spec.key,
        model_name=spec.name,
        best_val_score=float(best_score),
        best_params=best_params,
        n_tried=len(combos),
        n_failed=int(n_failed),
        elapsed_sec=float(dt),
    )


# =============================================================================
# CV eval
# =============================================================================
@dataclass
class CVMetrics:
    acc_mean: float
    acc_std: float
    f1_mean: float
    f1_std: float
    auc_mean: Optional[float]
    auc_std: Optional[float]
    n_folds: int
    elapsed_sec: float

def _mean_std(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std

def _iter_repeated_stratified_folds(X: np.ndarray, y: np.ndarray, n_splits: int, n_repeats: int, seed: int):
    minc = _min_class_count(y)
    n_splits_eff = int(min(max(2, n_splits), minc))

    if HAS_REPEATED_SKF:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits_eff,
            n_repeats=int(max(1, n_repeats)),
            random_state=int(seed),
        )
        for tr, te in cv.split(X, y):
            yield tr, te, n_splits_eff
        return

    for r in range(int(max(1, n_repeats))):
        skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=int(seed) + r)
        for tr, te in skf.split(X, y):
            yield tr, te, n_splits_eff

def cv_full_data_all_metrics(
    model_key: str,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> CVMetrics:
    accs: List[Optional[float]] = []
    f1s: List[Optional[float]] = []
    aucs: List[Optional[float]] = []

    t0 = time.time()
    n_folds = 0

    for tr_idx, te_idx, _ in _iter_repeated_stratified_folds(X, y, n_splits, n_repeats, seed):
        n_folds += 1
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        model = build_pipeline(model_key, seed=seed)
        supported = set(model.get_params(deep=True).keys())
        safe_params = {k: v for k, v in params.items() if k in supported}

        try:
            model.set_params(**safe_params)
        except Exception:
            accs.append(None); f1s.append(None); aucs.append(None)
            continue

        try:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            accs.append(float(accuracy_score(yte, pred)))
            f1s.append(float(f1_score(yte, pred, average="macro")))
            aucs.append(auc_binary_or_multiclass_ovr(model, Xte, yte))
        except Exception:
            accs.append(None); f1s.append(None); aucs.append(None)

    dt = time.time() - t0
    acc_mean, acc_std = _mean_std(accs)
    f1_mean, f1_std = _mean_std(f1s)
    auc_mean, auc_std = _mean_std(aucs)

    if acc_mean is None:
        acc_mean = float("nan"); acc_std = float("nan")
    if f1_mean is None:
        f1_mean = float("nan"); f1_std = float("nan")

    return CVMetrics(
        acc_mean=float(acc_mean),
        acc_std=float(acc_std),
        f1_mean=float(f1_mean),
        f1_std=float(f1_std),
        auc_mean=auc_mean,
        auc_std=auc_std,
        n_folds=int(n_folds),
        elapsed_sec=float(dt),
    )


# =============================================================================
# Clustering
# =============================================================================
@dataclass
class ClusterResult:
    method: str
    ari: float
    silhouette: Optional[float]

def _silhouette_safe(X: np.ndarray, labels: np.ndarray, seed: int) -> Optional[float]:
    try:
        n = X.shape[0]
        if n <= int(MAX_SIL_SAMPLES):
            return float(silhouette_score(X, labels, metric="euclidean"))
        return float(
            silhouette_score(
                X, labels, metric="euclidean",
                sample_size=int(MAX_SIL_SAMPLES),
                random_state=int(seed),
            )
        )
    except Exception:
        return None

def cluster_kmeans(X: np.ndarray, y: np.ndarray, *, seed: int) -> Tuple[ClusterResult, np.ndarray]:
    k = int(len(np.unique(y)))
    km = KMeans(n_clusters=k, n_init=int(KMEANS_N_INIT), random_state=int(seed))
    lab = km.fit_predict(X)
    ari = float(adjusted_rand_score(y, lab))
    sil = _silhouette_safe(X, lab, seed=seed)
    return ClusterResult(method="kmeans", ari=ari, silhouette=sil), lab

def cluster_spectral(X: np.ndarray, y: np.ndarray, *, seed: int) -> Tuple[ClusterResult, np.ndarray]:
    if not HAS_SPECTRAL:
        raise RuntimeError("SpectralClustering not available.")
    k = int(len(np.unique(y)))
    sc = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=min(10, max(2, X.shape[0] - 1)),
        assign_labels="kmeans",
        random_state=int(seed),
    )
    lab = sc.fit_predict(X)
    ari = float(adjusted_rand_score(y, lab))
    sil = _silhouette_safe(X, lab, seed=seed)
    return ClusterResult(method="spectral", ari=ari, silhouette=sil), lab


# =============================================================================
# Stability helpers
# =============================================================================
def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M = B.T @ A
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt

def mean_cosine_after_alignment(Z_ref: np.ndarray, Z_new: np.ndarray) -> float:
    R = orthogonal_procrustes(Z_ref, Z_new)
    Z_al = Z_new @ R
    nr = np.linalg.norm(Z_ref, axis=1) + 1e-12
    nn = np.linalg.norm(Z_al, axis=1) + 1e-12
    cos = np.sum(Z_ref * Z_al, axis=1) / (nr * nn)
    return float(np.mean(cos))


# =============================================================================
# Plot helpers
# =============================================================================
def _tsne_iter_kwarg_name() -> Optional[str]:
    try:
        from sklearn.manifold import TSNE
        params = set(inspect.signature(TSNE.__init__).parameters.keys())
        if "n_iter" in params:
            return "n_iter"
        if "max_iter" in params:
            return "max_iter"
        return None
    except Exception:
        return None

def _select_plot_indices(y: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    n = int(y.shape[0])
    max_points = int(max_points)
    if n <= max_points:
        return np.arange(n, dtype=np.int64)

    rng = np.random.RandomState(int(seed))
    classes, counts = np.unique(y, return_counts=True)

    selected = []
    remaining = max_points
    for c, cnt in zip(classes, counts):
        frac = float(cnt) / float(n)
        k = max(1, int(round(frac * max_points)))
        idx_c = np.where(y == c)[0]
        k = min(k, idx_c.size)
        pick = rng.choice(idx_c, size=k, replace=False)
        selected.extend(pick.tolist())
        remaining = max(0, remaining - k)

    if remaining > 0:
        pool = np.setdiff1d(np.arange(n), np.asarray(selected, dtype=np.int64))
        if pool.size > 0:
            extra = rng.choice(pool, size=min(remaining, pool.size), replace=False)
            selected.extend(extra.tolist())

    selected = np.asarray(sorted(set(selected)), dtype=np.int64)
    if selected.size > max_points:
        selected = rng.choice(selected, size=max_points, replace=False)
        selected = np.asarray(sorted(selected), dtype=np.int64)
    return selected

def make_tsne(Z: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = int(Z.shape[0])
    perp = float(TSNE_PERPLEXITY)
    if n <= 3:
        raise RuntimeError("Not enough points for t-SNE.")
    perp = min(perp, float(max(2, n - 1)))

    kwargs: Dict[str, Any] = dict(
        n_components=2,
        perplexity=perp,
        init="pca",
        random_state=int(seed),
        learning_rate="auto",
    )
    it_name = _tsne_iter_kwarg_name()
    if it_name is not None:
        kwargs[it_name] = int(TSNE_N_ITER)

    try:
        tsne = TSNE(**kwargs)
    except TypeError:
        kwargs.pop("learning_rate", None)
        tsne = TSNE(**kwargs)

    return tsne.fit_transform(Z)

def make_umap(Z: np.ndarray, seed: int) -> Optional[np.ndarray]:
    try:
        import umap  # type: ignore
    except Exception:
        return None
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(UMAP_N_NEIGHBORS),
        min_dist=float(UMAP_MIN_DIST),
        random_state=int(seed),
    )
    return reducer.fit_transform(Z)

def plot_2d_scatter(X2: np.ndarray, y: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(X2[:, 0], X2[:, 1], c=y, s=8)
    plt.title(title)
    plt.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    plt.savefig(out_path, dpi=220)
    plt.close()


# =============================================================================
# Records
# =============================================================================
@dataclass
class ModelEvalRecord:
    dataset_root: str
    run_name: str
    epoch: int
    D: int
    p: Optional[float]
    emb_path: str
    N: int
    n_classes: int

    model_key: str
    model_name: str
    fast_select_metric: str
    fast_best_val_score: float
    fast_best_params: Dict[str, Any]

    cv_acc_mean: float
    cv_acc_std: float
    cv_f1_mean: float
    cv_f1_std: float
    cv_auc_mean: Optional[float]
    cv_auc_std: Optional[float]
    cv_folds: int
    cv_time_sec: float

@dataclass
class EmbeddingClusterRecord:
    dataset_root: str
    run_name: str
    epoch: int
    D: int
    p: Optional[float]
    emb_path: str
    N: int
    n_classes: int

    kmeans_ari: Optional[float]
    kmeans_sil: Optional[float]
    spectral_ari: Optional[float]
    spectral_sil: Optional[float]

@dataclass
class EmbeddingSummaryRecord:
    dataset_root: str
    run_name: str
    epoch: int
    D: int
    p: Optional[float]
    emb_path: str
    emb_file: str
    N: int
    n_classes: int

    best_model_key: str
    best_model_name: str
    best_cv_acc_mean: float
    best_cv_f1_mean: float
    best_cv_auc_mean: Optional[float]

    fixed_svc_cv_acc_mean: Optional[float]
    fixed_svc_cv_f1_mean: Optional[float]
    fixed_svc_cv_auc_mean: Optional[float]

@dataclass
class DimSweepSummaryRecord:
    dataset_root: str
    run_name: str
    D: int
    best_epoch: int
    best_emb_file: str
    best_model_name: str
    best_cv_acc_mean: float
    best_cv_f1_mean: float
    best_cv_auc_mean: Optional[float]
    fixed_svc_cv_acc_mean: Optional[float]
    fixed_svc_cv_f1_mean: Optional[float]
    fixed_svc_cv_auc_mean: Optional[float]

@dataclass
class StabilityRecord:
    dataset_root: str
    run_name: str
    D: int
    epoch: int
    p: float
    clean_emb_file: str
    pert_emb_file: str
    mean_cosine_aligned: float
    delta_acc_best_model: Optional[float]
    delta_f1_best_model: Optional[float]
    delta_acc_fixed_svc: Optional[float]
    delta_f1_fixed_svc: Optional[float]

@dataclass
class PlotRecord:
    dataset_root: str
    run_name: str
    epoch: int
    D: int
    p: Optional[float]
    emb_file: str
    tsne_path: Optional[str]
    umap_path: Optional[str]
    n_points_plotted: int


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_jsonable(r), ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    rows2: List[Dict[str, Any]] = []
    for r in rows:
        r2 = dict(r)
        if "fast_best_params" in r2 and isinstance(r2["fast_best_params"], dict):
            r2["fast_best_params"] = json.dumps(_jsonable(r2["fast_best_params"]), ensure_ascii=False)
        rows2.append(r2)

    fieldnames: List[str] = sorted({k for r in rows2 for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows2:
            w.writerow(r)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    set_all_seeds(GLOBAL_SEED)

    if OVERWRITE_OUTPUT:
        wipe_dir(OUT_DIR)
    else:
        ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "plots"))

    model_eval_records: List[ModelEvalRecord] = []
    cluster_records: List[EmbeddingClusterRecord] = []
    emb_summary_records: List[EmbeddingSummaryRecord] = []
    dim_rows: List[DimSweepSummaryRecord] = []
    stab_rows: List[StabilityRecord] = []
    plot_rows: List[PlotRecord] = []

    emb_cache: Dict[str, np.ndarray] = {}

    print("=" * 120)
    print("[EVAL] REDDIT-MULTI-12K saved embeddings")
    print("=" * 120)
    print(f"dataset_roots         : {DATASET_ROOTS}")
    print(f"epochs_to_eval        : {EPOCHS_TO_EVAL if EPOCHS_TO_EVAL else '(auto-discover)'}")
    print(f"fast tune train_frac  : {FAST_TUNE_TRAIN_FRAC} (seed={FAST_TUNE_SEED}) metric={FAST_SELECT_METRIC}")
    print(f"CV                    : splits={CV_N_SPLITS} repeats={CV_N_REPEATS} seed={CV_SEED}")
    print(f"clustering            : kmeans={DO_KMEANS} spectral={DO_SPECTRAL and HAS_SPECTRAL}")
    print(f"stability             : {COMPUTE_STABILITY}")
    print(f"plots                 : {MAKE_PLOTS}")
    print(f"outputs               : {OUT_DIR}")
    print("")

    for dataset_root in DATASET_ROOTS:
        runs = discover_runs(dataset_root)
        if not runs:
            print(f"[WARN] No runs found under: {dataset_root}")
            continue

        print("#" * 120)
        print(f"[DATASET ROOT] {dataset_root}")
        print("#" * 120)

        for run_dir in runs:
            run_name = run_display_name(dataset_root, run_dir)

            labels_path = os.path.join(run_dir, "labels.npy")
            gids_path = os.path.join(run_dir, "graph_ids.npy")
            if not (os.path.exists(labels_path) and os.path.exists(gids_path)):
                print(f"[SKIP] missing labels/ids in: {run_dir}")
                continue

            y = np.load(labels_path, allow_pickle=False).astype(np.int64, copy=False)
            graph_ids = np.load(gids_path, allow_pickle=False).astype(np.int64, copy=False)
            if y.ndim != 1 or graph_ids.ndim != 1 or y.shape[0] != graph_ids.shape[0]:
                print(f"[SKIP] bad labels/ids shapes in: {run_dir}")
                continue

            emb_files = discover_embedding_files(run_dir)
            if not emb_files:
                print(f"[SKIP] no embeddings found in: {run_dir}")
                continue

            if EPOCHS_TO_EVAL:
                keep = set(int(e) for e in EPOCHS_TO_EVAL)
                emb_files = [p for p in emb_files if (infer_epoch_from_name(p) in keep)]

            emb_files = sorted(emb_files, key=lambda p: (infer_epoch_from_name(p) or 10**9, p))

            ncls = int(len(np.unique(y)))
            minc = _min_class_count(y)
            print(f"\n[RUN] {run_name} | N={y.shape[0]} | classes={ncls} | min_class_count={minc} | files={len(emb_files)}")

            for emb_path in emb_files:
                epoch = infer_epoch_from_name(emb_path)
                if epoch is None:
                    continue

                emb_file = os.path.basename(emb_path)
                p_val = infer_perturbation_p(f"{run_name}/{emb_file}")
                if p_val is None and ASSUME_MISSING_P_IS_CLEAN:
                    p_val = float(CLEAN_P_VALUE)

                try:
                    X = load_embeddings(emb_path)
                except Exception as e:
                    print(f"  [SKIP] failed loading {emb_file}: {e}")
                    continue

                if X.shape[0] != y.shape[0]:
                    print(f"  [SKIP] shape mismatch: {emb_file} X={X.shape} y={y.shape}")
                    continue

                D_name = infer_dim_from_name(emb_file)
                D = int(D_name) if D_name is not None else int(X.shape[1])

                print(f"\n  [EMB] epoch={epoch:03d} | D={D} | p={fmt(p_val)} | file={emb_file}")

                # FAST tune
                Xtr, Xva, ytr, yva = stratified_train_val_split(
                    X, y, train_frac=float(FAST_TUNE_TRAIN_FRAC), seed=int(FAST_TUNE_SEED + epoch)
                )

                fast_results: List[FastTuneResult] = []
                print("    [FAST TUNE]")
                for spec in MODEL_SPECS:
                    res = fast_tune_single_split(
                        spec, Xtr, ytr, Xva, yva,
                        metric=str(FAST_SELECT_METRIC),
                        cap_combos=HYPERPARAM_MAX_COMBOS_PER_MODEL,
                        seed=int(GLOBAL_SEED + epoch * 17),
                    )
                    fast_results.append(res)
                    print(
                        f"      {res.model_name:<18s} best_val_{FAST_SELECT_METRIC}={res.best_val_score:.4f} "
                        f"tried={res.n_tried:<4d} failed={res.n_failed:<3d} sec={res.elapsed_sec:.2f}"
                    )

                # CV all models
                print("    [CV] all models")
                current_rows_for_embedding: List[ModelEvalRecord] = []
                for fr in fast_results:
                    cvm = cv_full_data_all_metrics(
                        model_key=fr.model_key,
                        params=fr.best_params,
                        X=X,
                        y=y,
                        n_splits=int(CV_N_SPLITS),
                        n_repeats=int(CV_N_REPEATS),
                        seed=int(CV_SEED + epoch * 19),
                    )
                    print(
                        f"      {fr.model_name:<18s} "
                        f"cv_acc={fmt(cvm.acc_mean)}±{fmt(cvm.acc_std)} "
                        f"cv_f1={fmt(cvm.f1_mean)}±{fmt(cvm.f1_std)} "
                        f"cv_auc={fmt(cvm.auc_mean)}±{fmt(cvm.auc_std)} "
                        f"folds={cvm.n_folds:<3d} sec={cvm.elapsed_sec:.2f}"
                    )

                    rec = ModelEvalRecord(
                        dataset_root=dataset_root,
                        run_name=run_name,
                        epoch=int(epoch),
                        D=int(D),
                        p=p_val,
                        emb_path=emb_path,
                        N=int(X.shape[0]),
                        n_classes=int(ncls),

                        model_key=fr.model_key,
                        model_name=fr.model_name,
                        fast_select_metric=str(FAST_SELECT_METRIC),
                        fast_best_val_score=float(fr.best_val_score),
                        fast_best_params=dict(fr.best_params),

                        cv_acc_mean=float(cvm.acc_mean),
                        cv_acc_std=float(cvm.acc_std),
                        cv_f1_mean=float(cvm.f1_mean),
                        cv_f1_std=float(cvm.f1_std),
                        cv_auc_mean=float(cvm.auc_mean) if cvm.auc_mean is not None else None,
                        cv_auc_std=float(cvm.auc_std) if cvm.auc_std is not None else None,
                        cv_folds=int(cvm.n_folds),
                        cv_time_sec=float(cvm.elapsed_sec),
                    )
                    model_eval_records.append(rec)
                    current_rows_for_embedding.append(rec)

                fixed_svc_acc = fixed_svc_f1 = fixed_svc_auc = None
                if COMPUTE_FIXED_SVC_FOR_STABILITY:
                    try:
                        cvm_fix = cv_full_data_all_metrics(
                            model_key="svc_rbf",
                            params=dict(FIXED_SVC_PARAMS),
                            X=X,
                            y=y,
                            n_splits=int(CV_N_SPLITS),
                            n_repeats=int(CV_N_REPEATS),
                            seed=int(CV_SEED + epoch * 23),
                        )
                        fixed_svc_acc = float(cvm_fix.acc_mean)
                        fixed_svc_f1 = float(cvm_fix.f1_mean)
                        fixed_svc_auc = float(cvm_fix.auc_mean) if cvm_fix.auc_mean is not None else None
                        print(
                            f"    [CV-FIXED-SVC] cv_acc={fmt(fixed_svc_acc)} "
                            f"cv_f1={fmt(fixed_svc_f1)} cv_auc={fmt(fixed_svc_auc)}"
                        )
                    except Exception as e:
                        print(f"    [WARN] fixed SVC CV failed: {e}")

                best_row = sorted(
                    current_rows_for_embedding,
                    key=lambda rr: (-rr.cv_f1_mean, -rr.cv_acc_mean, rr.model_name)
                )[0]

                emb_summary_records.append(
                    EmbeddingSummaryRecord(
                        dataset_root=dataset_root,
                        run_name=run_name,
                        epoch=int(epoch),
                        D=int(D),
                        p=p_val,
                        emb_path=emb_path,
                        emb_file=emb_file,
                        N=int(X.shape[0]),
                        n_classes=int(ncls),
                        best_model_key=best_row.model_key,
                        best_model_name=best_row.model_name,
                        best_cv_acc_mean=float(best_row.cv_acc_mean),
                        best_cv_f1_mean=float(best_row.cv_f1_mean),
                        best_cv_auc_mean=float(best_row.cv_auc_mean) if best_row.cv_auc_mean is not None else None,
                        fixed_svc_cv_acc_mean=fixed_svc_acc,
                        fixed_svc_cv_f1_mean=fixed_svc_f1,
                        fixed_svc_cv_auc_mean=fixed_svc_auc,
                    )
                )

                # clustering
                kmeans_ari = kmeans_sil = None
                spectral_ari = spectral_sil = None

                if DO_KMEANS:
                    clu_km, _ = cluster_kmeans(X, y, seed=int(CLUSTER_SEED + epoch))
                    kmeans_ari = float(clu_km.ari)
                    kmeans_sil = float(clu_km.silhouette) if clu_km.silhouette is not None else None
                    print(f"    [CLUSTER] kmeans    ARI={fmt(kmeans_ari)} silhouette={fmt(kmeans_sil)}")

                if DO_SPECTRAL and HAS_SPECTRAL:
                    try:
                        clu_sp, _ = cluster_spectral(X, y, seed=int(CLUSTER_SEED + 1000 + epoch))
                        spectral_ari = float(clu_sp.ari)
                        spectral_sil = float(clu_sp.silhouette) if clu_sp.silhouette is not None else None
                        print(f"    [CLUSTER] spectral  ARI={fmt(spectral_ari)} silhouette={fmt(spectral_sil)}")
                    except Exception as e:
                        print(f"    [WARN] spectral clustering failed: {e}")

                cluster_records.append(
                    EmbeddingClusterRecord(
                        dataset_root=dataset_root,
                        run_name=run_name,
                        epoch=int(epoch),
                        D=int(D),
                        p=p_val,
                        emb_path=emb_path,
                        N=int(X.shape[0]),
                        n_classes=int(ncls),
                        kmeans_ari=kmeans_ari,
                        kmeans_sil=kmeans_sil,
                        spectral_ari=spectral_ari,
                        spectral_sil=spectral_sil,
                    )
                )

                if COMPUTE_STABILITY or MAKE_PLOTS:
                    emb_cache[emb_path] = X

    if not model_eval_records:
        print("\n[DONE] No embeddings evaluated.")
        return

    # dim sweep summary from clean embeddings
    if BUILD_DIM_SWEEP_SUMMARY:
        clean_rows = [
            r for r in emb_summary_records
            if r.p is not None and abs(float(r.p) - float(CLEAN_P_VALUE)) < 1e-12
        ]
        grp: Dict[Tuple[str, str, int], List[EmbeddingSummaryRecord]] = {}
        for r in clean_rows:
            grp.setdefault((r.dataset_root, r.run_name, r.D), []).append(r)

        for (root, run, d), rows in grp.items():
            best = sorted(rows, key=lambda z: (-z.best_cv_f1_mean, -z.best_cv_acc_mean, z.epoch, z.emb_file))[0]
            dim_rows.append(
                DimSweepSummaryRecord(
                    dataset_root=root,
                    run_name=run,
                    D=int(d),
                    best_epoch=int(best.epoch),
                    best_emb_file=str(best.emb_file),
                    best_model_name=str(best.best_model_name),
                    best_cv_acc_mean=float(best.best_cv_acc_mean),
                    best_cv_f1_mean=float(best.best_cv_f1_mean),
                    best_cv_auc_mean=float(best.best_cv_auc_mean) if best.best_cv_auc_mean is not None else None,
                    fixed_svc_cv_acc_mean=float(best.fixed_svc_cv_acc_mean) if best.fixed_svc_cv_acc_mean is not None else None,
                    fixed_svc_cv_f1_mean=float(best.fixed_svc_cv_f1_mean) if best.fixed_svc_cv_f1_mean is not None else None,
                    fixed_svc_cv_auc_mean=float(best.fixed_svc_cv_auc_mean) if best.fixed_svc_cv_auc_mean is not None else None,
                )
            )

    # stability
    if COMPUTE_STABILITY:
        clean_map: Dict[Tuple[str, str, int, int], EmbeddingSummaryRecord] = {}
        pert_rows: List[EmbeddingSummaryRecord] = []

        for r in emb_summary_records:
            if r.p is None:
                continue
            p = float(r.p)
            key = (r.dataset_root, r.run_name, int(r.D), int(r.epoch))
            if abs(p - float(CLEAN_P_VALUE)) < 1e-12:
                clean_map[key] = r
            elif p > 0.0:
                pert_rows.append(r)

        for pr in pert_rows:
            key = (pr.dataset_root, pr.run_name, int(pr.D), int(pr.epoch))
            cr = clean_map.get(key)
            if cr is None:
                continue

            X_clean = emb_cache.get(cr.emb_path)
            if X_clean is None:
                X_clean = load_embeddings(cr.emb_path)
                emb_cache[cr.emb_path] = X_clean

            X_pert = emb_cache.get(pr.emb_path)
            if X_pert is None:
                X_pert = load_embeddings(pr.emb_path)
                emb_cache[pr.emb_path] = X_pert

            if X_clean.shape != X_pert.shape:
                continue

            cos = mean_cosine_after_alignment(X_clean, X_pert)
            d_acc_best = float(pr.best_cv_acc_mean - cr.best_cv_acc_mean)
            d_f1_best = float(pr.best_cv_f1_mean - cr.best_cv_f1_mean)

            d_acc_svc = d_f1_svc = None
            if cr.fixed_svc_cv_acc_mean is not None and pr.fixed_svc_cv_acc_mean is not None:
                d_acc_svc = float(pr.fixed_svc_cv_acc_mean - cr.fixed_svc_cv_acc_mean)
            if cr.fixed_svc_cv_f1_mean is not None and pr.fixed_svc_cv_f1_mean is not None:
                d_f1_svc = float(pr.fixed_svc_cv_f1_mean - cr.fixed_svc_cv_f1_mean)

            stab_rows.append(
                StabilityRecord(
                    dataset_root=pr.dataset_root,
                    run_name=pr.run_name,
                    D=int(pr.D),
                    epoch=int(pr.epoch),
                    p=float(pr.p if pr.p is not None else 0.0),
                    clean_emb_file=cr.emb_file,
                    pert_emb_file=pr.emb_file,
                    mean_cosine_aligned=float(cos),
                    delta_acc_best_model=float(d_acc_best),
                    delta_f1_best_model=float(d_f1_best),
                    delta_acc_fixed_svc=float(d_acc_svc) if d_acc_svc is not None else None,
                    delta_f1_fixed_svc=float(d_f1_svc) if d_f1_svc is not None else None,
                )
            )

    # plots
    if MAKE_PLOTS:
        clean = [
            r for r in emb_summary_records
            if r.p is not None and abs(float(r.p) - float(CLEAN_P_VALUE)) < 1e-12
        ]
        if PLOT_ONLY_BEST_CLEAN_PER_RUN:
            per_run: Dict[Tuple[str, str], EmbeddingSummaryRecord] = {}
            for r in clean:
                k = (r.dataset_root, r.run_name)
                cur = per_run.get(k)
                if cur is None or (r.best_cv_f1_mean > cur.best_cv_f1_mean) or (
                    r.best_cv_f1_mean == cur.best_cv_f1_mean and r.best_cv_acc_mean > cur.best_cv_acc_mean
                ):
                    per_run[k] = r
            candidates = list(per_run.values())
        else:
            candidates = clean

        for r in sorted(candidates, key=lambda z: (z.dataset_root, z.run_name, z.epoch, z.D)):
            X = emb_cache.get(r.emb_path)
            if X is None:
                X = load_embeddings(r.emb_path)
                emb_cache[r.emb_path] = X

            run_dir = os.path.dirname(os.path.dirname(r.emb_path))
            labels_path = os.path.join(run_dir, "labels.npy")
            if not os.path.exists(labels_path):
                continue
            y = np.load(labels_path, allow_pickle=False).astype(np.int64, copy=False)

            idx = _select_plot_indices(y, max_points=int(PLOT_MAX_POINTS), seed=int(PLOT_SEED + r.epoch + r.D))
            Xs = X[idx]
            ys = y[idx]

            run_slug = r.run_name.replace("/", "__").replace("\\", "__")
            base_name = f"{os.path.basename(r.dataset_root.rstrip('/'))}__{run_slug}__epoch{r.epoch:03d}__D{r.D}"

            tsne_path = None
            umap_path = None

            try:
                X2_tsne = make_tsne(Xs, seed=int(PLOT_SEED + 1 + r.epoch))
                tsne_path = os.path.join(OUT_DIR, "plots", base_name + "__tsne.png")
                plot_2d_scatter(X2_tsne, ys, f"{r.run_name} | epoch={r.epoch} D={r.D} | t-SNE", tsne_path)
            except Exception as e:
                print(f"[WARN] t-SNE failed for {r.emb_file}: {e}")

            try:
                X2_umap = make_umap(Xs, seed=int(PLOT_SEED + 2 + r.epoch))
            except Exception:
                X2_umap = None

            if X2_umap is not None:
                try:
                    umap_path = os.path.join(OUT_DIR, "plots", base_name + "__umap.png")
                    plot_2d_scatter(X2_umap, ys, f"{r.run_name} | epoch={r.epoch} D={r.D} | UMAP", umap_path)
                except Exception as e:
                    print(f"[WARN] UMAP plot failed for {r.emb_file}: {e}")

            plot_rows.append(
                PlotRecord(
                    dataset_root=r.dataset_root,
                    run_name=r.run_name,
                    epoch=int(r.epoch),
                    D=int(r.D),
                    p=r.p,
                    emb_file=r.emb_file,
                    tsne_path=tsne_path,
                    umap_path=umap_path,
                    n_points_plotted=int(idx.size),
                )
            )

    # save outputs
    model_rows = [asdict(r) for r in model_eval_records]
    cluster_rows = [asdict(r) for r in cluster_records]
    emb_summary_rows = [asdict(r) for r in emb_summary_records]
    dim_rows_dict = [asdict(r) for r in dim_rows]
    stab_rows_dict = [asdict(r) for r in stab_rows]
    plot_rows_dict = [asdict(r) for r in plot_rows]

    if SAVE_JSONL:
        write_jsonl(os.path.join(OUT_DIR, "classification_all_models_cv_only.jsonl"), model_rows)
        write_jsonl(os.path.join(OUT_DIR, "clustering_results.jsonl"), cluster_rows)
        write_jsonl(os.path.join(OUT_DIR, "embedding_summary.jsonl"), emb_summary_rows)
        write_jsonl(os.path.join(OUT_DIR, "dimension_sweep_summary.jsonl"), dim_rows_dict)
        write_jsonl(os.path.join(OUT_DIR, "stability_results.jsonl"), stab_rows_dict)
        write_jsonl(os.path.join(OUT_DIR, "plots_index.jsonl"), plot_rows_dict)

    if SAVE_CSV:
        write_csv(os.path.join(OUT_DIR, "classification_all_models_cv_only.csv"), model_rows)
        write_csv(os.path.join(OUT_DIR, "clustering_results.csv"), cluster_rows)
        write_csv(os.path.join(OUT_DIR, "embedding_summary.csv"), emb_summary_rows)
        write_csv(os.path.join(OUT_DIR, "dimension_sweep_summary.csv"), dim_rows_dict)
        write_csv(os.path.join(OUT_DIR, "stability_results.csv"), stab_rows_dict)
        write_csv(os.path.join(OUT_DIR, "plots_index.csv"), plot_rows_dict)

    if SAVE_RESULTS_JSON:
        merged = {
            "config": {
                "dataset_roots": DATASET_ROOTS,
                "epochs_to_eval": EPOCHS_TO_EVAL,
                "fast_tune_train_frac": FAST_TUNE_TRAIN_FRAC,
                "fast_tune_seed": FAST_TUNE_SEED,
                "fast_select_metric": FAST_SELECT_METRIC,
                "cv_splits": CV_N_SPLITS,
                "cv_repeats": CV_N_REPEATS,
                "cv_seed": CV_SEED,
                "compute_stability": COMPUTE_STABILITY,
                "clean_p_value": CLEAN_P_VALUE,
                "make_plots": MAKE_PLOTS,
            },
            "counts": {
                "classification_rows": len(model_rows),
                "clustering_rows": len(cluster_rows),
                "embedding_summary_rows": len(emb_summary_rows),
                "dim_sweep_rows": len(dim_rows_dict),
                "stability_rows": len(stab_rows_dict),
                "plot_rows": len(plot_rows_dict),
            },
            "classification": model_rows,
            "clustering": cluster_rows,
            "embedding_summary": emb_summary_rows,
            "dimension_sweep_summary": dim_rows_dict,
            "stability_results": stab_rows_dict,
            "plots_index": plot_rows_dict,
        }
        with open(os.path.join(OUT_DIR, "results_merged.json"), "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print("  - Classification       :", os.path.abspath(os.path.join(OUT_DIR, "classification_all_models_cv_only.csv")))
    print("  - Clustering           :", os.path.abspath(os.path.join(OUT_DIR, "clustering_results.csv")))
    print("  - Embedding summary    :", os.path.abspath(os.path.join(OUT_DIR, "embedding_summary.csv")))
    print("  - Dimension sweep      :", os.path.abspath(os.path.join(OUT_DIR, "dimension_sweep_summary.csv")))
    print("  - Stability            :", os.path.abspath(os.path.join(OUT_DIR, "stability_results.csv")))
    print("  - Plots index          :", os.path.abspath(os.path.join(OUT_DIR, "plots_index.csv")))
    print("  - Merged JSON          :", os.path.abspath(os.path.join(OUT_DIR, "results_merged.json")))
    print("  - Plots directory      :", os.path.abspath(os.path.join(OUT_DIR, "plots")))


if __name__ == "__main__":
    main()
