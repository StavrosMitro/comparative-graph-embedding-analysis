#!/usr/bin/env python3
# all_in_one_wl_to_g2v_online.py
#
# Single-file pipeline:
#   A) Download TU dataset from internet (optionally force redownload)
#   B) Build per-graph WL BoW JSONL(.gz)
#   C) Train PV-DBOW Graph2Vec embeddings from that BoW
#   D) Save selected-epoch embeddings to disk
#
# Input dataset: TU Dortmund collection via torch_geometric.datasets.TUDataset
# Output:
#   - BoW JSONL(.gz): one graph per line
#   - Embeddings at selected epochs (npy or pt), plus graph_ids/labels arrays

import os

# -----------------------------------------------------------------------------
# CPU-ONLY: disable CUDA before importing torch
# -----------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DISABLE_CUDA"] = "1"

import gc
import gzip
import json
import random
import shutil
import hashlib
import warnings
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Literal, Any, Iterator
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, remove_self_loops

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIG
# =============================================================================
SEED = 42

# --------------------------
# Dataset download / loading
# --------------------------
DATASET_NAME = "ENZYMES"
DATASET_ROOT = "./data/TUDataset_online"
USE_NODE_ATTR = True

# If True, removes DATASET_ROOT/DATASET_NAME before loading, forcing internet download again.
FORCE_REDOWNLOAD_DATASET = True

# --------------------------
# Stage A output: WL->BoW
# --------------------------
BOW_OUTPUT_PATH = "bows_per_graph_enzymes.jsonl.gz"

# Initial WL labels:
#   - "constant": all nodes start with label 0
#   - "degree": node degree in processed undirected graph (no LCC restriction)
#   - "ftl": quantize node features using FTL (trained once on full dataset)
INITIAL_WL_FROM: Literal["constant", "degree", "ftl"] = "ftl"

WL_H = 1
KEEP_WL_FROM_EVERY_ITER = True

# Document-frequency filter across graphs:
# keep features with df > VOCAB_MIN_COUNT
VOCAB_MIN_COUNT = 1

# Count handling:
#   0 -> raw WL counts
#   1 -> presence only
#  >1 -> per-feature global binned counts in [1..COUNT_BINS]
COUNT_BINS = 10

# FTL config (only if INITIAL_WL_FROM == "ftl")
BASE_GROUPS = [list(range(0, 18)), [18, 19, 20]]
FTL_METHOD: Literal["kmeans", "uniform"] = "kmeans"
FTL_TOTAL_LABELS = 128

# --------------------------
# Stage B output: embeddings
# --------------------------
OUTPUT_DIR = "./g2v_embeddings_selected_epochs_named"
OVERWRITE_OUTPUT = True

SAVE_EPOCHS: List[int] = [20]
EMB_SAVE_FORMAT = "npy"  # "npy" or "pt"
SAVE_GRAPH_IDS_AND_LABELS = True

RUN_SWEEP: List[Dict[str, Any]] = [
    {"name": "tok_counts_occ",     "unique_embeddings_for_counts": False, "pair_thermometer": False},
    {"name": "pair_counts_unique", "unique_embeddings_for_counts": True,  "pair_thermometer": False},
    {"name": "pair_counts_thermo", "unique_embeddings_for_counts": True,  "pair_thermometer": True},
]

EMB_DIM = 128
NUM_NEG = 15
LR_ALPHA = 0.05
MAX_EPOCHS = max((int(e) for e in SAVE_EPOCHS if int(e) >= 1), default=1)

USE_LINEAR_LR_DECAY = True
MIN_LR_FRACTION = 0.1
SHUFFLE_GRAPHS_EACH_EPOCH = True
SHUFFLE_POSITIVES_WITHIN_GRAPH = True

SORT_FEATURE_VOCAB = True
MAX_OCC_PER_TOKEN: Optional[int] = None
NEG_POW: float = 0.75
NEGATIVE_SAMPLING_NO_TABLE: bool = True  # kept knob

DESCRIPTIVE_EMB_FILENAMES = True
ADD_RUNID_IN_FILENAME = True
# =============================================================================


# =============================================================================
# Small utilities
# =============================================================================
def now() -> float:
    return time.perf_counter()

def fmt_sec(x: float) -> str:
    if x < 1:
        return f"{x*1000:.1f} ms"
    if x < 60:
        return f"{x:.2f} s"
    return f"{x/60:.2f} min"

def human_bytes(x: int) -> str:
    if x < 1024:
        return f"{x} B"
    if x < 1024**2:
        return f"{x/1024:.2f} KiB"
    if x < 1024**3:
        return f"{x/(1024**2):.2f} MiB"
    return f"{x/(1024**3):.2f} GiB"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def wipe_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def stable_hash64(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()

def stable_int_hash(*parts: Any, mod: int = 2**31 - 1) -> int:
    s = "|".join(str(p) for p in parts)
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    x = int.from_bytes(h, byteorder="little", signed=False)
    return int(x % int(mod))

def file_stem(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".jsonl.gz"):
        return base[:-len(".jsonl.gz")]
    if base.endswith(".jsonl"):
        return base[:-len(".jsonl")]
    if "." in base:
        return base.rsplit(".", 1)[0]
    return base

def short_run_id(*parts: Any, n: int = 10) -> str:
    s = "|".join(str(p) for p in parts)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()[:int(n)]


# =============================================================================
# Dataset loading (internet-first)
# =============================================================================
def load_tu_dataset_from_internet(
    root: str,
    name: str,
    use_node_attr: bool,
    force_redownload: bool,
) -> List[Data]:
    """
    If force_redownload=True, deletes local dataset directory and triggers fresh download.
    TUDataset handles internet download automatically when files are missing.
    """
    ds_dir = os.path.join(root, name)
    if force_redownload and os.path.exists(ds_dir):
        shutil.rmtree(ds_dir)

    # Some PyG versions support force_reload; fallback if unavailable.
    try:
        ds = TUDataset(
            root=root,
            name=name,
            use_node_attr=use_node_attr,
            force_reload=bool(force_redownload),
        )
    except TypeError:
        ds = TUDataset(
            root=root,
            name=name,
            use_node_attr=use_node_attr,
        )

    return list(ds)


def hygienize_graph_data(data: Data) -> Data:
    edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index = edge_index

    if data.x is None:
        num_nodes = int(data.num_nodes) if data.num_nodes is not None else 0
        data.x = torch.ones((num_nodes, 1), dtype=torch.float32)

    return data

def pyg_to_nx_undirected(data: Data) -> nx.Graph:
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# =============================================================================
# FTL helpers (no sklearn)
# =============================================================================
class StandardScalerDense:
    def __init__(self, with_mean: bool = True):
        self.with_mean = with_mean
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScalerDense":
        X = np.asarray(X, dtype=np.float32)
        if self.with_mean:
            self.mean_ = X.mean(axis=0, keepdims=True)
        else:
            self.mean_ = np.zeros((1, X.shape[1]), dtype=np.float32)

        var = ((X - self.mean_) ** 2).mean(axis=0, keepdims=True)
        self.std_ = np.sqrt(var).astype(np.float32)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted.")
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean_) / self.std_).astype(np.float32)


class SimpleKMeans:
    def __init__(self, n_clusters: int, n_init: int = 5, max_iter: int = 50, seed: int = 42):
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.centers_: Optional[np.ndarray] = None

    @staticmethod
    def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        X2 = (X * X).sum(axis=1, keepdims=True)
        C2 = (C * C).sum(axis=1, keepdims=True).T
        return X2 + C2 - 2.0 * (X @ C.T)

    def fit(self, X: np.ndarray) -> "SimpleKMeans":
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)

        best_inertia = None
        best_centers = None

        for _ in range(self.n_init):
            idx = rng.choice(n, size=self.n_clusters, replace=(n < self.n_clusters))
            centers = X[idx].copy()

            for _it in range(self.max_iter):
                d2 = self._sq_dists(X, centers)
                labels = d2.argmin(axis=1)

                new_centers = centers.copy()
                for k in range(self.n_clusters):
                    mask = (labels == k)
                    if np.any(mask):
                        new_centers[k] = X[mask].mean(axis=0)
                    else:
                        new_centers[k] = X[rng.integers(0, n)]

                shift = float(np.mean((new_centers - centers) ** 2))
                centers = new_centers
                if shift < 1e-8:
                    break

            d2 = self._sq_dists(X, centers)
            inertia = float(np.sum(np.min(d2, axis=1)))
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()

        self.centers_ = best_centers
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("KMeans not fitted.")
        X = np.asarray(X, dtype=np.float32)
        d2 = self._sq_dists(X, self.centers_)
        return d2.argmin(axis=1).astype(np.int64)


@dataclass
class FTLConfig:
    method: str
    total_labels: int
    feature_groups: List[List[int]]


class FeaturesToDiscreteLabels:
    """
    transform_graph() returns per-node labels as strings (64-bit hex hash).
    """

    def __init__(self, cfg: FTLConfig, seed: int = 42):
        self.cfg = cfg
        self.seed = seed

        self._models: List[Optional[Tuple[SimpleKMeans, StandardScalerDense]]] = []
        self._bins: List[Optional[Tuple[List[np.ndarray], int, int]]] = []
        self._k_per_group: List[int] = []
        self._sel_idx: Optional[List[int]] = None
        self._groups_cols: Optional[List[List[int]]] = None

    def _select_features(self, X: np.ndarray) -> np.ndarray:
        idx = sorted({j for g in self.cfg.feature_groups for j in g})
        self._sel_idx = idx
        return X[:, idx]

    def fit(self, graphs_data_list: List[Data]) -> "FeaturesToDiscreteLabels":
        X_all = []
        for d in graphs_data_list:
            if d.x is None:
                raise ValueError("Dataset has no node features (x is None).")
            X_all.append(d.x.detach().cpu().numpy())
        X_all = np.vstack(X_all).astype(np.float32)

        X_sel = self._select_features(X_all)

        if self._sel_idx is None:
            raise RuntimeError("Feature selection failed.")

        orig_to_sel = {orig_i: k for k, orig_i in enumerate(self._sel_idx)}
        groups_cols: List[List[int]] = []
        for g in self.cfg.feature_groups:
            cols = [orig_to_sel[j] for j in g if j in orig_to_sel]
            if cols:
                groups_cols.append(cols)
        if not groups_cols:
            raise ValueError("No valid feature indices in feature_groups for this dataset.")
        self._groups_cols = groups_cols

        G = len(groups_cols)
        per_group = max(2, self.cfg.total_labels // G)
        labels_per_group = [per_group] * G
        labels_per_group[-1] = max(2, self.cfg.total_labels - per_group * (G - 1))

        self._models = []
        self._bins = []
        self._k_per_group = []

        for gi, cols in enumerate(groups_cols):
            k = int(labels_per_group[gi])
            Xg = X_sel[:, cols].astype(np.float32)
            self._k_per_group.append(k)

            if self.cfg.method == "kmeans":
                scaler = StandardScalerDense(with_mean=True).fit(Xg)
                Xg_std = scaler.transform(Xg)
                km = SimpleKMeans(n_clusters=k, n_init=5, max_iter=50, seed=self.seed).fit(Xg_std)
                self._models.append((km, scaler))
                self._bins.append(None)

            elif self.cfg.method == "uniform":
                d = Xg.shape[1]
                b = max(2, int(round(k ** (1.0 / max(1, d)))))
                edges = []
                for j in range(d):
                    col = Xg[:, j]
                    lo, hi = float(np.min(col)), float(np.max(col))
                    if lo == hi:
                        e = np.array([lo, hi], dtype=np.float32)
                    else:
                        e = np.linspace(lo, hi, b + 1).astype(np.float32)
                    edges.append(e)
                self._bins.append((edges, b, k))
                self._models.append(None)

            else:
                raise ValueError(f"Unknown method: {self.cfg.method}")

        return self

    def transform_graph(self, data: Data) -> List[str]:
        if data.x is None:
            raise ValueError("Dataset has no node features (x is None).")
        if self._sel_idx is None or self._groups_cols is None:
            raise RuntimeError("FTL not fitted.")

        x = data.x.detach().cpu().numpy().astype(np.float32)
        X_sel = x[:, self._sel_idx].astype(np.float32)

        n_nodes = X_sel.shape[0]
        per_group_codes: List[np.ndarray] = []

        for gi, cols in enumerate(self._groups_cols):
            k = int(self._k_per_group[gi])

            if self.cfg.method == "kmeans":
                km, scaler = self._models[gi]  # type: ignore[misc]
                Xg = X_sel[:, cols]
                z = km.predict(scaler.transform(Xg))
                per_group_codes.append(z.astype(np.int64))
            else:
                edges, b, _k = self._bins[gi]  # type: ignore[misc]
                Xg = X_sel[:, cols]
                digits = []
                for j in range(Xg.shape[1]):
                    e = edges[j]
                    dj = np.clip(np.digitize(Xg[:, j], e[1:-1], right=False), 0, b - 1)
                    digits.append(dj)
                digits = np.stack(digits, axis=1)

                code = np.zeros((digits.shape[0],), dtype=np.int64)
                mult = 1
                for j in range(digits.shape[1]):
                    code += digits[:, j] * mult
                    mult *= b
                code = code % k
                per_group_codes.append(code.astype(np.int64))

        labels: List[str] = []
        for i in range(n_nodes):
            sig = "|".join(str(int(arr[i])) for arr in per_group_codes)
            labels.append(stable_hash64(sig))
        return labels


# =============================================================================
# WL helpers
# =============================================================================
def constant_initial_labels(data: Data, value: int = 0) -> List[int]:
    n = int(data.num_nodes) if data.num_nodes is not None else 0
    return [int(value)] * n

def degree_initial_labels(data: Data, G: nx.Graph) -> List[int]:
    n = int(data.num_nodes) if data.num_nodes is not None else 0
    labels = [0] * n
    if G.number_of_nodes() == 0:
        return labels

    deg = dict(G.degree())
    for node, d in deg.items():
        ni = int(node)
        if 0 <= ni < n:
            labels[ni] = int(d)
    return labels

def initial_labels_for_graph(
    mode: str,
    data: Data,
    G: nx.Graph,
    ftl: Optional[FeaturesToDiscreteLabels],
) -> List[Any]:
    if mode == "constant":
        return constant_initial_labels(data, value=0)
    if mode == "degree":
        return degree_initial_labels(data, G)
    if mode == "ftl":
        if ftl is None:
            raise ValueError("INITIAL_WL_FROM='ftl' but FTL model is None.")
        return ftl.transform_graph(data)
    raise ValueError(f"Unknown INITIAL_WL_FROM: {mode}")

def wl_subtree_counts(G: nx.Graph, node_labels: List[Any], h: int, keep_from_every_iter: bool) -> Counter:
    if G.number_of_nodes() == 0:
        return Counter()

    nodes = sorted(G.nodes())
    cur = {n: str(node_labels[int(n)]) for n in nodes}

    feats = Counter()

    if keep_from_every_iter or h == 0:
        for n in nodes:
            feats[f"0:{cur[n]}"] += 1

    for it in range(1, h + 1):
        new = {}
        for n in nodes:
            neigh = sorted(cur[v] for v in G.neighbors(n))
            s = cur[n] + "|" + ",".join(neigh)
            new[n] = stable_hash64(f"{it}:{s}")
        cur = new

        if keep_from_every_iter:
            for n in nodes:
                feats[f"{it}:{cur[n]}"] += 1

    if (not keep_from_every_iter) and h > 0:
        for n in nodes:
            feats[f"{h}:{cur[n]}"] += 1

    return feats

def build_keep_set_docfreq(all_counters: List[Counter], min_count: int) -> Optional[set]:
    mc = int(min_count)
    if mc <= 0:
        return None

    df = Counter()
    for c in all_counters:
        for feat in c.keys():
            df[feat] += 1

    keep = {feat for feat, cnt in df.items() if cnt > mc}
    return keep

def make_per_feature_binner_global(
    all_counters: List[Counter],
    keep_set: Optional[set],
    count_bins: int,
) -> Tuple[Callable[[str, int], int], Dict[str, Any]]:
    B = int(count_bins)

    if B <= 1:
        return (lambda _feat, _v: 1), {"mode": "presence_only", "bins": max(0, B), "per_feature": True}

    vals: Dict[str, List[float]] = {}
    for c in all_counters:
        for feat, val in c.items():
            if keep_set is not None and feat not in keep_set:
                continue
            vals.setdefault(feat, []).append(float(val))

    feat_ranges: Dict[str, Tuple[float, float, bool]] = {}
    for feat, vlist in vals.items():
        if not vlist:
            feat_ranges[feat] = (1.0, 1.0, True)
            continue
        mn = float(min(vlist))
        mx = float(max(vlist))
        is_const = (mx <= mn + 1e-12)
        feat_ranges[feat] = (mn, mx, is_const)

    def bin_fn(feat: str, v: int) -> int:
        rec = feat_ranges.get(feat)
        if rec is None:
            return 1
        mn, mx, is_const = rec
        if is_const:
            return 1
        x = float(v)
        if x < mn:
            x = mn
        if x > mx:
            x = mx
        t = (x - mn) / (mx - mn)
        idx = int(np.floor(t * B))
        if idx >= B:
            idx = B - 1
        return int(idx + 1)

    info = {
        "mode": "binned_per_feature_global",
        "bins": B,
        "per_feature": True,
        "n_features_seen": int(len(feat_ranges)),
        "n_constant_features": int(sum(1 for (_mn, _mx, is_c) in feat_ranges.values() if is_c)),
    }
    return bin_fn, info

def _percentiles(x: List[float], ps=(0, 25, 50, 75, 90, 95, 99, 100)) -> Dict[int, float]:
    if not x:
        return {int(p): float("nan") for p in ps}
    arr = np.asarray(x, dtype=np.float64)
    return {int(p): float(np.percentile(arr, p)) for p in ps}

def print_vocab_report(
    counters: List[Counter],
    keep_set: Optional[set],
    dataset_size: int,
    top_k: int = 15,
):
    df_all = Counter()
    tf_all = Counter()
    for c in counters:
        tf_all.update(c)
        for feat in c.keys():
            df_all[feat] += 1

    all_feats = list(df_all.keys())
    n_vocab_all = len(all_feats)

    if keep_set is None:
        df_kept = df_all
        tf_kept = tf_all
        n_vocab_kept = n_vocab_all
        kept_feats = all_feats
    else:
        kept_feats = [f for f in all_feats if f in keep_set]
        df_kept = Counter({f: df_all[f] for f in kept_feats})
        tf_kept = Counter({f: tf_all[f] for f in kept_feats})
        n_vocab_kept = len(kept_feats)

    dist_all = Counter(df_all.values())
    dist_kept = Counter(df_kept.values())

    nnz_per_graph = []
    for c in counters:
        if keep_set is None:
            nnz_per_graph.append(len(c))
        else:
            nnz_per_graph.append(sum(1 for f in c.keys() if f in keep_set))

    nnz_stats = _percentiles([float(v) for v in nnz_per_graph])

    df_vals = [float(v) for v in df_kept.values()]
    df_stats = _percentiles(df_vals)

    singleton_cnt = int(dist_all.get(1, 0))
    singleton_share = (singleton_cnt / max(1, n_vocab_all)) * 100.0

    top_by_df = sorted(df_kept.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    top_by_tf = sorted(tf_kept.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

    def _iter_of_feat(f: str) -> Optional[int]:
        try:
            pref = f.split(":", 1)[0]
            if pref.isdigit():
                return int(pref)
        except Exception:
            return None
        return None

    df_by_iter = defaultdict(list)
    n_by_iter = Counter()
    for f, d in df_kept.items():
        it = _iter_of_feat(f)
        if it is not None:
            n_by_iter[it] += 1
            df_by_iter[it].append(float(d))

    sep = "=" * 80
    print("\n" + sep)
    print("REPORT")
    print(sep)

    print("CONFIG")
    print(f"graphs: {dataset_size}")
    print(f"wl_h: {WL_H}")
    print(f"keep_from_every_iter: {KEEP_WL_FROM_EVERY_ITER}")
    print(f"initial_wl_from: {INITIAL_WL_FROM}")
    if INITIAL_WL_FROM == "ftl":
        print(f"ftl_method: {FTL_METHOD}")
        print(f"ftl_total_labels: {FTL_TOTAL_LABELS}")
        print(f"ftl_feature_groups: {BASE_GROUPS}")
    print(f"vocab_min_count (drop df<=): {VOCAB_MIN_COUNT}")
    print()

    print("VOCAB SIZE")
    print(f"vocab_all (no df filter): {n_vocab_all}")
    if keep_set is None:
        print("vocab_kept (after df filter): same as vocab_all (VOCAB_MIN_COUNT<=0)")
    else:
        print(f"vocab_kept (after df>{VOCAB_MIN_COUNT}): {n_vocab_kept}")
        print(f"vocab_removed: {n_vocab_all - n_vocab_kept}")
    print()

    print("DOCFREQ DISTRIBUTION (ALL)  #graphs, #tokens")
    print("x_graphs,y_tokens")
    for x in sorted(dist_all.keys()):
        print(f"{x},{dist_all[x]}")
    print(f"singleton_tokens_df1: {singleton_cnt} ({singleton_share:.2f}%)")
    print()

    print("DOCFREQ DISTRIBUTION (KEPT)  #graphs, #tokens")
    print("x_graphs,y_tokens")
    for x in sorted(dist_kept.keys()):
        print(f"{x},{dist_kept[x]}")
    print()

    print("DOCFREQ SUMMARY (KEPT)")
    print("df_percentiles: " + ", ".join([f"p{p}={df_stats[p]:.0f}" for p in [0, 25, 50, 75, 90, 95, 99, 100]]))
    print()

    print("PER-GRAPH NNZ (KEPT)")
    print("nnz_percentiles: " + ", ".join([f"p{p}={nnz_stats[p]:.0f}" for p in [0, 25, 50, 75, 90, 95, 99, 100]]))
    print(f"nnz_mean: {float(np.mean(nnz_per_graph)):.2f}  nnz_std: {float(np.std(nnz_per_graph)):.2f}")
    print()

    print(f"TOP TOKENS BY DOCFREQ (KEPT)  (showing {len(top_by_df)})")
    print("rank,token,df_graphs,tf_total")
    for i, (tok, d) in enumerate(top_by_df, 1):
        print(f"{i},{tok},{int(d)},{int(tf_kept.get(tok, 0))}")
    print()

    print(f"TOP TOKENS BY TOTAL COUNT (KEPT)  (showing {len(top_by_tf)})")
    print("rank,token,tf_total,df_graphs")
    for i, (tok, t) in enumerate(top_by_tf, 1):
        print(f"{i},{tok},{int(t)},{int(df_kept.get(tok, 0))}")
    print()

    if n_by_iter:
        print("PER-ITERATION BREAKDOWN (KEPT)")
        print("iter,n_tokens,df_mean,df_median")
        for it in sorted(n_by_iter.keys()):
            dfs = df_by_iter[it]
            df_mean = float(np.mean(dfs)) if dfs else float("nan")
            df_med = float(np.median(dfs)) if dfs else float("nan")
            print(f"{it},{int(n_by_iter[it])},{df_mean:.2f},{df_med:.2f}")
        print()

    print("NOTES")
    print("- tokens are WL features (iteration_prefix:hash)")
    print("- df = number of graphs where token appears at least once")
    print("- tf = total count over all graphs")
    print(sep + "\n")


# =============================================================================
# Stage A: Build WL->BoW JSONL
# =============================================================================
def build_bow_jsonl_from_tu_dataset_online() -> Tuple[str, int, int]:
    """
    Returns:
        (bow_output_path, num_graphs, num_classes)
    """
    seed_everything(SEED)

    print("=" * 100)
    print("[STAGE A] WL -> BoW JSONL")
    print("=" * 100)
    print(f"[DATASET] name={DATASET_NAME} root={DATASET_ROOT}")
    print(f"[DATASET] force_redownload={FORCE_REDOWNLOAD_DATASET}")

    t0 = now()
    dataset = load_tu_dataset_from_internet(
        root=DATASET_ROOT,
        name=DATASET_NAME,
        use_node_attr=USE_NODE_ATTR,
        force_redownload=FORCE_REDOWNLOAD_DATASET,
    )
    dataset = [hygienize_graph_data(d) for d in dataset]
    print(f"[DATASET] loaded graphs={len(dataset)} | time={fmt_sec(now()-t0)}")

    y_raw = np.array([int(d.y.item()) for d in dataset], dtype=np.int64)
    _, y = np.unique(y_raw, return_inverse=True)

    nx_graphs = [pyg_to_nx_undirected(d) for d in dataset]

    ftl: Optional[FeaturesToDiscreteLabels]
    if INITIAL_WL_FROM == "ftl":
        print("[FTL] fitting on full dataset...")
        t_ftl = now()
        fcfg = FTLConfig(
            method=str(FTL_METHOD),
            total_labels=int(FTL_TOTAL_LABELS),
            feature_groups=BASE_GROUPS,
        )
        ftl = FeaturesToDiscreteLabels(fcfg, seed=SEED).fit(dataset)
        print(f"[FTL] done | time={fmt_sec(now()-t_ftl)}")
    else:
        ftl = None

    print("[WL] building per-graph counters...")
    t_wl = now()
    counters: List[Counter] = []
    for i in range(len(dataset)):
        labels = initial_labels_for_graph(INITIAL_WL_FROM, dataset[i], nx_graphs[i], ftl)
        c = wl_subtree_counts(
            nx_graphs[i],
            labels,
            h=int(WL_H),
            keep_from_every_iter=bool(KEEP_WL_FROM_EVERY_ITER),
        )
        counters.append(c)
    print(f"[WL] done | time={fmt_sec(now()-t_wl)}")

    keep_set = build_keep_set_docfreq(counters, min_count=int(VOCAB_MIN_COUNT))

    print_vocab_report(
        counters=counters,
        keep_set=keep_set,
        dataset_size=len(dataset),
        top_k=15,
    )

    bin_fn, bin_info = make_per_feature_binner_global(
        counters,
        keep_set=keep_set,
        count_bins=int(COUNT_BINS),
    )

    print(f"[WRITE] {BOW_OUTPUT_PATH}")
    opener = gzip.open if BOW_OUTPUT_PATH.endswith(".gz") else open
    n_written = 0
    total_nnz = 0
    B = int(COUNT_BINS)

    t_save = now()
    with opener(BOW_OUTPUT_PATH, "wt", encoding="utf-8") as f:
        for i, c in enumerate(counters):
            bow: Dict[str, int] = {}

            if B == 0:
                for feat, cnt in c.items():
                    if keep_set is not None and feat not in keep_set:
                        continue
                    bow[feat] = int(cnt)

            elif B == 1:
                for feat in c.keys():
                    if keep_set is not None and feat not in keep_set:
                        continue
                    bow[feat] = 1

            else:
                for feat, cnt in c.items():
                    if keep_set is not None and feat not in keep_set:
                        continue
                    bow[feat] = int(bin_fn(feat, int(cnt)))

            total_nnz += len(bow)
            rec = {
                "id": int(i),
                "class": int(y[i]),
                "class_raw": int(y_raw[i]),
                "bow": bow,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"[WRITE] done | time={fmt_sec(now()-t_save)}")

    n_graphs = len(dataset)
    n_classes = int(y.max()) + 1 if y.size else 0
    avg_nnz = total_nnz / max(1, n_written)

    print("Saved per-graph BOW JSONL:")
    print(f"  path: {BOW_OUTPUT_PATH}")
    print(f"  graphs: {n_graphs}  classes: {n_classes}")
    print(f"  wl_h: {WL_H}  keep_from_every_iter: {KEEP_WL_FROM_EVERY_ITER}")
    print(f"  initial_wl_from: {INITIAL_WL_FROM}")
    if INITIAL_WL_FROM == "ftl":
        print(f"  ftl: method={FTL_METHOD} total_labels={FTL_TOTAL_LABELS}")
    print(f"  vocab_min_count (drop df<=): {VOCAB_MIN_COUNT}  (0 means keep all)")
    print(f"  count_bins: {COUNT_BINS}")
    print(f"  count_info: {bin_info}")
    print(f"  avg_nonzeros_per_graph: {avg_nnz:.2f}")

    return BOW_OUTPUT_PATH, int(n_graphs), int(n_classes)


# =============================================================================
# Stage B: PV-DBOW from BoW JSONL
# =============================================================================
def _open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "rt", encoding="utf-8")

def iter_jsonl_records(path: str) -> Iterator[dict]:
    with _open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                raise ValueError(f"JSON parse error at line {line_no}: {e}")
            if not isinstance(rec, dict):
                raise ValueError(f"Expected dict at line {line_no}, got {type(rec)}")
            yield rec

def load_all_records(path: str) -> List[dict]:
    recs: List[dict] = []
    for rec in iter_jsonl_records(path):
        if "id" not in rec or "class" not in rec or "bow" not in rec:
            raise ValueError("Each record must have keys: id, class, bow")
        recs.append(rec)
    return recs

@dataclass
class GlobalGraphIndex:
    graph_ids: np.ndarray
    labels: np.ndarray
    labels_raw: np.ndarray
    gid_to_row: Dict[int, int]

def build_global_graph_index_from_recs(recs: List[dict]) -> GlobalGraphIndex:
    gids: List[int] = []
    y_raw: List[int] = []
    seen: set = set()

    for rec in recs:
        gid = int(rec["id"])
        if gid in seen:
            raise ValueError(f"Duplicate graph id detected: {gid}")
        seen.add(gid)

        lab = int(rec["class"])
        lab_raw = int(rec.get("class_raw", lab))
        gids.append(gid)
        y_raw.append(lab_raw)

    order = np.argsort(np.asarray(gids, dtype=np.int64))
    graph_ids = np.asarray(gids, dtype=np.int64)[order]
    labels_raw = np.asarray(y_raw, dtype=np.int64)[order]
    _, labels = np.unique(labels_raw, return_inverse=True)

    gid_to_row = {int(gid): int(i) for i, gid in enumerate(graph_ids.tolist())}
    return GlobalGraphIndex(
        graph_ids=graph_ids,
        labels=labels.astype(np.int64),
        labels_raw=labels_raw.astype(np.int64),
        gid_to_row=gid_to_row,
    )

def _iter_bow_items(rec: dict) -> List[Tuple[str, int]]:
    bow = rec.get("bow", {})
    if bow is None:
        return []
    if not isinstance(bow, dict):
        raise ValueError("bow must be a dict")
    items: List[Tuple[str, int]] = []
    for k, v in bow.items():
        if not isinstance(k, str):
            k = str(k)
        c = int(v)
        if c <= 0:
            continue
        items.append((k, c))
    return items

def build_vocab_from_recs(
    recs: List[dict],
    *,
    unique_embeddings_for_counts: bool,
    pair_thermometer: bool,
    sort_feature_vocab: bool,
) -> Tuple[Dict[Tuple[int, int], int], Dict[str, int], List[str]]:
    features: set = set()
    for rec in recs:
        for feat, _cnt in _iter_bow_items(rec):
            features.add(feat)

    feature_vocab = sorted(features) if sort_feature_vocab else list(features)
    tok_to_id = {feat: i for i, feat in enumerate(feature_vocab)}

    if not unique_embeddings_for_counts:
        return {}, tok_to_id, feature_vocab

    pairs: set = set()
    for rec in recs:
        for feat, cnt in _iter_bow_items(rec):
            tid = int(tok_to_id[feat])
            pairs.add((tid, int(cnt)))
            if pair_thermometer and int(cnt) > 1:
                pairs.add((tid, 1))

    pairs_list = sorted(pairs, key=lambda x: (x[0], x[1]))
    pair_to_ctxid = {p: i for i, p in enumerate(pairs_list)}
    return pair_to_ctxid, tok_to_id, feature_vocab

def build_pos_lists_and_context_freq(
    recs: List[dict],
    gi: GlobalGraphIndex,
    *,
    unique_embeddings_for_counts: bool,
    pair_thermometer: bool,
    tok_to_id: Dict[str, int],
    pair_to_ctxid: Dict[Tuple[int, int], int],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    N = int(gi.graph_ids.shape[0])
    Vctx = int(len(pair_to_ctxid) if unique_embeddings_for_counts else len(tok_to_id))

    pos_ctx_by_row: List[torch.Tensor] = [torch.zeros((0,), dtype=torch.long) for _ in range(N)]
    context_freq = torch.zeros((Vctx,), dtype=torch.float64)

    cap = MAX_OCC_PER_TOKEN
    if cap is not None:
        cap = int(cap)
        if cap <= 0:
            cap = None

    for rec in recs:
        gid = int(rec["id"])
        row = gi.gid_to_row.get(gid)
        if row is None:
            continue

        items = _iter_bow_items(rec)

        if unique_embeddings_for_counts:
            idxs: List[int] = []
            for feat, cnt in items:
                tid = int(tok_to_id[feat])
                p = (tid, int(cnt))
                idxs.append(int(pair_to_ctxid[p]))
                if pair_thermometer and int(cnt) > 1:
                    idxs.append(int(pair_to_ctxid[(tid, 1)]))

            if idxs:
                uniq = sorted(set(idxs))
                pos = torch.tensor(uniq, dtype=torch.long)
                pos_ctx_by_row[int(row)] = pos
                for ci in uniq:
                    context_freq[int(ci)] += 1.0
            else:
                pos_ctx_by_row[int(row)] = torch.zeros((0,), dtype=torch.long)

        else:
            idxs_occ: List[int] = []
            for feat, cnt in items:
                tid = int(tok_to_id[feat])
                c = int(cnt)
                if c <= 0:
                    continue
                m = c if cap is None else min(c, cap)
                if m <= 0:
                    continue
                idxs_occ.extend([tid] * m)
                context_freq[int(tid)] += float(m)

            pos_ctx_by_row[int(row)] = (
                torch.tensor(idxs_occ, dtype=torch.long) if idxs_occ else torch.zeros((0,), dtype=torch.long)
            )

    return pos_ctx_by_row, context_freq

def pos_stats(pos_ctx_by_row: List[torch.Tensor]) -> Tuple[int, float, int, int]:
    ps = [int(t.numel()) for t in pos_ctx_by_row]
    if not ps:
        return 0, 0.0, 0, 0
    s = int(sum(ps))
    mn = int(min(ps))
    mx = int(max(ps))
    mean = float(s) / float(len(ps))
    return s, mean, mn, mx


class PVDBOW(nn.Module):
    def __init__(self, num_graphs: int, context_vocab_size: int, emb_dim: int):
        super().__init__()
        self.graph_emb = nn.Embedding(num_graphs, emb_dim, sparse=True)
        self.ctx_emb = nn.Embedding(context_vocab_size, emb_dim, sparse=True)
        nn.init.normal_(self.graph_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.ctx_emb.weight, mean=0.0, std=0.02)

    def forward_scores(self, g_idx: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
        g = self.graph_emb(g_idx)
        c = self.ctx_emb(c_idx)
        return (g * c).sum(dim=1)

def build_neg_sampling_dist(context_freq: torch.Tensor, pow_: float) -> torch.Tensor:
    tf = context_freq.clone()
    tf[tf < 0] = 0
    if float(tf.sum().item()) <= 0:
        tf = torch.ones_like(tf)
    dist = tf.pow(float(pow_))
    dist = dist / dist.sum()
    return dist.to(torch.float32)

@torch.no_grad()
def sample_negatives_no_table(
    P: int,
    num_neg: int,
    dist: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    P = int(P)
    num_neg = int(num_neg)
    if P <= 0 or num_neg <= 0:
        return torch.zeros((max(P, 0), max(num_neg, 0)), dtype=torch.long)
    idx = torch.multinomial(dist, num_samples=P * num_neg, replacement=True, generator=generator)
    return idx.view(P, num_neg).to(torch.long)

def _set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = float(lr)

def save_embeddings_matrix(path: str, emb: torch.Tensor) -> int:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    e = emb.detach().to("cpu").float()
    if EMB_SAVE_FORMAT.lower() == "npy":
        np.save(path, e.numpy(), allow_pickle=False)
    elif EMB_SAVE_FORMAT.lower() == "pt":
        torch.save(e, path)
    else:
        raise ValueError(f"Unknown EMB_SAVE_FORMAT: {EMB_SAVE_FORMAT}")
    try:
        return int(os.path.getsize(path))
    except Exception:
        return int(e.numel() * 4)

def save_mapping_arrays(run_dir: str, gi: GlobalGraphIndex) -> None:
    ensure_dir(run_dir)
    np.save(os.path.join(run_dir, "graph_ids.npy"), gi.graph_ids.astype(np.int64), allow_pickle=False)
    np.save(os.path.join(run_dir, "labels.npy"), gi.labels.astype(np.int64), allow_pickle=False)
    np.save(os.path.join(run_dir, "labels_raw.npy"), gi.labels_raw.astype(np.int64), allow_pickle=False)

def build_embedding_filename(
    *,
    bows_stem: str,
    run_name: str,
    run_id: str,
    D: int,
    K: int,
    epoch: int,
) -> str:
    ext = "npy" if EMB_SAVE_FORMAT.lower() == "npy" else "pt"

    if not DESCRIPTIVE_EMB_FILENAMES:
        return f"epoch_{epoch:03d}.{ext}"

    rid = f"_{run_id}" if ADD_RUNID_IN_FILENAME and run_id else ""
    return f"{bows_stem}__{run_name}{rid}__D{D}_K{K}__epoch{epoch:03d}.{ext}"

@torch.no_grad()
def bench_dot_product_time_ns(D: int, reps: int = 200_000) -> float:
    D = int(D)
    reps = int(reps)
    if D <= 0 or reps <= 0:
        return 0.0

    a = torch.randn(D, dtype=torch.float32)
    b = torch.randn(D, dtype=torch.float32)

    for _ in range(200):
        _ = torch.dot(a, b)

    t0 = now()
    acc = 0.0
    for _ in range(reps):
        acc += float(torch.dot(a, b).item())
    dt = now() - t0
    if acc == 123456789.0:
        print("")
    return (dt / float(reps)) * 1e9


@dataclass(frozen=True)
class TrainConfig:
    emb_dim: int
    epochs: int
    alpha: float
    min_alpha: float
    num_neg: int
    use_lr_decay: bool
    shuffle_graphs_each_epoch: bool
    shuffle_positives_within_graph: bool

def train_pvdbow_embeddings_with_saving(
    cfg: TrainConfig,
    gi: GlobalGraphIndex,
    pos_ctx_by_row: List[torch.Tensor],
    context_freq: torch.Tensor,
    *,
    seed: int,
    save_epochs: List[int],
    run_dir: str,
    bows_stem: str,
    run_name: str,
    run_id: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    seed_everything(seed)
    rng = random.Random(int(seed))

    N = int(gi.graph_ids.shape[0])
    Vctx = int(context_freq.numel())

    dist = build_neg_sampling_dist(context_freq, NEG_POW)
    g_neg = torch.Generator()
    g_neg.manual_seed(int(seed) + 1337)

    model = PVDBOW(num_graphs=N, context_vocab_size=Vctx, emb_dim=cfg.emb_dim)
    opt = torch.optim.SGD(model.parameters(), lr=float(cfg.alpha))

    save_set = sorted({int(e) for e in save_epochs if int(e) >= 1})
    total_epochs = max(1, int(cfg.epochs))
    save_set = [e for e in save_set if e <= total_epochs]

    rows = list(range(N))
    epoch_stats: List[Dict[str, Any]] = []

    train_compute_total = 0.0
    train_save_total = 0.0
    train_total_updates = 0
    train_total_pos = 0

    t_train0 = now()
    for epoch in range(1, int(cfg.epochs) + 1):
        t0 = now()
        model.train()

        if cfg.use_lr_decay:
            progress = float(epoch) / float(total_epochs)
            lr = float(cfg.alpha) - (float(cfg.alpha) - float(cfg.min_alpha)) * progress
            lr = max(float(cfg.min_alpha), min(float(cfg.alpha), lr))
            _set_optimizer_lr(opt, lr)
        else:
            lr = float(cfg.alpha)

        if cfg.shuffle_graphs_each_epoch:
            rng.shuffle(rows)

        used_graphs = 0
        empty_graphs = 0
        total_pos = 0
        loss_sum = 0.0
        loss_count = 0

        for row in rows:
            c_pos = pos_ctx_by_row[int(row)]
            P = int(c_pos.numel())
            if P == 0:
                empty_graphs += 1
                continue

            used_graphs += 1
            total_pos += P

            if cfg.shuffle_positives_within_graph and P > 1:
                c_pos = c_pos[torch.randperm(P)]

            g_pos = torch.full((P,), int(row), dtype=torch.long)
            c_neg = sample_negatives_no_table(
                P=P,
                num_neg=int(cfg.num_neg),
                dist=dist,
                generator=g_neg,
            )

            pos_scores = model.forward_scores(g_pos, c_pos)
            g_rep = g_pos.unsqueeze(1).expand(-1, int(cfg.num_neg)).reshape(-1)
            neg_scores = model.forward_scores(g_rep, c_neg.reshape(-1))

            pos_term = F.logsigmoid(pos_scores)
            neg_term = F.logsigmoid(-neg_scores).view(P, int(cfg.num_neg)).sum(dim=1)
            loss = -(pos_term + neg_term).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item())
            loss_count += 1

        dt_compute = now() - t0
        train_compute_total += float(dt_compute)
        train_total_updates += int(loss_count)
        train_total_pos += int(total_pos)

        avg_loss = (loss_sum / max(1, loss_count))

        dt_save = 0.0
        saved_path = None
        saved_bytes = 0
        if epoch in set(save_set):
            model.eval()
            with torch.no_grad():
                emb_epoch = model.graph_emb.weight.detach().to("cpu").float()

            fname = build_embedding_filename(
                bows_stem=bows_stem,
                run_name=run_name,
                run_id=run_id,
                D=int(cfg.emb_dim),
                K=int(cfg.num_neg),
                epoch=int(epoch),
            )
            out_path = os.path.join(run_dir, "embeddings", fname)

            t_s0 = now()
            saved_bytes = save_embeddings_matrix(out_path, emb_epoch)
            dt_save = now() - t_s0
            saved_path = out_path
            train_save_total += float(dt_save)

        msg = (
            f"  epoch {epoch:>3d}/{cfg.epochs} | loss {avg_loss:.6f} | "
            f"used {used_graphs}/{N} | empty {empty_graphs}/{N} | "
            f"P_epoch={total_pos} | lr {lr:.6f} | compute {fmt_sec(dt_compute)}"
        )
        if saved_path is not None:
            msg += f" | saved {os.path.basename(saved_path)} ({human_bytes(int(saved_bytes))}) in {fmt_sec(dt_save)}"
        print(msg)

        epoch_stats.append({
            "epoch": int(epoch),
            "used_graphs": int(used_graphs),
            "empty_graphs": int(empty_graphs),
            "P_epoch": int(total_pos),
            "loss_steps": int(loss_count),
            "elapsed_compute_sec": float(dt_compute),
            "elapsed_save_sec": float(dt_save),
        })

    t_train_total = now() - t_train0

    model.eval()
    with torch.no_grad():
        emb_final = model.graph_emb.weight.detach().to("cpu").float().clone()

    train_report = {
        "train_time_sec_total": float(t_train_total),
        "train_compute_time_sec_total": float(train_compute_total),
        "train_save_time_sec_total": float(train_save_total),
        "train_total_updates": int(train_total_updates),
        "train_total_pos": int(train_total_pos),
        "epoch_stats": epoch_stats,
        "save_epochs": save_set,
    }
    return emb_final, train_report


def _bytes_of(dtype: str) -> int:
    if dtype == "float32":
        return 4
    if dtype == "float64":
        return 8
    if dtype == "int64":
        return 8
    raise ValueError(f"unknown dtype: {dtype}")

def print_symbol_mem_time_summary(
    *,
    run_name: str,
    unique_embeddings_for_counts: bool,
    pair_thermometer: bool,
    N: int,
    V: int,
    D: int,
    K: int,
    pos_ctx_by_row: List[torch.Tensor],
    train_report: Dict[str, Any],
    run_wall_sec: float,
) -> None:
    mode = "unique" if unique_embeddings_for_counts else "occurrence"

    sumP, meanP, minP, maxP = pos_stats(pos_ctx_by_row)

    mem_graph_emb = int(N * D * _bytes_of("float32"))
    mem_ctx_emb = int(V * D * _bytes_of("float32"))
    mem_pos = int(int(sumP) * _bytes_of("int64"))
    mem_vocab = int(V * _bytes_of("int64"))

    mem_scores_avg = int(round(float(meanP)) * K * _bytes_of("float32"))
    mem_scores_worst = int(int(maxP) * K * _bytes_of("float32"))

    t_total = float(train_report.get("train_time_sec_total", 0.0) or 0.0)
    t_compute = float(train_report.get("train_compute_time_sec_total", 0.0) or 0.0)
    t_save = float(train_report.get("train_save_time_sec_total", 0.0) or 0.0)
    updates = int(train_report.get("train_total_updates", 0) or 0)
    epochs = int(len(train_report.get("epoch_stats", []) or []))

    avg_epoch = (t_total / float(epochs)) if epochs > 0 else 0.0
    avg_epoch_compute = (t_compute / float(epochs)) if epochs > 0 else 0.0
    avg_sgd_step = (t_compute / float(updates)) if updates > 0 else 0.0

    dot_ns = bench_dot_product_time_ns(D=D, reps=200_000)
    dot_us = dot_ns / 1000.0

    print("\n" + "=" * 100)
    print(f"[CONFIG SUMMARY] {run_name}")
    print("=" * 100)

    print("\n[SYMBOLS]")
    print(f"  N : number of graphs = {N}")
    print(f"  V : context vocabulary size = {V}")
    print(f"  D : embedding dimension = {D}")
    print(f"  K : negative samples per positive = {K}")
    print(f"  Pᵢ: positives for graph i (mode={mode})")
    print(f"      ΣPᵢ={int(sumP)} | mean(Pᵢ)={float(meanP):.2f} | min={int(minP)} | max={int(maxP)}")
    if unique_embeddings_for_counts:
        print(f"      pair_thermometer: {pair_thermometer}")

    print("\n[THEORETICAL MEMORY] (ignores Python overhead)")
    print("  Assumptions: float32=4B, int64=8B")
    print(f"  graph embeddings Θ(N*D) : {human_bytes(mem_graph_emb)}")
    print(f"  token embeddings Θ(V*D) : {human_bytes(mem_ctx_emb)}")
    print(f"  positives Θ(ΣPᵢ)        : {human_bytes(mem_pos)}")
    print(f"  vocabulary Θ(V)         : {human_bytes(mem_vocab)}")
    print("  ---")
    print(f"  per SGD step: scores Θ(Pᵢ*K) float32")
    print(f"    avg-case  (Pᵢ≈mean) : approx {human_bytes(mem_scores_avg)}")
    print(f"    worst-case(Pᵢ=max)  : approx {human_bytes(mem_scores_worst)}")

    print("\n[TIMING]")
    print(f"  run wall time (this config): {fmt_sec(run_wall_sec)}")
    print(f"  training total time: {fmt_sec(t_total)}")
    print(f"    compute: {fmt_sec(t_compute)}")
    print(f"    save:    {fmt_sec(t_save)}")
    print(f"  avg / epoch (total):   {fmt_sec(avg_epoch)}")
    print(f"  avg / epoch (compute): {fmt_sec(avg_epoch_compute)}")
    print(f"  avg / SGD step (compute): {fmt_sec(avg_sgd_step)}   (#steps={updates})")

    print("\n[INDEPENDENT MICROBENCHMARK]")
    print(f"  avg dot-product time for D={D} (float32, CPU): ~{dot_us:.3f} µs / dot\n")


def run_g2v_from_bow(bow_jsonl_path: str) -> None:
    seed_everything(SEED)

    bows_stem = file_stem(bow_jsonl_path)

    if OVERWRITE_OUTPUT:
        print(f"[OVERWRITE] Deleting previous outputs under: {OUTPUT_DIR}")
        wipe_dir(OUTPUT_DIR)
    else:
        ensure_dir(OUTPUT_DIR)

    save_epochs = sorted({int(e) for e in SAVE_EPOCHS if int(e) >= 1})
    save_epochs = [e for e in save_epochs if e <= int(MAX_EPOCHS)]

    print("=" * 100)
    print("[STAGE B] Graph2Vec (PV-DBOW) from WL->BoW JSONL")
    print("=" * 100)
    print(f"input_jsonl_gz : {bow_jsonl_path}")
    print(f"bows_stem      : {bows_stem}")
    print(f"output_dir     : {OUTPUT_DIR}")
    print(f"overwrite      : {OVERWRITE_OUTPUT}")
    print(f"sweep_runs     : {[r['name'] for r in RUN_SWEEP]}")
    print(f"train_cfg      : D={EMB_DIM} K={NUM_NEG} lr={LR_ALPHA} epochs={MAX_EPOCHS}")
    print(f"save_epochs    : {save_epochs} format={EMB_SAVE_FORMAT}")
    print("")

    t_script0 = now()

    print("[LOAD] Reading BoW JSONL...")
    t_load0 = now()
    recs = load_all_records(bow_jsonl_path)
    print(f"[LOAD] records={len(recs)} | time={fmt_sec(now() - t_load0)}\n")

    print("[INDEX] Building global graph index...")
    t_idx0 = now()
    gi = build_global_graph_index_from_recs(recs)
    y = gi.labels.copy()
    n_classes = int(y.max() + 1) if y.size else 0
    print(f"[INDEX] N={len(y)} classes={n_classes} | time={fmt_sec(now() - t_idx0)}\n")

    for run in RUN_SWEEP:
        run_name = str(run.get("name", "run"))
        unique_embeddings_for_counts = bool(run.get("unique_embeddings_for_counts", False))
        pair_thermometer = bool(run.get("pair_thermometer", False)) if unique_embeddings_for_counts else False

        run_id = short_run_id(
            bows_stem,
            run_name,
            "PAIR" if unique_embeddings_for_counts else "TOKEN",
            bool(pair_thermometer),
            EMB_DIM,
            NUM_NEG,
            LR_ALPHA,
            MAX_EPOCHS,
            "MAX_OCC_NONE" if MAX_OCC_PER_TOKEN is None else int(MAX_OCC_PER_TOKEN),
            float(NEG_POW),
        )

        run_dir = os.path.join(OUTPUT_DIR, bows_stem, run_name)

        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        ensure_dir(run_dir)
        ensure_dir(os.path.join(run_dir, "embeddings"))

        print("\n" + "#" * 110)
        print(f"[RUN] {run_name} (run_id={run_id})")
        print("#" * 110)
        print(f"  unique_embeddings_for_counts: {unique_embeddings_for_counts}")
        print(f"  pair_thermometer           : {pair_thermometer if unique_embeddings_for_counts else '(inactive)'}")
        print(f"  sort_feature_vocab         : {SORT_FEATURE_VOCAB}")
        print(f"  run_dir                    : {run_dir}")
        print("")

        t_run0 = now()

        if SAVE_GRAPH_IDS_AND_LABELS:
            save_mapping_arrays(run_dir, gi)
            print(f"[SAVE] mapping arrays in {run_dir}")

        print("[VOCAB] Building vocab...")
        t_v0 = now()
        pair_to_ctxid, tok_to_id, _feature_vocab = build_vocab_from_recs(
            recs,
            unique_embeddings_for_counts=unique_embeddings_for_counts,
            pair_thermometer=pair_thermometer,
            sort_feature_vocab=bool(SORT_FEATURE_VOCAB),
        )
        Vctx = len(pair_to_ctxid) if unique_embeddings_for_counts else len(tok_to_id)
        print(f"[VOCAB] Vtok={len(tok_to_id)} Vctx={Vctx} | time={fmt_sec(now() - t_v0)}\n")

        print("[DATA] Building positives + frequencies...")
        t_d0 = now()
        pos_ctx_by_row, context_freq = build_pos_lists_and_context_freq(
            recs,
            gi,
            unique_embeddings_for_counts=unique_embeddings_for_counts,
            pair_thermometer=pair_thermometer,
            tok_to_id=tok_to_id,
            pair_to_ctxid=pair_to_ctxid,
        )
        print(f"[DATA] done | time={fmt_sec(now() - t_d0)}\n")

        cfg_seed = stable_int_hash(
            "all_in_one_wl_to_g2v_online",
            SEED,
            bows_stem,
            run_name,
            "PAIR" if unique_embeddings_for_counts else "TOKEN",
            bool(pair_thermometer),
            EMB_DIM,
            NUM_NEG,
            LR_ALPHA,
            MAX_EPOCHS,
            "MAX_OCC_NONE" if MAX_OCC_PER_TOKEN is None else int(MAX_OCC_PER_TOKEN),
            float(NEG_POW),
            "NEG_NO_TABLE",
        )
        min_alpha = float(LR_ALPHA) * float(MIN_LR_FRACTION) if USE_LINEAR_LR_DECAY else float(LR_ALPHA)

        cfg = TrainConfig(
            emb_dim=int(EMB_DIM),
            epochs=int(MAX_EPOCHS),
            alpha=float(LR_ALPHA),
            min_alpha=float(min_alpha),
            num_neg=int(NUM_NEG),
            use_lr_decay=bool(USE_LINEAR_LR_DECAY),
            shuffle_graphs_each_epoch=bool(SHUFFLE_GRAPHS_EACH_EPOCH),
            shuffle_positives_within_graph=bool(SHUFFLE_POSITIVES_WITHIN_GRAPH),
        )

        print("=" * 100)
        print(f"[TRAIN] PV-DBOW (run={run_name})")
        print("=" * 100)

        _emb_final, train_report = train_pvdbow_embeddings_with_saving(
            cfg=cfg,
            gi=gi,
            pos_ctx_by_row=pos_ctx_by_row,
            context_freq=context_freq,
            seed=int(cfg_seed),
            save_epochs=save_epochs,
            run_dir=run_dir,
            bows_stem=bows_stem,
            run_name=run_name,
            run_id=run_id,
        )

        run_wall = now() - t_run0

        print_symbol_mem_time_summary(
            run_name=run_name,
            unique_embeddings_for_counts=unique_embeddings_for_counts,
            pair_thermometer=pair_thermometer,
            N=int(gi.graph_ids.shape[0]),
            V=int(Vctx),
            D=int(EMB_DIM),
            K=int(NUM_NEG),
            pos_ctx_by_row=pos_ctx_by_row,
            train_report=train_report,
            run_wall_sec=float(run_wall),
        )

        emb_dir = os.path.join(run_dir, "embeddings")
        if train_report.get("save_epochs"):
            print(f"[SAVED] epochs={train_report.get('save_epochs')} -> {emb_dir}")
        else:
            print("[SAVED] no embeddings saved (save_epochs empty or out of range).")

        del pos_ctx_by_row, context_freq, tok_to_id, pair_to_ctxid, _emb_final
        gc.collect()

    print("\n" + "=" * 100)
    print("[GLOBAL]")
    print("=" * 100)
    print(f"total stage-B wall time: {fmt_sec(now() - t_script0)}")

    del recs
    gc.collect()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    seed_everything(SEED)

    t0 = now()

    bow_path, n_graphs, n_classes = build_bow_jsonl_from_tu_dataset_online()
    print(f"[PIPELINE] Stage A complete: {bow_path} | graphs={n_graphs}, classes={n_classes}\n")

    run_g2v_from_bow(bow_path)

    print("\n" + "=" * 100)
    print("[DONE]")
    print("=" * 100)
    print(f"total pipeline wall time: {fmt_sec(now() - t0)}")


if __name__ == "__main__":
    main()
