#!/usr/bin/env python3
# reddit_multi12k_wl_g2v_save_embeddings_disk_safe.py
#
# RAM-safe pipeline for REDDIT-MULTI-12K exported dataset:
#   - Load graphs from disk export (metadata.tsv + graphs/*.edgelist)
#   - WL -> BoW (no FTL)
#   - Graph2Vec (PV-DBOW) in 3 modes
#   - Save selected-epoch embeddings (+ labels / graph_ids)
#
# Exported dataset format expected:
#   DATASET_DIR/
#     metadata.tsv                  # graph_id, num_nodes, label
#     graphs/
#       graph_00000.edgelist
#       graph_00001.edgelist
#       ...
#
# Notes:
# - This avoids PyG TUDataset processing RAM spikes.
# - Node ids in edge files can be global; loader relabels per graph to 0..k-1.
# - Isolated nodes are preserved using metadata num_nodes.

import os
import gc
import json
import math
import time
import random
import hashlib
import shutil
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Iterator, Literal
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


# =============================================================================
# USER CONFIG
# =============================================================================

SEED = 42

# Exported dataset directory (from your safe downloader/exporter)
DATASET_DIR = "./reddit_multi_12k_disk"
DATASET_NAME = "REDDIT-MULTI-12K"

# WL init labels
#   - "constant": all nodes label 0
#   - "degree": node degree
INITIAL_WL_FROM: Literal["constant", "degree"] = "constant"

# WL params
WL_H = 1
KEEP_WL_FROM_EVERY_ITER = True
VOCAB_MIN_COUNT = 1   # keep features with df > VOCAB_MIN_COUNT
COUNT_BINS = 3        # 0 raw counts, 1 presence, >1 per-feature global binning

# Loader batching (RAM-friendly)
LOAD_BATCH_SIZE = 128
SHOW_PROGRESS = True

# Graph2Vec modes (same 3 modes)
RUN_SWEEP: List[Dict[str, Any]] = [
    {"name": "tok_counts_occ",     "unique_embeddings_for_counts": False, "pair_thermometer": False},
    {"name": "pair_counts_unique", "unique_embeddings_for_counts": True,  "pair_thermometer": False},
    {"name": "pair_counts_thermo", "unique_embeddings_for_counts": True,  "pair_thermometer": True},
]

# Graph2Vec training
EMB_DIMS: List[int] = [128]
NUM_NEG = 15
LR_ALPHA = 0.05
SAVE_EPOCHS: List[int] = [2]
MAX_EPOCHS = max((int(e) for e in SAVE_EPOCHS if int(e) >= 1), default=1)

USE_LINEAR_LR_DECAY = True
MIN_LR_FRACTION = 0.1
SHUFFLE_GRAPHS_EACH_EPOCH = True
SHUFFLE_POSITIVES_WITHIN_GRAPH = True
NEG_POW = 0.75

# Optional cap only for occurrence mode
MAX_OCC_PER_TOKEN: Optional[int] = None

SORT_FEATURE_VOCAB = True

# Optional perturbation embeddings for stability evaluation
PERTURB_PS: List[float] = []  # e.g. [0.05, 0.10]

# Output
OUTPUT_BOW_JSONL = "bows_per_graph_reddit_multi12k.jsonl.gz"
OUTPUT_DIR = "./g2v_reddit_multi12k_embeddings_selected_epochs_named"
OVERWRITE_OUTPUT = True
SAVE_GRAPH_IDS_AND_LABELS = True
SAVE_PERTURBED_BOWS = False

EMB_SAVE_FORMAT = "npy"  # "npy" or "pt"
DESCRIPTIVE_EMB_FILENAMES = True
ADD_RUNID_IN_FILENAME = True


# =============================================================================
# RAM-safe loader for exported dataset
# =============================================================================
@dataclass(frozen=True)
class GraphRecord:
    graph_id: int
    num_nodes: int
    label: int
    path: str


def load_metadata(dataset_dir: str) -> List[GraphRecord]:
    meta_path = os.path.join(dataset_dir, "metadata.tsv")
    graphs_dir = os.path.join(dataset_dir, "graphs")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.tsv not found: {meta_path}")
    if not os.path.isdir(graphs_dir):
        raise FileNotFoundError(f"graphs dir not found: {graphs_dir}")

    records: List[GraphRecord] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        expected = ["graph_id", "num_nodes", "label"]
        if header != expected:
            raise ValueError(f"Unexpected metadata header: {header} (expected {expected})")

        for line in f:
            line = line.strip()
            if not line:
                continue
            gid_s, nn_s, y_s = line.split("\t")
            gid = int(gid_s)
            nn = int(nn_s)
            y = int(y_s)
            p = os.path.join(graphs_dir, f"graph_{gid:05d}.edgelist")
            records.append(GraphRecord(graph_id=gid, num_nodes=nn, label=y, path=p))

    records.sort(key=lambda r: r.graph_id)
    return records


def read_edgelist_to_nx(path: str, num_nodes: int) -> nx.Graph:
    # Pass 1: unique node ids seen in edges
    node_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            u = int(a)
            v = int(b)
            if u == v:
                continue
            node_ids.add(u)
            node_ids.add(v)

    id_list = sorted(node_ids)
    mapping = {nid: i for i, nid in enumerate(id_list)}

    G = nx.Graph()
    G.add_nodes_from(range(len(id_list)))

    # Pass 2: add edges with relabeled ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            u = int(a)
            v = int(b)
            if u == v:
                continue
            G.add_edge(mapping[u], mapping[v])

    # Preserve isolates
    cur_n = G.number_of_nodes()
    if num_nodes > cur_n:
        G.add_nodes_from(range(cur_n, num_nodes))

    return G


def iter_graph_batches(
    dataset_dir: str,
    batch_size: int = 64,
    show_progress: bool = True,
) -> Iterator[Tuple[List[nx.Graph], np.ndarray, np.ndarray]]:
    records = load_metadata(dataset_dir)
    n = len(records)

    starts = range(0, n, int(batch_size))
    if show_progress and HAS_TQDM:
        starts = tqdm(list(starts), desc="Loading graph batches", unit="batch", dynamic_ncols=True)

    for start in starts:
        end = min(start + int(batch_size), n)
        batch_recs = records[start:end]

        graphs: List[nx.Graph] = []
        labels = np.empty(len(batch_recs), dtype=np.int64)
        gids = np.empty(len(batch_recs), dtype=np.int64)

        for j, r in enumerate(batch_recs):
            G = read_edgelist_to_nx(r.path, r.num_nodes)
            G.remove_edges_from(nx.selfloop_edges(G))
            graphs.append(G)
            labels[j] = int(r.label)
            gids[j] = int(r.graph_id)

        yield graphs, labels, gids
        del graphs, labels, gids
        gc.collect()


# =============================================================================
# Utils
# =============================================================================
def now() -> float:
    return time.perf_counter()

def fmt_sec(x: float) -> str:
    if x < 1:
        return f"{x*1000:.1f} ms"
    if x < 60:
        return f"{x:.2f} s"
    return f"{x/60:.2f} min"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def wipe_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def set_all_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def stable_hash64(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()

def short_run_id(*parts: Any, n: int = 10) -> str:
    s = "|".join(str(p) for p in parts)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()[:int(n)]

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


# =============================================================================
# WL / BoW
# =============================================================================
def constant_initial_labels(num_nodes: int, value: int = 0) -> List[int]:
    return [int(value)] * int(num_nodes)

def degree_initial_labels(num_nodes: int, G: nx.Graph) -> List[int]:
    labels = [0] * int(num_nodes)
    deg = dict(G.degree())
    for node, d in deg.items():
        ni = int(node)
        if 0 <= ni < num_nodes:
            labels[ni] = int(d)
    return labels

def initial_labels_for_graph(mode: str, G: nx.Graph) -> List[Any]:
    n = int(G.number_of_nodes())
    if mode == "constant":
        return constant_initial_labels(n, value=0)
    if mode == "degree":
        return degree_initial_labels(n, G)
    raise ValueError(f"Unknown INITIAL_WL_FROM={mode}")

def wl_subtree_counts(G: nx.Graph, node_labels: List[Any], h: int, keep_from_every_iter: bool) -> Counter:
    if G.number_of_nodes() == 0:
        return Counter()

    nodes = sorted(G.nodes())
    cur = {n: str(node_labels[int(n)]) for n in nodes}
    feats = Counter()

    if keep_from_every_iter or h == 0:
        for n in nodes:
            feats[f"0:{cur[n]}"] += 1

    for it in range(1, int(h) + 1):
        new = {}
        for n in nodes:
            neigh = sorted(cur[v] for v in G.neighbors(n))
            s = cur[n] + "|" + ",".join(neigh)
            new[n] = stable_hash64(f"{it}:{s}")
        cur = new

        if keep_from_every_iter:
            for n in nodes:
                feats[f"{it}:{cur[n]}"] += 1

    if (not keep_from_every_iter) and int(h) > 0:
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
    return {feat for feat, cnt in df.items() if cnt > mc}

def make_per_feature_binner_global(
    all_counters: List[Counter],
    keep_set: Optional[set],
    count_bins: int,
):
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
        mn = float(min(vlist)) if vlist else 1.0
        mx = float(max(vlist)) if vlist else 1.0
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
        x = max(mn, min(mx, x))
        t = (x - mn) / (mx - mn)
        idx = int(math.floor(t * B))
        idx = min(B - 1, max(0, idx))
        return int(idx + 1)

    info = {
        "mode": "binned_per_feature_global",
        "bins": B,
        "per_feature": True,
        "n_features_seen": int(len(feat_ranges)),
        "n_constant_features": int(sum(1 for (_mn, _mx, is_c) in feat_ranges.values() if is_c)),
    }
    return bin_fn, info

def counters_to_bows(
    counters: List[Counter],
    keep_set: Optional[set],
    count_bins: int,
    bin_fn,
) -> List[Dict[str, int]]:
    B = int(count_bins)
    bows: List[Dict[str, int]] = []

    for c in counters:
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
        bows.append(bow)

    return bows

def save_bows_jsonl(path: str, bows: List[Dict[str, int]], labels: np.ndarray, labels_raw: np.ndarray, graph_ids: np.ndarray) -> None:
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    with opener(path, "wt", encoding="utf-8") as f:
        for i, bow in enumerate(bows):
            rec = {
                "id": int(i),
                "graph_id": int(graph_ids[i]),
                "class": int(labels[i]),
                "class_raw": int(labels_raw[i]),
                "bow": bow,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =============================================================================
# Perturbation
# =============================================================================
def perturb_remove_edges_nx(G: nx.Graph, p: float, seed: int) -> nx.Graph:
    p = float(p)
    rng = np.random.default_rng(int(seed))
    H = G.copy()
    edges = list(H.edges())
    m = len(edges)
    if m == 0 or p <= 0.0:
        return H
    k = int(round(p * m))
    k = min(m, max(0, k))
    if k > 0:
        idx = rng.choice(m, size=k, replace=False)
        to_remove = [edges[i] for i in idx]
        H.remove_edges_from(to_remove)
    return H


# =============================================================================
# Build counters via batch loading (RAM-safe)
# =============================================================================
def build_wl_counters_from_disk(
    dataset_dir: str,
    batch_size: int,
    show_progress: bool,
    initial_wl_from: str,
    wl_h: int,
    keep_from_every_iter: bool,
    perturb_p: Optional[float] = None,
    perturb_seed_base: int = 0,
) -> Tuple[List[Counter], np.ndarray, np.ndarray]:
    counters: List[Counter] = []
    labels_raw: List[int] = []
    graph_ids: List[int] = []

    t0 = now()
    batch_iter = iter_graph_batches(
        dataset_dir=dataset_dir,
        batch_size=int(batch_size),
        show_progress=bool(show_progress),
    )

    global_i = 0
    for graphs, labels, gids in batch_iter:
        for j, G in enumerate(graphs):
            if perturb_p is not None:
                G_use = perturb_remove_edges_nx(
                    G, p=float(perturb_p), seed=int(perturb_seed_base + int(gids[j]) + global_i)
                )
            else:
                G_use = G

            init_labels = initial_labels_for_graph(initial_wl_from, G_use)
            c = wl_subtree_counts(
                G_use, init_labels, h=int(wl_h), keep_from_every_iter=bool(keep_from_every_iter)
            )
            counters.append(c)
            labels_raw.append(int(labels[j]))
            graph_ids.append(int(gids[j]))
            global_i += 1

            if G_use is not G:
                del G_use

        del graphs, labels, gids
        gc.collect()

    dt = now() - t0
    print(f"[WL] built {len(counters)} counters in {fmt_sec(dt)} (perturb_p={perturb_p})")

    return counters, np.asarray(labels_raw, dtype=np.int64), np.asarray(graph_ids, dtype=np.int64)


# =============================================================================
# Graph2Vec: vocab + positives
# =============================================================================
def iter_bow_items(bow: Dict[str, int]) -> Iterator[Tuple[str, int]]:
    for k, v in bow.items():
        c = int(v)
        if c <= 0:
            continue
        yield str(k), c

def build_vocab_from_bows(
    bows: List[Dict[str, int]],
    *,
    unique_embeddings_for_counts: bool,
    pair_thermometer: bool,
    sort_feature_vocab: bool,
) -> Tuple[Dict[Tuple[int, int], int], Dict[str, int], List[str]]:
    features: set = set()
    for bow in bows:
        for feat, _cnt in iter_bow_items(bow):
            features.add(feat)

    feature_vocab = sorted(features) if sort_feature_vocab else list(features)
    tok_to_id = {feat: i for i, feat in enumerate(feature_vocab)}

    if not unique_embeddings_for_counts:
        return {}, tok_to_id, feature_vocab

    pairs: set = set()
    for bow in bows:
        for feat, cnt in iter_bow_items(bow):
            tid = int(tok_to_id[feat])
            pairs.add((tid, int(cnt)))
            if pair_thermometer and int(cnt) > 1:
                pairs.add((tid, 1))

    pairs_list = sorted(pairs, key=lambda x: (x[0], x[1]))
    pair_to_ctxid = {p: i for i, p in enumerate(pairs_list)}
    return pair_to_ctxid, tok_to_id, feature_vocab

def build_pos_lists_and_context_freq(
    bows: List[Dict[str, int]],
    *,
    unique_embeddings_for_counts: bool,
    pair_thermometer: bool,
    tok_to_id: Dict[str, int],
    pair_to_ctxid: Dict[Tuple[int, int], int],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    N = len(bows)
    Vctx = int(len(pair_to_ctxid) if unique_embeddings_for_counts else len(tok_to_id))

    pos_ctx_by_row: List[torch.Tensor] = [torch.zeros((0,), dtype=torch.long) for _ in range(N)]
    context_freq = torch.zeros((Vctx,), dtype=torch.float64)

    cap = MAX_OCC_PER_TOKEN
    if cap is not None:
        cap = int(cap)
        if cap <= 0:
            cap = None

    for row, bow in enumerate(bows):
        if unique_embeddings_for_counts:
            idxs: List[int] = []
            for feat, cnt in iter_bow_items(bow):
                tid = int(tok_to_id[feat])
                idxs.append(int(pair_to_ctxid[(tid, int(cnt))]))
                if pair_thermometer and int(cnt) > 1:
                    idxs.append(int(pair_to_ctxid[(tid, 1)]))
            if idxs:
                uniq = sorted(set(idxs))
                pos = torch.tensor(uniq, dtype=torch.long)
                pos_ctx_by_row[row] = pos
                for ci in uniq:
                    context_freq[int(ci)] += 1.0
            else:
                pos_ctx_by_row[row] = torch.zeros((0,), dtype=torch.long)

        else:
            idxs_occ: List[int] = []
            for feat, cnt in iter_bow_items(bow):
                tid = int(tok_to_id[feat])
                c = int(cnt)
                m = c if cap is None else min(c, cap)
                if m <= 0:
                    continue
                idxs_occ.extend([tid] * m)
                context_freq[int(tid)] += float(m)
            pos_ctx_by_row[row] = torch.tensor(idxs_occ, dtype=torch.long) if idxs_occ else torch.zeros((0,), dtype=torch.long)

    return pos_ctx_by_row, context_freq


# =============================================================================
# PV-DBOW model + training
# =============================================================================
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
def sample_negatives(P: int, num_neg: int, dist: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if P <= 0 or num_neg <= 0:
        return torch.zeros((max(P, 0), max(num_neg, 0)), dtype=torch.long)
    idx = torch.multinomial(dist, num_samples=P * num_neg, replacement=True, generator=generator)
    return idx.view(P, num_neg).to(torch.long)

def set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = float(lr)

def save_embeddings_matrix(path: str, emb: torch.Tensor) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    e = emb.detach().to("cpu").float()
    if EMB_SAVE_FORMAT.lower() == "npy":
        np.save(path, e.numpy(), allow_pickle=False)
    elif EMB_SAVE_FORMAT.lower() == "pt":
        torch.save(e, path)
    else:
        raise ValueError(f"Unknown EMB_SAVE_FORMAT: {EMB_SAVE_FORMAT}")

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

def build_embedding_filename(
    *,
    bows_stem: str,
    run_name: str,
    run_id: str,
    D: int,
    K: int,
    epoch: int,
    p: Optional[float],
) -> str:
    ext = "npy" if EMB_SAVE_FORMAT.lower() == "npy" else "pt"
    if not DESCRIPTIVE_EMB_FILENAMES:
        if p is None:
            return f"epoch_{epoch:03d}.{ext}"
        return f"epoch_{epoch:03d}__p{p:.2f}.{ext}"

    rid = f"_{run_id}" if ADD_RUNID_IN_FILENAME and run_id else ""
    core = f"{bows_stem}__{run_name}{rid}__D{D}_K{K}"
    if p is not None:
        core += f"__p{p:.2f}"
    return f"{core}__epoch{epoch:03d}.{ext}"

def train_pvdbow_with_saving(
    cfg: TrainConfig,
    pos_ctx_by_row: List[torch.Tensor],
    context_freq: torch.Tensor,
    *,
    seed: int,
    save_epochs: List[int],
    run_dir: str,
    bows_stem: str,
    run_name: str,
    run_id: str,
    p_value: Optional[float],
) -> Dict[str, Any]:
    set_all_seeds(seed)
    rng = random.Random(int(seed))

    N = len(pos_ctx_by_row)
    Vctx = int(context_freq.numel())

    if N <= 0 or Vctx <= 0:
        return {"ok": False, "reason": f"empty N or Vctx (N={N}, Vctx={Vctx})", "save_epochs": []}

    dist = build_neg_sampling_dist(context_freq, NEG_POW)
    g_neg = torch.Generator()
    g_neg.manual_seed(int(seed) + 1337)

    model = PVDBOW(num_graphs=N, context_vocab_size=Vctx, emb_dim=cfg.emb_dim)
    opt = torch.optim.SGD(model.parameters(), lr=float(cfg.alpha))

    save_set = sorted({int(e) for e in save_epochs if 1 <= int(e) <= int(cfg.epochs)})
    rows = list(range(N))

    t0 = now()
    for epoch in range(1, int(cfg.epochs) + 1):
        ep_t0 = now()

        if cfg.use_lr_decay:
            progress = float(epoch) / float(max(1, cfg.epochs))
            lr = float(cfg.alpha) - (float(cfg.alpha) - float(cfg.min_alpha)) * progress
            lr = max(float(cfg.min_alpha), min(float(cfg.alpha), lr))
            set_optimizer_lr(opt, lr)
        else:
            lr = float(cfg.alpha)

        if cfg.shuffle_graphs_each_epoch:
            rng.shuffle(rows)

        used_graphs = 0
        empty_graphs = 0
        total_pos = 0
        loss_sum = 0.0
        steps = 0

        for row in rows:
            c_pos = pos_ctx_by_row[row]
            P = int(c_pos.numel())
            if P == 0:
                empty_graphs += 1
                continue

            used_graphs += 1
            total_pos += P

            if cfg.shuffle_positives_within_graph and P > 1:
                c_pos = c_pos[torch.randperm(P)]

            g_pos = torch.full((P,), int(row), dtype=torch.long)
            c_neg = sample_negatives(P=P, num_neg=int(cfg.num_neg), dist=dist, generator=g_neg)

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
            steps += 1

        avg_loss = loss_sum / max(1, steps)
        msg = (
            f"      epoch {epoch:>3d}/{cfg.epochs} | loss={avg_loss:.6f} | "
            f"used={used_graphs}/{N} empty={empty_graphs}/{N} P={total_pos} lr={lr:.6f} "
            f"| {fmt_sec(now()-ep_t0)}"
        )

        if epoch in set(save_set):
            with torch.no_grad():
                emb_epoch = model.graph_emb.weight.detach().to("cpu").float()

            fname = build_embedding_filename(
                bows_stem=bows_stem,
                run_name=run_name,
                run_id=run_id,
                D=int(cfg.emb_dim),
                K=int(cfg.num_neg),
                epoch=int(epoch),
                p=p_value,
            )
            out_path = os.path.join(run_dir, "embeddings", fname)
            save_embeddings_matrix(out_path, emb_epoch)
            msg += f" | saved={os.path.basename(out_path)}"

        print(msg)

    total_time = now() - t0
    return {"ok": True, "train_time_sec": float(total_time), "save_epochs": save_set}

def save_mapping_arrays(run_dir: str, graph_ids: np.ndarray, labels: np.ndarray, labels_raw: np.ndarray) -> None:
    ensure_dir(run_dir)
    np.save(os.path.join(run_dir, "graph_ids.npy"), graph_ids.astype(np.int64), allow_pickle=False)
    np.save(os.path.join(run_dir, "labels.npy"), labels.astype(np.int64), allow_pickle=False)
    np.save(os.path.join(run_dir, "labels_raw.npy"), labels_raw.astype(np.int64), allow_pickle=False)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    set_all_seeds(SEED)

    if OVERWRITE_OUTPUT:
        print(f"[OVERWRITE] deleting: {OUTPUT_DIR}")
        wipe_dir(OUTPUT_DIR)
    else:
        ensure_dir(OUTPUT_DIR)

    print("=" * 110)
    print("REDDIT-MULTI-12K | disk-safe loader | WL + Graph2Vec | no FTL")
    print("=" * 110)
    print(f"DATASET_DIR={DATASET_DIR}")
    print(f"INITIAL_WL_FROM={INITIAL_WL_FROM} WL_H={WL_H} KEEP_EVERY_ITER={KEEP_WL_FROM_EVERY_ITER}")
    print(f"VOCAB_MIN_COUNT={VOCAB_MIN_COUNT} COUNT_BINS={COUNT_BINS}")
    print(f"RUNS={[r['name'] for r in RUN_SWEEP]}")
    print(f"DIMS={EMB_DIMS} EPOCHS={MAX_EPOCHS} SAVE_EPOCHS={SAVE_EPOCHS}")
    print(f"PERTURB_PS={PERTURB_PS if PERTURB_PS else '(disabled)'}")
    print("")

    records = load_metadata(DATASET_DIR)
    print(f"[DATA] metadata loaded: {len(records)} graphs")

    # Clean counters + labels + ids
    counters_clean, labels_raw, graph_ids = build_wl_counters_from_disk(
        dataset_dir=DATASET_DIR,
        batch_size=int(LOAD_BATCH_SIZE),
        show_progress=bool(SHOW_PROGRESS),
        initial_wl_from=str(INITIAL_WL_FROM),
        wl_h=int(WL_H),
        keep_from_every_iter=bool(KEEP_WL_FROM_EVERY_ITER),
        perturb_p=None,
        perturb_seed_base=int(SEED + 1000),
    )

    # map labels to 0..C-1
    uniq_raw, labels = np.unique(labels_raw, return_inverse=True)
    labels = labels.astype(np.int64)
    N = int(labels.shape[0])
    C = int(len(uniq_raw))

    print(f"[DATA] N={N} classes={C} raw_labels={list(map(int, uniq_raw.tolist()))[:20]}")

    keep_set = build_keep_set_docfreq(counters_clean, min_count=int(VOCAB_MIN_COUNT))
    bin_fn, bin_info = make_per_feature_binner_global(
        counters_clean, keep_set=keep_set, count_bins=int(COUNT_BINS)
    )
    bows_clean = counters_to_bows(
        counters_clean, keep_set=keep_set, count_bins=int(COUNT_BINS), bin_fn=bin_fn
    )

    print(f"[WL] kept_vocab={len(keep_set) if keep_set is not None else 'all'}")
    print(f"[WL] bin_info={bin_info}")

    save_bows_jsonl(OUTPUT_BOW_JSONL, bows_clean, labels=labels, labels_raw=labels_raw, graph_ids=graph_ids)
    print(f"[BOW] saved clean bows: {OUTPUT_BOW_JSONL}")

    bows_stem = file_stem(OUTPUT_BOW_JSONL)
    root_for_dataset = os.path.join(OUTPUT_DIR, DATASET_NAME)
    ensure_dir(root_for_dataset)

    # optional perturbed bows cache (shared across run modes)
    perturbed_bows_cache: Dict[float, List[Dict[str, int]]] = {}

    for run in RUN_SWEEP:
        run_name = str(run["name"])
        unique_counts = bool(run["unique_embeddings_for_counts"])
        thermo = bool(run["pair_thermometer"]) if unique_counts else False

        run_id = short_run_id(
            DATASET_NAME, run_name, unique_counts, thermo,
            EMB_DIMS, NUM_NEG, LR_ALPHA, MAX_EPOCHS, SAVE_EPOCHS
        )

        run_dir = os.path.join(root_for_dataset, run_name)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        ensure_dir(run_dir)
        ensure_dir(os.path.join(run_dir, "embeddings"))

        if SAVE_GRAPH_IDS_AND_LABELS:
            save_mapping_arrays(run_dir, graph_ids=graph_ids, labels=labels, labels_raw=labels_raw)

        print("\n" + "#" * 110)
        print(f"[RUN] {run_name} | unique_counts={unique_counts} thermo={thermo} | run_id={run_id}")
        print(f"      run_dir={run_dir}")
        print("#" * 110)

        p_values: List[Optional[float]] = [None] + [float(p) for p in PERTURB_PS]

        for p in p_values:
            if p is None:
                bows = bows_clean
                p_label = "clean"
            else:
                p = float(p)
                p_label = f"p={p:.2f}"
                if p in perturbed_bows_cache:
                    bows = perturbed_bows_cache[p]
                else:
                    counters_p, labels_raw_p, graph_ids_p = build_wl_counters_from_disk(
                        dataset_dir=DATASET_DIR,
                        batch_size=int(LOAD_BATCH_SIZE),
                        show_progress=bool(SHOW_PROGRESS),
                        initial_wl_from=str(INITIAL_WL_FROM),
                        wl_h=int(WL_H),
                        keep_from_every_iter=bool(KEEP_WL_FROM_EVERY_ITER),
                        perturb_p=float(p),
                        perturb_seed_base=int(SEED + 50000 + int(round(p * 10000))),
                    )

                    # sanity: ordering should match
                    if not np.array_equal(graph_ids_p, graph_ids):
                        raise RuntimeError("Perturbed loader order mismatch in graph_ids.")
                    if not np.array_equal(labels_raw_p, labels_raw):
                        raise RuntimeError("Perturbed loader order mismatch in labels.")

                    bows = counters_to_bows(
                        counters_p,
                        keep_set=keep_set,
                        count_bins=int(COUNT_BINS),
                        bin_fn=bin_fn,
                    )
                    perturbed_bows_cache[p] = bows

                    if SAVE_PERTURBED_BOWS:
                        p_bow_path = os.path.join(OUTPUT_DIR, f"bows_per_graph_reddit_multi12k__p{p:.2f}.jsonl.gz")
                        save_bows_jsonl(p_bow_path, bows, labels=labels, labels_raw=labels_raw, graph_ids=graph_ids)
                        print(f"[BOW] saved perturbed bows: {p_bow_path}")

                    del counters_p, labels_raw_p, graph_ids_p
                    gc.collect()

            print(f"  [MODE DATA] building vocab/positives ({p_label}) ...")
            t_mode0 = now()
            pair_to_ctxid, tok_to_id, _feature_vocab = build_vocab_from_bows(
                bows,
                unique_embeddings_for_counts=unique_counts,
                pair_thermometer=thermo,
                sort_feature_vocab=bool(SORT_FEATURE_VOCAB),
            )
            pos_ctx_by_row, context_freq = build_pos_lists_and_context_freq(
                bows,
                unique_embeddings_for_counts=unique_counts,
                pair_thermometer=thermo,
                tok_to_id=tok_to_id,
                pair_to_ctxid=pair_to_ctxid,
            )

            Vctx = len(pair_to_ctxid) if unique_counts else len(tok_to_id)
            non_empty = int(sum(1 for t in pos_ctx_by_row if int(t.numel()) > 0))
            mean_pi = float(np.mean([int(t.numel()) for t in pos_ctx_by_row])) if pos_ctx_by_row else 0.0
            print(
                f"  [MODE DATA] done in {fmt_sec(now()-t_mode0)} | Vtok={len(tok_to_id)} Vctx={Vctx} "
                f"| non_empty={non_empty}/{len(pos_ctx_by_row)} mean(P_i)={mean_pi:.2f}"
            )

            if Vctx <= 0 or non_empty <= 0:
                print(f"  [SKIP] no trainable contexts for {run_name} ({p_label}).")
                del pair_to_ctxid, tok_to_id, _feature_vocab, pos_ctx_by_row, context_freq
                gc.collect()
                continue

            for D in EMB_DIMS:
                cfg_seed = stable_int_hash(
                    "reddit_multi12k_wl_g2v",
                    SEED,
                    DATASET_NAME,
                    INITIAL_WL_FROM,
                    WL_H,
                    KEEP_WL_FROM_EVERY_ITER,
                    VOCAB_MIN_COUNT,
                    COUNT_BINS,
                    run_name,
                    unique_counts,
                    thermo,
                    f"p={p:.2f}" if p is not None else "clean",
                    D,
                    NUM_NEG,
                    LR_ALPHA,
                    MAX_EPOCHS,
                )

                min_alpha = float(LR_ALPHA) * float(MIN_LR_FRACTION) if USE_LINEAR_LR_DECAY else float(LR_ALPHA)
                cfg = TrainConfig(
                    emb_dim=int(D),
                    epochs=int(MAX_EPOCHS),
                    alpha=float(LR_ALPHA),
                    min_alpha=float(min_alpha),
                    num_neg=int(NUM_NEG),
                    use_lr_decay=bool(USE_LINEAR_LR_DECAY),
                    shuffle_graphs_each_epoch=bool(SHUFFLE_GRAPHS_EACH_EPOCH),
                    shuffle_positives_within_graph=bool(SHUFFLE_POSITIVES_WITHIN_GRAPH),
                )

                print(f"  [TRAIN] {run_name} | {p_label} | D={D} K={NUM_NEG}")
                rep = train_pvdbow_with_saving(
                    cfg=cfg,
                    pos_ctx_by_row=pos_ctx_by_row,
                    context_freq=context_freq,
                    seed=int(cfg_seed),
                    save_epochs=SAVE_EPOCHS,
                    run_dir=run_dir,
                    bows_stem=bows_stem,
                    run_name=run_name,
                    run_id=run_id,
                    p_value=p,
                )

                if not rep.get("ok", False):
                    print(f"  [WARN] training skipped/failed: {rep.get('reason', 'unknown')}")
                else:
                    print(
                        f"  [DONE] {run_name} | {p_label} | D={D} "
                        f"| train_time={fmt_sec(float(rep.get('train_time_sec', 0.0)))} "
                        f"| saved_epochs={rep.get('save_epochs', [])}"
                    )

            del pair_to_ctxid, tok_to_id, _feature_vocab, pos_ctx_by_row, context_freq
            gc.collect()

    print("\n" + "=" * 110)
    print("[DONE]")
    print("=" * 110)
    print(f"Embeddings root: {os.path.abspath(os.path.join(OUTPUT_DIR, DATASET_NAME))}")
    print(f"Clean BoW JSONL : {os.path.abspath(OUTPUT_BOW_JSONL)}")


if __name__ == "__main__":
    main()
