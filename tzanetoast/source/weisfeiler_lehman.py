# source/weisfeiler_lehman.py

from __future__ import annotations
from typing import Dict, Hashable, List
import networkx as nx


def initial_node_labels(
    G: nx.Graph,
    attr_name: str = "label",
    default_label: str = "0",
) -> Dict[Hashable, str]:
    """
    Initialize labels for each node in the graph.

    If the node has an attribute `attr_name`, use that.
    Otherwise, fall back to `default_label`.

    Returns:
        dict: {node: label_str}
    """
    labels = {}
    for v in G.nodes():
        value = G.nodes[v].get(attr_name, default_label)
        labels[v] = str(value)
    return labels


def wl_step(
    G: nx.Graph,
    labels: Dict[Hashable, str],
) -> Dict[Hashable, str]:
    """
    Perform one Weisfeiler–Lehman refinement step.

    new_label(v) = concat(
        label(v),
        sorted multiset of neighbor labels
    )

    We keep labels as strings to make them easy to inspect/print.
    """
    new_labels: Dict[Hashable, str] = {}

    for v in G.nodes():
        # Current label of the node
        my_label = labels[v]

        # Labels of neighbors (multiset)
        neigh_labels = [labels[u] for u in G.neighbors(v)]
        neigh_labels.sort()

        # Build a canonical string representation
        # Example: "a|b_c_c"
        neigh_part = "_".join(neigh_labels)
        new_labels[v] = f"{my_label}|{neigh_part}"

    return new_labels


def wl_iterations(
    G: nx.Graph,
    h: int,
    attr_name: str = "label",
    default_label: str = "0",
) -> List[Dict[Hashable, str]]:
    """
    Run WL for `h` iterations and return all labelings.

    Returns a list of dictionaries:
        labels_per_iter[0] = labels at iteration 0 (initial)
        labels_per_iter[1] = labels after 1 WL step
        ...
        labels_per_iter[h] = labels after h WL steps
    """
    if h < 0:
        raise ValueError("h must be >= 0")

    labels_per_iter: List[Dict[Hashable, str]] = []

    # Iteration 0: initial labels
    labels = initial_node_labels(G, attr_name=attr_name, default_label=default_label)
    labels_per_iter.append(labels)

    # Iterations 1..h
    for _ in range(h):
        labels = wl_step(G, labels)
        labels_per_iter.append(labels)

    return labels_per_iter


def wl_subtree_bag(
    G: nx.Graph,
    h: int,
    attr_name: str = "label",
    default_label: str = "0",
) -> List[str]:
    """
    Return the 'bag of subtree labels' for this graph up to height h.

    This is what Graph2Vec uses as the 'words' of the graph:
    we collect all node labels from iterations 0..h into a single list.
    """
    labels_per_iter = wl_iterations(
        G,
        h=h,
        attr_name=attr_name,
        default_label=default_label,
    )

    bag: List[str] = []
    for labels in labels_per_iter:
        bag.extend(labels.values())
    return bag
