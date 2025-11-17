# main_wl_demo.py

import networkx as nx
from source.weisfeiler_lehman import wl_subtree_bag, wl_iterations
from source.learn_embeddings_from_bags import learn_embeddings_from_bags


def build_toy_graph() -> nx.Graph:
    """
    Build a small toy graph to visualize WL steps.

    Structure:
        0 - 1 - 2
            |   |
            3 - 4

    So there is a little square (1-2-4-3) with a tail node 0.
    """
    G = nx.Graph()

    # Add nodes 0..4
    G.add_nodes_from(range(5))

    # Add edges
    edges = [
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 4),
        (2, 4),
    ]
    G.add_edges_from(edges)

    return G


def print_wl_labels(labels_per_iter):
    """
    Pretty-print WL labels per iteration.
    """
    for it, labels in enumerate(labels_per_iter):
        print(f"\n=== WL iteration {it} ===")
        for node in sorted(labels.keys()):
            print(f"node {node}: {labels[node]}")


def main():
    # 1) Build some toy graphs
    G1 = build_toy_graph()     # square + tail
    G2 = nx.path_graph(5)      # simple chain: 0-1-2-3-4
    G3 = nx.star_graph(4)      # star: 0 connected to 1,2,3,4

    graphs = [G1, G2, G3]

    # 2) Run WL on G1 and print labels at each iteration
    h = 3
    labels_per_iter = wl_iterations(G1, h=h)
    print("Weisfeiler–Lehman label refinement on G1:")
    print_wl_labels(labels_per_iter)

    # 3) Build bags of WL subtree labels for each graph
    print(f"\n\nBag-of-subtrees for each graph (h={h}):")
    bags = []
    for idx, G in enumerate(graphs):
        bow = wl_subtree_bag(G, h=h)
        bags.append(bow)
        print(f"\nGraph {idx}:")
        print(f"  bag size = {len(bow)}")
        print(f"  first 10 tokens = {bow[:10]}")

    # 4) Learn embeddings from bags using our custom Doc2Vec-like function
    print("\n\nLearning embeddings from bags...")
    Z, W, token_to_id = learn_embeddings_from_bags(
        bags,
        dim=50,
        epochs=5000,
        lr=0.001,
        negative=3,
        min_count=1,
        seed=0,
    )

    # 5) Print resulting graph embeddings
    print("\nGraph embeddings (rows = graphs):")
    for i, vec in enumerate(Z):
        print(f"Graph {i}: {vec}")

    print("\nVocab (subtree tokens):")
    for tok, idx in token_to_id.items():
        print(f"  {idx}: {tok}")


if __name__ == "__main__":
    main()

