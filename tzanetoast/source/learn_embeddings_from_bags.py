# source/learn_embeddings_from_bags.py

from __future__ import annotations

from typing import List, Sequence, Dict, Tuple
import numpy as np
import random


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def build_vocab(
    bags: List[Sequence[str]],
    min_count: int = 1,
) -> Tuple[Dict[str, int], List[str], np.ndarray, np.ndarray]:
    """
    Build vocabulary from bags of tokens.

    Returns:
        token_to_id: dict mapping token -> index [0..M-1]
        id_to_token: inverse list
        freqs:       frequency of each token (aligned with id)
        noise_dist:  negative-sampling distribution over token ids
                     proportional to freq^0.75
    """
    freq: Dict[str, int] = {}
    for bag in bags:
        for tok in bag:
            freq[tok] = freq.get(tok, 0) + 1

    # apply min_count
    tokens = [t for t, c in freq.items() if c >= min_count]
    tokens.sort()
    if not tokens:
        raise ValueError("Vocab is empty; try lowering min_count.")

    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = tokens

    freqs = np.array([freq[t] for t in tokens], dtype=np.float64)

    # noise dist ~ freq^0.75 (word2vec heuristic)
    pow_freqs = freqs ** 0.75
    noise_dist = pow_freqs / pow_freqs.sum()

    return token_to_id, id_to_token, freqs, noise_dist


def bags_to_pairs(
    bags: List[Sequence[str]],
    token_to_id: Dict[str, int],
) -> List[Tuple[int, int]]:
    """
    Build the set S of (doc_id, word_id) positive pairs.

    For each occurrence of token in a bag, we add one pair.
    """
    pairs: List[Tuple[int, int]] = []
    for doc_id, bag in enumerate(bags):
        for tok in bag:
            if tok in token_to_id:
                wid = token_to_id[tok]
                pairs.append((doc_id, wid))
    if not pairs:
        raise ValueError("No (doc, word) pairs (maybe all bags are empty?).")
    return pairs


def learn_embeddings_from_bags(
    bags: List[Sequence[str]],
    dim: int = 128,
    epochs: int = 5,
    lr: float = 0.01,
    negative: int = 5,
    min_count: int = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Core function:

        Input:  list of bags-of-words (one bag per graph/document)
        Output: 
            Z: graph/document embeddings [N, dim]
            W: word/subtree embeddings [M, dim]
            token_to_id: vocab mapping

    Follows the algorithm:

        - build vocab
        - build positive (doc, word) pairs
        - negative sampling with noise_dist ~ freq^0.75
        - SGD updates per pair
    """
    random.seed(seed)
    np.random.seed(seed)

    N = len(bags)               # number of documents/graphs
    token_to_id, id_to_token, freqs, noise_dist = build_vocab(
        bags, min_count=min_count
    )
    M = len(id_to_token)        # vocabulary size

    # Build positive pairs S
    pairs = bags_to_pairs(bags, token_to_id)

    # Initialize embeddings: Z for docs, W for words
    Z = 0.01 * np.random.randn(N, dim).astype(np.float64)
    W = 0.01 * np.random.randn(M, dim).astype(np.float64)

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0

        for (i, j) in pairs:
            # Sample K negative word indices according to noise_dist
            neg_ids = np.random.choice(M, size=negative, p=noise_dist)

            z_i = Z[i]            # shape (dim,)
            w_j = W[j]            # shape (dim,)
            w_negs = W[neg_ids]   # shape (K, dim)

            # ----- forward -----
            # positive score and prob
            s_pos = float(z_i @ w_j)           # scalar
            p_pos = sigmoid(s_pos)             # scalar

            # negative scores and probs
            s_negs = w_negs @ z_i              # shape (K,)
            p_negs = sigmoid(s_negs)           # shape (K,)

            # loss for monitoring (optional)
            # ℓ = -log σ(s_pos) - Σ log σ(-s_neg)
            # log σ(-x) = log(1 - σ(x))
            loss_pos = -np.log(p_pos + 1e-12)
            loss_neg = -np.log(1.0 - p_negs + 1e-12).sum()
            total_loss += loss_pos + loss_neg

            # ----- gradients -----
            # grad z_i
            # ∂ℓ/∂z_i = (p_pos - 1)*w_j + Σ p_neg_k * w_neg_k
            grad_z = (p_pos - 1.0) * w_j + (p_negs[:, None] * w_negs).sum(axis=0)

            # grad w_j (positive word)
            # ∂ℓ/∂w_j = (p_pos - 1)*z_i
            grad_w_pos = (p_pos - 1.0) * z_i

            # grad w_neg_k = p_neg_k * z_i
            grad_w_negs = p_negs[:, None] * z_i[None, :]  # shape (K, dim)

            # ----- SGD updates -----
            Z[i] -= lr * grad_z
            W[j] -= lr * grad_w_pos
            for k_idx, neg_id in enumerate(neg_ids):
                W[neg_id] -= lr * grad_w_negs[k_idx]

        avg_loss = total_loss / len(pairs)
        print(f"Epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.4f}")

    return Z, W, token_to_id
