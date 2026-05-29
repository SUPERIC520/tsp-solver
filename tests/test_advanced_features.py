"""Tests for advanced features of the TSP solver, including Alpha-values and MST."""

import numpy as np

from src.core.preprocessing import build_candidate_sets, refine_candidate_set_with_alpha
from src.core.validation import (
    _build_undirected_adj,
    compute_alpha_values,
    compute_hk_lower_bound,
    compute_mst_weight,
)


def test_hk_pi_extraction() -> None:
    # Create a simple square of cities
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    candidate_set = build_candidate_sets(coords, k=3)
    lb, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=10)

    assert isinstance(lb, float)
    assert isinstance(pi, np.ndarray)
    assert pi.shape == (4,)
    assert lb > 0


def test_alpha_calculation() -> None:
    # Create 10 random cities
    np.random.seed(42)
    coords = np.random.rand(10, 2)
    n = 10
    candidate_set = build_candidate_sets(coords, k=5)

    _lb, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=20)
    alphas = compute_alpha_values(n, coords, candidate_set, pi)

    assert alphas.shape == candidate_set.shape
    assert np.all(alphas >= -1e-9)  # Alpha values should be non-negative

    # Check that edges in the 1-tree have alpha = 0
    # Actually, let's just check the MST of n-1 nodes
    root = 0
    adj_ptr, adj_indices, adj_dists = _build_undirected_adj(n, coords, candidate_set)

    # Pre-allocate buffers for MST call
    min_dist = np.empty(n, dtype=np.float64)
    parent = np.empty(n, dtype=np.int32)
    visited = np.empty(n, dtype=np.bool_)
    degrees = np.empty(n, dtype=np.int32)
    max_heap_size = int(adj_ptr[n])
    heap_val = np.empty(max_heap_size, dtype=np.float64)
    heap_node = np.empty(max_heap_size, dtype=np.int32)

    _w = compute_mst_weight(
        n, adj_ptr, adj_indices, adj_dists, pi, root,
        min_dist, parent, visited, degrees, heap_val, heap_node
    )

    for i in range(n):
        if i == root or parent[i] == -1:
            continue
        p = parent[i]
        # Find (i, p) or (p, i) in candidate_set
        found = False
        for k in range(candidate_set.shape[1]):
            if candidate_set[i, k] == p:
                assert abs(alphas[i, k]) < 1e-7
                found = True
                break
        if not found:
            # It might be that i is in p's candidate set but not vice-versa
            for k in range(candidate_set.shape[1]):
                if candidate_set[p, k] == i:
                    assert abs(alphas[p, k]) < 1e-7
                    found = True
                    break


def test_refine_candidate_set() -> None:
    np.random.seed(42)
    coords = np.random.rand(50, 2)
    n = 50
    candidate_set = build_candidate_sets(coords, k=10)

    _lb, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=20)
    refined_set = refine_candidate_set_with_alpha(coords, candidate_set, pi, top_k=10)

    assert refined_set.shape == candidate_set.shape
    # Check that it's different (likely, because alpha is better than distance)
    # or at least that it's still a valid candidate set
    assert np.all(np.isin(refined_set, np.append(candidate_set, -1)))

    # Check that for each node, the new candidates are sorted by alpha
    alphas = compute_alpha_values(n, coords, candidate_set, pi)

    for i in range(n):
        last_alpha = -1.0
        for k in range(refined_set.shape[1]):
            v = refined_set[i, k]
            if v == -1:
                break

            # Find alpha of (i, v)
            # We need to find v in original candidate_set to get its alpha
            alpha_v = -1.0
            for m in range(candidate_set.shape[1]):
                if candidate_set[i, m] == v:
                    alpha_v = alphas[i, m]
                    break

            assert alpha_v >= last_alpha - 1e-9
            last_alpha = alpha_v


if __name__ == "__main__":
    # Quick manual run
    test_hk_pi_extraction()
    test_alpha_calculation()
    test_refine_candidate_set()
