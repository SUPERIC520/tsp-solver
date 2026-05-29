"""Edge frequency analysis for soft-backbone optimization.

This module provides functions to calculate edge frequencies across multiple
tours, which can be used to bias candidate set ordering.
"""


import numpy as np
import numpy.typing as npt
from numba import njit


@njit(fastmath=True)  # type: ignore
def _count_edges_sorted(
    tours: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int64], int, int]:
    """Encode all edges in all tours as sorted (u < v) integers.

    Edge encoding: u * n + v (with u < v guaranteed).
    Returns the sorted edge array along with m (num tours) and n (num cities).
    """
    m, n = tours.shape
    all_edges = np.empty(m * n, dtype=np.int64)
    for i in range(m):
        for j in range(n):
            u, v = int(tours[i, j]), int(tours[i, (j + 1) % n])
            if u < v:
                all_edges[i * n + j] = np.int64(u) * n + v
            else:
                all_edges[i * n + j] = np.int64(v) * n + u
    all_edges.sort()
    return all_edges, m, n


def compute_edge_frequencies(
    tours: npt.NDArray[np.int32],
    candidate_set: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Compute normalized edge frequency matrix aligned to the candidate set.

    For each node i and each candidate j = candidate_set[i, k], returns the
    fraction of tours in which edge (i, j) appears.

    Returns:
        freq: np.ndarray, shape (n, K), dtype float64
            freq[i, k] = frequency of edge (i, candidate_set[i, k]) in [0, 1].
    """
    m, n = tours.shape
    k_dim = candidate_set.shape[1]

    if m == 0:
        return np.zeros((n, k_dim), dtype=np.float64)

    # Step 1: build sorted encoded edge list
    all_edges_sorted, m_val, n_cities = _count_edges_sorted(tours)

    # Step 2: count occurrences using a Python dict
    # This is fast enough for the orchestration layer
    edge_count: dict[int, int] = {}
    if len(all_edges_sorted) > 0:
        cur = int(all_edges_sorted[0])
        cnt = 1
        for idx in range(1, len(all_edges_sorted)):
            e = int(all_edges_sorted[idx])
            if e == cur:
                cnt += 1
            else:
                edge_count[cur] = cnt
                cur = e
                cnt = 1
        edge_count[cur] = cnt

    # Step 3: build (n, K) frequency matrix aligned to candidate_set
    freq = np.zeros((n_cities, k_dim), dtype=np.float64)
    inv_m = 1.0 / m_val
    for i in range(n_cities):
        for k in range(k_dim):
            j = int(candidate_set[i, k])
            if j == -1:
                continue
            u, v = (i, j) if i < j else (j, i)
            code = u * n_cities + v
            count = edge_count.get(code, 0)
            freq[i, k] = count * inv_m

    return freq
