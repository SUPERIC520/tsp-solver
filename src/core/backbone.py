import numpy as np
from numba import njit
from typing import Any, Tuple


@njit(cache=True)  # type: ignore
def _count_edges(
    tours: np.ndarray[Any, np.dtype[np.int32]],
) -> Tuple[np.ndarray[Any, np.dtype[np.int32]], np.ndarray[Any, np.dtype[np.int32]]]:
    """
    Count edge frequencies across multiple tours.
    Returns edges and their counts.
    To avoid N^2 memory, we only store edges that actually appear.
    Since each tour has N edges, M tours have M*N edges total.
    """
    m, n = tours.shape
    # Maximum possible unique edges is M*N
    # We use a simple hash map or just a sorted list to count.
    # Numba doesn't have a built-in dict that works well in @njit for this scale easily,
    # but we can use a flattened array and sort it.

    # Store edges as (min(u,v), max(u,v)) encoded as u*N + v
    all_edges = np.empty(m * n, dtype=np.int64)
    for i in range(m):
        for j in range(n):
            u = tours[i, j]
            v = tours[i, (j + 1) % n]
            if u < v:
                all_edges[i * n + j] = np.int64(u) * n + v
            else:
                all_edges[i * n + j] = np.int64(v) * n + u

    all_edges.sort()

    return all_edges, np.array([m, n], dtype=np.int32)  # Return meta info


def extract_consensus_edges(
    tours: np.ndarray[Any, np.dtype[np.int32]], threshold: float = 0.95
) -> np.ndarray[Any, np.dtype[np.int32]]:
    """
    Identify edges that appear in at least threshold*M tours.
    Returns locked_edges array of shape (N, 2) initialized with -1.
    """
    m, n = tours.shape
    all_edges_encoded, _ = _count_edges(tours)

    locked_edges = np.full((n, 2), -1, dtype=np.int32)

    if m == 0:
        return locked_edges

    count_threshold = int(threshold * m)

    # Count occurrences in the sorted array
    if len(all_edges_encoded) == 0:
        return locked_edges

    current_edge = all_edges_encoded[0]
    current_count = 1

    for i in range(1, len(all_edges_encoded)):
        if all_edges_encoded[i] == current_edge:
            current_count += 1
        else:
            if current_count >= count_threshold:
                # Lock this edge
                u = int(current_edge // n)
                v = int(current_edge % n)
                # Add v to u's locked list
                if locked_edges[u, 0] == -1:
                    locked_edges[u, 0] = v
                elif locked_edges[u, 1] == -1:
                    locked_edges[u, 1] = v

                # Add u to v's locked list
                if locked_edges[v, 0] == -1:
                    locked_edges[v, 0] = u
                elif locked_edges[v, 1] == -1:
                    locked_edges[v, 1] = u

            current_edge = all_edges_encoded[i]
            current_count = 1

    # Check last edge
    if current_count >= count_threshold:
        u = int(current_edge // n)
        v = int(current_edge % n)
        if locked_edges[u, 0] == -1:
            locked_edges[u, 0] = v
        elif locked_edges[u, 1] == -1:
            locked_edges[u, 1] = v
        if locked_edges[v, 0] == -1:
            locked_edges[v, 0] = u
        elif locked_edges[v, 1] == -1:
            locked_edges[v, 1] = u

    return locked_edges
