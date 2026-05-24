import numpy as np
from numba import njit
from typing import Any


@njit  # type: ignore
def _xy2d(n: int, x: int, y: int) -> int:
    """
    Convert (x, y) to d (distance along Hilbert curve).
    n must be a power of 2.
    """
    d = 0
    s = n // 2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * int(rx)) ^ int(ry))

        # Rotate/flip
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


@njit(cache=True)  # type: ignore
def _get_hilbert_indices(
    coords: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.int32]]:
    """
    Sort indices based on Hilbert curve.
    """
    num_points = coords.shape[0]

    # Normalize coordinates to [0, 2^k - 1]
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf

    for i in range(num_points):
        if coords[i, 0] < min_x:
            min_x = coords[i, 0]
        if coords[i, 1] < min_y:
            min_y = coords[i, 1]
        if coords[i, 0] > max_x:
            max_x = coords[i, 0]
        if coords[i, 1] > max_y:
            max_y = coords[i, 1]

    range_x = max_x - min_x
    range_y = max_y - min_y
    max_range = max(range_x, range_y)

    if max_range == 0:
        return np.arange(num_points, dtype=np.int32)

    # Choose n as a power of 2 large enough for precision
    # For 115k cities, 2^20 is more than enough
    n = 2**20

    hilbert_distances = np.empty(num_points, dtype=np.int64)
    for i in range(num_points):
        ix = int((coords[i, 0] - min_x) / max_range * (n - 1))
        iy = int((coords[i, 1] - min_y) / max_range * (n - 1))
        hilbert_distances[i] = _xy2d(n, ix, iy)

    return np.argsort(hilbert_distances).astype(np.int32)


def generate_hilbert_seeds(
    coords: np.ndarray[Any, np.dtype[np.float64]], num_seeds: int = 8
) -> np.ndarray[Any, np.dtype[np.int32]]:
    """
    Generate diverse initial tours using rotated/reflected Hilbert curves.
    Ensures maximum diversity by applying all 8 symmetries of the square.
    """
    n = coords.shape[0]
    seeds = np.empty((num_seeds, n), dtype=np.int32)

    # Use up to 8 Hilbert symmetries
    num_hilbert = min(num_seeds, 8)

    for i in range(num_hilbert):
        transformed_coords = coords.copy()
        sym = i % 8
        if sym >= 4:
            transformed_coords[:, 0] = -transformed_coords[:, 0]
            sym -= 4

        for _ in range(sym):
            old_x = transformed_coords[:, 0].copy()
            old_y = transformed_coords[:, 1].copy()
            transformed_coords[:, 0] = -old_y
            transformed_coords[:, 1] = old_x

        seeds[i] = _get_hilbert_indices(transformed_coords)

    for i in range(num_hilbert, num_seeds):
        # If more than 8 seeds are requested, add some noise to coords for more Hilbert variants
        transformed_coords = coords.copy()
        transformed_coords += np.random.normal(0, 1e-6, transformed_coords.shape)
        seeds[i] = _get_hilbert_indices(transformed_coords)

    return seeds


@njit(fastmath=True, cache=True)  # type: ignore
def _greedy_nn_tour(
    coords: np.ndarray,
    candidate_set: np.ndarray,
    start: int,
) -> np.ndarray:
    """
    Build a greedy nearest-neighbor tour starting from `start`.
    Uses the candidate set for O(N*k) construction instead of O(N^2).
    Falls back to a linear scan for any node not reachable via candidates.
    """
    n = coords.shape[0]
    tour = np.empty(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)

    tour[0] = start
    visited[start] = True

    for step in range(1, n):
        curr = tour[step - 1]
        best_d = np.inf
        best_next = -1

        # Search candidate set first
        for k in range(candidate_set.shape[1]):
            nxt = candidate_set[curr, k]
            if nxt == -1:
                break
            if not visited[nxt]:
                dx = coords[curr, 0] - coords[nxt, 0]
                dy = coords[curr, 1] - coords[nxt, 1]
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d = d
                    best_next = nxt

        # Fallback: linear scan if no unvisited candidate found
        if best_next == -1:
            for j in range(n):
                if not visited[j]:
                    dx = coords[curr, 0] - coords[j, 0]
                    dy = coords[curr, 1] - coords[j, 1]
                    d = dx * dx + dy * dy
                    if d < best_d:
                        best_d = d
                        best_next = j

        tour[step] = best_next
        visited[best_next] = True

    return tour


def generate_greedy_nn_seeds(
    coords: np.ndarray,
    candidate_set: np.ndarray,
    num_seeds: int = 1,
    start_nodes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate greedy nearest-neighbor seeds from diverse starting cities.
    Useful for complementing Hilbert seeds with different basin exploration.
    """
    n = coords.shape[0]
    seeds = np.empty((num_seeds, n), dtype=np.int32)

    if start_nodes is None:
        # Space starting nodes evenly across the tour
        step = max(1, n // num_seeds)
        start_nodes = np.array([i * step for i in range(num_seeds)], dtype=np.int32)

    for i in range(num_seeds):
        seeds[i] = _greedy_nn_tour(coords, candidate_set, int(start_nodes[i]))

    return seeds
