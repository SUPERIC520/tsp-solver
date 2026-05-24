import numpy as np
from scipy.spatial import KDTree
from numba import njit, prange
from typing import Any, cast, Tuple
from src.core.validation import compute_alpha_values


@njit(cache=True)  # type: ignore
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
def get_hilbert_indices(
    coords: np.ndarray,
) -> np.ndarray:
    """
    Sort indices based on Hilbert curve.
    """
    num_points = coords.shape[0]
    if num_points == 0:
        return np.zeros(0, dtype=np.int32)

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
    n = 2**20

    hilbert_distances = np.empty(num_points, dtype=np.int64)
    for i in range(num_points):
        ix = int((coords[i, 0] - min_x) / max_range * (n - 1))
        iy = int((coords[i, 1] - min_y) / max_range * (n - 1))
        hilbert_distances[i] = _xy2d(n, ix, iy)

    return np.argsort(hilbert_distances).astype(np.int32)


def hilbert_reorder_cities(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder cities according to a Hilbert curve to improve cache locality.
    Returns (reordered_coords, original_to_new_mapping).
    """
    indices = get_hilbert_indices(coords)
    reordered_coords = coords[indices].copy()
    
    n = coords.shape[0]
    original_to_new = np.empty(n, dtype=np.int32)
    original_to_new[indices] = np.arange(n, dtype=np.int32)
    
    return reordered_coords, original_to_new


@njit(parallel=True, fastmath=True, cache=True)  # type: ignore
def _sort_by_alpha(
    candidate_set: np.ndarray[Any, np.dtype[np.int32]],
    alpha_values: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.int32]]:
    n, k = candidate_set.shape
    refined = np.empty_like(candidate_set)

    for i in prange(n):
        # Only sort the non -1 neighbors
        num_neighbors = 0
        for m in range(k):
            if candidate_set[i, m] == -1:
                break
            num_neighbors += 1

        if num_neighbors > 0:
            curr_cands = candidate_set[i, :num_neighbors].copy()
            curr_alphas = alpha_values[i, :num_neighbors].copy()

            # Sort
            sort_idx = np.argsort(curr_alphas)

            for m in range(num_neighbors):
                refined[i, m] = curr_cands[sort_idx[m]]
            for m in range(num_neighbors, k):
                refined[i, m] = -1
        else:
            refined[i, :] = -1

    return refined


def refine_candidate_set_with_alpha(
    coords: np.ndarray[Any, np.dtype[np.float64]],
    candidate_set: np.ndarray[Any, np.dtype[np.int32]],
    pi: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.int32]]:
    """
    Re-sort candidate set based on Alpha-values.
    Small Alpha-values are prioritized.
    """
    n = coords.shape[0]
    alpha_values = compute_alpha_values(n, coords, candidate_set, pi)
    refined = _sort_by_alpha(candidate_set, alpha_values)
    return cast(np.ndarray[Any, np.dtype[np.int32]], refined)


@njit(parallel=True, fastmath=True, cache=True)  # type: ignore
def _filter_nearest_neighbors(
    coords: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    kdtree_indices: np.ndarray,
    k: int,
) -> np.ndarray:
    n = coords.shape[0]
    candidate_set = np.full((n, k), -1, dtype=np.int32)

    for i in prange(n):
        # 1. Collect Delaunay neighbors
        start, end = indptr[i], indptr[i + 1]
        d_neighbors = indices[start:end]
        num_d = d_neighbors.shape[0]

        # Calculate distances for Delaunay neighbors
        d_dists = np.empty(num_d, dtype=np.float64)
        for j in range(num_d):
            neighbor = d_neighbors[j]
            dx = coords[i, 0] - coords[neighbor, 0]
            dy = coords[i, 1] - coords[neighbor, 1]
            d_dists[j] = dx * dx + dy * dy

        # Sort Delaunay neighbors by distance
        sort_idx = np.argsort(d_dists)

        added = 0
        # Add up to k Delaunay neighbors
        for j in range(num_d):
            if added >= k:
                break
            neighbor = d_neighbors[sort_idx[j]]
            candidate_set[i, added] = neighbor
            added += 1

        # 2. Fill with KDTree neighbors if needed
        if added < k:
            for j in range(kdtree_indices.shape[1]):
                neighbor = kdtree_indices[i, j]
                if neighbor == i or neighbor == -1:
                    continue  # skip itself or invalid

                # Check if already added
                is_new = True
                for m in range(added):
                    if candidate_set[i, m] == neighbor:
                        is_new = False
                        break

                if is_new:
                    candidate_set[i, added] = neighbor
                    added += 1
                    if added >= k:
                        break

    return candidate_set


def build_candidate_sets(coords: np.ndarray, k: int = 16) -> np.ndarray:
    """
    Build candidate sets using KDTree nearest neighbors.
    Bypasses Delaunay triangulation entirely.
    """
    n = coords.shape[0]
    tree = KDTree(coords)
    _, kdtree_indices = tree.query(coords, k=min(k + 1, n))
    
    # Slice off the first neighbor (which is the node itself)
    candidate_set = kdtree_indices[:, 1:].astype(np.int32)
    return cast(np.ndarray, candidate_set)


def refine_with_alpha(
    coords: np.ndarray, candidate_set: np.ndarray, pi: np.ndarray
) -> np.ndarray:
    """
    Re-sort the existing candidate set by Alpha-values.
    """
    n = coords.shape[0]
    alpha_values = compute_alpha_values(n, coords, candidate_set, pi)
    refined = _sort_by_alpha(candidate_set, alpha_values)
    return cast(np.ndarray, refined)
