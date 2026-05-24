import numpy as np
from scipy.spatial import Delaunay, KDTree  # type: ignore
from numba import njit, prange  # type: ignore
from typing import Any, cast
from src.core.validation import compute_alpha_values


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
    Build candidate sets using Delaunay triangulation,
    filled with KDTree nearest neighbors up to k.
    Accelerated with Numba.
    """
    n = coords.shape[0]
    # Delaunay triangulation for geometric neighbors
    tri = Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices

    # KDTree for nearest neighbor filling
    tree = KDTree(coords)
    _, kdtree_indices = tree.query(coords, k=min(k + 1, n))

    # Numba-accelerated filtering and merging
    candidate_set = _filter_nearest_neighbors(
        coords.astype(np.float64),
        indptr.astype(np.int32),
        indices.astype(np.int32),
        kdtree_indices.astype(np.int32),
        k,
    )

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
