"""Preprocessing utilities for TSP.

This module provides functions for reordering cities using Hilbert curves,
building candidate sets using KD-Trees, and refining candidate sets using
Alpha-values.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from scipy.spatial import KDTree

from src.config import K_NEIGHBORS, KD_TREE_QUERY_SIZE
from src.core.validation import compute_alpha_values
from src.utils.memory_utils import ensure_alignment


@njit(cache=True)  # type: ignore
def _xy2d(n: int, x: int, y: int) -> int:
    """Convert (x, y) to d (distance along Hilbert curve).

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
    coords: npt.NDArray[np.float64],
) -> npt.NDArray[np.int32]:
    """Sort indices based on Hilbert curve."""
    num_points = coords.shape[0]
    if num_points == 0:
        return np.zeros(0, dtype=np.int32)

    # Normalize coordinates to [0, 2^k - 1]
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf

    for i in range(num_points):
        min_x = min(min_x, coords[i, 0])
        min_y = min(min_y, coords[i, 1])
        max_x = max(max_x, coords[i, 0])
        max_y = max(max_y, coords[i, 1])

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


def hilbert_reorder_cities(
    coords: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Reorder cities according to a Hilbert curve to improve cache locality.

    Returns:
        reordered_coords: float64, C-contiguous, 64-byte aligned, shape (N, 2)
        original_to_new:  int32,   C-contiguous, 64-byte aligned, shape (N,)
            Maps original city index -> new (Hilbert-ordered) index.
    """
    coords = coords.astype(np.float64, copy=False)
    indices = get_hilbert_indices(np.ascontiguousarray(coords))
    reordered_coords = coords[indices]

    n = coords.shape[0]
    original_to_new = np.empty(n, dtype=np.int32)
    original_to_new[indices] = np.arange(n, dtype=np.int32)

    reordered_coords_aligned = ensure_alignment(
        np.ascontiguousarray(reordered_coords.astype(np.float64)),
        alignment=64,
    )
    original_to_new_aligned = ensure_alignment(
        np.ascontiguousarray(original_to_new),
        alignment=64,
    )

    return reordered_coords_aligned, original_to_new_aligned


@njit(parallel=True, fastmath=True, cache=True)  # type: ignore
def _sort_by_alpha(
    candidate_set: npt.NDArray[np.int32],
    alpha_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.int32]:
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
    coords: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    pi: npt.NDArray[np.float64],
    top_k: int = K_NEIGHBORS,
) -> npt.NDArray[np.int32]:
    """Re-sort candidate set based on Alpha-values.

    Small Alpha prioritised. Slices the result to the first ``top_k`` elements.
    Returns a 64-byte aligned C-contiguous array of shape (N, top_k), dtype int32.
    """
    coords_aligned = ensure_alignment(
        np.ascontiguousarray(coords.astype(np.float64, copy=False)), alignment=64
    )
    candidate_set_aligned = ensure_alignment(
        np.ascontiguousarray(candidate_set.astype(np.int32, copy=False)), alignment=64
    )
    pi_aligned = ensure_alignment(
        np.ascontiguousarray(pi.astype(np.float64, copy=False)), alignment=64
    )

    n = coords_aligned.shape[0]
    if n == 0:
        empty: npt.NDArray[np.int32] = ensure_alignment(
            np.empty((0, top_k), dtype=np.int32), alignment=64
        )
        return empty

    alpha_values = compute_alpha_values(
        n,
        coords_aligned.astype(np.float64),
        candidate_set_aligned.astype(np.int32),
        pi_aligned.astype(np.float64),
    )
    refined = _sort_by_alpha(
        candidate_set_aligned.astype(np.int32), alpha_values.astype(np.float64)
    )

    c_cols = refined.shape[1]
    if c_cols == top_k:
        sliced: npt.NDArray[np.int32] = refined
    elif c_cols > top_k:
        sliced = np.ascontiguousarray(refined[:, :top_k])
    else:
        padded: npt.NDArray[np.int32] = np.full((n, top_k), -1, dtype=np.int32)
        padded[:, :c_cols] = refined
        sliced = padded

    result: npt.NDArray[np.int32] = ensure_alignment(
        np.ascontiguousarray(sliced), alignment=64
    )
    return result


def build_candidate_sets(
    coords: npt.NDArray[np.float64], k: int = KD_TREE_QUERY_SIZE
) -> npt.NDArray[np.int32]:
    """Build candidate sets using KDTree nearest neighbors.

    Returns a 64-byte aligned C-contiguous matrix of shape (N, k), dtype int32.
    """
    coords_aligned = ensure_alignment(
        np.ascontiguousarray(coords.astype(np.float64, copy=False)), alignment=64
    )
    n = coords_aligned.shape[0]

    candidate_set: npt.NDArray[np.int32] = np.full((n, k), -1, dtype=np.int32)
    if n <= 1:
        return ensure_alignment(candidate_set, alignment=64)

    tree = KDTree(coords_aligned)
    query_k = min(k + 1, n)
    _, indices_raw = tree.query(coords_aligned, k=query_k)
    indices = np.asarray(indices_raw)

    if indices.ndim == 1:
        indices = indices.reshape(-1, 1)

    num_cols_to_copy = query_k - 1
    if num_cols_to_copy > 0:
        candidate_set[:, :num_cols_to_copy] = indices[:, 1:query_k].astype(np.int32)

    # Ensure connectivity by adding Hilbert tour edges (i, i-1) and (i, i+1)
    # Since coordinates are Hilbert-ordered, these are just adjacent indices.
    for i in range(n):
        for neighbor in [(i - 1 + n) % n, (i + 1) % n]:
            if neighbor == i:
                continue
            # Check if already in candidate set
            exists = False
            for col in range(k):
                if candidate_set[i, col] == neighbor:
                    exists = True
                    break
                if candidate_set[i, col] == -1:
                    break

            if not exists:
                # Find an empty slot or replace the last one if full
                for col in range(k):
                    if candidate_set[i, col] == -1:
                        candidate_set[i, col] = neighbor
                        break
                else:
                    # Replace the last KD-Tree neighbor if full
                    candidate_set[i, k - 1] = neighbor

    return ensure_alignment(np.ascontiguousarray(candidate_set), alignment=64)


def refine_with_alpha(
    coords: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    pi: npt.NDArray[np.float64],
    top_k: int = K_NEIGHBORS,
) -> npt.NDArray[np.int32]:
    """Re-sort the existing candidate set by Alpha-values.

    Delegates to refine_candidate_set_with_alpha.
    """
    return refine_candidate_set_with_alpha(coords, candidate_set, pi, top_k=top_k)
