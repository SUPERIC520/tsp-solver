"""Validation utilities for TSP, including Held-Karp lower bounds and Alpha-values.

This module provides functions to compute the Held-Karp lower bound using
1-tree relaxations and subgradient optimization, and to calculate Alpha-values
used for candidate set refinement.
"""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from src.config import HK_BOUNDS_CACHE
from src.utils.data_io import load_hk_cache, save_hk_cache

# Project root is two levels up from src/core/validation.py
_root = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = HK_BOUNDS_CACHE


def load_hk_cache_json(n: int) -> tuple[float, npt.NDArray[np.float64]] | None:
    """Load Held-Karp lower bound and pi values from JSON cache.

    Args:
        n: Number of cities.

    Returns:
        A tuple (lower_bound, pi_array) if found, else None.
    """
    if not CACHE_PATH.exists():
        return None
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        key = str(n)
        if key in data:
            entry = data[key]
            # Support both old "lower_bound" key and new canonical "lb" key
            lb_val = entry.get("lb", entry.get("lower_bound"))
            if lb_val is None:
                return None
            lb = float(lb_val)
            pi = np.array(entry["pi"], dtype=np.float64)
            return lb, pi
    except (OSError, ValueError, KeyError):
        pass
    return None


def save_hk_cache_json(n: int, lb: float, pi: npt.NDArray[np.float64]) -> None:
    """Save Held-Karp lower bound and pi values to JSON cache.

    Args:
        n: Number of cities.
        lb: Lower bound value.
        pi: Pi values array.
    """
    data = {}
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            pass

    data[str(n)] = {"lb": lb, "pi": pi.tolist()}

    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass


@njit(cache=True, fastmath=True)  # type: ignore
def _get_dist(
    i: int, j: int, coords: npt.NDArray[np.float64], pi: npt.NDArray[np.float64]
) -> float:
    """Get transformed distance: d'(i,j) = d(i,j) + pi[i] + pi[j]."""
    dx = coords[i, 0] - coords[j, 0]
    dy = coords[i, 1] - coords[j, 1]
    return float(np.sqrt(dx * dx + dy * dy) + pi[i] + pi[j])


@njit(cache=True, fastmath=True)  # type: ignore
def _heap_push(
    heap_val: npt.NDArray[np.float64],
    heap_node: npt.NDArray[np.int32],
    size: int,
    d: float,
    node: int,
) -> int:
    idx = size
    heap_val[idx] = d
    heap_node[idx] = node
    size += 1
    while idx > 0:
        p = (idx - 1) // 2
        if heap_val[idx] < heap_val[p]:
            heap_val[idx], heap_val[p] = heap_val[p], heap_val[idx]
            heap_node[idx], heap_node[p] = heap_node[p], heap_node[idx]
            idx = p
        else:
            break
    return size


@njit(cache=True, fastmath=True)  # type: ignore
def _heap_pop(
    heap_val: npt.NDArray[np.float64], heap_node: npt.NDArray[np.int32], size: int
) -> tuple[float, int, int]:
    res_d = heap_val[0]
    res_node = heap_node[0]
    size -= 1
    if size > 0:
        heap_val[0] = heap_val[size]
        heap_node[0] = heap_node[size]
        idx = 0
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx
            if left < size and heap_val[left] < heap_val[smallest]:
                smallest = left
            if right < size and heap_val[right] < heap_val[smallest]:
                smallest = right
            if smallest != idx:
                heap_val[idx], heap_val[smallest] = heap_val[smallest], heap_val[idx]
                heap_node[idx], heap_node[smallest] = (
                    heap_node[smallest],
                    heap_node[idx],
                )
                idx = smallest
            else:
                break
    return res_d, int(res_node), size


@njit(cache=True, fastmath=True)  # type: ignore
def _build_undirected_adj(
    n: int, candidate_set: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    adj_ptr = np.zeros(n + 1, dtype=np.int32)
    for u in range(n):
        for k in range(candidate_set.shape[1]):
            v = int(candidate_set[u, k])
            if v == -1:
                break
            adj_ptr[u + 1] += 1
            adj_ptr[v + 1] += 1

    for i in range(n):
        adj_ptr[i + 1] += adj_ptr[i]

    adj_indices = np.empty(int(adj_ptr[n]), dtype=np.int32)
    curr_ptr = adj_ptr.copy()

    for u in range(n):
        for k in range(candidate_set.shape[1]):
            v = int(candidate_set[u, k])
            if v == -1:
                break
            adj_indices[curr_ptr[u]] = v
            curr_ptr[u] += 1
            adj_indices[curr_ptr[v]] = u
            curr_ptr[v] += 1

    return adj_ptr, adj_indices


@njit(cache=True, fastmath=True)  # type: ignore
def compute_mst_weight(
    n: int,
    coords: npt.NDArray[np.float64],
    adj_ptr: npt.NDArray[np.int32],
    adj_indices: npt.NDArray[np.int32],
    pi: npt.NDArray[np.float64],
    root: int,
) -> tuple[float, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Compute the Minimum Spanning Tree (MST) weight of a 1-tree relaxation.

    Args:
        n: Number of cities.
        coords: Coordinates of the cities.
        adj_ptr: Adjacency list pointer.
        adj_indices: Adjacency list indices.
        pi: Pi values for distance transformation.
        root: The root node for the 1-tree.

    Returns:
        A tuple (total_weight, degrees, parent_array).
    """
    min_dist = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)

    start_node = 1 if root == 0 else 0
    min_dist[start_node] = 0.0

    total_weight = 0.0
    degrees = np.zeros(n, dtype=np.int32)

    max_heap_size = int(adj_ptr[n])
    heap_val = np.empty(max_heap_size, dtype=np.float64)
    heap_node = np.empty(max_heap_size, dtype=np.int32)
    heap_size = 0

    heap_size = _heap_push(heap_val, heap_node, heap_size, 0.0, start_node)

    nodes_added = 0
    while heap_size > 0 and nodes_added < n - 1:
        d, u, heap_size = _heap_pop(heap_val, heap_node, heap_size)
        if visited[u]:
            continue
        visited[u] = True
        total_weight += d
        nodes_added += 1
        if parent[u] != -1:
            degrees[u] += 1
            degrees[int(parent[u])] += 1
        for k in range(int(adj_ptr[u]), int(adj_ptr[u + 1])):
            v = int(adj_indices[k])
            if v == root or visited[v]:
                continue
            dist_uv = _get_dist(u, v, coords, pi)
            if dist_uv < min_dist[v]:
                min_dist[v] = dist_uv
                parent[v] = u
                heap_size = _heap_push(heap_val, heap_node, heap_size, dist_uv, v)
    if nodes_added < n - 1:
        return -1.0, degrees, parent
    return total_weight, degrees, parent


@njit(cache=True, fastmath=True, parallel=True)  # type: ignore
def _compute_hk_impl(
    coords: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    max_iter: int,
    initial_pi: npt.NDArray[np.float64],
    target_ub: float,
) -> tuple[float, npt.NDArray[np.float64]]:
    n = coords.shape[0]
    coords = np.ascontiguousarray(coords)
    candidate_set = np.ascontiguousarray(candidate_set)
    pi = np.ascontiguousarray(initial_pi).copy()
    best_lb = -np.inf
    best_pi = pi.copy()
    adj_ptr, adj_indices = _build_undirected_adj(n, candidate_set)
    lambda_val = 2.0
    period = max(100, max_iter // 50)
    last_improvement = 0
    upper_bound = target_ub
    for iteration in range(max_iter):
        root = 0
        mst_weight = -1.0
        degrees = np.zeros(n, dtype=np.int32)
        for r_try in range(3):
            root = (r_try * (n // 3)) % n
            mst_weight, degrees, _ = compute_mst_weight(
                n, coords, adj_ptr, adj_indices, pi, root
            )
            if mst_weight != -1.0:
                break
        if mst_weight == -1.0:
            if best_lb == -np.inf:
                return 0.0, pi
            return best_lb, best_pi
        d1, d2 = np.inf, np.inf
        n1, n2 = -1, -1
        for k in range(int(adj_ptr[root]), int(adj_ptr[root + 1])):
            v = int(adj_indices[k])
            d = _get_dist(root, v, coords, pi)
            if d < d1:
                d2 = d1
                n2 = n1
                d1 = d
                n1 = v
            elif d < d2:
                d2 = d
                n2 = v
        if n2 == -1:
            if best_lb == -np.inf:
                return 0.0, pi
            return best_lb, best_pi
        one_tree_weight = mst_weight + d1 + d2
        degrees[root] = 2
        degrees[n1] += 1
        degrees[n2] += 1
        lb = one_tree_weight - 2.0 * np.sum(pi)
        if lb > best_lb + 1e-6:
            best_lb = lb
            best_pi = pi.copy()
            last_improvement = iteration
        if upper_bound == np.inf:
            upper_bound = lb * 1.2
        subgradient = degrees - 2
        norm_sq = np.sum(subgradient * subgradient)
        if norm_sq == 0:
            break
        if iteration - last_improvement >= period:
            lambda_val *= 0.7
            last_improvement = iteration
            if lambda_val < 0.0001:
                break
        target = (
            min(upper_bound, best_lb * 1.05)
            if upper_bound > best_lb
            else (best_lb + 1.0)
        )
        step = lambda_val * (target - lb) / norm_sq
        pi += step * subgradient
    return best_lb, best_pi


def compute_hk_lower_bound(
    coords: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    max_iter: int = 500,
    initial_pi: npt.NDArray[np.float64] | None = None,
    target_ub: float = np.inf,
    sample_name: str | None = None,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Compute the Held-Karp lower bound.

    Args:
        coords: Coordinates of the cities.
        candidate_set: Candidate set for neighbor pruning.
        max_iter: Maximum number of subgradient iterations.
        initial_pi: Initial pi values for subgradient optimization.
        target_ub: Target upper bound for step size calculation.
        sample_name: Name of the sample for caching.

    Returns:
        A tuple (lower_bound, pi_array).
    """
    n = coords.shape[0]
    cached_json = load_hk_cache_json(n)
    if cached_json is not None:
        return cached_json

    if sample_name:
        cached = load_hk_cache(sample_name)
        if cached:
            save_hk_cache_json(n, cached[0], cached[1])
            return cached

    if initial_pi is None:
        init_pi = np.zeros(n, dtype=np.float64)
    else:
        init_pi = np.ascontiguousarray(initial_pi, dtype=np.float64)

    best_lb, best_pi = _compute_hk_impl(
        coords, candidate_set, max_iter, init_pi, target_ub
    )

    save_hk_cache_json(n, best_lb, best_pi)
    if sample_name:
        save_hk_cache(sample_name, best_lb, best_pi)

    return best_lb, best_pi


@njit(cache=True, fastmath=True, parallel=True)  # type: ignore
def compute_alpha_values(
    n: int,
    coords: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    pi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute Alpha-values for candidate set refinement.

    Alpha-values measure the 'nearness' of an edge to the MST.

    Args:
        n: Number of cities.
        coords: Coordinates of the cities.
        candidate_set: Candidate set for neighbor pruning.
        pi: Pi values from Held-Karp relaxation.

    Returns:
        A matrix of Alpha-values of shape (n, num_candidates).
    """
    coords = np.ascontiguousarray(coords)
    candidate_set = np.ascontiguousarray(candidate_set)
    pi = np.ascontiguousarray(pi)
    adj_ptr, adj_indices = _build_undirected_adj(n, candidate_set)
    root = 0
    _mst_weight, _degrees, parent = compute_mst_weight(
        n, coords, adj_ptr, adj_indices, pi, root
    )
    d1, d2 = np.inf, np.inf
    n1, n2 = -1, -1
    for k in range(int(adj_ptr[root]), int(adj_ptr[root + 1])):
        v = int(adj_indices[k])
        d = _get_dist(root, v, coords, pi)
        if d < d1:
            d2 = d1
            n2 = n1
            d1 = d
            n1 = v
        elif d < d2:
            d2 = d
            n2 = v
    mst_adj_ptr = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        if i == root or parent[i] == -1:
            continue
        mst_adj_ptr[i + 1] += 1
        mst_adj_ptr[int(parent[i]) + 1] += 1
    for i in range(n):
        mst_adj_ptr[i + 1] += mst_adj_ptr[i]
    mst_adj_indices = np.empty(int(mst_adj_ptr[n]), dtype=np.int32)
    mst_adj_weights = np.empty(int(mst_adj_ptr[n]), dtype=np.float64)
    curr_ptr = mst_adj_ptr.copy()
    for i in range(n):
        if i == root or parent[i] == -1:
            continue
        p = int(parent[i])
        w = _get_dist(i, p, coords, pi)
        mst_adj_indices[curr_ptr[i]] = p
        mst_adj_weights[curr_ptr[i]] = w
        curr_ptr[i] += 1
        mst_adj_indices[curr_ptr[p]] = i
        mst_adj_weights[curr_ptr[p]] = w
        curr_ptr[p] += 1
    log_n = int(np.log2(n)) + 1
    up = np.full((n, log_n), -1, dtype=np.int32)
    max_edge = np.zeros((n, log_n), dtype=np.float64)
    depth = np.zeros(n, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)
    head = 0
    tail = 0
    start_node = 1 if root == 0 else 0
    queue[tail] = start_node
    tail += 1
    depth[start_node] = 1
    while head < tail:
        u = int(queue[head])
        head += 1
        for idx in range(int(mst_adj_ptr[u]), int(mst_adj_ptr[u + 1])):
            v = int(mst_adj_indices[idx])
            if depth[v] == 0:
                depth[v] = depth[u] + 1
                up[v, 0] = u
                max_edge[v, 0] = mst_adj_weights[idx]
                queue[tail] = v
                tail += 1
    for j in range(1, log_n):
        for i in range(n):
            if up[i, j - 1] != -1:
                up[i, j] = up[int(up[i, j - 1]), j - 1]
                max_edge[i, j] = max(
                    max_edge[i, j - 1], max_edge[int(up[i, j - 1]), j - 1]
                )

    num_cands = candidate_set.shape[1]
    alphas = np.full((n, num_cands), np.inf, dtype=np.float64)

    # We use explicit integer types to avoid Numba parallel inference issues
    for node_i in prange(n):
        u_idx = np.int32(node_i)
        for cand_k in range(num_cands):
            v_node = candidate_set[u_idx, cand_k]
            if v_node == -1:
                break

            v_idx = np.int32(v_node)
            if root in (int(u_idx), int(v_idx)):
                val_other = int(v_idx) if int(u_idx) == root else int(u_idx)
                if val_other in (n1, n2):
                    alphas[int(u_idx), cand_k] = 0.0
                else:
                    alphas[int(u_idx), cand_k] = (
                        _get_dist(root, val_other, coords, pi) - d2
                    )
            else:
                curr_u = int(u_idx)
                curr_v = int(v_idx)
                if depth[curr_u] < depth[curr_v]:
                    curr_u, curr_v = curr_v, curr_u

                max_e = -1e15
                diff = depth[curr_u] - depth[curr_v]
                for lca_step in range(log_n):
                    if (diff >> lca_step) & 1:
                        max_e = max(max_e, max_edge[curr_u, lca_step])
                        curr_u = int(up[curr_u, lca_step])

                if curr_u != curr_v:
                    for lca_step in range(log_n - 1, -1, -1):
                        if up[curr_u, lca_step] != up[curr_v, lca_step]:
                            max_e = max(max_e, max_edge[curr_u, lca_step])
                            max_e = max(max_e, max_edge[curr_v, lca_step])
                            curr_u = int(up[curr_u, lca_step])
                            curr_v = int(up[curr_v, lca_step])
                    max_e = max(max_e, max_edge[curr_u, 0])
                    max_e = max(max_e, max_edge[curr_v, 0])

                alphas[int(u_idx), cand_k] = (
                    _get_dist(int(u_idx), int(v_idx), coords, pi) - max_e
                )
    return alphas


def validate_result(best_length: float, lower_bound: float) -> float:
    """Calculate the percentage optimality gap between the best length and lower bound.

    Args:
        best_length: The length of the best tour found.
        lower_bound: The lower bound on the optimal tour length.

    Returns:
        The percentage gap (0-100). Returns 100.0 if lower_bound <= 0.
    """
    if lower_bound <= 0:
        return 100.0
    return (best_length - lower_bound) / lower_bound * 100.0
