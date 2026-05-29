"""Validation utilities for TSP, including Held-Karp lower bounds and Alpha-values.

This module provides functions to compute the Held-Karp lower bound using
1-tree relaxations and subgradient optimization, and to calculate Alpha-values
used for candidate set refinement.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from src.utils.data_io import load_hk_cache

# Minimum lambda threshold for Held-Karp subgradient optimization
HK_MIN_LAMBDA: float = 0.0001


def compute_hk_lower_bound(
    coords: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    max_iter: int = 500,
    initial_pi: npt.NDArray[np.float64] | None = None,
    initial_lb: float = -np.inf,
    target_ub: float = np.inf,
    sample_name: str | None = None,
    *,
    use_cache: bool = True,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Compute the Held-Karp lower bound.

    Args:
        coords: Coordinates of the cities.
        candidate_set: Candidate set for neighbor pruning.
        max_iter: Maximum number of subgradient iterations.
        initial_pi: Initial pi values for subgradient optimization.
        initial_lb: Initial best lower bound for subgradient optimization.
        target_ub: Target upper bound for step size calculation.
        sample_name: Name of the sample for caching.
        use_cache: Whether to load the cached bound if available.

    Returns:
        A tuple (lower_bound, pi_array).
    """
    n = coords.shape[0]
    name = sample_name or str(n)

    if use_cache:
        cached = load_hk_cache(name)
        if cached:
            return cached

    if initial_pi is None:
        init_pi = np.zeros(n, dtype=np.float64)
    else:
        init_pi = np.ascontiguousarray(initial_pi, dtype=np.float64)

    best_lb, best_pi = _compute_hk_impl(
        coords, candidate_set, max_iter, init_pi, initial_lb, target_ub
    )

    return best_lb, best_pi


@njit(fastmath=True)  # type: ignore
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


@njit(fastmath=True)  # type: ignore
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


@njit(fastmath=True)  # type: ignore
def _build_undirected_adj(
    n: int, coords: npt.NDArray[np.float64], candidate_set: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    """Build undirected adjacency list and pre-calculate distances."""
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
    adj_dists = np.empty(int(adj_ptr[n]), dtype=np.float64)
    curr_ptr = adj_ptr.copy()

    for u in range(n):
        for k in range(candidate_set.shape[1]):
            v = int(candidate_set[u, k])
            if v == -1:
                break

            # Calculate original Euclidean distance
            dx = coords[u, 0] - coords[v, 0]
            dy = coords[u, 1] - coords[v, 1]
            d = float(np.sqrt(dx * dx + dy * dy))

            idx_u = curr_ptr[u]
            adj_indices[idx_u] = v
            adj_dists[idx_u] = d
            curr_ptr[u] += 1

            idx_v = curr_ptr[v]
            adj_indices[idx_v] = u
            adj_dists[idx_v] = d
            curr_ptr[v] += 1

    return adj_ptr, adj_indices, adj_dists


@njit(fastmath=True)  # type: ignore
def compute_mst_weight(
    n: int,
    adj_ptr: npt.NDArray[np.int32],
    adj_indices: npt.NDArray[np.int32],
    adj_dists: npt.NDArray[np.float64],
    pi: npt.NDArray[np.float64],
    root: int,
    # Pre-allocated buffers
    min_dist: npt.NDArray[np.float64],
    parent: npt.NDArray[np.int32],
    visited: npt.NDArray[np.bool_],
    degrees: npt.NDArray[np.int32],
    heap_val: npt.NDArray[np.float64],
    heap_node: npt.NDArray[np.int32],
) -> float:
    """Compute the Minimum Spanning Tree (MST) weight of a 1-tree relaxation.

    Uses pre-allocated buffers and pre-calculated distances for performance.
    """
    min_dist.fill(np.inf)
    parent.fill(-1)
    visited.fill(False)
    degrees.fill(0)

    start_node = 1 if root == 0 else 0
    min_dist[start_node] = 0.0

    total_weight = 0.0
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

        p = parent[u]
        if p != -1:
            degrees[u] += 1
            degrees[p] += 1

        for k in range(int(adj_ptr[u]), int(adj_ptr[u + 1])):
            v = int(adj_indices[k])
            if v == root or visited[v]:
                continue

            # Transformed distance: d'(u,v) = d(u,v) + pi[u] + pi[v]
            dist_uv = adj_dists[k] + pi[u] + pi[v]

            if dist_uv < min_dist[v]:
                min_dist[v] = dist_uv
                parent[v] = u
                heap_size = _heap_push(heap_val, heap_node, heap_size, dist_uv, v)

    if nodes_added < n - 1:
        return -1.0
    return total_weight


@njit(fastmath=True)  # type: ignore
def _compute_hk_impl(
    coords: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    max_iter: int,
    initial_pi: npt.NDArray[np.float64],
    initial_lb: float,
    target_ub: float,
) -> tuple[float, npt.NDArray[np.float64]]:
    n = coords.shape[0]
    coords = np.ascontiguousarray(coords)
    candidate_set = np.ascontiguousarray(candidate_set)
    pi = np.ascontiguousarray(initial_pi).copy()

    best_lb = initial_lb
    best_pi = pi.copy()

    adj_ptr, adj_indices, adj_dists = _build_undirected_adj(n, coords, candidate_set)

    # Pre-allocate MST buffers
    min_dist = np.empty(n, dtype=np.float64)
    parent = np.empty(n, dtype=np.int32)
    visited = np.empty(n, dtype=np.bool_)
    degrees = np.empty(n, dtype=np.int32)
    max_heap_size = int(adj_ptr[n])
    heap_val = np.empty(max_heap_size, dtype=np.float64)
    heap_node = np.empty(max_heap_size, dtype=np.int32)

    lambda_val = 2.0
    period = max(100, max_iter // 50)
    last_improvement = 0
    upper_bound = target_ub

    for iteration in range(max_iter):
        root = 0
        mst_weight = -1.0

        # Try a few roots if connectivity issues occur
        for r_try in range(3):
            root = (r_try * (n // 3)) % n
            mst_weight = compute_mst_weight(
                n, adj_ptr, adj_indices, adj_dists, pi, root,
                min_dist, parent, visited, degrees, heap_val, heap_node
            )
            if mst_weight != -1.0:
                break

        if mst_weight == -1.0:
            if best_lb == -np.inf:
                return 0.0, pi
            return best_lb, best_pi

        # Find two shortest edges incident to root in the candidate set
        d1, d2 = np.inf, np.inf
        n1, n2 = -1, -1
        for k in range(int(adj_ptr[root]), int(adj_ptr[root + 1])):
            v = int(adj_indices[k])
            # d'(root, v) = d(root, v) + pi[root] + pi[v]
            d = adj_dists[k] + pi[root] + pi[v]
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
            upper_bound = lb * 1.1 # Closer guess for UB

        subgradient = degrees - 2
        norm_sq = np.sum(subgradient * subgradient)
        if norm_sq == 0:
            break

        if iteration - last_improvement >= period:
            lambda_val *= 0.8 # More conservative decay
            last_improvement = iteration
            if lambda_val < HK_MIN_LAMBDA:
                break

        target = (
            min(upper_bound, best_lb * 1.05)
            if upper_bound > best_lb
            else (best_lb + 1.0)
        )
        step = lambda_val * (target - lb) / norm_sq
        pi += step * subgradient

    return best_lb, best_pi


@njit(fastmath=True, parallel=True)  # type: ignore
def compute_alpha_values(
    n: int,
    coords: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    pi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute Alpha-values for candidate set refinement.

    Alpha-values measure the 'nearness' of an edge to the MST.
    """
    coords = np.ascontiguousarray(coords)
    candidate_set = np.ascontiguousarray(candidate_set)
    pi = np.ascontiguousarray(pi)

    adj_ptr, adj_indices, adj_dists = _build_undirected_adj(n, coords, candidate_set)

    # Pre-allocate MST buffers for alpha calculation
    min_dist = np.empty(n, dtype=np.float64)
    parent = np.empty(n, dtype=np.int32)
    visited = np.empty(n, dtype=np.bool_)
    degrees = np.empty(n, dtype=np.int32)
    max_heap_size = int(adj_ptr[n])
    heap_val = np.empty(max_heap_size, dtype=np.float64)
    heap_node = np.empty(max_heap_size, dtype=np.int32)

    root = 0
    _mst_weight = compute_mst_weight(
        n, adj_ptr, adj_indices, adj_dists, pi, root,
        min_dist, parent, visited, degrees, heap_val, heap_node
    )

    # Find two shortest edges incident to root
    d1, d2 = np.inf, np.inf
    n1, n2 = -1, -1
    for k in range(int(adj_ptr[root]), int(adj_ptr[root + 1])):
        v = int(adj_indices[k])
        d = adj_dists[k] + pi[root] + pi[v]
        if d < d1:
            d2 = d1
            n2 = n1
            d1 = d
            n1 = v
        elif d < d2:
            d2 = d
            n2 = v

    # Build MST adjacency for LCA preprocessing
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
        # Find weight in original adj list to avoid re-calculating
        w = 0.0
        for k in range(int(adj_ptr[i]), int(adj_ptr[i + 1])):
            if adj_indices[k] == p:
                w = adj_dists[k] + pi[i] + pi[p]
                break

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
                    # Find d'(root, other)
                    d_ro = 0.0
                    for k in range(int(adj_ptr[root]), int(adj_ptr[root + 1])):
                        if adj_indices[k] == val_other:
                            d_ro = adj_dists[k] + pi[root] + pi[val_other]
                            break
                    alphas[int(u_idx), cand_k] = d_ro - d2
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

                # Find d'(u, v)
                d_uv = 0.0
                for k in range(int(adj_ptr[u_idx]), int(adj_ptr[u_idx + 1])):
                    if adj_indices[k] == v_idx:
                        d_uv = adj_dists[k] + pi[u_idx] + pi[v_idx]
                        break
                alphas[int(u_idx), cand_k] = d_uv - max_e
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
