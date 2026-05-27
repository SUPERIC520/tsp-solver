"""Seed generation for TSP optimization.

This module provides functions for generating initial tours (seeds) using
greedy nearest-neighbor heuristics, with support for parallel execution.
"""

import multiprocessing as mp
from typing import cast

import numpy as np
from numba import njit

from src.config import NUM_PROCESSES_SEEDING
from src.utils.memory_utils import ensure_alignment


@njit(fastmath=True, cache=True)  # type: ignore
def _greedy_nn_tour(
    coords: np.ndarray,
    candidate_set: np.ndarray,
    start: int,
) -> np.ndarray:
    """Build a greedy nearest-neighbor tour starting from `start`.

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


def _greedy_nn_worker(args: tuple[np.ndarray, np.ndarray, int]) -> np.ndarray:
    """Worker function for parallel greedy NN seed generation."""
    coords, candidate_set, start_node = args
    return cast("np.ndarray", _greedy_nn_tour(coords, candidate_set, int(start_node)))


@njit(cache=True)  # type: ignore
def _find_index_jit(arr: np.ndarray, val: int) -> int:
    for i in range(arr.shape[0]):
        if arr[i] == val:
            return i
    return -1


@njit(cache=True)  # type: ignore
def _rotate_tour_jit(tour: np.ndarray, start_idx: int) -> np.ndarray:
    n = tour.shape[0]
    out = np.empty(n, dtype=np.int32)
    out[: n - start_idx] = tour[start_idx:]
    out[n - start_idx :] = tour[:start_idx]
    return out


def rotate_tour(tour: np.ndarray, start_node: int) -> np.ndarray:
    """Rotate starting node sequence.

    Keeps the path cycle topology unchanged.
    Ensure the output array is C-contiguous and 64-byte aligned.
    """
    tour = np.ascontiguousarray(tour, dtype=np.int32)
    start_idx = _find_index_jit(tour, start_node)
    if start_idx == -1:
        msg = f"start_node {start_node} not found in tour"
        raise ValueError(msg)

    rotated = _rotate_tour_jit(tour, start_idx)
    return ensure_alignment(rotated, alignment=64)


def generate_greedy_nn_seeds(
    coords: np.ndarray,
    candidate_set: np.ndarray,
    num_seeds: int = 1,
    start_nodes: np.ndarray | None = None,
) -> np.ndarray:
    """Generate greedy nearest-neighbor seeds in parallel.

    Diverse starting cities are used.
    Uses multiprocessing.Pool (bounded by NUM_PROCESSES_SEEDING).
    Returns a 64-byte aligned contiguous matrix of shape (num_seeds, N).
    """
    n = coords.shape[0]

    if num_seeds <= 0:
        res = np.empty((0, n), dtype=np.int32)
        return ensure_alignment(res, alignment=64)

    if start_nodes is None:
        # Space starting nodes evenly across the tour
        step = max(1, n // num_seeds)
        start_nodes = np.array([i * step for i in range(num_seeds)], dtype=np.int32)
    elif start_nodes.shape[0] != num_seeds:
        msg = (
            f"start_nodes length {start_nodes.shape[0]} "
            f"does not match num_seeds {num_seeds}"
        )
        raise ValueError(msg)

    # Determine process count
    num_procs = NUM_PROCESSES_SEEDING
    if num_procs <= 0:
        num_procs = mp.cpu_count()
    num_procs = min(num_procs, num_seeds)

    # Ensure inputs are C-contiguous and correctly typed
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    candidate_set = np.ascontiguousarray(candidate_set, dtype=np.int32)
    start_nodes = np.ascontiguousarray(start_nodes, dtype=np.int32)

    # Prepare tasks
    tasks = [(coords, candidate_set, int(start_nodes[i])) for i in range(num_seeds)]

    # Run in parallel pool
    if num_procs > 1:
        with mp.Pool(processes=num_procs) as pool:
            tours = pool.map(_greedy_nn_worker, tasks)
    else:
        # Fallback to serial execution if only 1 process is used
        tours = [_greedy_nn_worker(task) for task in tasks]

    # Convert list of tours to a 2D numpy array
    seeds = np.empty((num_seeds, n), dtype=np.int32)
    for i in range(num_seeds):
        seeds[i] = tours[i]

    # Ensure the returned matrix is 64-byte aligned and contiguous
    return ensure_alignment(seeds, alignment=64)
