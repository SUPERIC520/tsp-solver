import numpy as np
import pytest
from src.core.kopt_engine import (
    _optimize_2opt,
    _optimize_3opt_sequential,
    cascading_kopt_optimize,
    compute_tour_length,
    _update_pos,
)


def _compute_candidate_dists(
    coords_x: np.ndarray, coords_y: np.ndarray, candidate_set: np.ndarray
) -> np.ndarray:
    n = coords_x.shape[0]
    num_cand = candidate_set.shape[1]
    candidate_dists = np.zeros(candidate_set.shape, dtype=np.float64)
    for i in range(n):
        for k in range(num_cand):
            w = candidate_set[i, k]
            if w != -1:
                dx = coords_x[i] - coords_x[w]
                dy = coords_y[i] - coords_y[w]
                candidate_dists[i, k] = np.sqrt(dx * dx + dy * dy)
    return candidate_dists


def test_kopt_engine_basic() -> None:
    # Create a small set of 20 cities in a circle
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords_x = np.cos(theta)
    coords_y = np.sin(theta)

    # Random initial tour
    tour = np.arange(n)
    np.random.shuffle(tour)

    # Candidate set: just all other cities for simplicity in this small test
    # (In real use, it would be top 16)
    candidate_set = np.full((n, n - 1), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, : len(cands)] = cands

    candidate_dists = _compute_candidate_dists(coords_x, coords_y, candidate_set)

    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    initial_length = compute_tour_length(tour, coords_x, coords_y)

    # Run 2-opt
    improved_2opt = _optimize_2opt(
        tour, coords_x, coords_y, candidate_set, candidate_dists, locked_edges, pos, dlb
    )
    length_after_2opt = compute_tour_length(tour, coords_x, coords_y)

    if improved_2opt:
        assert length_after_2opt < initial_length + 1e-9

    # Run 3-opt
    dlb.fill(False)
    improved_3opt = _optimize_3opt_sequential(
        tour, coords_x, coords_y, candidate_set, candidate_dists, locked_edges, pos, dlb
    )
    length_after_3opt = compute_tour_length(tour, coords_x, coords_y)

    if improved_3opt:
        assert length_after_3opt < length_after_2opt + 1e-9


def test_cascading_kopt_optimize() -> None:
    n = 20
    coords_x = np.random.rand(n)
    coords_y = np.random.rand(n)
    tour = np.arange(n)
    np.random.shuffle(tour)

    candidate_set = np.full((n, n - 1), -1, dtype=np.int32)
    for i in range(n):
        # Use simple distance-based candidates for testing
        dists = np.sqrt((coords_x - coords_x[i]) ** 2 + (coords_y - coords_y[i]) ** 2)
        indices = np.argsort(dists)
        cands = [idx for idx in indices if idx != i]
        candidate_set[i, : len(cands)] = cands

    locked_edges = np.full((n, 2), -1, dtype=np.int32)

    initial_length = compute_tour_length(tour, coords_x, coords_y)
    best_tour, best_length = cascading_kopt_optimize(
        tour, coords_x, coords_y, candidate_set, locked_edges, num_kicks=5
    )

    assert best_length <= initial_length + 1e-9
    # For N=20, it should find a very good tour, likely the optimal circle or close to it
    assert len(np.unique(best_tour)) == n


def test_2opt_circle() -> None:
    # A simple case where 2-opt SHOULD improve
    # 4 points in a square, but tour is criss-crossed
    coords_x = np.array([0, 1, 0, 1], dtype=np.float64)
    coords_y = np.array([0, 1, 1, 0], dtype=np.float64)
    # Tour: 0-1-2-3-0 (length 1.41 + 1 + 1.41 + 1 = 4.82)
    # Better tour: 0-2-1-3-0 (length 1 + 1 + 1 + 1 = 4.0)
    tour = np.array([0, 1, 2, 3], dtype=np.int32)
    n = 4
    candidate_set = np.full((n, 3), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, : len(cands)] = cands

    candidate_dists = _compute_candidate_dists(coords_x, coords_y, candidate_set)

    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    initial_length = compute_tour_length(tour, coords_x, coords_y)
    improved = _optimize_2opt(
        tour, coords_x, coords_y, candidate_set, candidate_dists, locked_edges, pos, dlb
    )
    final_length = compute_tour_length(tour, coords_x, coords_y)

    assert improved
    assert final_length < initial_length - 0.1
    assert final_length == pytest.approx(4.0)
