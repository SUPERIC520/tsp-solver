import numpy as np
import pytest
from src.core.kopt_engine import (
    _optimize_2opt,
    _optimize_3opt_sequential,
    cascading_kopt_optimize,
    compute_tour_length,
    _update_pos,
)


def test_kopt_engine_basic():
    # Create a small set of 20 cities in a circle
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    # Random initial tour
    tour = np.arange(n)
    np.random.shuffle(tour)

    # Candidate set: just all other cities for simplicity in this small test
    # (In real use, it would be top 16)
    candidate_set = np.full((n, n - 1), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, : len(cands)] = cands

    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    initial_length = compute_tour_length(tour, coords)

    # Run 2-opt
    improved_2opt = _optimize_2opt(tour, coords, candidate_set, locked_edges, pos, dlb)
    length_after_2opt = compute_tour_length(tour, coords)

    if improved_2opt:
        assert length_after_2opt < initial_length + 1e-9

    # Run 3-opt
    dlb.fill(False)
    improved_3opt = _optimize_3opt_sequential(
        tour, coords, candidate_set, locked_edges, pos, dlb
    )
    length_after_3opt = compute_tour_length(tour, coords)

    if improved_3opt:
        assert length_after_3opt < length_after_2opt + 1e-9


def test_cascading_kopt_optimize():
    n = 20
    coords = np.random.rand(n, 2)
    tour = np.arange(n)
    np.random.shuffle(tour)

    candidate_set = np.full((n, n - 1), -1, dtype=np.int32)
    for i in range(n):
        # Use simple distance-based candidates for testing
        dists = np.linalg.norm(coords - coords[i], axis=1)
        indices = np.argsort(dists)
        cands = [idx for idx in indices if idx != i]
        candidate_set[i, : len(cands)] = cands

    locked_edges = np.full((n, 2), -1, dtype=np.int32)

    initial_length = compute_tour_length(tour, coords)
    best_tour, best_length = cascading_kopt_optimize(
        tour, coords, candidate_set, locked_edges, num_kicks=5
    )

    assert best_length <= initial_length + 1e-9
    # For N=20, it should find a very good tour, likely the optimal circle or close to it
    assert len(np.unique(best_tour)) == n


def test_2opt_circle():
    # A simple case where 2-opt SHOULD improve
    # 4 points in a square, but tour is criss-crossed
    coords = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float64)
    # Tour: 0-1-2-3-0 (length 1.41 + 1 + 1.41 + 1 = 4.82)
    # Better tour: 0-2-1-3-0 (length 1 + 1 + 1 + 1 = 4.0)
    tour = np.array([0, 1, 2, 3], dtype=np.int32)
    n = 4
    candidate_set = np.full((n, 3), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, : len(cands)] = cands

    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    initial_length = compute_tour_length(tour, coords)
    improved = _optimize_2opt(tour, coords, candidate_set, locked_edges, pos, dlb)
    final_length = compute_tour_length(tour, coords)

    assert improved
    assert final_length < initial_length - 0.1
    assert final_length == pytest.approx(4.0)
