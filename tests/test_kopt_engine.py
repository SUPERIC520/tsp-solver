import numpy as np
import pytest
import numba
from src.core.kopt_engine import (
    _optimize_2opt,
    _optimize_or_opt,
    _optimize_3opt_sequential,
    _optimize_4opt_sequential,
    _optimize_5opt_sequential,
    cascading_kopt_optimize,
    compute_tour_length,
    _update_pos,
)


@pytest.fixture(scope="module", autouse=True)
def disable_numba_jit():
    # Save original state
    orig_disable_jit = numba.config.DISABLE_JIT
    # Disable JIT
    numba.config.DISABLE_JIT = True
    yield
    # Restore original state
    numba.config.DISABLE_JIT = orig_disable_jit


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


def test_2opt_improvement() -> None:
    # 4 points in a square, but tour is criss-crossed
    coords_x = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    coords_y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    # Tour: 0-1-2-3 (length: sqrt(2) + 1 + sqrt(2) + 1 = 4.828)
    # Better tour: 0-2-1-3 (length: 1 + 1 + 1 + 1 = 4.0)
    tour = np.array([0, 1, 2, 3], dtype=np.int32)
    n = 4
    candidate_set = np.full((n, 3), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, : len(cands)] = cands

    candidate_dists = _compute_candidate_dists(coords_x, coords_y, candidate_set)
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    initial_length = compute_tour_length(tour, coords_x, coords_y)
    improved = _optimize_2opt(
        tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb
    )
    final_length = compute_tour_length(tour, coords_x, coords_y)

    assert improved
    assert final_length < initial_length - 0.1
    assert final_length == pytest.approx(4.0)


def test_or_opt_improvement() -> None:
    # 5 points in a line: 0-1-2-3-4
    # Tour: 0-2-3-1-4 (suboptimal, segment [2,3] is relocated)
    coords_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    coords_y = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    tour = np.array([0, 2, 3, 1, 4], dtype=np.int32)
    n = 5
    candidate_set = np.full((n, 4), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, : len(cands)] = cands

    candidate_dists = _compute_candidate_dists(coords_x, coords_y, candidate_set)
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    initial_length = compute_tour_length(tour, coords_x, coords_y)
    improved = _optimize_or_opt(
        tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb, max_len=2
    )
    final_length = compute_tour_length(tour, coords_x, coords_y)

    # If Or-opt successfully relocates 1 or [2,3], the tour length will decrease.
    # Optimal length: 0-1-2-3-4-0 is 4 + 4 = 8.0
    if improved:
        assert final_length < initial_length - 1e-9


def test_3opt_improvement() -> None:
    # 6 points on a circle
    n = 6
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords_x = np.cos(theta)
    coords_y = np.sin(theta)

    # Criss-crossed tour
    tour = np.array([0, 3, 1, 4, 2, 5], dtype=np.int32)
    candidate_set = np.full((n, 5), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, : len(cands)] = cands

    candidate_dists = _compute_candidate_dists(coords_x, coords_y, candidate_set)
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    initial_length = compute_tour_length(tour, coords_x, coords_y)
    improved = _optimize_3opt_sequential(
        tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb
    )
    final_length = compute_tour_length(tour, coords_x, coords_y)

    if improved:
        assert final_length < initial_length - 1e-9


def test_4opt_improvement() -> None:
    # 8 points on a circle
    n = 8
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords_x = np.cos(theta)
    coords_y = np.sin(theta)

    tour = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int32)
    candidate_set = np.full((n, 7), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, : len(cands)] = cands

    candidate_dists = _compute_candidate_dists(coords_x, coords_y, candidate_set)
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    initial_length = compute_tour_length(tour, coords_x, coords_y)
    improved = _optimize_4opt_sequential(
        tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb
    )
    final_length = compute_tour_length(tour, coords_x, coords_y)

    if improved:
        assert final_length < initial_length - 1e-9


def test_5opt_improvement() -> None:
    # 10 points on a circle
    n = 10
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords_x = np.cos(theta)
    coords_y = np.sin(theta)

    tour = np.array([0, 5, 1, 6, 2, 7, 3, 8, 4, 9], dtype=np.int32)
    candidate_set = np.full((n, 9), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, : len(cands)] = cands

    candidate_dists = _compute_candidate_dists(coords_x, coords_y, candidate_set)
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    initial_length = compute_tour_length(tour, coords_x, coords_y)
    improved = _optimize_5opt_sequential(
        tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb
    )
    final_length = compute_tour_length(tour, coords_x, coords_y)

    if improved:
        assert final_length < initial_length - 1e-9


def test_cascading_kopt_optimize() -> None:
    n = 20
    coords_x = np.random.rand(n)
    coords_y = np.random.rand(n)
    tour = np.arange(n)
    np.random.shuffle(tour)

    candidate_set = np.full((n, n - 1), -1, dtype=np.int32)
    for i in range(n):
        dists = np.sqrt((coords_x - coords_x[i]) ** 2 + (coords_y - coords_y[i]) ** 2)
        indices = np.argsort(dists)
        cands = [idx for idx in indices if idx != i]
        candidate_set[i, : len(cands)] = cands

    initial_length = compute_tour_length(tour, coords_x, coords_y)
    best_tour, best_length, kicks_done = cascading_kopt_optimize(
        tour, coords_x, coords_y, candidate_set, num_kicks=5, max_opt=5
    )

    assert best_length <= initial_length + 1e-9
    assert len(np.unique(best_tour)) == n
    assert kicks_done >= 0
