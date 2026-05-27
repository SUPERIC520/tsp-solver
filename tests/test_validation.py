"""Tests for Held-Karp lower bounds and solution validation."""

import numpy as np

from src.core.preprocessing import build_candidate_sets
from src.core.validation import compute_hk_lower_bound, validate_result


def test_compute_hk_lower_bound_square() -> None:
    # 4 points in a square
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )

    candidate_set = build_candidate_sets(coords, k=3)
    lb, _pi = compute_hk_lower_bound(coords, candidate_set, max_iter=50)

    # Lower bound for square should be close to 4.0
    assert lb > 3.9
    assert lb <= 4.000001


def test_validate_result() -> None:
    gap = validate_result(101.0, 100.0)
    assert np.isclose(gap, 1.0)


def test_hk_lower_bound_small() -> None:
    """T6.4: Run HK on a tiny coord set (N=5).

    Verify lb > 0 and pi has correct shape.
    """
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.866],
            [1.5, 0.866],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )
    candidate_set = build_candidate_sets(coords, k=4)
    lb, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=200)

    assert lb > 0.0, f"Expected lb > 0, got {lb}"
    assert pi.shape == (5,), f"Expected pi shape (5,), got {pi.shape}"


def test_alpha_values_shape() -> None:
    """T6.4: compute_alpha_values returns array of shape (N, K), all values >= 0."""
    from src.core.validation import compute_alpha_values

    np.random.seed(0)
    n = 8
    k = 5
    coords = np.random.rand(n, 2).astype(np.float64)
    candidate_set = build_candidate_sets(coords, k=k)

    _lb, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=100)
    alphas = compute_alpha_values(n, coords, candidate_set, pi)

    assert isinstance(alphas, np.ndarray)
    assert alphas.shape == (n, k), f"Expected shape ({n}, {k}), got {alphas.shape}"
    assert alphas.dtype == np.float64

    # All valid (non-sentinel) alpha values must be >= 0
    valid_mask = candidate_set != -1
    assert np.all(alphas[valid_mask] >= -1e-9), "Alpha values must be non-negative"


def test_hk_bound_precision_and_cache() -> None:
    import os

    # 5 points in a line — use a fresh unique set of coords to avoid N=5 cache collision
    # We use N=6 to avoid caching overlap with test_hk_lower_bound_small (N=5)
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=np.float64,
    )
    candidate_set = build_candidate_sets(coords, k=5)
    lb, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=200)

    # For a 6-point line the HK lower bound should be positive and pi has correct shape
    assert lb > 0.0
    assert pi.shape == (6,)

    from src.core.validation import load_hk_cache_json, save_hk_cache_json

    dummy_n = 9999
    dummy_lb = 42.42
    dummy_pi = np.ones(dummy_n, dtype=np.float64) * 0.123

    save_hk_cache_json(dummy_n, dummy_lb, dummy_pi)

    loaded = load_hk_cache_json(dummy_n)
    assert loaded is not None
    assert np.isclose(loaded[0], dummy_lb)
    assert np.allclose(loaded[1], dummy_pi)

    dummy_coords = np.zeros((dummy_n, 2), dtype=np.float64)
    dummy_candidates = np.zeros((dummy_n, 3), dtype=np.int32)

    lb_cached, pi_cached = compute_hk_lower_bound(dummy_coords, dummy_candidates)
    assert np.isclose(lb_cached, dummy_lb)
    assert np.allclose(pi_cached, dummy_pi)

    # Clean up dummy entry
    import json

    from src.core.validation import CACHE_PATH

    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                data = json.load(f)
            if str(dummy_n) in data:
                del data[str(dummy_n)]
                with open(CACHE_PATH, "w") as f:
                    json.dump(data, f)
        except Exception:
            pass


def test_compute_alpha_values_dimensions_and_shape() -> None:
    # 10 random points
    np.random.seed(42)
    coords = np.random.rand(10, 2).astype(np.float64)
    candidate_set = build_candidate_sets(coords, k=5)

    _lb, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=50)

    from src.core.validation import compute_alpha_values

    alphas = compute_alpha_values(len(coords), coords, candidate_set, pi)

    assert isinstance(alphas, np.ndarray)
    assert alphas.shape == candidate_set.shape
    assert alphas.shape == (10, 5)
    assert alphas.dtype == np.float64

    valid_mask = candidate_set != -1
    valid_alphas = alphas[valid_mask]
    assert np.all(valid_alphas >= -1e-9)
    assert np.any(np.isclose(valid_alphas, 0.0, atol=1e-9))
