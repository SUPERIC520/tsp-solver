import numpy as np
from src.core.validation import compute_hk_lower_bound, validate_result
from src.core.preprocessing import build_candidate_sets


def test_compute_hk_lower_bound_square():
    # 4 points in a square
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )

    candidate_set = build_candidate_sets(coords, k=3)
    lb, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=50)

    # Lower bound for square should be close to 4.0
    assert lb > 3.9
    assert lb <= 4.000001


def test_validate_result():
    gap = validate_result(101.0, 100.0)
    assert np.isclose(gap, 1.0)
