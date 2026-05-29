"""Robustness tests for data I/O and pi alignment."""

from pathlib import Path

import numpy as np
import pytest

from src.core.seed_generation import rotate_tour
from src.utils.data_io import load_hk_cache, load_tour, save_hk_cache


def test_hk_cache_shape_robustness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that we can handle both 0D and 1D shapes in HK cache."""
    # Mock CACHE_DIR to use temp path
    monkeypatch.setattr("src.utils.data_io.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.utils.data_io.CACHE_VERSION", "test_v1")

    sample_name = "test_city_99"
    lb_val = 1234.56
    pi = np.random.rand(99)

    # Save using standard utility
    save_hk_cache(sample_name, lb_val, pi)

    # Verify we can load it back
    res = load_hk_cache(sample_name)
    assert res is not None
    loaded_lb, loaded_pi = res
    assert float(loaded_lb) == pytest.approx(lb_val)
    assert np.allclose(loaded_pi, pi)

    # Manually overwrite with a 1D array for LB to simulate "buggy" or older format
    bound_path = tmp_path / "test_v1" / f"sample_{sample_name}_hk.npy"
    np.save(bound_path, np.array([lb_val]))  # 1D instead of 0D

    # Verify the loader handles 1D
    res_1d = load_hk_cache(sample_name)
    assert res_1d is not None
    loaded_lb_1d, _ = res_1d
    assert float(np.array(loaded_lb_1d).reshape(-1)[0]) == pytest.approx(lb_val)


def test_start_tour_rotation() -> None:
    """Verify that rotate_tour correctly handles different start nodes."""
    original_tour = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    # Rotate to start with node 2
    rotated = np.empty_like(original_tour)
    rotate_tour(original_tour, 2, out=rotated)
    assert np.array_equal(rotated, [2, 3, 4, 0, 1])

    # Rotate to start with node 0 (no change in order, just start)
    rotate_tour(original_tour, 0, out=rotated)
    assert np.array_equal(rotated, [0, 1, 2, 3, 4])

    # Rotate to start with node 4
    rotate_tour(original_tour, 4, out=rotated)
    assert np.array_equal(rotated, [4, 0, 1, 2, 3])


def test_invalid_tour_file(tmp_path: Path) -> None:
    """Test behavior when loading a non-existent tour file."""
    fake_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_tour(str(fake_path))


def test_pi_alignment_simulation() -> None:
    """Simulate the reordering logic in main.py to ensure pi vectors stay aligned."""
    n = 10

    orig_to_new = np.random.permutation(n).astype(np.int32)
    new_to_orig = np.empty(n, dtype=np.int32)
    new_to_orig[orig_to_new] = np.arange(n, dtype=np.int32)

    # Original pi values (node 0 has pi 0.0, node 1 has pi 0.1, etc.)
    pi_orig = np.arange(n, dtype=np.float64) / 10.0

    # In main.py, we load pi_orig and then:
    # pi = pi_orig[new_to_orig]
    pi_new = pi_orig[new_to_orig]

    # To check if node 'i' in the NEW ordering has the correct pi:
    # The new node 'j' corresponds to original node new_to_orig[j]
    for j in range(n):
        orig_node_idx = new_to_orig[j]
        assert pi_new[j] == pi_orig[orig_node_idx]
