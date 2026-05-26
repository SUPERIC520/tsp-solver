"""
Extended tests for the Data I/O module.
"""

from pathlib import Path
import numpy as np
from src.utils.data_io import (
    save_solution_csv,
    load_best_length_from_csv,
    get_hk_cache_paths,
    load_hk_cache,
    save_hk_cache,
)


def test_save_solution_csv_and_load_best_length(tmp_path: Path) -> None:
    """Test saving solutions to CSV and loading the best length."""
    filepath = str(tmp_path / "solutions.csv")
    tour = np.array([0, 2, 4, 1, 3], dtype=np.int32)
    length = 123.456

    # Test load from non-existent file
    assert load_best_length_from_csv(filepath) == float("inf")

    # Save
    save_solution_csv(filepath, tour, length)
    assert Path(filepath).exists()

    # Load best length
    loaded_length = load_best_length_from_csv(filepath)
    assert np.isclose(loaded_length, length)

    # Test invalid file content
    with Path(filepath).open("w") as f:
        f.write("invalid content\n")
    assert load_best_length_from_csv(filepath) == float("inf")


def test_hk_cache() -> None:
    """Test Held-Karp bound and Pi vector caching."""
    sample_name = "test_sample_123"
    bound = 100.5
    pi = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Ensure paths are what we expect
    bound_path_str, pi_path_str = get_hk_cache_paths(sample_name)
    bound_path = Path(bound_path_str)
    pi_path = Path(pi_path_str)
    assert "test_sample_123" in bound_path_str
    assert "test_sample_123" in pi_path_str

    # Clean up before test if they exist (unlikely)
    bound_path.unlink(missing_ok=True)
    pi_path.unlink(missing_ok=True)

    try:
        # Load from empty cache
        assert load_hk_cache(sample_name) is None

        # Save to cache
        save_hk_cache(sample_name, bound, pi)
        assert bound_path.exists()
        assert pi_path.exists()

        # Load from cache
        result = load_hk_cache(sample_name)
        assert result is not None
        loaded_bound, loaded_pi = result
        assert np.isclose(loaded_bound, bound)
        assert np.allclose(loaded_pi, pi)

    finally:
        # Clean up
        bound_path.unlink(missing_ok=True)
        pi_path.unlink(missing_ok=True)
