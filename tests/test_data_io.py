"""Tests for basic data loading and saving functionality."""

from pathlib import Path

import numpy as np

from src.utils.data_io import load_cities, load_tour, save_solution_csv


def test_load_cities() -> None:
    """Test loading city data from a CSV file."""
    # This just checks that it doesn't crash and returns expected shape
    # (assuming cities.csv exists in the default data path)
    coords = load_cities("data/cities.csv")
    assert coords.ndim == 2
    assert coords.shape[1] == 2


def test_save_and_load_tour(tmp_path: Path) -> None:
    """Test saving a tour to disk and loading it back."""
    # Setup dummy data
    tour = np.array([0, 1, 2, 3], dtype=np.int32)
    length = 10.0
    file_path = tmp_path / "test_tour.csv"

    # Save using the standard production saver
    save_solution_csv(str(file_path), tour, length)

    # Load
    loaded_tour, loaded_length = load_tour(str(file_path))

    # Verify
    np.testing.assert_array_equal(loaded_tour, tour)
    assert abs(loaded_length - length) < 1e-6
