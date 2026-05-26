import os
from typing import Any

import numpy as np

from src.core.kopt_engine import compute_tour_length
from src.utils.data_io import load_cities, load_tour, save_tour


def test_load_cities() -> None:
    filepath = "tests/data/sample_cities.csv"
    coords = load_cities(filepath)

    assert coords.shape == (5, 2)
    assert coords.dtype == np.float64
    assert np.allclose(coords[0], [0.0, 0.0])
    assert np.allclose(coords[-1], [0.5, 0.5])


def test_save_and_load_tour(tmp_path: Any) -> None:
    # Setup dummy data
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64
    )
    tour = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    # Calculate length
    length = compute_tour_length(tour, coords[:, 0], coords[:, 1])

    filepath = tmp_path / "test_tour.txt"

    # Save
    save_tour(str(filepath), tour, length)

    assert os.path.exists(filepath)

    # Load
    loaded_tour, loaded_length = load_tour(str(filepath))

    # Verify
    assert np.array_equal(tour, loaded_tour)
    assert np.isclose(length, loaded_length)

    # Recalculate length from loaded tour and original coords
    recalculated_length = compute_tour_length(loaded_tour, coords[:, 0], coords[:, 1])
    assert np.isclose(recalculated_length, loaded_length)
