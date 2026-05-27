"""Utilities for persisting the best found tours and managing results."""

import csv
from pathlib import Path

import numpy as np

from src.utils.data_io import load_best_length_from_csv


def update_best_tour(
    best_tour_file: str,
    new_tour: np.ndarray,
    new_length: float,
    *,
    is_full_run: bool = True,
) -> bool:
    """Update the optimal tour on disk if the new length is a global improvement.

    Args:
        best_tour_file: Path to the CSV file storing the best tour.
        new_tour: Array of city indices for the new tour.
        new_length: Total length of the new tour.
        is_full_run: Whether this is a full production run. If False,
            persistence is skipped to prevent overwriting global solutions.

    Returns:
        bool: True if the file was updated, False otherwise.
    """
    if not is_full_run:
        return False

    current_best = load_best_length_from_csv(best_tour_file)
    if new_length < current_best:
        path = Path(best_tour_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tour_str = ",".join(map(str, new_tour.tolist()))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tour", "length"])
            writer.writerow([tour_str, str(new_length)])
        return True

    return False
