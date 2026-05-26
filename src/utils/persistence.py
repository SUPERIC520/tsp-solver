import csv
import numpy as np
import os
from src.utils.data_io import load_best_length_from_csv

def update_best_tour(best_tour_file: str, new_tour: np.ndarray, new_length: float, is_full_run: bool = True) -> bool:
    """
    Updates the optimal tour on disk if length is a global improvement.
    Skipped if is_full_run is False to prevent overwriting global solutions with sub-scale test runs.
    """
    if not is_full_run:
        return False
        
    current_best = load_best_length_from_csv(best_tour_file)
    if new_length < current_best:
        if os.path.dirname(best_tour_file):
            os.makedirs(os.path.dirname(best_tour_file), exist_ok=True)
        tour_str = ",".join(map(str, new_tour.tolist()))
        with open(best_tour_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tour", "length"])
            writer.writerow([tour_str, str(new_length)])
        return True
        
    return False
