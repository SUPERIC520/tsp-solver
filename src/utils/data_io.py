import csv
import pickle
from pathlib import Path

import numpy as np

from src.config import CACHE_VERSION


def load_cities(filepath: str) -> np.ndarray:
    """Load city coordinates from a CSV file.

    Format: Index X Y (Space-separated)
    Returns: np.ndarray of shape (N, 2) with dtype float64.
    """
    # Using np.loadtxt which handles space-separated values by default
    # Usecols (1, 2) to skip the Index column
    return np.loadtxt(filepath, usecols=(1, 2), dtype=np.float64)


def save_solution_csv(filepath: str, tour: np.ndarray, length: float) -> None:
    """Save the optimized tour and its length to a CSV file.

    Format:
    length,L
    index,I1,I2,...,IN.
    """
    path = Path(filepath)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"length,{length}\n")
        # Joining indices with commas for CSV format
        indices_str = ",".join(map(str, tour))
        f.write(f"indices,{indices_str}\n")


def load_best_length_from_csv(filepath: str) -> float:
    """Reads a CSV file and parses the best length.

    Returns np.inf if the file is missing or corrupt.
    Supports:
    1. Custom format:
       length,L
       indices,I1,I2...
    2. Pandas format:
       tour,length
       "I1,I2...",L.
    """
    path = Path(filepath)
    if not path.exists():
        return float("inf")
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            row1 = next(reader, None)
            min_csv_columns = 2
            if row1 is None or len(row1) < min_csv_columns:
                return float("inf")

            # Format 1: length,L
            if row1[0] == "length":
                return float(row1[1])

            # Format 2: tour,length
            if "length" in row1:
                idx = row1.index("length")
                row2 = next(reader, None)
                if row2 is not None and len(row2) > idx:
                    return float(row2[idx])

            return float("inf")
    except (OSError, ValueError, csv.Error):
        return float("inf")


def save_tour(filepath: str, tour: np.ndarray, length: float) -> None:
    """Save the optimized tour and its length to a file."""
    path = Path(filepath)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"Total Length: {length}\n")
        f.write("Tour Indices:\n")
        # Save indices one per line or space-separated.
        f.write(" ".join(map(str, tour)))
        f.write("\n")


def load_tour(filepath: str) -> tuple[np.ndarray, float]:
    r"""Load a tour and its saved length from a file.

    Supports simple file format (length\nindices) and CSV format
    (length,L\nindices,I1,I2...).
    Returns: (tour_indices, length).
    """
    path = Path(filepath)
    if filepath.endswith(".csv"):
        # CSV format:
        # length,L
        # indices,I1,I2,...,IN
        with path.open("r", encoding="utf-8") as f:
            line1 = f.readline()
            length = float(line1.split(",")[1])
            line2 = f.readline()
            tour = np.array([int(i) for i in line2.split(",")[1:]], dtype=np.int32)
        return tour, length
    # Simple format
    with path.open("r", encoding="utf-8") as f:
        line1 = f.readline()
        length = float(line1.split(": ")[1])
        f.readline()  # Skip "Tour Indices:"
        line3 = f.readline()
        tour = np.array([int(i) for i in line3.split()], dtype=np.int32)
    return tour, length


def get_hk_cache_paths(sample_name: str) -> tuple[str, str]:
    """Generate paths for HK bound and Pi vector cache."""
    bound_path = f"data/cache/{CACHE_VERSION}/sample_{sample_name}_hk.npy"
    pi_path = f"data/cache/{CACHE_VERSION}/sample_{sample_name}_pi.npy"
    return bound_path, pi_path


def load_hk_cache(sample_name: str) -> tuple[float, np.ndarray] | None:
    """Load HK bound and Pi vector from cache if they exist."""
    bound_path, pi_path = get_hk_cache_paths(sample_name)
    bp = Path(bound_path)
    pp = Path(pi_path)
    if bp.exists() and pp.exists():
        try:
            bound = float(np.load(bound_path))
            pi = np.load(pi_path)
        except (ValueError, OSError, pickle.UnpicklingError):
            return None
        else:
            return bound, pi
    return None


def save_hk_cache(sample_name: str, bound: float, pi: np.ndarray) -> None:
    """Save HK bound and Pi vector to cache."""
    bound_path, pi_path = get_hk_cache_paths(sample_name)
    bp = Path(bound_path)
    bp.parent.mkdir(parents=True, exist_ok=True)
    np.save(bound_path, np.array(bound))
    np.save(pi_path, pi)
