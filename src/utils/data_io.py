import numpy as np
import os


def load_cities(filepath: str) -> np.ndarray:
    """
    Load city coordinates from a CSV file.
    Format: Index X Y (Space-separated)
    Returns: np.ndarray of shape (N, 2) with dtype float64.
    """
    # Using np.loadtxt which handles space-separated values by default
    # Usecols (1, 2) to skip the Index column
    data = np.loadtxt(filepath, usecols=(1, 2), dtype=np.float64)
    return data


def save_tour(filepath: str, tour: np.ndarray, length: float) -> None:
    """
    Save the optimized tour and its length to a file.
    """
    with open(filepath, "w") as f:
        f.write(f"Total Length: {length}\n")
        f.write("Tour Indices:\n")
        # Save indices one per line or space-separated.
        f.write(" ".join(map(str, tour)))
        f.write("\n")


def load_tour(filepath: str) -> tuple[np.ndarray, float]:
    """
    Load a tour and its saved length from a file.
    Returns: (tour_indices, length)
    """
    with open(filepath, "r") as f:
        line1 = f.readline()
        length = float(line1.split(": ")[1])
        f.readline()  # Skip "Tour Indices:"
        line3 = f.readline()
        tour = np.array([int(i) for i in line3.split()], dtype=np.int32)
    return tour, length


def get_hk_cache_paths(sample_name: str) -> tuple[str, str]:
    """
    Generate paths for HK bound and Pi vector cache.
    """
    bound_path = f"data/sample_{sample_name}_hk.npy"
    pi_path = f"data/sample_{sample_name}_pi.npy"
    return bound_path, pi_path


def load_hk_cache(sample_name: str) -> tuple[float, np.ndarray] | None:
    """
    Load HK bound and Pi vector from cache if they exist.
    """
    bound_path, pi_path = get_hk_cache_paths(sample_name)
    if os.path.exists(bound_path) and os.path.exists(pi_path):
        try:
            bound = float(np.load(bound_path))
            pi = np.load(pi_path)
            return bound, pi
        except Exception:
            return None
    return None


def save_hk_cache(sample_name: str, bound: float, pi: np.ndarray) -> None:
    """
    Save HK bound and Pi vector to cache.
    """
    bound_path, pi_path = get_hk_cache_paths(sample_name)
    os.makedirs(os.path.dirname(bound_path), exist_ok=True)
    np.save(bound_path, np.array(bound))
    np.save(pi_path, pi)
