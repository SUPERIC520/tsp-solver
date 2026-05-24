import os
import sys
from typing import Tuple, Optional
import numpy as np
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets
from src.core.validation import compute_hk_lower_bound


def load_cached_hk(n_sample: int) -> Tuple[Optional[float], Optional[np.ndarray]]:
    hk_file = f"data/sample_{n_sample}_hk.npy"
    pi_file = f"data/sample_{n_sample}_pi.npy"
    if os.path.exists(hk_file) and os.path.exists(pi_file):
        hk = float(np.load(hk_file)[0])
        pi = np.load(pi_file)
        return hk, pi
    return None, None


def save_cached_hk(n_sample: int, hk: float, pi: np.ndarray) -> None:
    hk_file = f"data/sample_{n_sample}_hk.npy"
    pi_file = f"data/sample_{n_sample}_pi.npy"
    np.save(hk_file, np.array([hk]))
    np.save(pi_file, pi)


def compute_all_hk(force: bool = False) -> None:
    coords_full = load_cities("data/cities.csv")
    full_n = coords_full.shape[0]

    samples = [500, 1000, 5000, 10000, full_n]
    max_iters = [50000, 50000, 20000, 10000, 2000]  # Adjust iter count based on size

    for n_sample, iters in zip(samples, max_iters):
        print(f"\n--- Computing HK for N={n_sample} with {iters} iterations ---")
        coords = coords_full[:n_sample]

        # Check cache if not forced
        if not force:
            cached_hk, _ = load_cached_hk(n_sample)
            if cached_hk is not None:
                print(
                    f"HK already cached for N={n_sample}: {cached_hk:.2f}. Skipping..."
                )
                continue

        print("Building candidate set...")
        candidate_set = build_candidate_sets(coords, k=32)

        print("Computing HK lower bound...")
        lb_val, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=iters)
        save_cached_hk(n_sample, lb_val, pi)
        print(f"Done. HK Lower Bound: {lb_val:.2f}")


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    compute_all_hk(force=force)
