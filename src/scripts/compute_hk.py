"""Script to compute and cache Held-Karp lower bounds for TSP.

This script allows for pre-computing the HK bound and pi vector for large
datasets, ensuring they are available for the main solver iterations.
"""

import argparse
import time

from src.config import DATA_PATH, KD_TREE_QUERY_SIZE
from src.core.preprocessing import build_candidate_sets
from src.core.validation import compute_hk_lower_bound
from src.utils.data_io import load_cities, save_hk_cache


def main() -> None:
    """Compute and cache Held-Karp lower bound."""
    parser = argparse.ArgumentParser(description="Compute Held-Karp lower bound.")
    parser.add_argument("--n", type=int, default=0, help="Number of cities (0 for all)")
    parser.add_argument("--iters", type=int, default=10000, help="HK iterations")
    parser.add_argument(
        "--k", type=int, default=KD_TREE_QUERY_SIZE, help="Candidate set size"
    )
    args = parser.parse_args()

    print(f"Loading data (n={args.n if args.n > 0 else 'Full'})...")
    coords_full = load_cities(str(DATA_PATH))
    coords = coords_full[: args.n] if args.n > 0 else coords_full
    n = coords.shape[0]

    print(f"Building candidate set (k={args.k})...")
    t0 = time.time()
    candidate_set = build_candidate_sets(coords, k=args.k)
    print(f"Candidate set built in {time.time() - t0:.2f}s")

    print(f"Computing Held-Karp lower bound ({args.iters} iterations)...")
    t0 = time.time()
    # Note: We use use_cache=False to force re-computation or extension
    lb, pi = compute_hk_lower_bound(
        coords, candidate_set, max_iter=args.iters, use_cache=False
    )
    duration = time.time() - t0

    print(f"\nFinal Result for N={n}:")
    print(f"  - Lower Bound: {lb:.2f}")
    print(f"  - Time taken: {duration:.2f}s ({duration/60:.2f}m)")
    print(f"  - Speed: {duration/args.iters:.4f}s/iter")

    # Save to .npy format
    sample_name = str(n)
    print(f"Saving to .npy cache (sample_{sample_name}_hk.npy)...")
    save_hk_cache(sample_name, lb, pi)


if __name__ == "__main__":
    main()
