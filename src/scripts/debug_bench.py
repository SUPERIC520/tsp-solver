import time
import numpy as np
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets
from src.core.seed_generation import generate_hilbert_seeds
from src.core.kopt_engine import cascading_kopt_optimize
from src.core.validation import compute_hk_lower_bound, validate_result


def benchmark(n: int = 500) -> None:
    print(f"--- Debug Benchmarking TSP Solver (N={n}) ---")

    # 1. Load Data
    coords_full = load_cities("data/cities.csv")
    coords = coords_full[:n]

    # 2. Preprocessing
    candidate_set = build_candidate_sets(coords, k=16)

    # 3. HK Bound
    lb_val, pi = compute_hk_lower_bound(
        coords, candidate_set, max_iter=200, sample_name=f"bench_{n}"
    )

    # 4. Seed Generation
    seeds = generate_hilbert_seeds(coords, num_seeds=1)

    # 5. Single Solve
    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    start_solve = time.time()
    tour, length, kicks = cascading_kopt_optimize(
        seeds[0], coords[:, 0], coords[:, 1], candidate_set, num_kicks=0
    )
    solve_time = time.time() - start_solve
    print(f"Single Solve completed in {solve_time:.2f}s. Length: {length:.2f}")

    gap = validate_result(length, lb_val)
    print(f"Gap: {gap:.4f}%")


if __name__ == "__main__":
    benchmark(50)
