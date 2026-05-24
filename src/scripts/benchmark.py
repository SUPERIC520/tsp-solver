import time
import numpy as np
import sys
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets
from src.core.seed_generation import generate_hilbert_seeds
from src.core.orchestration import parallel_solve
from src.core.validation import compute_hk_lower_bound, validate_result


def benchmark(n: int = 500, kicks: int | None = None) -> None:
    print(f"--- Benchmarking TSP Solver (N={n}) ---")
    start_total = time.time()

    # 1. Load Data
    coords_full = load_cities("data/cities.csv")
    coords = coords_full[:n]
    print(f"Loaded {n} cities.")

    # 2. Preprocessing
    print("Step 2: Building candidate sets (Delaunay top 16)...")
    candidate_set = build_candidate_sets(coords, k=16)

    # 3. HK Bound
    print("Step 3: Computing or loading HK lower bound...")
    sample_name = f"bench_{n}"
    lb_val, pi = compute_hk_lower_bound(
        coords, candidate_set, max_iter=200, sample_name=sample_name
    )
    print(f"HK Lower Bound: {lb_val:.2f}")

    print("Step 3.5: Refining candidate sets with Alpha-values...")
    from src.core.preprocessing import refine_with_alpha

    candidate_set = refine_with_alpha(coords, candidate_set, pi)

    # 4. Seed Generation
    print("Step 4: Generating 8 Hilbert seeds...")
    seeds = generate_hilbert_seeds(coords, num_seeds=8)

    # 5. Parallel Solve
    print("Step 5: Running Full K-Opt Parallel Solve...")
    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    start_solve = time.time()

    if kicks is not None:
        num_kicks = kicks
    else:
        num_kicks = 10000
        if n >= 5000:
            num_kicks = 150000
        elif n >= 1000:
            num_kicks = 30000
        else:
            num_kicks = max(1000, n * 2)

    print(f"Using num_kicks={num_kicks}")
    results = parallel_solve(
        seeds, coords, candidate_set, locked_edges, num_kicks=num_kicks
    )
    solve_time = time.time() - start_solve
    print(f"Solve completed in {solve_time:.2f}s.")

    # 6. Aggregate
    best_length = min(r[1] for r in results)
    gap = validate_result(best_length, lb_val)

    print(f"\nResults for N={n}:")
    print(f"Best Length: {best_length:.2f}")
    print(f"Gap: {gap:.4f}%")
    print(f"Total Time: {time.time() - start_total:.2f}s")

    if n == 500:
        if solve_time < 10 and gap < 5:
            print("SUCCESS: N=500 target met!")
        else:
            print(
                f"FAILURE: N=500 target not met. (Time: {solve_time:.2f}s, Gap: {gap:.4f}%)"
            )


if __name__ == "__main__":
    n_arg = 500
    kicks_arg = None
    if len(sys.argv) > 1:
        n_arg = int(sys.argv[1])
    if len(sys.argv) > 2:
        kicks_arg = int(sys.argv[2])
    benchmark(n_arg, kicks_arg)
