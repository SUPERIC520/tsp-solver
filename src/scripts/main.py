"""Main entry point for the TSP Solver production run.

This script orchestrates the full TSP solving pipeline, including data loading,
Hilbert reordering, Held-Karp lower bound computation, candidate set refinement,
parallel K-Opt optimization, and results persistence.
"""

import argparse
import multiprocessing as mp
import time

import numpy as np

from src.config import (
    BEST_TOUR_PATH,
    CACHE_DIR,
    CACHE_VERSION,
    DATA_PATH,
    K_NEIGHBORS,
    KD_TREE_QUERY_SIZE,
    SOLUTIONS_PATH,
)
from src.core.orchestration import parallel_solve
from src.core.preprocessing import (
    build_candidate_sets,
    hilbert_reorder_cities,
    refine_candidate_set_with_alpha,
)
from src.core.seed_generation import generate_greedy_nn_seeds, rotate_tour
from src.core.validation import compute_hk_lower_bound, validate_result
from src.utils.data_io import load_cities, load_tour, save_solution_csv
from src.utils.persistence import update_best_tour


def save_cached_hk_main(n: int, hk: float, pi: np.ndarray) -> None:
    """Save computed Held-Karp lower bound and pi vector to cache.

    Args:
        n: Number of cities.
        hk: Computed lower bound.
        pi: Computed pi vector.
    """
    cache_subdir = CACHE_DIR / CACHE_VERSION
    hk_file = cache_subdir / f"sample_{n}_hk.npy"
    pi_file = cache_subdir / f"sample_{n}_pi.npy"
    hk_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(hk_file, np.array([hk]))
    np.save(pi_file, pi)


def main() -> None:
    """Run the main TSP solver optimization loop."""
    parser = argparse.ArgumentParser(description="TSP Solver - Final Production Run")
    parser.add_argument(
        "--kicks", type=int, default=25000, help="Number of kicks per seed"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Number of full iterations (0 for infinite)",
    )
    parser.add_argument("--seeds", type=int, default=8, help="Total number of seeds")
    parser.add_argument(
        "--max_opt",
        type=int,
        default=3,
        help="Max k for k-opt (3 is standard successful config)",
    )
    parser.add_argument(
        "--n", type=int, default=0, help="Number of cities to subset (0 for all)"
    )
    parser.add_argument(
        "--hk_iter", type=int, default=10000, help="HK lower bound iterations"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable using cached HK bounds"
    )
    parser.add_argument(
        "--start_tour",
        type=str,
        default=None,
        help="Path to an existing tour file to start from",
    )
    args = parser.parse_args()

    # 1. Load Data
    coords_full = load_cities(str(DATA_PATH))
    total_cities = coords_full.shape[0]

    coords_orig = coords_full[:args.n] if args.n > 0 else coords_full
    n = coords_orig.shape[0]

    # Check if we are running on full data for persistence
    # We only update best_tour.csv if we are solving the complete problem
    is_full_run = n == total_cities

    # 1.5 Hilbert Reorder for Cache Locality
    coords, orig_to_new = hilbert_reorder_cities(coords_orig)
    new_to_orig = np.empty(n, dtype=np.int32)
    new_to_orig[orig_to_new] = np.arange(n, dtype=np.int32)

    # 2. Preprocessing
    candidate_set = build_candidate_sets(coords, k=KD_TREE_QUERY_SIZE)

    # 2.5 HK Bound
    lb_val = None
    pi = None

    # Try loading cache by default
    if not args.no_cache:
        cache_subdir = CACHE_DIR / CACHE_VERSION
        hk_npy = cache_subdir / f"sample_{n}_hk.npy"
        pi_npy = cache_subdir / f"sample_{n}_pi.npy"
        if hk_npy.exists() and pi_npy.exists():
            lb_val = float(np.load(hk_npy)[0])
            pi_orig = np.load(pi_npy)
            # Map cached pi (which was in original order) to new order
            pi = pi_orig[new_to_orig]

    if lb_val is None:
        lb_val, pi = compute_hk_lower_bound(
            coords, candidate_set, max_iter=args.hk_iter
        )
        # Map pi back to original order for saving
        pi_orig = pi[orig_to_new]
        save_cached_hk_main(n, lb_val, pi_orig)

    # 2.6 Refinement
    assert pi is not None
    assert (
        K_NEIGHBORS <= KD_TREE_QUERY_SIZE
    ), "K_NEIGHBORS must be <= KD_TREE_QUERY_SIZE for proper refinement"
    candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)
    candidate_set = candidate_set[:, :K_NEIGHBORS]

    # 3. Seed Generation
    if args.start_tour:
        start_tour_indices, _ = load_tour(args.start_tour)
        # Map tour to new order
        start_tour_new = orig_to_new[start_tour_indices]

        # [HLD 3.3] 100% Exploit strategy using rotated versions of the best tour
        seeds = np.empty((args.seeds, n), dtype=np.int32)
        step = max(1, n // args.seeds)
        for i in range(args.seeds):
            start_node = int(start_tour_new[i * step % n])
            seeds[i] = rotate_tour(start_tour_new, start_node)
    else:
        # Initial seeds are all Greedy NN from different starting points
        seeds = generate_greedy_nn_seeds(coords, candidate_set, num_seeds=args.seeds)

    # 4. Optimization
    num_processes = min(mp.cpu_count(), args.seeds)

    start_opt = time.time()

    global_best_tour_new = None
    global_best_length = np.inf

    current_iter = 0
    while args.iters == 0 or current_iter < args.iters:
        current_iter += 1
        iter_start = time.time()

        results = parallel_solve(
            seeds,
            coords,
            candidate_set,
            num_processes=num_processes,
            num_kicks=args.kicks,
            max_opt=args.max_opt,
            iteration_start_time=iter_start,
            total_start_time=start_opt,
        )
        # Track iteration best
        iter_best_tour = None
        iter_best_length = np.inf
        for tour, length in results:
            if length < iter_best_length:
                iter_best_length = length
                iter_best_tour = tour.copy()

        if iter_best_length < global_best_length:
            global_best_length = iter_best_length
            global_best_tour_new = (
                iter_best_tour.copy() if iter_best_tour is not None else None
            )

            validate_result(global_best_length, lb_val)

            # Save intermediate result
            if global_best_tour_new is not None:
                temp_best_tour = new_to_orig[global_best_tour_new]
                save_solution_csv(
                    str(SOLUTIONS_PATH), temp_best_tour, global_best_length
                )
                if is_full_run:
                    update_best_tour(
                        str(BEST_TOUR_PATH),
                        temp_best_tour,
                        global_best_length,
                        is_full_run=is_full_run,
                    )
        else:
            validate_result(iter_best_length, lb_val)

        # [HLD 3.3] 100% Exploit strategy using rotated versions of the best tour
        # for ALL subsequent seeds
        if global_best_tour_new is not None:
            step = max(1, n // args.seeds)
            for i in range(args.seeds):
                start_node = int(global_best_tour_new[i * step % n])
                seeds[i] = rotate_tour(global_best_tour_new, start_node)

    if global_best_tour_new is None:
        return

    # Map best tour back to original indices
    best_tour = new_to_orig[global_best_tour_new]

    # 5. Validation
    validate_result(global_best_length, lb_val)

    # 6. Save Results
    save_solution_csv(str(SOLUTIONS_PATH), best_tour, global_best_length)
    if is_full_run:
        update_best_tour(
            str(BEST_TOUR_PATH),
            best_tour,
            global_best_length,
            is_full_run=is_full_run,
        )


if __name__ == "__main__":
    main()
