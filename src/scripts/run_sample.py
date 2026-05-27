"""Benchmark script for testing the TSP Solver on smaller subsets.

This script allows for quick experimentation and validation of the solver's
performance on subsets of the city data.
"""

import argparse
import time

import numpy as np

from src.config import BEST_TOUR_PATH, DATA_PATH, NOTES_PATH, SOLUTIONS_PATH
from src.core.kopt_engine import cascading_kopt_optimize
from src.core.orchestration import parallel_solve
from src.core.preprocessing import (
    build_candidate_sets,
    hilbert_reorder_cities,
    refine_candidate_set_with_alpha,
)
from src.core.seed_generation import generate_greedy_nn_seeds, rotate_tour
from src.core.validation import compute_hk_lower_bound, validate_result
from src.utils.data_io import load_cities, save_solution_csv
from src.utils.persistence import update_best_tour

SUCCESS_GAP_THRESHOLD = 5.0


def warmup() -> None:
    """Trigger Numba JIT compilation with a small problem."""
    coords_x = np.array([0, 1, 1, 0, 0.5], dtype=np.float64)
    coords_y = np.array([0, 0, 1, 1, 0.5], dtype=np.float64)
    cs = np.array(
        [[1, 3, 4], [0, 2, 4], [1, 3, 4], [0, 2, 4], [0, 1, 2]], dtype=np.int32
    )

    cascading_kopt_optimize(
        np.array([0, 1, 2, 3, 4], dtype=np.int32),
        coords_x,
        coords_y,
        cs,
        num_kicks=2,
        max_opt=3,
    )


def run_benchmark(
    n_sample: int,
    num_seeds: int = 8,
    num_kicks: int = 1500,
    num_iterations: int = 3,
    hk_iter: int = 2000,
    *,
    max_opt: int = 3,
) -> tuple[float, float]:
    """Run a TSP solver benchmark on a city subset.

    Args:
        n_sample: Number of cities to sample.
        num_seeds: Number of initial seeds.
        num_kicks: Number of kicks per seed.
        num_iterations: Number of optimization iterations.
        hk_iter: Held-Karp iterations.
        max_opt: Maximum K for K-Opt.

    Returns:
        A tuple (gap_percentage, total_time_seconds).
    """
    warmup()
    start_total = time.time()

    # 1. Load Data
    coords_full = load_cities(str(DATA_PATH))
    total_cities = coords_full.shape[0]

    coords_orig = coords_full[:n_sample] if n_sample > 0 else coords_full
    n = coords_orig.shape[0]

    # Check if we are running on full data for persistence
    is_full_run = n == total_cities

    # 1.5 Hilbert Reorder
    coords, orig_to_new = hilbert_reorder_cities(coords_orig)
    new_to_orig = np.empty(n, dtype=np.int32)
    new_to_orig[orig_to_new] = np.arange(n, dtype=np.int32)

    # 2. Preprocessing
    candidate_set = build_candidate_sets(coords, k=64)

    lb_val, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=hk_iter)

    candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)
    # Keep top 40 after refinement
    candidate_set = candidate_set[:, :40]

    # 3. Seed Generation
    seeds = generate_greedy_nn_seeds(coords, candidate_set, num_seeds=num_seeds)

    # 4. Iterative Optimization
    global_best_length = np.inf
    global_best_tour_new = None

    for _iter_idx in range(num_iterations):
        results = parallel_solve(
            seeds,
            coords,
            candidate_set,
            num_kicks=num_kicks,
            max_opt=max_opt,
        )

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
            pass

        # [HLD 3.3] 100% Exploit strategy using rotated versions of the best tour
        # for next seeds
        if global_best_tour_new is not None:
            step = max(1, n // num_seeds)
            for i in range(num_seeds):
                start_node = int(global_best_tour_new[i * step % n])
                seeds[i] = rotate_tour(global_best_tour_new, start_node)

    # 5. Validation
    if global_best_tour_new is None:
        return 0.0, 0.0

    # Map best tour back to original indices
    best_tour = new_to_orig[global_best_tour_new]

    # Save final results
    save_solution_csv(str(SOLUTIONS_PATH), best_tour, global_best_length)
    if is_full_run:
        update_best_tour(
            str(BEST_TOUR_PATH),
            best_tour,
            global_best_length,
            is_full_run=is_full_run,
        )

    gap = validate_result(global_best_length, lb_val)

    total_time = time.time() - start_total

    # Log to notes.md (append)
    with NOTES_PATH.open("a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        f.write(f"\n## [{timestamp}] - Benchmark N={n_sample} (Refined Architecture)\n")
        f.write(
            f"- **Params**: seeds={num_seeds}, kicks={num_kicks}, "
            f"iterations={num_iterations}, hk_iter={hk_iter}, max_opt={max_opt}\n"
        )
        f.write(
            f"- **Results**: Gap={gap:.4f}%, LB={lb_val:.2f}, "
            f"Best={global_best_length:.2f}, Time={total_time:.2f}s\n"
        )
        status = "SUCCESS" if gap < SUCCESS_GAP_THRESHOLD else "FAILURE"
        f.write(f"- **Status**: {status}\n")

    return gap, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--kicks", type=int, default=1500)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--hk_iter", type=int, default=2000)
    parser.add_argument("--force_hk", action="store_true")
    parser.add_argument("--max_opt", type=int, default=3)
    args = parser.parse_args()

    run_benchmark(
        args.n,
        args.seeds,
        args.kicks,
        args.iters,
        args.hk_iter,
        max_opt=args.max_opt,
    )
