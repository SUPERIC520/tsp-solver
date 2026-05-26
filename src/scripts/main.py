import time
import numpy as np
import os
import argparse
import multiprocessing as mp
import sys
from typing import Tuple, Optional
from src.utils.data_io import load_cities, save_solution_csv, load_tour
from src.utils.persistence import update_best_tour
from src.utils.time_utils import format_duration
from src.core.preprocessing import (
    build_candidate_sets,
    refine_candidate_set_with_alpha,
    hilbert_reorder_cities,
)
from src.core.seed_generation import generate_greedy_nn_seeds, rotate_tour
from src.core.orchestration import parallel_solve
from src.core.validation import compute_hk_lower_bound, validate_result


def save_cached_hk_main(n: int, hk: float, pi: np.ndarray) -> None:
    hk_file = f"data/sample_{n}_hk.npy"
    pi_file = f"data/sample_{n}_pi.npy"
    np.save(hk_file, np.array([hk]))
    np.save(pi_file, pi)


def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="TSP Solver - Final Production Run")
    parser.add_argument("--kicks", type=int, default=25000, help="Number of kicks per seed")
    parser.add_argument("--iters", type=int, default=1, help="Number of full iterations (0 for infinite)")
    parser.add_argument("--seeds", type=int, default=8, help="Total number of seeds")
    parser.add_argument("--max_opt", type=int, default=3, help="Max k for k-opt (3 is standard successful config)")
    parser.add_argument("--n", type=int, default=0, help="Number of cities to subset (0 for all)")
    parser.add_argument("--hk_iter", type=int, default=10000, help="HK lower bound iterations")
    parser.add_argument("--no_cache", action="store_true", help="Disable using cached HK bounds")
    parser.add_argument("--start_tour", type=str, default=None, help="Path to an existing tour file to start from")
    args = parser.parse_args()

    start_total = time.time()

    # 1. Load Data
    print("\n[Step 1] Loading city data...")
    t0 = time.time()
    coords_orig = load_cities("data/cities.csv")
    if args.n > 0:
        coords_orig = coords_orig[:args.n]
    n = coords_orig.shape[0]
    
    # Check if we are running on full data for persistence
    is_full_run = (args.n == 0 or args.n == coords_orig.shape[0])
    
    dt = time.time() - t0
    print(f"  - Loaded {n} cities in {dt:.2f}s.")

    # 1.5 Hilbert Reorder for Cache Locality
    print("[Step 1.5] Reordering cities for cache locality (Hilbert)...")
    t0 = time.time()
    coords, orig_to_new = hilbert_reorder_cities(coords_orig)
    new_to_orig = np.empty(n, dtype=np.int32)
    new_to_orig[orig_to_new] = np.arange(n, dtype=np.int32)
    dt = time.time() - t0
    print(f"  - Reordered in {dt:.2f}s.")

    # 2. Preprocessing
    print("[Step 2] Building initial candidate sets (k=64)...")
    t0 = time.time()
    candidate_set = build_candidate_sets(coords, k=64)
    dt = time.time() - t0
    print(f"  - Initial preprocessing done in {dt:.2f}s.")

    # 2.5 HK Bound
    print(f"[Step 3] Obtaining Held-Karp lower bound...")
    t0 = time.time()
    lb_val = None
    pi = None
    
    # Try loading cache by default
    if not args.no_cache:
        hk_npy = f"data/sample_{n}_hk.npy"
        pi_npy = f"data/sample_{n}_pi.npy"
        if os.path.exists(hk_npy) and os.path.exists(pi_npy):
            lb_val = float(np.load(hk_npy)[0])
            pi_orig = np.load(pi_npy)
            # Map cached pi (which was in original order) to new order
            pi = pi_orig[new_to_orig]
            print(f"  - Loaded cached LB: {lb_val:.2f}")

    if lb_val is None:
        print(f"  - Cache not found or disabled. Computing HK LB ({args.hk_iter} iterations)...")
        lb_val, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=args.hk_iter)
        # Map pi back to original order for saving
        pi_orig = pi[orig_to_new]
        save_cached_hk_main(n, lb_val, pi_orig)
        dt = time.time() - t0
        print(f"  - HK LB computed: {lb_val:.2f} (Time: {dt:.2f}s)")
    else:
        dt = time.time() - t0
        print(f"  - Used cached HK (Load time: {dt:.2f}s)")

    # 2.6 Refinement
    print("[Step 4] Refining candidate sets with Alpha-values...")
    t0 = time.time()
    candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)
    candidate_set = candidate_set[:, :40]
    dt = time.time() - t0
    print(f"  - Refinement to top 40 done in {dt:.2f}s.")

    # 3. Seed Generation
    print(f"[Step 5] Generating {args.seeds} seeds...")
    t0 = time.time()
    
    if args.start_tour:
        print(f"  - Loading starting tour from {args.start_tour}...")
        start_tour_indices, _ = load_tour(args.start_tour)
        # Map tour to new order
        start_tour_new = orig_to_new[start_tour_indices]
        
        # [HLD 3.3] 100% Exploit strategy using rotated versions of the best tour
        seeds = np.empty((args.seeds, n), dtype=np.int32)
        step = max(1, n // args.seeds)
        for i in range(args.seeds):
            start_node = start_tour_new[i * step % n]
            seeds[i] = rotate_tour(start_tour_new, start_node)
    else:
        # Initial seeds are all Greedy NN from different starting points
        seeds = generate_greedy_nn_seeds(coords, candidate_set, num_seeds=args.seeds)
    
    dt = time.time() - t0
    print(f"  - Seeds generated in {dt:.2f}s.")

    # 4. Optimization
    print(f"[Step 6] Parallel optimization (kicks={args.kicks}, max_opt={args.max_opt}, iters={args.iters})...")
    num_processes = min(mp.cpu_count(), args.seeds)
    print(f"  - Running {args.seeds} parallel solvers on {num_processes} processes...")

    start_opt = time.time()

    global_best_tour_new = None
    global_best_length = np.inf

    current_iter = 0
    while args.iters == 0 or current_iter < args.iters:
        current_iter += 1
        iter_start = time.time()
        print(f"\n  [Iteration {current_iter}/{args.iters if args.iters > 0 else '∞'}]")

        results = parallel_solve(
            seeds,
            coords,
            candidate_set,
            num_processes=num_processes,
            num_kicks=args.kicks,
            max_opt=args.max_opt,
            iteration_start_time=iter_start,
            total_start_time=start_opt
        )        
        # Track iteration best
        iter_best_tour = None
        iter_best_length = np.inf
        for tour, length in results:
            if length < iter_best_length:
                iter_best_length = length
                iter_best_tour = tour.copy()
        
        iter_duration = time.time() - iter_start
        total_elapsed = time.time() - start_opt
        print(f"  - Iteration duration: {format_duration(iter_duration)}")
        print(f"  - Total elapsed: {format_duration(total_elapsed)}")
        
        if iter_best_length < global_best_length:
            global_best_length = iter_best_length
            global_best_tour_new = iter_best_tour.copy()
            print(f"  - Found new best length: {global_best_length:.2f}")
            
            # Save intermediate result
            temp_best_tour = new_to_orig[global_best_tour_new]
            save_solution_csv("data/solutions.csv", temp_best_tour, global_best_length)
            if is_full_run:
                update_best_tour("data/best_tour.csv", temp_best_tour, global_best_length)
        else:
            print(f"  - Iteration best: {iter_best_length:.2f} (No improvement)")
            
        # [HLD 3.3] 100% Exploit strategy using rotated versions of the best tour for ALL subsequent seeds
        if global_best_tour_new is not None:
            step = max(1, n // args.seeds)
            for i in range(args.seeds):
                start_node = global_best_tour_new[i * step % n]
                seeds[i] = rotate_tour(global_best_tour_new, start_node)

    dt_opt = time.time() - start_opt
    print(f"\n  - Optimization completed in {dt_opt:.2f}s.")
    
    best_tour_new = global_best_tour_new
    best_length = global_best_length

    # Map best tour back to original indices
    assert best_tour_new is not None
    best_tour = new_to_orig[best_tour_new]

    # 5. Validation
    print("\n[Step 7] Final Validation")
    gap = validate_result(best_length, lb_val)
    print(f"  - Best Length:     {best_length:.2f}")
    print(f"  - HK Lower Bound:  {lb_val:.2f}")
    print(f"  - Solution Gap:    {gap:.4f}%")

    # 6. Save Results
    print("\n[Step 8] Saving results...")
    save_solution_csv("data/solutions.csv", best_tour, best_length)
    if is_full_run:
        update_best_tour("data/best_tour.csv", best_tour, best_length)
        print("  - Saved to data/solutions.csv and data/best_tour.csv")
    else:
        print("  - Saved to data/solutions.csv (Persistence skipped: not a full run)")

    total_time = time.time() - start_total
    print(f"\n[Done] Total Wall-Clock Runtime: {total_time:.2f}s ({total_time/60:.2f}m)")


if __name__ == "__main__":
    main()
