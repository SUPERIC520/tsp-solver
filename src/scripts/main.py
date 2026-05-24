import time
import numpy as np
import os
import argparse
import multiprocessing as mp
import sys
from typing import Tuple, Optional
from src.utils.data_io import load_cities, save_solution_csv
from src.core.preprocessing import (
    build_candidate_sets,
    refine_candidate_set_with_alpha,
    hilbert_reorder_cities,
)
from src.core.seed_generation import generate_hilbert_seeds, generate_greedy_nn_seeds
from src.core.orchestration import parallel_solve
from src.core.validation import compute_hk_lower_bound, validate_result


def save_cached_hk_main(n: int, hk: float, pi: np.ndarray) -> None:
    hk_file = f"data/sample_{n}_hk.npy"
    pi_file = f"data/sample_{n}_pi.npy"
    np.save(hk_file, np.array([hk]))
    np.save(pi_file, pi)


def main() -> None:
    parser = argparse.ArgumentParser(description="TSP Solver - Final Production Run")
    parser.add_argument("--kicks", type=int, default=25000, help="Number of kicks per seed")
    parser.add_argument("--seeds", type=int, default=8, help="Total number of seeds")
    parser.add_argument("--greedy_seeds", type=int, default=4, help="Number of greedy NN seeds")
    parser.add_argument("--max_opt", type=int, default=3, help="Max k for k-opt (3 is standard successful config)")
    parser.add_argument("--n", type=int, default=0, help="Number of cities to subset (0 for all)")
    parser.add_argument("--hk_iter", type=int, default=10000, help="HK lower bound iterations")
    parser.add_argument("--no_cache", action="store_true", help="Disable using cached HK bounds")
    args = parser.parse_args()

    start_total = time.time()

    # 1. Load Data
    print("\n[Step 1] Loading city data...")
    t0 = time.time()
    coords_orig = load_cities("data/cities.csv")
    if args.n > 0:
        coords_orig = coords_orig[:args.n]
    n = coords_orig.shape[0]
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
    num_hilbert = args.seeds - args.greedy_seeds
    print(f"[Step 5] Generating {num_hilbert} Hilbert + {args.greedy_seeds} greedy-NN seeds...")
    t0 = time.time()
    seeds_list = []
    if num_hilbert > 0:
        seeds_list.append(generate_hilbert_seeds(coords, num_seeds=num_hilbert))
    if args.greedy_seeds > 0:
        seeds_list.append(generate_greedy_nn_seeds(coords, candidate_set, num_seeds=args.greedy_seeds))
    seeds = np.vstack(seeds_list) if len(seeds_list) > 1 else seeds_list[0]
    dt = time.time() - t0
    print(f"  - Seeds generated in {dt:.2f}s.")

    # 4. Optimization
    print(f"[Step 6] Parallel optimization (kicks={args.kicks}, max_opt={args.max_opt})...")
    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    num_processes = min(mp.cpu_count(), args.seeds)
    print(f"  - Running {args.seeds} parallel solvers on {num_processes} processes...")

    start_opt = time.time()
    
    results = parallel_solve(
        seeds, 
        coords, 
        candidate_set, 
        locked_edges, 
        num_processes=num_processes,
        num_kicks=args.kicks,
        max_opt=args.max_opt
    )
    
    dt_opt = time.time() - start_opt
    print(f"\n  - Optimization completed in {dt_opt:.2f}s.")

    # Find best tour
    best_tour_new = None
    best_length = np.inf
    for tour, length in results:
        if length < best_length:
            best_length = length
            best_tour_new = tour.copy()

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
    print("  - Saved to data/solutions.csv")

    total_time = time.time() - start_total
    print(f"\n[Done] Total Wall-Clock Runtime: {total_time:.2f}s ({total_time/60:.2f}m)")


if __name__ == "__main__":
    main()
