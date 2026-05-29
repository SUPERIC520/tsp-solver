"""Main entry point for the TSP Solver production run.

This script orchestrates the full TSP solving pipeline, including data loading,
Hilbert reordering, Held-Karp lower bound computation, candidate set refinement,
parallel K-Opt optimization, and results persistence.
"""

import argparse
import multiprocessing as mp
import time

import numpy as np

import src.core.orchestration as orch
from src.config import (
    BEST_TOUR_PATH,
    CACHE_DIR,
    CACHE_VERSION,
    DATA_PATH,
    K_NEIGHBORS,
    KD_TREE_QUERY_SIZE,
    SOLUTIONS_PATH,
)
from src.core.backbone import compute_edge_frequencies
from src.core.orchestration import parallel_solve
from src.core.preprocessing import (
    apply_backbone_bias,
    build_candidate_sets,
    hilbert_reorder_cities,
    refine_candidate_set_with_alpha,
)
from src.core.seed_generation import (
    generate_greedy_nn_seeds,
    generate_random_seeds,
    rotate_tour,
)
from src.core.validation import (
    compute_alpha_values,
    compute_hk_lower_bound,
    validate_result,
)
from src.utils.data_io import load_cities, load_tour, save_hk_cache, save_solution_csv
from src.utils.persistence import update_best_tour


def str2bool(v: str) -> bool:
    """Parse string to boolean."""
    val = v.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    if val in ("no", "false", "f", "n", "0"):
        return False
    msg = "Boolean value expected."
    raise argparse.ArgumentTypeError(msg)


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
        default=5,
        help="Max k for k-opt (5 is recommended for quality)",
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
        "--no_alpha",
        action="store_true",
        help="Skip Alpha-value refinement and use raw KD-tree",
    )
    parser.add_argument(
        "--start_tour",
        type=str,
        default=None,
        help="Path to an existing tour file to start from",
    )
    parser.add_argument(
        "--seed_strategy",
        type=str,
        default="greedy",
        choices=["greedy", "random"],
        help="Initial seeding strategy: greedy (default), random",
    )
    parser.add_argument(
        "--backbone_weight",
        type=float,
        default=0.0,
        help="Soft backbone strength (0.0 to 1.0). "
        "Biases candidate sorting by edge frequency in population.",
    )
    args = parser.parse_args()

    start_total = time.time()

    # 1. Load Data
    print("\n[Step 1] Loading city data...")
    t0 = time.time()
    coords_full = load_cities(str(DATA_PATH))
    total_cities = coords_full.shape[0]

    coords_orig = coords_full[: args.n] if args.n > 0 else coords_full
    n = coords_orig.shape[0]

    is_full_run = n == total_cities
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
    print(
        f"[Step 2] Building initial candidate sets (k={KD_TREE_QUERY_SIZE})..."
    )
    t0 = time.time()
    candidate_set = build_candidate_sets(coords, k=KD_TREE_QUERY_SIZE)
    dt = time.time() - t0
    print(f"  - Initial preprocessing done in {dt:.2f}s.")

    # 2.5 HK Bound
    print("[Step 2.5] Obtaining Held-Karp lower bound...")
    t0 = time.time()
    lb_val = None
    pi = None

    if not args.no_cache:
        cache_subdir = CACHE_DIR / CACHE_VERSION
        hk_npy = cache_subdir / f"sample_{n}_hk.npy"
        pi_npy = cache_subdir / f"sample_{n}_pi.npy"
        if hk_npy.exists() and pi_npy.exists():
            lb_val = float(np.load(hk_npy)[0])
            pi_orig = np.load(pi_npy)
            pi = pi_orig[new_to_orig]
            print(f"  - Loaded cached LB: {lb_val:.2f}")

    if lb_val is None:
        print(
            "  - Cache not found or disabled. "
            f"Computing HK LB ({args.hk_iter} iterations)..."
        )
        lb_val, pi = compute_hk_lower_bound(
            coords, candidate_set, max_iter=args.hk_iter, use_cache=not args.no_cache
        )
        pi_orig = pi[orig_to_new]
        save_hk_cache(str(n), lb_val, pi_orig)
        dt = time.time() - t0
        print(f"  - HK LB computed: {lb_val:.2f} (Time: {dt:.2f}s)")
    else:
        dt = time.time() - t0
        print(f"  - Used cached HK (Load time: {dt:.2f}s)")

    # 2.6 Refinement
    alpha_values = None
    if args.no_alpha:
        print("  - Skipped Alpha-refinement per --no_alpha flag.")
        candidate_set = candidate_set[:, :K_NEIGHBORS]
    else:
        print("[Step 2.6] Refining candidate sets with Alpha-values...")
        t0 = time.time()
        assert pi is not None
        # Cache alpha values for backbone bias reuse
        alpha_values = compute_alpha_values(n, coords, candidate_set, pi)
        candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)
        candidate_set = candidate_set[:, :K_NEIGHBORS]
        dt = time.time() - t0
        print(f"  - Refinement to top {K_NEIGHBORS} done in {dt:.2f}s.")

    # 3. Seed Generation
    print(f"[Step 3] Generating {args.seeds} seeds...")
    t0 = time.time()

    num_processes = min(mp.cpu_count(), args.seeds)
    print(
        f"  - Initializing persistent multiprocessing pool with {num_processes} workers..."
    )
    shared_progress_array = mp.Array("i", args.seeds)
    solver_pool = mp.Pool(
        processes=num_processes,
        initializer=orch._init_worker,
        initargs=(shared_progress_array,)
    )

    if args.start_tour:
        start_tour_indices, _ = load_tour(args.start_tour)
        start_tour_new = orig_to_new[start_tour_indices]
        seeds = np.empty((args.seeds, n), dtype=np.int32)
        step = max(1, n // args.seeds)
        for i in range(args.seeds):
            start_node = int(start_tour_new[i * step % n])
            rotate_tour(start_tour_new, start_node, out=seeds[i])
    elif args.seed_strategy == "random":
        seeds = generate_random_seeds(n, args.seeds)
    else:
        seeds = generate_greedy_nn_seeds(coords, candidate_set, num_seeds=args.seeds, pool=solver_pool)
    dt = time.time() - t0
    print(f"  - Seeds generated in {dt:.2f}s.")

    # 4. Optimization
    print(
        f"[Step 4] Optimization (iters={args.iters}, "
        f"max_opt={args.max_opt}, kicks={args.kicks})..."
    )

    start_opt = time.time()
    global_best_tour_new = None
    global_best_length = np.inf

    current_iter = 0
    try:
        while args.iters == 0 or current_iter < args.iters:
            current_iter += 1
            if args.iters != 1:
                print(f"\n--- Iteration {current_iter} ---")
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
                pool=solver_pool,
                progress_array=shared_progress_array,
            )

            this_iter_best_length = min(length for _, length in results)
            global_best_so_far = min(global_best_length, this_iter_best_length)
            print(
                f"  - Iteration best: {this_iter_best_length:.2f}  "
                f"(global best: {global_best_so_far:.2f})"
            )

            if this_iter_best_length < global_best_length:
                global_best_length = this_iter_best_length
                for tour, length in results:
                    if length == this_iter_best_length:
                        global_best_tour_new = tour.copy()
                        break

                validate_result(global_best_length, lb_val)

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
                validate_result(this_iter_best_length, lb_val)

            # [HLD 3.3] Standard Uniform Rotation strategy
            assert global_best_tour_new is not None

            # --- Soft Backbone: Bias candidate ordering for next iteration ---
            if args.iters != 1 and args.backbone_weight > 0.0 and alpha_values is not None:
                tours_pop = np.array([tour for tour, _ in results], dtype=np.int32)
                freq_matrix = compute_edge_frequencies(tours_pop, candidate_set)
                candidate_set = apply_backbone_bias(
                    candidate_set, alpha_values, freq_matrix, args.backbone_weight
                )
                print(f"  - Applied soft-backbone bias (weight={args.backbone_weight}).")

            step = max(1, n // args.seeds)
            for i in range(args.seeds):
                start_node = int(global_best_tour_new[i * step % n])
                rotate_tour(global_best_tour_new, start_node, out=seeds[i])
            if args.iters != 1:
                print("  - Reseeded seeds from best tour (uniform rotation).")
    finally:
        solver_pool.close()
        solver_pool.join()

    dt_opt = time.time() - start_opt
    print(f"\n  - Optimization completed in {dt_opt:.2f}s.")

    if global_best_tour_new is None:
        return

    best_tour = new_to_orig[global_best_tour_new]

    # 5. Validation
    print("\n[Step 5] Final Validation")
    gap = validate_result(global_best_length, lb_val)
    print(f"  - Best Length:     {global_best_length:.2f}")
    print(f"  - HK Lower Bound:  {lb_val:.2f}")
    print(f"  - Solution Gap:    {gap:.4f}%")

    # 6. Save Results
    print("\n[Step 6] Saving results...")
    save_solution_csv(str(SOLUTIONS_PATH), best_tour, global_best_length)
    if is_full_run:
        update_best_tour(
            str(BEST_TOUR_PATH), best_tour, global_best_length, is_full_run=is_full_run
        )

    total_time = time.time() - start_total
    print(f"\n[Done] Total Wall-Clock Runtime: {total_time:.2f}s ({total_time / 60:.2f}m)")


if __name__ == "__main__":
    main()
