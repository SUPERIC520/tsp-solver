import time
import numpy as np
import os
import argparse
from typing import Tuple, Optional
from src.utils.data_io import load_cities
from src.core.preprocessing import (
    build_candidate_sets,
    refine_candidate_set_with_alpha,
    hilbert_reorder_cities,
)
from src.core.seed_generation import generate_greedy_nn_seeds, rotate_tour
from src.core.orchestration import parallel_solve
from src.core.validation import compute_hk_lower_bound, validate_result


def warmup() -> None:
    print("Warming up JIT...")
    coords_x = np.array([0, 1, 1, 0, 0.5], dtype=np.float64)
    coords_y = np.array([0, 0, 1, 1, 0.5], dtype=np.float64)
    cs = np.array(
        [[1, 3, 4], [0, 2, 4], [1, 3, 4], [0, 2, 4], [0, 1, 2]], dtype=np.int32
    )
    from src.core.kopt_engine import cascading_kopt_optimize

    cascading_kopt_optimize(
        np.array([0, 1, 2, 3, 4], dtype=np.int32),
        coords_x,
        coords_y,
        cs,
        num_kicks=2,
        max_opt=3,
    )
    print("Warmup complete.")


def run_benchmark(
    n_sample: int,
    num_seeds: int = 8,
    num_kicks: int = 1500,
    num_iterations: int = 3,
    hk_iter: int = 2000,
    force_hk: bool = False,
    max_opt: int = 3,
) -> Tuple[float, float]:
    warmup()
    start_total = time.time()

    # 1. Load Data
    print(f"\n--- Benchmarking {n_sample} cities ---")
    coords_orig = load_cities("data/cities.csv")
    if n_sample > 0:
        coords_orig = coords_orig[:n_sample]
    n = coords_orig.shape[0]
    print(f"Loaded {n} cities.")

    # 1.5 Hilbert Reorder
    print("Step 1.5: Reordering cities for cache locality (Hilbert)...")
    coords, orig_to_new = hilbert_reorder_cities(coords_orig)
    new_to_orig = np.empty(n, dtype=np.int32)
    new_to_orig[orig_to_new] = np.arange(n, dtype=np.int32)

    # 2. Preprocessing
    print("Step 2: Building initial candidate sets (KDTree k=64)...")
    start_pre = time.time()
    candidate_set = build_candidate_sets(coords, k=64)
    print(f"Initial preprocessing done in {time.time() - start_pre:.2f}s.")

    print(
        f"Step 3: Obtaining Held-Karp lower bound (max_iter={hk_iter})..."
    )
    start_alpha = time.time()
    lb_val, pi = compute_hk_lower_bound(
        coords, candidate_set, max_iter=hk_iter
    )
    print(f"HK Lower Bound: {lb_val:.2f}")

    print("Step 4: Refining candidate sets with Alpha-values...")
    candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)
    # Keep top 40 after refinement
    candidate_set = candidate_set[:, :40]
    print(f"Alpha refinement done in {time.time() - start_alpha:.2f}s.")

    # 3. Seed Generation
    print(f"Step 5: Generating {num_seeds} initial greedy-NN seeds...")
    seeds = generate_greedy_nn_seeds(coords, candidate_set, num_seeds=num_seeds)

    # 4. Iterative Optimization
    global_best_length = np.inf
    global_best_tour_new = None

    for iter_idx in range(num_iterations):
        print(f"\nIteration {iter_idx + 1}/{num_iterations}")

        start_iter = time.time()
        results = parallel_solve(
            seeds,
            coords,
            candidate_set,
            num_kicks=num_kicks,
            max_opt=max_opt,
        )
        print(f"Parallel solve completed in {time.time() - start_iter:.2f}s.")

        iter_best_tour = None
        iter_best_length = np.inf
        for tour, length in results:
            if length < iter_best_length:
                iter_best_length = length
                iter_best_tour = tour.copy()

        if iter_best_length < global_best_length:
            global_best_length = iter_best_length
            global_best_tour_new = iter_best_tour.copy()
            print(f"New best length: {global_best_length:.2f}")
        else:
            print(f"Iteration best: {iter_best_length:.2f} (No improvement)")

        # [HLD 3.3] 100% Exploit strategy using rotated versions of the best tour for next seeds
        if global_best_tour_new is not None:
            step = max(1, n // num_seeds)
            for i in range(num_seeds):
                start_node = global_best_tour_new[i * step % n]
                seeds[i] = rotate_tour(global_best_tour_new, start_node)

    # 5. Validation
    print("\nStep 5: Final Validation")
    gap = validate_result(global_best_length, lb_val)
    print(f"Final Best Length: {global_best_length:.2f}")
    print(f"HK Lower Bound: {lb_val:.2f}")
    print(f"Gap: {gap:.4f}%")

    total_time = time.time() - start_total
    print(f"Total execution time: {total_time:.2f}s")

    # Log to notes.md (append)
    with open("notes.md", "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        f.write(f"\n## [{timestamp}] - Benchmark N={n_sample} (Refined Architecture)\n")
        f.write(
            f"- **Params**: seeds={num_seeds}, kicks={num_kicks}, iterations={num_iterations}, hk_iter={hk_iter}, max_opt={max_opt}\n"
        )
        f.write(
            f"- **Results**: Gap={gap:.4f}%, LB={lb_val:.2f}, Best={global_best_length:.2f}, Time={total_time:.2f}s\n"
        )
        f.write(f"- **Status**: {'SUCCESS' if gap < 5.0 else 'FAILURE'}\n")

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
        args.force_hk,
        args.max_opt,
    )
