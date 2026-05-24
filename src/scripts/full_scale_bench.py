"""
full_scale_bench.py
-------------------
Full-scale benchmark script optimized for large N (default: 115,475 cities).

Key differences from run_sample.py:
- No backbone consensus (threshold=0.99 minimises locked edges at full scale)
- num_processes = min(cpu_count, num_seeds) for efficient parallelism
- Logs results with timestamp to notes.md
- SUCCESS threshold: gap < 5.0%
"""

import time
import numpy as np
import os
import argparse
import multiprocessing as mp
from typing import Tuple, Optional
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets, refine_candidate_set_with_alpha
from src.core.seed_generation import generate_hilbert_seeds
from src.core.orchestration import parallel_solve
from src.core.backbone import extract_consensus_edges
from src.core.validation import compute_hk_lower_bound, validate_result


def load_cached_hk(n_sample: int) -> Tuple[Optional[float], Optional[np.ndarray]]:
    hk_file = f"data/sample_{n_sample}_hk.npy"
    pi_file = f"data/sample_{n_sample}_pi.npy"
    if os.path.exists(hk_file) and os.path.exists(pi_file):
        hk_arr = np.load(hk_file)
        hk = float(hk_arr.item()) if hk_arr.size == 1 else float(hk_arr[0])
        pi = np.load(pi_file)
        return hk, pi
    return None, None


def save_cached_hk(n_sample: int, hk: float, pi: np.ndarray) -> None:
    hk_file = f"data/sample_{n_sample}_hk.npy"
    pi_file = f"data/sample_{n_sample}_pi.npy"
    np.save(hk_file, np.array([hk]))
    np.save(pi_file, pi)


def warmup() -> None:
    print("Warming up JIT...")
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=np.float64)
    cs = np.array(
        [[1, 3, 4], [0, 2, 4], [1, 3, 4], [0, 2, 4], [0, 1, 2]], dtype=np.int32
    )
    from src.core.kopt_engine import cascading_kopt_optimize

    locked = np.full((5, 2), -1, dtype=np.int32)
    cascading_kopt_optimize(
        np.array([0, 1, 2, 3, 4], dtype=np.int32),
        coords,
        cs,
        locked,
        num_kicks=2,
        max_opt=3,
    )
    print("Warmup complete.")


def run_full_scale_bench(
    n: int = 115475,
    num_seeds: int = 8,
    num_kicks: int = 3000,
    num_iterations: int = 1,
    hk_iter: int = 2000,
    max_opt: int = 3,
    force_hk: bool = False,
    no_backbone: bool = False,
    backbone_threshold: float = 0.99,
) -> Tuple[float, float]:
    warmup()
    start_total = time.time()

    print(f"\n--- Full Scale Benchmark: {n} cities ---")
    print(
        f"Params: seeds={num_seeds}, kicks={num_kicks}, iters={num_iterations}, "
        f"hk_iter={hk_iter}, max_opt={max_opt}, backbone_threshold={backbone_threshold}, "
        f"no_backbone={no_backbone}"
    )

    # 1. Load Data
    coords_full = load_cities("data/cities.csv")
    coords = coords_full[:n]
    actual_n = coords.shape[0]
    print(f"Loaded {actual_n} cities.")

    # 2. Preprocessing
    print("Step 2: Building initial candidate sets (Delaunay+KDTree)...")
    start_pre = time.time()
    candidate_set = build_candidate_sets(coords, k=32)
    print(f"Initial preprocessing done in {time.time() - start_pre:.2f}s.")

    print(
        f"Step 2.5: Refining candidate sets with Alpha-values (HK max_iter={hk_iter})..."
    )
    start_alpha = time.time()
    lb_val, pi = load_cached_hk(n)

    if lb_val is None or pi is None or force_hk:
        print(f"Computing HK lower bound (force_hk={force_hk})...")
        initial_pi = pi if pi is not None else None
        lb_val, pi = compute_hk_lower_bound(
            coords, candidate_set, max_iter=hk_iter, initial_pi=initial_pi
        )
        save_cached_hk(n, lb_val, pi)
    else:
        print(f"Loaded cached HK lower bound: {lb_val:.2f}")

    candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)
    # Keep top 40 after refinement for better quality vs speed trade-off
    candidate_set = candidate_set[:, :40]
    print(
        f"Alpha refinement done in {time.time() - start_alpha:.2f}s. HK Lower Bound: {lb_val:.2f}"
    )

    # 3. Seed Generation
    print(f"Step 3: Generating {num_seeds} Hilbert seeds...")
    seeds = generate_hilbert_seeds(coords, num_seeds=num_seeds)

    # 4. Iterative Optimization
    # Backbone disabled: threshold=0.99 means virtually no edges are locked at full scale
    # until the approach is proven to be beneficial.
    locked_edges = np.full((actual_n, 2), -1, dtype=np.int32)
    best_length = np.inf

    # Set num_processes to min(cpu_count, num_seeds) for efficient resource use
    num_processes = min(mp.cpu_count(), num_seeds)
    print(f"Using {num_processes} parallel processes for {num_seeds} seeds.")
    print(f"Estimated time: ~{num_kicks * 0.3:.0f}s per seed (heuristic)")

    for iter_idx in range(num_iterations):
        print(f"\nIteration {iter_idx + 1}/{num_iterations}")

        start_iter = time.time()
        results = parallel_solve(
            seeds,
            coords,
            candidate_set,
            locked_edges,
            num_processes=num_processes,
            num_kicks=num_kicks,
            max_opt=max_opt,
        )
        print(f"Parallel solve completed in {time.time() - start_iter:.2f}s.")

        iteration_tours = np.empty((len(results), actual_n), dtype=np.int32)
        for i, (tour, length) in enumerate(results):
            iteration_tours[i] = tour
            if length < best_length:
                best_length = length

        print(f"Best length so far: {best_length:.2f}")

        if iter_idx < num_iterations - 1:
            if no_backbone:
                # Skip backbone consensus entirely when --no_backbone is set
                num_locked = 0
                print("Backbone skipped (--no_backbone). No edges locked.")
            else:
                locked_edges = extract_consensus_edges(
                    iteration_tours, threshold=backbone_threshold
                )
                num_locked = np.sum(locked_edges[:, 0] != -1) // 2
                print(
                    f"Locked {num_locked} consensus edges for next iteration (threshold={backbone_threshold})."
                )
            # Use current best tours as seeds for next iteration
            seeds = iteration_tours.copy()

    # 5. Validation
    print("\nStep 5: Validation")
    gap = validate_result(best_length, lb_val)
    print(f"Final Best Length: {best_length:.2f}")
    print(f"HK Lower Bound:    {lb_val:.2f}")
    print(f"Gap:               {gap:.4f}%")

    total_time = time.time() - start_total
    print(f"Total execution time: {total_time:.2f}s")

    status = "SUCCESS" if gap < 5.0 else "FAILURE"
    print(f"Status: {status}")

    # Log to notes.md
    with open("notes.md", "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        f.write(f"\n## [{timestamp}] - Full Scale Bench N={n}\n")
        f.write(
            f"- **Params**: seeds={num_seeds}, kicks={num_kicks}, iterations={num_iterations}, "
            f"hk_iter={hk_iter}, max_opt={max_opt}, processes={num_processes}, "
            f"backbone_threshold={backbone_threshold}, no_backbone={no_backbone}\n"
        )
        f.write(
            f"- **Results**: Gap={gap:.4f}%, LB={lb_val:.2f}, Best={best_length:.2f}, Time={total_time:.2f}s\n"
        )
        f.write(f"- **Status**: {status}\n")

    return gap, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-scale TSP benchmark")
    parser.add_argument("--n", type=int, default=115475, help="Number of cities")
    parser.add_argument("--seeds", type=int, default=8, help="Number of Hilbert seeds")
    parser.add_argument(
        "--kicks", type=int, default=3000, help="Double-bridge kicks per seed"
    )
    parser.add_argument(
        "--iters", type=int, default=1, help="Number of outer iterations"
    )
    parser.add_argument(
        "--hk_iter", type=int, default=2000, help="HK lower bound max iterations"
    )
    parser.add_argument(
        "--max_opt", type=int, default=3, help="Max k for cascading k-opt (2..max_opt)"
    )
    parser.add_argument(
        "--force_hk", action="store_true", help="Force recompute HK lower bound"
    )
    parser.add_argument(
        "--no_backbone",
        action="store_true",
        help="Skip backbone consensus entirely (no edges locked between iterations)",
    )
    parser.add_argument(
        "--backbone_threshold",
        type=float,
        default=0.99,
        help="Fraction of tours that must share an edge to lock it (default: 0.99)",
    )
    args = parser.parse_args()

    run_full_scale_bench(
        n=args.n,
        num_seeds=args.seeds,
        num_kicks=args.kicks,
        num_iterations=args.iters,
        hk_iter=args.hk_iter,
        max_opt=args.max_opt,
        force_hk=args.force_hk,
        no_backbone=args.no_backbone,
        backbone_threshold=args.backbone_threshold,
    )
