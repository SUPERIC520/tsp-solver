import time
import numpy as np
import os
from typing import Tuple, Optional
from src.utils.data_io import load_cities, save_tour
from src.core.preprocessing import build_candidate_sets, refine_candidate_set_with_alpha
from src.core.seed_generation import generate_hilbert_seeds
from src.core.orchestration import parallel_solve
from src.core.backbone import extract_consensus_edges
from src.core.validation import compute_hk_lower_bound, validate_result


def load_cached_hk_main(n: int) -> Tuple[Optional[float], Optional[np.ndarray]]:
    # Try sample files first as they are specifically for the 115475 dataset
    hk_npy = "data/sample_115475_hk.npy"
    pi_npy = "data/sample_115475_pi.npy"
    if os.path.exists(hk_npy) and os.path.exists(pi_npy):
        hk = float(np.load(hk_npy)[0])
        pi = np.load(pi_npy)
        return hk, pi

    hk_file = "data/lower_bound.txt"
    pi_file = "data/main_pi.npy"
    if os.path.exists(hk_file) and os.path.exists(pi_file):
        with open(hk_file, "r") as f:
            hk = float(f.read().strip())
        pi = np.load(pi_file)
        return hk, pi
    return None, None


def save_cached_hk_main(hk: float, pi: np.ndarray) -> None:
    hk_file = "data/lower_bound.txt"
    pi_file = "data/main_pi.npy"
    with open(hk_file, "w") as f:
        f.write(str(hk))
    np.save(pi_file, pi)


def main() -> None:
    start_total = time.time()

    # 1. Load Data
    print("Step 1: Loading cities...")
    coords = load_cities("data/cities.csv")
    n = coords.shape[0]
    print(f"Loaded {n} cities.")

    # 2. Preprocessing
    print("Step 2: Building candidate sets (Delaunay+KDTree)...")
    start_pre = time.time()
    candidate_set = build_candidate_sets(coords, k=32)
    print(f"Initial preprocessing done in {time.time() - start_pre:.2f}s.")

    print("Step 2.5: Refining candidate sets with Alpha-values...")
    start_alpha = time.time()
    lb_val, pi = load_cached_hk_main(n)
    if lb_val is None or pi is None:
        print("Computing HK lower bound (not cached)...")
        lb_val, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=100)
        save_cached_hk_main(lb_val, pi)
    else:
        print("Loaded cached HK lower bound and pi vector.")

    candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)
    # Keep top 20 after refinement for speed
    candidate_set = candidate_set[:, :20]
    print(
        f"Alpha refinement done in {time.time() - start_alpha:.2f}s. HK Lower Bound: {lb_val:.2f}"
    )

    # 3. Seed Generation
    print("Step 3: Generating 8 Hilbert seeds...")
    seeds = generate_hilbert_seeds(coords, num_seeds=8)

    # 4. Iterative Optimization
    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    best_tour = None
    best_length = np.inf

    num_iterations = 3
    kick_schedule = [800, 1500, 2500]

    for iter_idx in range(num_iterations):
        num_kicks = kick_schedule[iter_idx]
        print(f"\nIteration {iter_idx + 1}/{num_iterations}")
        print(f"Locked edges: {np.sum(locked_edges != -1) // 2}")
        print(f"Num kicks: {num_kicks}")

        start_iter = time.time()
        results = parallel_solve(
            seeds, coords, candidate_set, locked_edges, num_kicks=num_kicks
        )
        print(f"Parallel solve completed in {time.time() - start_iter:.2f}s.")

        # Aggregate results
        iteration_tours = np.empty((len(results), n), dtype=np.int32)
        for i, (tour, length) in enumerate(results):
            iteration_tours[i] = tour
            if length < best_length:
                best_length = length
                best_tour = tour.copy()

        print(f"Best length this iteration: {best_length:.2f}")

        # Extract consensus for next iteration
        if iter_idx < num_iterations - 1:
            print("Extracting consensus edges...")
            locked_edges = extract_consensus_edges(iteration_tours, threshold=0.85)
            # Update seeds for the next iteration to start from the currently optimized tours
            seeds = iteration_tours.copy()

    # 5. Validation
    print("\nStep 5: Validation")
    lower_bound = lb_val
    gap = validate_result(best_length, lower_bound)
    print(f"Final Best Length: {best_length:.2f}")
    print(f"Estimated Lower Bound: {lower_bound:.2f}")
    print(f"Gap: {gap:.4f}%")

    # 6. Save Results
    print("\nStep 6: Saving results...")
    assert best_tour is not None
    save_tour("data/solution.txt", best_tour, best_length)

    print(f"\nTotal execution time: {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    main()
