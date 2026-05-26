import time
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets, refine_candidate_set_with_alpha
from src.core.seed_generation import generate_hilbert_seeds
from src.core.orchestration import parallel_solve, run_iterative_optimization
from src.core.backbone import extract_consensus_edges
from src.core.validation import compute_hk_lower_bound, validate_result

def run_hard_lock(coords, candidate_set, pi, num_seeds, num_kicks, num_iterations):
    n = coords.shape[0]
    seeds = generate_hilbert_seeds(coords, num_seeds=num_seeds)
    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    best_length = np.inf
    start_time = time.time()
    
    current_seeds = seeds.copy()
    
    for iter_idx in range(num_iterations):
        print(f"\nHard Lock Iteration {iter_idx + 1}/{num_iterations}")
        results = parallel_solve(current_seeds, coords, candidate_set, locked_edges, num_kicks=num_kicks)
        iteration_tours = np.empty((len(results), n), dtype=np.int32)
        for i, (tour, length) in enumerate(results):
            iteration_tours[i] = tour
            if length < best_length:
                best_length = length
        
        if iter_idx < num_iterations - 1:
            locked_edges = extract_consensus_edges(iteration_tours, threshold=0.9)
            num_locked = np.sum(locked_edges[:, 0] != -1) // 2
            print(f"  - Locked {num_locked} edges.")
            current_seeds = iteration_tours.copy()
            
    return best_length, time.time() - start_time

def run_soft_backbone(coords, candidate_set, pi, num_seeds, num_kicks, num_iterations):
    seeds = generate_hilbert_seeds(coords, num_seeds=num_seeds)
    start_time = time.time()
    best_tour, best_length = run_iterative_optimization(
        seeds, coords, candidate_set, pi, 
        num_iterations=num_iterations, num_kicks=num_kicks
    )
    return best_length, time.time() - start_time

def main():
    n = 500
    num_seeds = 8
    num_kicks = 1000
    num_iterations = 5
    
    print(f"Loading {n} cities...")
    coords_full = load_cities("data/cities.csv")
    coords = coords_full[:n]
    
    print("Preprocessing...")
    candidate_set = build_candidate_sets(coords, k=40)
    lb_val, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=500)
    candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)
    
    print("\n=== COMPARISON: HARD LOCK vs SOFT BACKBONE ===")
    
    print("\n--- Phase 1: Hard Lock ---")
    len_hard, time_hard = run_hard_lock(coords, candidate_set, pi, num_seeds, num_kicks, num_iterations)
    
    print("\n--- Phase 2: Soft Backbone ---")
    len_soft, time_soft = run_soft_backbone(coords, candidate_set, pi, num_seeds, num_kicks, num_iterations)
    
    print(f"\nFinal Results (N={n}, Iterations={num_iterations}, Seeds={num_seeds}, Kicks={num_kicks}):")
    print(f"Hard Lock:     Length={len_hard:.2f}, Time={time_hard:.2f}s, Gap={validate_result(len_hard, lb_val):.4f}%")
    print(f"Soft Backbone: Length={len_soft:.2f}, Time={time_soft:.2f}s, Gap={validate_result(len_soft, lb_val):.4f}%")
    
    if len_soft <= len_hard:
        print("\nSUCCESS: Soft Backbone performed as well or better than Hard Lock.")
    else:
        print("\nINFO: Soft Backbone length was slightly higher, but check gap.")

if __name__ == "__main__":
    main()
