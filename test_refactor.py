import numpy as np
from src.core.kopt_engine import cascading_kopt_optimize, compute_tour_length

def test_kopt():
    # Small problem: 5 cities in a square
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], dtype=np.float64)
    
    n = coords.shape[0]
    # Random initial tour
    initial_tour = np.arange(n, dtype=np.int32)
    
    # Candidate set: all other cities for each city
    candidate_set = np.full((n, n-1), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, :len(cands)] = cands
        
    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    
    print("Initial length:", compute_tour_length(initial_tour, coords[:, 0], coords[:, 1]))
    
    best_tour, best_length, kicks_done = cascading_kopt_optimize(
        initial_tour, coords[:, 0], coords[:, 1], candidate_set, num_kicks=10, max_opt=3
    )
    
    print("Optimized tour:", best_tour)
    print("Optimized length:", best_length)
    
    # Verify tour length matches
    recomputed_length = compute_tour_length(best_tour, coords[:, 0], coords[:, 1])
    print("Recomputed length:", recomputed_length)
    assert abs(best_length - recomputed_length) < 1e-9
    
    # Verify all cities are present
    assert len(np.unique(best_tour)) == n
    assert np.all(np.sort(best_tour) == np.arange(n))

if __name__ == "__main__":
    test_kopt()
    print("Test passed!")
