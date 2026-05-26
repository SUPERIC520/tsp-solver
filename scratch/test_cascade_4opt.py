import numpy as np
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets
from src.core.seed_generation import generate_hilbert_seeds
from src.core.kopt_engine import cascading_kopt_optimize, compute_tour_length

def main():
    n = 100
    coords = load_cities('data/cities.csv')[:n]
    cx = coords[:, 0]
    cy = coords[:, 1]
    candidate_set = build_candidate_sets(coords, k=8)
    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    
    # max_opt = 3
    np.random.seed(42)
    tour3 = generate_hilbert_seeds(coords, 1)[0]
    best_tour3, best_len3 = cascading_kopt_optimize(
        tour3, cx, cy, candidate_set, locked_edges, num_kicks=10, max_opt=3
    )
    
    # max_opt = 4
    np.random.seed(42)
    tour4 = generate_hilbert_seeds(coords, 1)[0]
    best_tour4, best_len4 = cascading_kopt_optimize(
        tour4, cx, cy, candidate_set, locked_edges, num_kicks=10, max_opt=4
    )
    
    # max_opt = 5
    np.random.seed(42)
    tour5 = generate_hilbert_seeds(coords, 1)[0]
    best_tour5, best_len5 = cascading_kopt_optimize(
        tour5, cx, cy, candidate_set, locked_edges, num_kicks=10, max_opt=5
    )
    
    print(f"Final length with max_opt=3: {best_len3}")
    print(f"Final length with max_opt=4: {best_len4}")
    print(f"Final length with max_opt=5: {best_len5}")

if __name__ == "__main__":
    main()
