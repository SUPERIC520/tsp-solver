import numpy as np
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets
from src.core.seed_generation import generate_hilbert_seeds
from src.core.kopt_engine import (
    _optimize_3opt_sequential,
    _optimize_4opt_sequential,
    _optimize_5opt_sequential,
    compute_tour_length,
    _update_pos
)

def _compute_candidate_dists(coords_x, coords_y, candidate_set):
    n, num_cand = candidate_set.shape
    candidate_dists = np.zeros(candidate_set.shape, dtype=np.float64)
    for i in range(n):
        for k in range(num_cand):
            w = candidate_set[i, k]
            if w != -1:
                dx = coords_x[i] - coords_x[w]
                dy = coords_y[i] - coords_y[w]
                candidate_dists[i, k] = np.sqrt(dx * dx + dy * dy)
    return candidate_dists

def main():
    n = 100
    coords = load_cities('data/cities.csv')[:n]
    cx = coords[:, 0]
    cy = coords[:, 1]
    candidate_set = build_candidate_sets(coords, k=8)
    candidate_dists = _compute_candidate_dists(cx, cy, candidate_set)
    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    
    # 3-opt verification
    print("=== Testing 3-opt sequential ===")
    np.random.seed(42)
    tour = generate_hilbert_seeds(coords, 1)[0]
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)
    
    len_before = compute_tour_length(tour, cx, cy)
    improved_3 = _optimize_3opt_sequential(tour, cx, cy, candidate_set, candidate_dists, locked_edges, pos, dlb)
    len_after = compute_tour_length(tour, cx, cy)
    print(f"3-opt improved: {improved_3}")
    print(f"Length before: {len_before}, after: {len_after}, diff: {len_after - len_before}")
    
    # 4-opt verification
    print("\n=== Testing 4-opt sequential ===")
    np.random.seed(42)
    tour = generate_hilbert_seeds(coords, 1)[0]
    _update_pos(tour, pos)
    dlb.fill(False)
    
    len_before = compute_tour_length(tour, cx, cy)
    improved_4 = _optimize_4opt_sequential(tour, cx, cy, candidate_set, candidate_dists, locked_edges, pos, dlb)
    len_after = compute_tour_length(tour, cx, cy)
    print(f"4-opt improved: {improved_4}")
    print(f"Length before: {len_before}, after: {len_after}, diff: {len_after - len_before}")

    # 5-opt verification
    print("\n=== Testing 5-opt sequential ===")
    np.random.seed(42)
    tour = generate_hilbert_seeds(coords, 1)[0]
    _update_pos(tour, pos)
    dlb.fill(False)
    
    len_before = compute_tour_length(tour, cx, cy)
    improved_5 = _optimize_5opt_sequential(tour, cx, cy, candidate_set, candidate_dists, locked_edges, pos, dlb)
    len_after = compute_tour_length(tour, cx, cy)
    print(f"5-opt improved: {improved_5}")
    print(f"Length before: {len_before}, after: {len_after}, diff: {len_after - len_before}")

if __name__ == "__main__":
    main()
