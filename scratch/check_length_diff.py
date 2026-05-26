import numpy as np
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets
from src.core.seed_generation import generate_greedy_nn_seeds

def _dist(c1, c2, coords_x, coords_y):
    dx = coords_x[c1] - coords_x[c2]
    dy = coords_y[c1] - coords_y[c2]
    return float(np.sqrt(dx * dx + dy * dy))

def _apply_2opt(tour, i, j):
    n = tour.shape[0]
    if i == j: return
    size = (j - i + n) % n + 1
    count = size // 2
    for k in range(count):
        idx1 = (i + k) % n
        idx2 = (j - k + n) % n
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

def _apply_2opt_indices(tour, t1_idx, t2_idx, t3_idx, t4_idx):
    n = tour.shape[0]
    if (t1_idx + 1) % n == t2_idx:
        _apply_2opt(tour, t2_idx, t3_idx)
    else:
        _apply_2opt(tour, t1_idx, t4_idx)

def _update_pos(tour, pos):
    for i in range(tour.shape[0]):
        pos[tour[i]] = i

def compute_tour_length(tour, coords_x, coords_y):
    length = 0.0
    n = tour.shape[0]
    for i in range(n):
        length += _dist(tour[i], tour[(i + 1) % n], coords_x, coords_y)
    return length

def _optimize_4opt_sequential_corrected(
    tour: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    candidate_set: np.ndarray,
    candidate_dists: np.ndarray,
    locked_edges: np.ndarray,
    pos: np.ndarray,
    dlb: np.ndarray,
) -> bool:
    n = tour.shape[0]
    if n < 8:
        return False
    globally_improved = False
    num_cand = candidate_set.shape[1]

    for t1_idx in range(n):
        t1 = tour[t1_idx]
        if dlb[t1]:
            continue

        t2_idx = (t1_idx + 1) % n
        t2 = tour[t2_idx]
        if locked_edges[t1, 0] == t2 or locked_edges[t1, 1] == t2:
            continue
        dist_t1_t2 = _dist(t1, t2, coords_x, coords_y)

        found = False
        for k3 in range(num_cand):
            t3 = candidate_set[t1, k3]  # t3 is candidate of t1
            if t3 == -1:
                break
            if t3 == t1 or t3 == t2:
                continue
            dist_t1_t3 = candidate_dists[t1, k3]
            g1 = dist_t1_t2 - dist_t1_t3
            if g1 <= 1e-9:
                continue

            t3_idx = pos[t3]
            for d1 in [1, -1]:
                t4_idx = (t3_idx + d1 + n) % n
                t4 = tour[t4_idx]
                if t4 == t1 or t4 == t2:
                    continue
                if locked_edges[t3, 0] == t4 or locked_edges[t3, 1] == t4:
                    continue
                dist_t3_t4 = _dist(t3, t4, coords_x, coords_y)
                
                # Intermediate check: Swap 1 gain must be positive
                g1_swap = g1 + dist_t3_t4 - _dist(t2, t4, coords_x, coords_y)
                if g1_swap <= 1e-9:
                    continue

                for k5 in range(num_cand):
                    t5 = candidate_set[t2, k5]  # t5 is candidate of t2
                    if t5 == -1:
                        break
                    if t5 in (t1, t2, t3, t4):
                        continue
                    dist_t2_t5 = candidate_dists[t2, k5]
                    g2 = g1 + dist_t3_t4 - dist_t2_t5
                    if g2 <= 1e-9:
                        continue

                    t5_idx = pos[t5]
                    for d2 in [1, -1]:
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = tour[t6_idx]
                        if t6 in (t1, t2, t3, t4, t5):
                            continue
                        if locked_edges[t5, 0] == t6 or locked_edges[t5, 1] == t6:
                            continue
                        dist_t5_t6 = _dist(t5, t6, coords_x, coords_y)
                        
                        # Intermediate check: Swap 2 gain must be positive
                        g2_swap = g2 + dist_t5_t6 - _dist(t4, t6, coords_x, coords_y)
                        if g2_swap <= 1e-9:
                            continue

                        for k7 in range(num_cand):
                            t7 = candidate_set[t4, k7]  # t7 is candidate of t4
                            if t7 == -1:
                                break
                            if t7 in (t1, t2, t3, t4, t5, t6):
                                continue
                            dist_t4_t7 = candidate_dists[t4, k7]
                            g3 = g2 + dist_t5_t6 - dist_t4_t7
                            if g3 <= 1e-9:
                                continue

                            t7_idx = pos[t7]
                            for d3 in [1, -1]:
                                t8_idx = (t7_idx + d3 + n) % n
                                t8 = tour[t8_idx]
                                if t8 in (t1, t2, t3, t4, t5, t6, t7):
                                    continue
                                if locked_edges[t7, 0] == t8 or locked_edges[t7, 1] == t8:
                                    continue
                                dist_t7_t8 = _dist(t7, t8, coords_x, coords_y)

                                gain = g3 + dist_t7_t8 - _dist(t6, t8, coords_x, coords_y)
                                if gain > 1e-9:
                                    print("Found improvement!")
                                    print("t1:", t1, "t2:", t2, "t3:", t3, "t4:", t4, "t5:", t5, "t6:", t6, "t7:", t7, "t8:", t8)
                                    print("Broken distances:")
                                    print("t1-t2:", dist_t1_t2)
                                    print("t3-t4:", dist_t3_t4)
                                    print("t5-t6:", dist_t5_t6)
                                    print("t7-t8:", dist_t7_t8)
                                    print("Sum broken:", dist_t1_t2 + dist_t3_t4 + dist_t5_t6 + dist_t7_t8)
                                    print("Added distances:")
                                    print("t1-t3:", dist_t1_t3)
                                    print("t2-t5:", dist_t2_t5)
                                    print("t4-t7:", dist_t4_t7)
                                    print("t6-t8:", _dist(t6, t8, coords_x, coords_y))
                                    print("Sum added:", dist_t1_t3 + dist_t2_t5 + dist_t4_t7 + _dist(t6, t8, coords_x, coords_y))
                                    print("Calculated gain:", gain)
                                    
                                    # Apply corrected swaps
                                    _apply_2opt_indices(tour, t1_idx, t2_idx, t3_idx, t4_idx)
                                    _update_pos(tour, pos)
                                    _apply_2opt_indices(tour, pos[t2], pos[t4], pos[t5], pos[t6])
                                    _update_pos(tour, pos)
                                    _apply_2opt_indices(tour, pos[t4], pos[t6], pos[t7], pos[t8])
                                    _update_pos(tour, pos)

                                    dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = dlb[t5] = dlb[t6] = dlb[t7] = dlb[t8] = False
                                    globally_improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            dlb[t1] = True
    return globally_improved

def _compute_candidate_dists(coords_x, coords_y, candidate_set):
    n = coords_x.shape[0]
    num_cand = candidate_set.shape[1]
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
    
    # Generate seed
    cs_init = build_candidate_sets(coords, k=10)
    tour = generate_greedy_nn_seeds(coords, cs_init, 1)[0]
    
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)
    
    # Measure length before
    len_before = compute_tour_length(tour, cx, cy)
    print("Initial length:", len_before)
    
    # Let's run a single call to _optimize_4opt_sequential_corrected
    improved = _optimize_4opt_sequential_corrected(
        tour, cx, cy, candidate_set, candidate_dists, locked_edges, pos, dlb
    )
    len_after = compute_tour_length(tour, cx, cy)
    print("Improved return value:", improved)
    print("Length after:", len_after)
    print("Difference (after - before):", len_after - len_before)

if __name__ == "__main__":
    main()
