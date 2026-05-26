import numpy as np
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets
from src.core.seed_generation import generate_greedy_nn_seeds
from src.core.kopt_engine import compute_tour_length

def _dist(c1, c2, cx, cy):
    dx = cx[c1] - cx[c2]
    dy = cy[c1] - cy[c2]
    return np.sqrt(dx * dx + dy * dy)

def _apply_2opt(tour, i, j):
    n = tour.shape[0]
    if i == j:
        return
    size = (j - i + n) % n + 1
    count = size // 2
    for k in range(count):
        idx1 = (i + k) % n
        idx2 = (j - k + n) % n
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

def debug_or_opt(tour, cx, cy, candidate_set, candidate_dists, pos, dlb):
    n = tour.shape[0]
    globally_improved = False
    for i_idx in range(n):
        u = tour[i_idx]
        found = False
        for length in range(1, 6):
            j_idx = (i_idx + length - 1) % n
            v = tour[j_idx]
            
            p_u_idx = (i_idx - 1 + n) % n
            p_u = tour[p_u_idx]
            s_v_idx = (j_idx + 1) % n
            s_v = tour[s_v_idx]
            
            base_g = _dist(p_u, u, cx, cy) + _dist(v, s_v, cx, cy) - _dist(p_u, s_v, cx, cy)
            if base_g <= 1e-9:
                continue
                
            for k in range(candidate_set.shape[1]):
                w = candidate_set[u, k]
                if w == -1 or w == u or w == v or w == p_u or w == s_v:
                    continue
                
                w_idx = pos[w]
                is_inside = False
                if i_idx <= j_idx:
                    if i_idx <= w_idx <= j_idx:
                        is_inside = True
                else:
                    if w_idx >= i_idx or w_idx <= j_idx:
                        is_inside = True
                if is_inside:
                    continue
                    
                s_w_idx = (w_idx + 1) % n
                s_w = tour[s_w_idx]
                
                dist_wu = candidate_dists[u, k]
                if base_g + _dist(w, s_w, cx, cy) > dist_wu + _dist(v, s_w, cx, cy) + 1e-9:
                    print(f"Executing non-reversed relocate: i={i_idx}({u}), j={j_idx}({v}), w={w_idx}({w})")
                    print("Before:", tour.copy())
                    len_before = compute_tour_length(tour, cx, cy)
                    
                    if i_idx <= j_idx < w_idx:
                        _apply_2opt(tour, i_idx, j_idx)
                        print("After swap 1:", tour.copy())
                        _apply_2opt(tour, s_v_idx, w_idx)
                        print("After swap 2:", tour.copy())
                        _apply_2opt(tour, i_idx, w_idx)
                        print("After swap 3:", tour.copy())
                    elif w_idx < i_idx <= j_idx:
                        _apply_2opt(tour, i_idx, j_idx)
                        print("After swap 1:", tour.copy())
                        _apply_2opt(tour, s_w_idx, p_u_idx)
                        print("After swap 2:", tour.copy())
                        _apply_2opt(tour, s_w_idx, j_idx)
                        print("After swap 3:", tour.copy())
                    else:
                        continue
                        
                    len_after = compute_tour_length(tour, cx, cy)
                    print(f"Length before: {len_before:.2f}, after: {len_after:.2f}, diff: {len_after - len_before:.2f}")
                    # Update pos dictionary
                    pos = {node: idx for idx, node in enumerate(tour)}
                    globally_improved = True
                    found = True
                    break
            if found:
                break
    return globally_improved

def main():
    n = 100
    coords = load_cities('data/cities.csv')[:n]
    cx = coords[:, 0]
    cy = coords[:, 1]
    candidate_set = build_candidate_sets(coords, k=8)
    # compute dists
    cand_dists = np.zeros(candidate_set.shape)
    for i in range(n):
        for k in range(candidate_set.shape[1]):
            w = candidate_set[i, k]
            if w != -1:
                cand_dists[i, k] = _dist(i, w, cx, cy)
                
    cs_init = build_candidate_sets(coords, k=10)
    tour = generate_greedy_nn_seeds(coords, cs_init, 1)[0]
    pos = {node: idx for idx, node in enumerate(tour)}
    dlb = np.zeros(n, dtype=np.bool_)
    
    debug_or_opt(tour, cx, cy, candidate_set, cand_dists, pos, dlb)

if __name__ == '__main__':
    main()
