import numpy as np

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

def main():
    n = 10
    tour = np.arange(n, dtype=np.int32)
    print("Original:", tour)
    
    # We want to relocate segment [u, v] = [2, 3] (indices 2 to 3)
    # to be after w = 6 (index 6) in reversed order.
    # p_u = 1, s_v = 4, s_w = 7.
    # We expect: 0 1 -> 4 5 6 -> 3 2 -> 7 8 9
    
    # According to the code for i_idx <= j_idx < w_idx:
    # _apply_2opt(tour, s_v_idx, w_idx)   # s_v_idx = 4, w_idx = 6
    # _apply_2opt(tour, i_idx, w_idx)     # i_idx = 2, w_idx = 6
    
    _apply_2opt(tour, 4, 6)
    print("After swap 1:", tour)
    _apply_2opt(tour, 2, 6)
    print("After swap 2:", tour)
    
    print("Unique:", len(np.unique(tour)))

if __name__ == '__main__':
    main()
