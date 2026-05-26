import numpy as np

def _apply_2opt(tour: np.ndarray, i: int, j: int) -> None:
    n = tour.shape[0]
    if i == j:
        return
    size = (j - i + n) % n + 1
    count = size // 2
    for k in range(count):
        idx1 = (i + k) % n
        idx2 = (j - k + n) % n
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

def _apply_2opt_indices(tour: np.ndarray, t1_idx: int, t2_idx: int, t3_idx: int, t4_idx: int) -> None:
    n = tour.shape[0]
    if (t1_idx + 1) % n == t2_idx:
        _apply_2opt(tour, t2_idx, t3_idx)
    else:
        _apply_2opt(tour, t1_idx, t4_idx)

def main():
    n = 10
    tour = np.arange(n, dtype=np.int32)
    print("Original tour:", tour)
    pos = {node: idx for idx, node in enumerate(tour)}
    
    # Let's say we have t1, t2, t3, t4, t5, t6, t7, t8
    # In a sequential 4-opt, we walk along the tour:
    # (t1, t2) is an edge. Let's say t1_idx = 0, t2_idx = 1.
    # (t3, t4) is an edge. Let's say t3_idx = 3, t4_idx = 4.
    # (t5, t6) is an edge. Let's say t5_idx = 6, t6_idx = 7.
    # (t7, t8) is an edge. Let's say t7_idx = 8, t8_idx = 9.
    
    t1, t2 = tour[0], tour[1]
    t3, t4 = tour[3], tour[4]
    t5, t6 = tour[6], tour[7]
    t7, t8 = tour[8], tour[9]
    
    print(f"t1={t1}, t2={t2}, t3={t3}, t4={t4}, t5={t5}, t6={t6}, t7={t7}, t8={t8}")
    
    # Apply first swap
    _apply_2opt_indices(tour, 0, 1, 3, 4)
    print("After swap 1:", tour)
    
    # Update pos
    pos = {node: idx for idx, node in enumerate(tour)}
    
    # Apply second swap
    _apply_2opt_indices(tour, pos[t1], pos[t4], pos[t5], pos[t6])
    print("After swap 2:", tour)
    
    # Update pos
    pos = {node: idx for idx, node in enumerate(tour)}
    
    # Apply third swap
    _apply_2opt_indices(tour, pos[t1], pos[t6], pos[t7], pos[t8])
    print("After swap 3:", tour)
    
    print("Unique elements:", len(np.unique(tour)))
    if len(np.unique(tour)) != n:
        print("TOUR IS BROKEN (has duplicates/missing elements)!")
    else:
        print("Tour is valid.")

if __name__ == "__main__":
    main()
