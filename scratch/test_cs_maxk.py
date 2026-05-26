import time
import numpy as np
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets
from src.core.seed_generation import generate_greedy_nn_seeds
from src.core.kopt_engine import cascading_kopt_optimize

def run_experiment(n, cs_size, max_k, kicks, initial_seed_tour, mode="standard"):
    coords = load_cities('data/cities.csv')[:n]
    cx = coords[:, 0]
    cy = coords[:, 1]
    candidate_set = build_candidate_sets(coords, k=cs_size)
    locked_edges = np.full((coords.shape[0], 2), -1, dtype=np.int32)
    
    cascading = (mode == "cascading")
    
    start = time.time()
    best_tour, best_len = cascading_kopt_optimize(
        initial_tour=initial_seed_tour,
        coords_x=cx,
        coords_y=cy,
        candidate_set=candidate_set,
        locked_edges=locked_edges,
        num_kicks=kicks,
        max_opt=max_k,
        cascading=cascading,
    )
    t = time.time() - start
    return best_len, t

if __name__ == "__main__":
    n = 500
    kicks = 500
    print(f"--- CS vs Max-K Grid Search (N={n}, Kicks={kicks}) ---")
    print(f"{'CS':<5} | {'Max-K':<6} | {'Mode':<10} | {'Best Len':<12} | {'Time (s)':<10}")
    print("-" * 58)
    
    coords = load_cities('data/cities.csv')[:n]
    cs_init = build_candidate_sets(coords, k=10)
    base_seed = generate_greedy_nn_seeds(coords, cs_init, 1)[0]
    
    coords_warmup = load_cities('data/cities.csv')[:100]
    cs_init_warmup = build_candidate_sets(coords_warmup, k=10)
    base_seed_warmup = generate_greedy_nn_seeds(coords_warmup, cs_init_warmup, 1)[0]
    run_experiment(100, 10, 3, 10, base_seed_warmup, mode="standard")
    run_experiment(100, 10, 3, 10, base_seed_warmup, mode="cascading")
    
    for cs in [8, 16, 32, 64]:
        for mk in [3, 4, 5]:
            for mode in ["standard", "cascading"]:
                blen, t = run_experiment(n, cs, mk, kicks, base_seed, mode)
                print(f"{cs:<5} | {mk:<6} | {mode:<10} | {blen:<12.2f} | {t:<10.2f}", flush=True)
