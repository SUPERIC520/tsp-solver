import multiprocessing as mp
import numpy as np
import time
from typing import Tuple, List, cast, Any
from src.core.kopt_engine import cascading_kopt_optimize as kopt_optimize_tour


def _kopt_worker(
    args: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int],
) -> Tuple[np.ndarray, float]:
    """
    Worker function for multiprocessing.
    """
    seed, coords, candidate_set, locked_edges, num_kicks, max_opt = args
    return cast(
        Tuple[np.ndarray, float],
        kopt_optimize_tour(
            seed, coords, candidate_set, locked_edges, num_kicks, max_opt
        ),
    )


def parallel_solve(
    seeds: np.ndarray,
    coords: np.ndarray,
    candidate_set: np.ndarray,
    locked_edges: np.ndarray,
    num_processes: int = -1,
    num_kicks: int = 100,
    max_opt: int = 3,
) -> List[Tuple[np.ndarray, float]]:
    """
    Run K-Opt optimization in parallel for multiple seeds.
    """
    if num_processes == -1:
        num_processes = mp.cpu_count()

    num_seeds = seeds.shape[0]
    tasks = []
    for i in range(num_seeds):
        tasks.append(
            (seeds[i], coords, candidate_set, locked_edges, num_kicks, max_opt)
        )

    results = []
    with mp.Pool(processes=min(num_processes, num_seeds)) as pool:
        # Use apply_async to allow periodic heartbeat prints
        async_results = [pool.apply_async(_kopt_worker, (task,)) for task in tasks]
        
        completed = [False] * num_seeds
        while not all(completed):
            for i, res in enumerate(async_results):
                if not completed[i] and res.ready():
                    result = res.get()
                    results.append(result)
                    completed[i] = True
                    print(f"  - Completed seed {len(results)}/{num_seeds} (Length: {result[1]:.2f})")
            
            if not all(completed):
                time.sleep(10)  # Heartbeat every 10 seconds
                print(f"  ... still solving ({sum(completed)}/{num_seeds} seeds done) ...")

    return results
