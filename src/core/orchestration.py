import multiprocessing as mp
import numpy as np
import time
import sys
from typing import Tuple, List, cast
from src.core.kopt_engine import cascading_kopt_optimize as kopt_optimize_tour


def _kopt_worker(
    args: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float],
) -> Tuple[np.ndarray, float]:
    """
    Worker function for multiprocessing.
    """
    seed, coords_x, coords_y, candidate_set, locked_edges, num_kicks, max_opt, time_limit_s = args
    return cast(
        Tuple[np.ndarray, float],
        kopt_optimize_tour(
            seed, coords_x, coords_y, candidate_set, locked_edges, num_kicks, max_opt,
            time_limit_s=time_limit_s,
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
    time_limit_s: float = -1.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Run K-Opt optimization in parallel for multiple seeds.
    If time_limit_s > 0, each worker will stop kicking after that many
    wall-clock seconds (with a 3% safety margin built into the engine).
    """
    if num_processes == -1:
        num_processes = mp.cpu_count()

    coords_x = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
    coords_y = np.ascontiguousarray(coords[:, 1], dtype=np.float64)

    num_seeds = seeds.shape[0]
    tasks = []
    for i in range(num_seeds):
        tasks.append(
            (seeds[i], coords_x, coords_y, candidate_set, locked_edges, num_kicks, max_opt, time_limit_s)
        )

    results = []
    start_time_solve = time.time()
    with mp.Pool(processes=min(num_processes, num_seeds)) as pool:
        # Use apply_async to allow periodic heartbeat prints
        async_results = [pool.apply_async(_kopt_worker, (task,)) for task in tasks]
        
        completed = [False] * num_seeds
        last_print = time.time()
        while not all(completed):
            for i, res in enumerate(async_results):
                if not completed[i] and res.ready():
                    result = res.get()
                    results.append(result)
                    completed[i] = True
                    # Clear the status line and print completion on a new line
                    sys.stdout.write("\r" + " " * 60 + "\r")
                    print(f"  - Completed seed {len(results)}/{num_seeds} (Length: {result[1]:.2f})")
            
            if not all(completed):
                time.sleep(0.5)
                # Show status on a single line using carriage return
                done = sum(completed)
                elapsed = time.time() - start_time_solve
                status = f"\r    [Progress] {done}/{num_seeds} seeds solved | Elapsed: {elapsed:.1f}s ..."
                sys.stdout.write(status)
                sys.stdout.flush()

    return results
