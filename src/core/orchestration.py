import multiprocessing as mp
import numpy as np
import time
import sys
from typing import Tuple, List, cast, Optional
from src.core.kopt_engine import cascading_kopt_optimize as kopt_optimize_tour
from src.utils.time_utils import format_duration



def _kopt_worker(
    args: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float, List[int], int],
) -> Tuple[np.ndarray, float, int]:
    """
    Worker function for multiprocessing. Returns (tour, length, kicks_completed).
    """
    seed, coords_x, coords_y, candidate_set, locked_edges, num_kicks, max_opt, time_limit_s, progress_list, seed_idx = args
    # Convert list proxy to something the engine can use if needed, or pass it directly.
    # The engine expects an object it can update index-wise.
    return cast(
        Tuple[np.ndarray, float, int],
        kopt_optimize_tour(
            seed, coords_x, coords_y, candidate_set, locked_edges, num_kicks, max_opt,
            time_limit_s=time_limit_s,
            progress_array=progress_list,
            seed_idx=seed_idx,
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
    iteration_start_time: float = 0.0,
    total_start_time: float = 0.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Run K-Opt optimization in parallel for multiple seeds.
    """
    if num_processes == -1:
        num_processes = mp.cpu_count()

    coords_x = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
    coords_y = np.ascontiguousarray(coords[:, 1], dtype=np.float64)

    num_seeds = seeds.shape[0]
    
    with mp.Manager() as manager:
        # Create shared list proxy
        progress_list = manager.list([0] * num_seeds)
        
        tasks = []
        for i in range(num_seeds):
            tasks.append(
                (seeds[i], coords_x, coords_y, candidate_set, locked_edges, num_kicks, max_opt, time_limit_s, progress_list, i)
            )

        results = []
        
        with mp.Pool(processes=min(num_processes, num_seeds)) as pool:
            # Use apply_async
            async_results = [pool.apply_async(_kopt_worker, (task,)) for task in tasks]
            
            completed = [False] * num_seeds
            while not all(completed):
                for i, res in enumerate(async_results):
                    if not completed[i] and res.ready():
                        result = res.get()
                        # result is (tour, length, kicks_completed_by_worker)
                        results.append((result[0], result[1]))
                        completed[i] = True
                        sys.stdout.write("\r" + " " * 100 + "\r")
                        print(f"  - Completed seed {len(results)}/{num_seeds} (Length: {result[1]:.2f})")
                
                if not all(completed):
                    time.sleep(0.5)
                    # Calculate real-time progress from shared list proxy
                    total_kicks_done = sum(progress_list)
                    total_kicks_target = num_seeds * num_kicks
                    percent = (total_kicks_done / total_kicks_target) * 100

                    done = sum(completed)
                    iter_elapsed = time.time() - iteration_start_time
                    total_elapsed = time.time() - total_start_time
                    status = f"\r    [Progress] {done}/{num_seeds} seeds solved | kicks: {total_kicks_done}/{total_kicks_target} ({percent:.2f}%) | Iter Elapsed: {format_duration(iter_elapsed)} | Total Elapsed: {format_duration(total_elapsed)} ..."
                    sys.stdout.write(status)
                    sys.stdout.flush()
    return results
