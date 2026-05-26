import multiprocessing as mp
import numpy as np
import time
import sys
from typing import Tuple, List, Any
from src.core.kopt_engine import cascading_kopt_optimize as kopt_optimize_tour
from src.utils.time_utils import format_duration

# Global variable to store the shared progress array in worker processes
_shared_progress_array: Any = None


def _init_worker(progress_array: Any) -> None:
    """
    Initializer for worker processes. Sets the shared progress array.
    """
    global _shared_progress_array
    _shared_progress_array = progress_array


def _kopt_worker(
    args: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float, int],
) -> Tuple[np.ndarray, float, int] | Exception:
    """
    Worker function for multiprocessing. Returns (tour, length, kicks_completed)
    or catches exceptions and returns the exception object.
    """
    try:
        seed, coords_x, coords_y, candidate_set, num_kicks, max_opt, time_limit_s, seed_idx = args
        
        global _shared_progress_array
        
        # Check signature of kopt_optimize_tour for locked_edges compatibility (defensive)
        import inspect
        sig = inspect.signature(kopt_optimize_tour)
        func: Any = kopt_optimize_tour
        if "locked_edges" in sig.parameters:
            n = seed.shape[0]
            locked_edges = np.full((n, 2), -1, dtype=np.int32)
            res = func(
                seed, coords_x, coords_y, candidate_set, locked_edges, num_kicks, max_opt,
                time_limit_s=time_limit_s,
                progress_array=_shared_progress_array,
                seed_idx=seed_idx,
            )
        else:
            res = func(
                seed, coords_x, coords_y, candidate_set, num_kicks, max_opt,
                time_limit_s=time_limit_s,
                progress_array=_shared_progress_array,
                seed_idx=seed_idx,
            )
        return res  # type: ignore[no-any-return]
    except Exception as e:
        return e


def parallel_solve(
    seeds: np.ndarray,
    coords: np.ndarray,
    candidate_set: np.ndarray,
    locked_edges: Any = None,
    num_processes: int = -1,
    num_kicks: int = 100,
    max_opt: int = 3,
    time_limit_s: float = -1.0,
    iteration_start_time: float = 0.0,
    total_start_time: float = 0.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Run K-Opt optimization in parallel for multiple seeds.
    Uses multiprocessing.Array for low-overhead progress tracking.
    """
    from src.config import NUM_PROCESSES_SOLVER

    # Handle transitions / backward compatibility where locked_edges might be passed
    actual_num_processes = num_processes
    if locked_edges is not None:
        if isinstance(locked_edges, (int, np.integer)):
            actual_num_processes = int(locked_edges)

    p_count = actual_num_processes
    if p_count == -1:
        p_count = NUM_PROCESSES_SOLVER
    if p_count <= 0:
        p_count = mp.cpu_count()

    coords_x = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
    coords_y = np.ascontiguousarray(coords[:, 1], dtype=np.float64)

    num_seeds = seeds.shape[0]
    
    # [T5.1] Shared memory counter array
    progress_array = mp.Array('i', num_seeds)

    # Disable inter-process tour synchronization to prevent locks.
    # Distribute slice seeds and the progress Array.
    tasks = []
    for i in range(num_seeds):
        tasks.append(
            (seeds[i], coords_x, coords_y, candidate_set, num_kicks, max_opt, time_limit_s, i)
        )

    results = []
    completed = [False] * num_seeds
    start_time = time.time()

    # [T5.2] Setup worker processes initialization under mp.Pool
    pool = mp.Pool(
        processes=min(p_count, num_seeds),
        initializer=_init_worker,
        initargs=(progress_array,),
    )

    try:
        # [T5.3] Implement robust pool manager loop using apply_async
        async_results = [pool.apply_async(_kopt_worker, (task,)) for task in tasks]

        while not all(completed):
            # Check elapsed time for timeout
            if time_limit_s > 0 and (time.time() - start_time) > time_limit_s:
                sys.stdout.write("\r" + " " * 100 + "\r")
                print(f"    [Timeout] limit of {time_limit_s}s exceeded. Terminating workers...", file=sys.stderr)
                pool.terminate()
                pool.join()
                break

            for i, res in enumerate(async_results):
                if not completed[i] and res.ready():
                    result = res.get()
                    if isinstance(result, Exception):
                        raise result
                    
                    # result is (tour, length, kicks_completed)
                    results.append((result[0], result[1]))
                    completed[i] = True
                    sys.stdout.write("\r" + " " * 100 + "\r")
                    print(f"  - Completed seed {len(results)}/{num_seeds} (Length: {result[1]:.2f})")

            if not all(completed):
                time.sleep(0.1)
                
                # Real-time progress reporting (T5.1)
                total_kicks_done = sum(progress_array[:])
                total_kicks_target = num_seeds * num_kicks
                percent = (total_kicks_done / total_kicks_target) * 100 if total_kicks_target > 0 else 0.0

                done = sum(completed)
                iter_elapsed = time.time() - iteration_start_time
                total_elapsed = time.time() - total_start_time
                status = f"\r    [Progress] {done}/{num_seeds} seeds solved | kicks: {total_kicks_done}/{total_kicks_target} ({percent:.2f}%) | Iter Elapsed: {format_duration(iter_elapsed)} | Total Elapsed: {format_duration(total_elapsed)} ..."
                sys.stdout.write(status)
                sys.stdout.flush()

        pool.close()
        pool.join()

    except Exception as e:
        print(f"\n    [Exception] Encountered exception in orchestration: {e}. Terminating workers...", file=sys.stderr)
        pool.terminate()
        pool.join()
        # Return best intermediate results collected so far
        return results

    return results
