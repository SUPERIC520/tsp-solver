"""Orchestration for parallel TSP optimization.

This module provides the infrastructure to run multiple optimization seeds
in parallel using Python's multiprocessing module, with real-time progress
tracking through shared memory.
"""

import multiprocessing as mp
import multiprocessing.pool
import sys
import time
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt

from src.config import NUM_PROCESSES_SOLVER
from src.core.kopt_engine import cascading_kopt_optimize
from src.utils.time_utils import format_duration

# Global variable to store the shared progress array in worker processes
_shared_progress_array: npt.NDArray[np.int32] | None = None


def _init_worker(progress_array: npt.NDArray[np.int32]) -> None:
    """Initializer for worker processes. Sets the shared progress array.

    Args:
        progress_array: Shared memory array for tracking kick progress.
    """
    global _shared_progress_array  # noqa: PLW0603
    _shared_progress_array = progress_array


def _solver_worker(
    args: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float, int],
) -> tuple[np.ndarray, float, int] | Exception:
    """General worker function that dispatches to k-opt engine."""
    try:
        (
            seed,
            coords_x,
            coords_y,
            candidate_set,
            num_kicks,
            max_opt,
            time_limit_s,
            seed_idx,
        ) = args

        global _shared_progress_array  # noqa: PLW0602

        return cascading_kopt_optimize(
            seed,
            coords_x,
            coords_y,
            candidate_set,
            num_kicks,
            max_opt=max_opt,
            time_limit_s=time_limit_s,
            progress_array=_shared_progress_array,
            seed_idx=seed_idx,
        )
    except Exception as e:  # noqa: BLE001
        return e


def _raise_exception(e: Exception) -> NoReturn:
    """Helper to abstract raise for TRY301."""
    raise e


def parallel_solve(
    seeds: np.ndarray,
    coords: np.ndarray,
    candidate_set: np.ndarray,
    num_processes: int = -1,
    num_kicks: int = 100,
    max_opt: int = 5,
    time_limit_s: float = -1.0,
    iteration_start_time: float = 0.0,
    total_start_time: float = 0.0,
    pool: multiprocessing.pool.Pool | None = None,
    progress_array: Any = None,
) -> list[tuple[np.ndarray, float]]:
    """Run optimization in parallel for multiple seeds using cascading K-opt."""
    p_count = num_processes
    if p_count <= 0:
        p_count = NUM_PROCESSES_SOLVER
    if p_count <= 0:
        p_count = mp.cpu_count()

    coords_x = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
    coords_y = np.ascontiguousarray(coords[:, 1], dtype=np.float64)

    num_seeds = seeds.shape[0]

    created_pool = False
    if progress_array is None:
        progress_array = mp.Array("i", num_seeds)
        if pool is None:
            pool = mp.Pool(
                processes=min(p_count, num_seeds),
                initializer=_init_worker,
                initargs=(progress_array,),
            )
            created_pool = True
    else:
        # Reset progress array for the new iteration
        for i in range(num_seeds):
            progress_array[i] = 0

    if pool is None:
        msg = "Persistent pool must be initialized before calling parallel_solve"
        raise RuntimeError(msg)
    # Distribute slice seeds
    tasks = [
        (
            seeds[i],
            coords_x,
            coords_y,
            candidate_set,
            num_kicks,
            max_opt,
            time_limit_s,
            i,
        )
        for i in range(num_seeds)
    ]

    results: list[tuple[np.ndarray, float]] = []
    completed = [False] * num_seeds
    start_time = time.time()

    try:
        async_results = [pool.apply_async(_solver_worker, (task,)) for task in tasks]

        while not all(completed):
            if time_limit_s > 0 and (time.time() - start_time) > time_limit_s:
                sys.stdout.write("\r" + " " * 120 + "\r")
                if created_pool:
                    pool.terminate()
                    pool.join()
                break

            for i, res in enumerate(async_results):
                if not completed[i] and res.ready():
                    result = res.get()
                    if isinstance(result, Exception):
                        _raise_exception(result)

                    results.append((result[0], result[1]))
                    completed[i] = True
                    sys.stdout.write("\r" + " " * 120 + "\r")
                    print(
                        f"  - Completed seed {len(results)}/{num_seeds} "
                        f"(Length: {result[1]:.2f})"
                    )

            if not all(completed):
                time.sleep(0.1)
                total_kicks_done = sum(progress_array[:])
                total_kicks_target = num_seeds * num_kicks
                percent = (
                    (total_kicks_done / total_kicks_target) * 100
                    if total_kicks_target > 0
                    else 0.0
                )
                done = sum(completed)
                iter_elapsed = time.time() - iteration_start_time
                total_elapsed = time.time() - total_start_time
                status = (
                    f"\r    [Optimization Progress] {done}/{num_seeds} solved | "
                    f"kicks: {total_kicks_done}/{total_kicks_target} "
                    f"({percent:.2f}%) | "
                    f"Iter Elapsed: {format_duration(iter_elapsed)} | "
                    f"Total Elapsed: {format_duration(total_elapsed)} ..."
                )
                sys.stdout.write(status.ljust(120))
                sys.stdout.flush()

        if created_pool:
            pool.close()
            pool.join()

    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"Exception in parallel_solve: {e}\n")
        if created_pool:
            pool.terminate()
            pool.join()
        return results

    return results
