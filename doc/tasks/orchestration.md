# Orchestration Module Task List

- [x] **[T5.1]** Reimplement worker concurrency in [orchestration.py](src/core/orchestration.py) using `multiprocessing.Array('i', num_seeds)` for low-overhead progress counters, eliminating the Manager proxy process.
- [x] **[T5.2]** Set up worker process initialization under `multiprocessing.Pool` (concurrency bounded by `NUM_PROCESSES_SOLVER`), distributing slice seeds and the progress Array. Ensure inter-process tour synchronization is disabled to prevent locks.
- [x] **[T5.3]** Implement a robust pool manager loop utilizing `apply_async`. Catch exceptions (Numba compiles, MemoryErrors) inside worker tasks using `try-except` blocks, and check elapsed times. In case of timeouts or exceptions, invoke `pool.terminate()`, join processes, and return the best intermediate results. *(Depends on [T5.2])*
- [x] **[T5.4]** Write unit tests in `tests/test_orchestration.py` verifying progress reporting accuracy, concurrent solver scaling, and child process failure recovery.
