# Seed Generation Module Task List

- [x] **[T3.1]** Remove standalone Hilbert-based seed generation (`generate_hilbert_seeds`) and random seed generation (`generate_random_seeds`) from [seed_generation.py](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/src/core/seed_generation.py).
- [x] **[T3.2]** Implement `rotate_tour(tour, start_node)` to rotate starting node sequence while keeping the path cycle topology unchanged. Ensure the output array is C-contiguous and 64-byte aligned.
- [x] **[T3.3]** Implement multi-process parallel seed generation `generate_greedy_nn_seeds` using `multiprocessing.Pool` (bounded by `NUM_PROCESSES_SEEDING` processes) to compute Greedy NN tours starting from distinct nodes in parallel. Returns a 64-byte aligned contiguous matrix of shape `(num_seeds, N)`.
- [x] **[T3.4]** Write unit tests in `tests/test_seed_generation.py` checking `rotate_tour` output and parallel NN seed generation dimensions, non-duplication, and coordinate mapping correctness under disabled JIT test runs.
