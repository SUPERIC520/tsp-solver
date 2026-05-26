# K-Opt Engine Module Task List

- [x] **[T4.1]** Search and completely remove `locked_edges` from all function signatures, variable layouts, and local search routines inside [kopt_engine.py](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/src/core/kopt_engine.py).
- [x] **[T4.2]** Apply `@njit(fastmath=True, parallel=True)` for parallel precomputations and array calculations like `_precompute_candidate_dists` to compile optimal LLVM SIMD code.
- [x] **[T4.3]** Refine `_optimize_2opt` using 64-byte aligned contiguous distance arrays and DLB pruning, iterating candidates up to `K_NEIGHBORS`.
- [x] **[T4.4]** Implement `_optimize_or_opt` for relocate moves of segments up to length `max_len = OR_OPT_MAX_LEN` utilizing C-contiguous data layouts. *(Depends on [T4.3])*
- [x] **[T4.5]** Refine `_optimize_3opt_sequential` to implement dynamic funneling: loop `k3` searches up to `K_NEIGHBORS`, but loop `k5` is restricted to `K_3OPT`. Compute gains incrementally and prune early if the first cut has negative gain ($g_1 \le 0$). *(Depends on [T4.4])*
- [x] **[T4.6]** Refine `_optimize_4opt_sequential` implementing dynamic funneling: outer loop `k3` is restricted to `K_NEIGHBORS`, loop `k5` is restricted to `K_3OPT`, and loop `k7` is restricted to `K_4OPT`. *(Depends on [T4.5])*
- [x] **[T4.7]** Refine `_optimize_5opt_sequential` implementing dynamic funneling: outer loop `k3` is restricted to `K_NEIGHBORS`, loop `k5` is restricted to `K_3OPT`, loop `k7` is restricted to `K_4OPT`, and loop `k9` is restricted to `K_5OPT`. *(Depends on [T4.6])*
- [x] **[T4.8]** Implement `_full_cascade` cascade controller calling `2-opt -> or-opt -> 3-opt -> 4-opt -> 5-opt` sequentially until convergence, checking `max_opt` constraint bounds. *(Depends on [T4.7])*
- [x] **[T4.9]** Implement `_cascading_kopt_inner` outer ILS kick loop to run in JIT-compiled chunks. *(Depends on [T4.8])*
- [x] **[T4.10]** Implement Python wrapper `cascading_kopt_optimize` asserting 64-byte memory alignments and C-contiguity on input coordinates and path arrays, then calling `_cascading_kopt_inner` in chunks, tracking and returning exact `kicks_done`.
- [x] **[T4.11]** Write unit tests in `tests/test_kopt_engine.py` (under JIT-disabled mode) verifying correct cost delta computations and local search improvements for all five operators on simple coordinates (e.g., square and circle tours).
