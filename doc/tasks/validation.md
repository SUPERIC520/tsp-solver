# Validation Module Task List

- [x] **[T6.1]** Completely clean up `locked_edges` parameters and variables from [validation.py](src/core/validation.py).
- [x] **[T6.2]** Optimize `_compute_hk_impl` subgradient descent loop with `@njit(fastmath=True, parallel=True)` for distance calculations and MST weight summaries. Implement file-based caching inside `compute_hk_lower_bound` to store/retrieve lower bounds and Pi vectors.
- [x] **[T6.3]** Optimize `compute_alpha_values` using `@njit(fastmath=True, parallel=True)` to parallelize edge Alpha evaluations across nodes. *(Depends on [T6.2])*
- [x] **[T6.4]** Write unit tests in `tests/test_validation.py` checking HK bound precision and computed Alpha arrays dimensions and shape matching.
