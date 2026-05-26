# Preprocessing Module Task List

- [x] **[T2.1]** Clean up Delaunay neighbor filtering remnants from [preprocessing.py](src/core/preprocessing.py) (Delete `_filter_nearest_neighbors` and any Delaunay variables).
- [x] **[T2.2]** Implement a 64-byte alignment check and correction helper `ensure_alignment(arr, alignment=64)` to align coordinate and tour arrays to Cache Lines for optimal JIT performance.
- [x] **[T2.3]** Refine `hilbert_reorder_cities` to sort coordinates along the Hilbert space-filling curve, returning 64-byte aligned, C-contiguous coordinate matrices and inverse mapping arrays. *(Depends on [T2.2])*
- [x] **[T2.4]** Refine `build_candidate_sets` to compute base nearest-neighbors using `scipy.spatial.KDTree` returning a 64-byte aligned C-contiguous matrix of shape `(N, KD_TREE_QUERY_SIZE)`. *(Depends on [T2.2])*
- [x] **[T2.5]** Refine `refine_candidate_set_with_alpha` to compute Alpha values, sort candidates, slice to the first `K_NEIGHBORS` elements, and return a 64-byte aligned C-contiguous array of shape `(N, K_NEIGHBORS)`. *(Depends on [T2.4])*
- [x] **[T2.6]** Write unit tests in `tests/test_preprocessing.py` validating memory alignment constraints, Hilbert reordering index correctness, KD-Tree neighbors validation, and Alpha-sorted slice correctness under JIT disabled test runs.
