# Preprocessing Module Tasks

- [x] **Task 1: Delaunay Triangulation Wrapper**
  - Use `scipy.spatial.Delaunay` to get initial edge connectivity.
  - Convert Delaunay simplices into an adjacency list (sparse representation).
- [x] **Task 2: Numba-Accelerated Neighbor Filtering (Core)**
  - Implement `@njit(parallel=True)` function `_filter_nearest_neighbors`.
  - Input: `coords`, `adjacency_list`, `offset_list`, `k=16`.
  - For each city, sort its Delaunay neighbors by Euclidean distance and pick top 16.
- [x] **Task 3: Memory Optimization**
  - Ensure the output `candidate_set` is `np.int32` to minimize memory footprint for 115k cities.
  - Verify that the candidate set shape is exactly `(N, 16)`.
- [x] **Task 4: Unit Testing Neighbors**
  - Create a test with 10 cities in a known grid.
  - Manually calculate nearest neighbors and compare with `build_candidate_sets` output.
- [x] **Task 5: Performance Benchmarking**
  - Run the module on 115k cities and measure execution time.
  - Aim for < 10 seconds for the entire preprocessing phase. (Result: ~0.7s)
