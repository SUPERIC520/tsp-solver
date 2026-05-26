# LKH+ TSP Solver - Technical Specifications

This document outlines the architectural configurations, computational boundaries, and algorithmic specifications of the project.

## 1. System Parameters and Configuration (`src/config.py`)

### Structural Limits
* `KD_TREE_QUERY_SIZE`: 64
* `K_NEIGHBORS`: 64
* `K_3OPT`: 48
* `K_4OPT`: 32
* `K_5OPT`: 16
* `OR_OPT_MAX_LEN`: 8
* `MAX_OPT`: 5

### Execution Settings
* `CACHE_VERSION`: Resolves to Git HEAD branch or 7-character commit hash.
* `NUM_PROCESSES_SOLVER`: -1
* `NUM_PROCESSES_SEEDING`: -1
* `SEED_STRATEGY`: "Greedy"

## 2. Memory Allocations and Data Structures

### Primary Arrays
* **Coordinate Matrix**: NumPy `ndarray`, shape `(N, 2)`, dtype `np.float64`, C-contiguous.
* **Candidate Set Matrix**: NumPy `ndarray`, shape `(N, K_NEIGHBORS)`, dtype `np.int32`, C-contiguous.
* **Seed Tours Matrix**: NumPy `ndarray`, shape `(num_seeds, N)`, dtype `np.int32`, C-contiguous.
* **Tour Representation**: NumPy `ndarray`, shape `(N,)`, dtype `np.int32`.
* **Don't Look Bits (DLB)**: NumPy `ndarray`, shape `(N,)`, dtype `np.bool_`.
* **Adjacency Graphs**: Arrays for pointers `(N + 1,)` and indices `(edges,)`, dtype `np.int32`.

### Memory Management
* Memory boundary alignment is executed using 64-byte offsets: `np.empty(nbytes + 64, dtype=np.uint8)`.
* Slice offset calculation is defined as `(64 - (address % 64)) % 64`.
* Cross-process synchronization employs `multiprocessing.Array('i', num_seeds)`.

## 3. Algorithmic Complexity

* **Distance Metric**: L2-norm Euclidean distance calculation `sqrt(dx^2 + dy^2)`, O(1) per evaluation.
* **Coordinate Transformation (`get_hilbert_indices`)**: Normalizes bounds to a `[0, 2^20 - 1]` grid. Evaluates in O(N log N) time via `np.argsort`.
* **Candidate Set Generation (`build_candidate_sets`)**: KD-Tree construction is O(N log N). Nearest neighbor queries return `min(k + 1, n)` elements. Appends `(i, i-1)` and `(i, i+1)` indexes.
* **Alpha-Value Calculation (`compute_alpha_values`)**:
  * 1-Tree generation utilizes Prim's Algorithm with binary min-heap, resolving in O(N * K log N) time.
  * Lowest Common Ancestor (LCA) via binary lifting pre-computes in O(N log N) time. Space complexity is bounded to `(N, log2(n) + 1)`.
  * LCA edge queries resolve in O(log N) time.
  * Parallel alpha-sorting pass via `np.argsort` resolves in O(N * K log K) time.
* **Seed Generation (`_greedy_nn_tour`)**: Requires O(N * K) utilizing candidate sets, defaults to O(N^2) linear scan for disjoint sets.
* **Local Search Segment Manipulation (`_apply_2opt`)**: O(L) time where segment length `L = (j - i + n) % n + 1`. Array element swap loops execute `L // 2` times.

## 4. Computational Boundaries

* **Floating Point Epsilon Constraints**:
  * Acceptance threshold distance delta: `1e-9`.
  * Held-Karp lower bound minimum improvement delta: `1e-6`.
  * LCA initialization bounds: `-1e15`.
* **Sub-Graph Constraints**:
  * 2-opt evaluates strictly on `n >= 4`.
  * 3-opt evaluates strictly on `n >= 6`.
  * 4-opt evaluates strictly on `n >= 8`.
* **Candidate Set Traversal Widths**:
  * 2-opt neighbor limit: `min(candidate_set.shape[1], K_NEIGHBORS)`.
  * 3-opt neighbor limit: `min(candidate_set.shape[1], K_3OPT)`.
  * Or-opt iteration limits: Hardcoded variable limit set to `40`.
  * Or-opt segment length constraint (`max_len`): Fixed default 5, strictly bound to `min(max_len + 1, n - 2)`.
* **Held-Karp Subgradient Evaluation (`_compute_hk_impl`)**:
  * Iteration limit: 500.
  * Search origins: `0`, `n // 3`, `(2 * n // 3) % n`.
  * Initial Lambda Scalar: 2.0.
  * Lambda Decay Rate: 0.7.
  * Lambda Decay Condition: Triggers after `max(100, max_iter // 50)` successive iterations yielding 0 bound improvement.
  * Lambda Floor Limit: 0.0001.

## 5. Execution Pipeline and Multi-Processing

### Orchestration Sequence (`main.py`)
1. Instantiates subset array using length constraint `--n`.
2. Computes mapping matrices `orig_to_new` and `new_to_orig` via `hilbert_reorder_cities`.
3. Pre-computes candidate sets mapped to column depth `K_NEIGHBORS`.
4. Retrieves `float` and `np.ndarray` cache state for Held-Karp bounds and `pi` values, conditionally executing calculations bound by `--hk_iter`.
5. Iterates process spawning via `multiprocessing` utilizing the limits `min(mp.cpu_count(), args.seeds)`.
6. Validates iterations, extracting `iter_best_length` against `global_best_length` state logic.

### CLI Default Arguments (`argparse`)
* `--kicks` (int): 25000
* `--iters` (int): 1
* `--seeds` (int): 8
* `--max_opt` (int): 3
* `--n` (int): 0
* `--hk_iter` (int): 10000
* `--no_cache` (bool): False
* `--start_tour` (str): None

## 6. I/O Mechanics and Persistence Data Types

### Data Parsing and Output Models
* **Source Coordinates**: Sliced dynamically using `np.loadtxt(filepath, usecols=(1, 2), dtype=np.float64)`. Space-separated parsing.
* **Vector Serialization**: Output array `np.ndarray` dimensions joined to strings explicitly via `",".join(map(str, tour))` for CSV output.
* **Binary Store Cache Paths**: 
  * Path formatting strings: `data/cache/{CACHE_VERSION}/sample_{sample_name}_hk.npy` and `data/cache/{CACHE_VERSION}/sample_{sample_name}_pi.npy`.
  * File instantiation conditional logic: Evaluates `os.path.exists` prior to boolean evaluation. Target directories resolved via `os.makedirs`.
* **Update Validations**: Persistence writes constrained logically to parameters `is_full_run == True` and numeric evaluations `new_length < current_best`.
* **Type Parsing Constraints**: Load functions resolve strings conditionally `int(i) for i in line.split(",")`. Missing or corrupted paths return `float("inf")`.