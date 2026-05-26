# LKH+ TSP Solver - Technical Specifications

This document outlines the architectural configurations, mathematical boundaries, and algorithmic specifications of the project.

## 1. Mathematical Foundations and Objective Formulation
* The Traveling Salesperson Problem (TSP) is formulated as an Integer Linear Programming (ILP) problem.
* The ILP is relaxed into a Linear Programming (LP) problem.
* The algorithm evaluates the dual problem. Subgradient ascent estimates the dual solutions due to the exponential dimensionality of sub-tour elimination constraints.
* The subgradient evaluation yields a Lagrange multiplier (`pi` vector) for each node representing degree constraint penalties.
* Edge weights are subsequently modified applying the Lagrange multipliers: `c'_{ij} = c_{ij} + pi_i + pi_j`.

## 2. Initialization and Seeding Strategies
* **Iteration 0 Initialization**: Seeded exclusively using a Greedy Nearest Neighbor logic. Documented variance in `doc/archive_notes.md` and `doc/archive_trials.md` demonstrates statistical significance (ANOVA: p < 2.07 x 10^-85), yielding an initial gap of 5.71%.
* **Continuation Seeding (Iteration > 0)**: Applies 100% exploitation of the global best tour generated from the previous iteration. Statistical significance confirmed (ANOVA: p < 2.74 x 10^-19).
* **Diversification Mechanism**: Implements a uniform index rotation on the best tour, calculated via `shift = (i * n) // num_seeds`.

## 3. Iterated Local Search (ILS) Core Engine
* **Local Search**: Executes a cascading k-opt sequence. Evaluates 2-opt, followed by Or-opt relocations, terminating at 3-opt (or up to `MAX_OPT` limits). 
* **Perturbation Operator**: Executes a double bridge kick.
* **Kick Limits**: Bounded computationally by a maximum kick count per iteration (`MAX_KICKS_PER_ITER = 100,000`) and an early-exit stagnation limit (`NON_IMPROVE_LIMIT = 5000`).

## 4. Evolutionary Backbone Mechanism (Gamma Penalty)
* **Frequency Analysis**: Post-iteration evaluation extracts edge usage frequency across all generated seed tours.
* **Cost Matrix Mutation**: Evaluates a gamma penalty multiplier proportional to edge appearance frequency.
* **Candidate Re-sorting**: Candidate set sorting keys are modified applying the gamma penalty to the Lagrange multipliers (`sort_key = alpha[i, k] - backbone_weight * freq[i, k]`).

## 5. Iterative Sequence and Orchestration
* Executes parallel sequence via `multiprocessing` evaluating up to `min(mp.cpu_count(), args.seeds)` processes.
* The core engine loops sequentially over the evolutionary backbone mechanism and uniformly rotated seeds.
* Execution terminates at `args.iters` or iterates infinitely if `args.iters == 0`.

## 6. System Parameters and Configuration (`src/config.py`)
* `KD_TREE_QUERY_SIZE`: 64
* `K_NEIGHBORS`: 64
* `K_3OPT`: 48
* `K_4OPT`: 32
* `K_5OPT`: 16
* `OR_OPT_MAX_LEN`: 8
* `MAX_OPT`: 5
* `NUM_PROCESSES_SOLVER`: -1
* `NUM_PROCESSES_SEEDING`: -1
* `SEED_STRATEGY`: "Greedy"
* `CACHE_VERSION`: Resolves to Git HEAD branch or 7-character commit hash.

## 7. Memory Allocations and Algorithmic Complexity
* **Coordinate Matrix**: NumPy `ndarray`, shape `(N, 2)`, dtype `np.float64`, C-contiguous.
* **Candidate Set Matrix**: NumPy `ndarray`, shape `(N, K_NEIGHBORS)`, dtype `np.int32`, C-contiguous.
* **Distance Metric**: L2-norm Euclidean distance calculation `sqrt(dx^2 + dy^2)`, O(1) per evaluation.
* **Candidate Set Generation (`build_candidate_sets`)**: KD-Tree construction is O(N log N). Nearest neighbor queries return `min(k + 1, n)` elements. 
* **Alpha-Value Calculation (`compute_alpha_values`)**:
  * 1-Tree generation utilizes Prim's Algorithm with binary min-heap: O(N * K log N) time.
  * Lowest Common Ancestor (LCA) via binary lifting pre-computes in O(N log N) time. Space complexity is bounded to `(N, log2(n) + 1)`.
  * LCA edge queries resolve in O(log N) time.
  * Parallel alpha-sorting pass via `np.argsort` resolves in O(N * K log K) time.
* **Local Search Segment Manipulation (`_apply_2opt`)**: O(L) time where segment length `L = (j - i + n) % n + 1`. Array element swap loops execute `L // 2` times.

## 8. I/O Mechanics and Persistence
* **Source Coordinates**: Sliced dynamically using `np.loadtxt(filepath, usecols=(1, 2), dtype=np.float64)`.
* **Binary Store Cache Paths**: 
  * Path formatting strings: `data/cache/{CACHE_VERSION}/sample_{sample_name}_hk.npy` and `data/cache/{CACHE_VERSION}/sample_{sample_name}_pi.npy`.
* **Vector Serialization**: Output array `np.ndarray` dimensions joined to strings explicitly via `",".join(map(str, tour))` for CSV output.