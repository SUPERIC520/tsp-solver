# High-Level Design: TSP Solver (Full Cascading K-Opt)

## 1. Introduction
This document outlines the high-level design for a high-performance Traveling Salesperson Problem (TSP) solver. The system is designed to handle ~115,000 cities with a target precision of <5% margin from the Held-Karp lower bound, utilizing Numba-accelerated Full Cascading K-Opt optimization (2/3/4/5-opt) and parallel processing.

## 2. System Architecture
The system follows a modular architecture where components communicate primarily via NumPy arrays for performance and compatibility with Numba and SciPy.

### 2.1 Module Overview
- **Data I/O Module**: Manages file operations and coordinate normalization.
- **Preprocessing Module**: Generates candidate sets using Delaunay triangulation.
- **Seed Generation Module**: Creates diverse initial tours via Hilbert curves.
- **Full Cascading K-Opt Engine**: Numba-JIT accelerated engine for path refinement, implementing a cascading strategy with 2, 3, 4, and 5-edge swaps. For initial testing, at most 3-opt moves are considered to balance performance.
- **Orchestration Module**: Manages parallel execution across CPU cores.
- **Backbone Consensus Module**: Implements edge locking and iterative refinement.
- **Validation Module**: Benchmarks results against the Held-Karp lower bound, with caching support.

## 3. Module Definitions

### 3.1 Data I/O Module
- **Responsibility**: Load city coordinates from `data/cities.csv` and export final results.
- **Input**: `cities.csv` (Format: `Index, X, Y`).
- **Output**: NumPy array `float64[N, 2]` of coordinates.

### 3.2 Preprocessing Module
- **Responsibility**: Reduce search space by identifying neighbor candidates.
- **Input**: Coordinate array.
- **Process**: Compute Delaunay triangulation using `scipy.spatial`.
- **Output**: NumPy array `int32[N, 16]` containing the indices of the 16 nearest neighbors for each city, derived from Delaunay edges.

### 3.3 Seed Generation Module
- **Responsibility**: Provide diverse starting points for the optimization engine.
- **Process**: Generate up to 8 unique seeds using rotated Hilbert curves.
- **Output**: NumPy array `int32[8, N]` representing initial permutations.

### 3.4 Full Cascading K-Opt Engine
- **Responsibility**: Perform high-speed K-Opt swaps (2, 3, 4, and 5-opt) to find local optima in a cascading fashion (trying simpler moves first).
- **Technology**: Numba `@njit` with SIMD vectorization.
- **Input**: Tour array, candidate set, coordinate array.
- **Output**: Optimized tour array, path length.

### 3.5 Orchestration Module
- **Responsibility**: Distribute seeds to worker processes and collect results.
- **Technology**: `multiprocessing.Pool`.
- **Process**: Map 8 seeds to Cascading K-Opt instances across all physical cores.

### 3.6 Backbone Consensus Module
- **Responsibility**: Lock high-confidence edges to simplify the problem in subsequent iterations.
- **Process**: Analyze the top-performing tours; lock edges appearing in >95% of solutions.
- **Output**: Set of locked edges, list of unresolved nodes for refinement.

### 3.7 Validation Module
- **Responsibility**: Verify the quality of the solution.
- **Process**: Calculate Held-Karp lower bound (using 1-tree relaxation) and compare with the best found tour length. Results are cached to avoid redundant computation.

## 4. Interface and Data Flow

### 4.1 Data Structures
- **Coordinates**: `np.ndarray (shape=(N, 2), dtype=float64)`
- **Candidate Set**: `np.ndarray (shape=(N, 16), dtype=int32)`
- **Tours**: `np.ndarray (shape=(8, N), dtype=int32)`
- **Edge Locks**: `np.ndarray (shape=(N, 2), dtype=int32)` adjacency list representation.

### 4.2 High-Level Sequence
1. **Data I/O** loads coordinates.
2. **Preprocessing** builds the candidate neighbor matrix (Delaunay top 16).
3. **Seed Generation** creates up to 8 initial paths.
4. **Orchestration** dispatches seeds to **Full Cascading K-Opt Engine**.
5. **Backbone Consensus** identifies stable edges and triggers iterative refinement if necessary.
6. **Validation** computes final metrics and **Data I/O** saves the optimal path.

## 5. Testing & Validation Protocol
- **MANDATORY**: ALL tasks, implementation, and tests MUST be executed by a specialized Sub-Agent.
- **Sequential Scale-up**: Start testing with N=100. Progress only if successful.
- **Initial Limit**: Use at most 3-opt for initial validation phases.

## 6. Performance Considerations
- **Memory**: Use NumPy's memory-efficient dtypes (`int32`, `float64`).
- **CPU**: Maximize cache locality within K-Opt loops to leverage Numba's SIMD optimizations. Take advantage of Euclidean triangle inequality for early pruning.
- **Parallelism**: Use 8 seeds max to stay within resource limits while ensuring diversity.
