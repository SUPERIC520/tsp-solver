# TSP Solver - Technical Specifications

This repository contains a high-performance Traveling Salesperson Problem (TSP) solver based on cascading k-opt heuristic, optimized with Numba, and parallelized for large-scale datasets.

## How to Use

### 1. Environment Setup

#### Option A: `uv`
```powershell
uv sync
```

#### Option B: `pip` and `venv`
```powershell
# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix/macOS:
source .venv/bin/activate

# Install production dependencies
pip install .

# Install development tools
pip install -e ".[dev]"
```

*Note: Core dependencies include Numba, NumPy, and SciPy. Development tools (Mypy, Ruff, Pytest, Pyright) are in the `dev` group.*

### 2. Basic Execution (Main Solver)
The main production script orchestrates the full pipeline: loading cities, computing lower bounds, and running parallel K-opt optimization.

```powershell
# Run on a subset of 500 cities for a quick test
uv run python -m src.scripts.main --n 500 --kicks 100 --iters 1 --seeds 8
```

#### Key Arguments:
*   `--n`: Number of cities to sample from the dataset (0 for all).
*   `--kicks`: Number of double-bridge kicks per seed.
*   `--seeds`: Number of parallel optimization starts.
*   `--iters`: Number of full re-optimization passes (0 for infinite).
*   `--max_opt`: Maximum K-opt level (default 3, range [2, 5]).
*   `--hk_iter`: Iterations for Held-Karp lower bound computation.
*   `--no_cache`: Disable using cached Held-Karp results.
*   `--start_tour`: Path to a tour CSV/text file to initialize optimization.

### 3. Benchmarking
Use the `run_sample` script to evaluate performance and log results to `notes.md`.

```powershell
uv run python -m src.scripts.run_sample --n 1000 --kicks 500 --iters 3
```

### 4. Verification and Quality Control
The project maintains strict standards for type safety and linting:

```powershell
# Run unit tests
uv run pytest

# Run strict type checking
uv run mypy --strict src tests
uv run pyright src tests

# Run linter
uv run ruff check .
```

### 5. Data Outputs
*   `data/solutions.csv`: Contains the results of the current run.
*   `data/best_tour.csv`: Stores the global best tour found (only updated during full-scale runs).

---

## 1. Mathematical Foundations and Objective Formulation

### Integer Linear Programming (ILP) Formulation
The Traveling Salesperson Problem (TSP) on a graph $G = (V, E)$ with vertex set $V = \{1, 2, \dots, n\}$ and edge costs $c_{ij}$ for $(i,j) \in E$ is formulated using decision variables $x_{ij} \in \{0, 1\}$. The variable $x_{ij} = 1$ if edge $(i, j)$ is included in the tour, and $x_{ij} = 0$ otherwise.

Objective function:
$$\min \sum_{(i,j) \in E} c_{ij} x_{ij}$$

Subject to degree constraints:
$$\sum_{j \in V, j \neq i} x_{ij} = 2 \quad \forall i \in V$$

Subject to subtour elimination constraints (SECs):
$$\sum_{i,j \in S} x_{ij} \le |S| - 1 \quad \forall S \subset V, 2 \le |S| \le n-1$$

### 1-Tree Relaxation
A 1-tree is defined as a spanning tree on the vertex set $V \setminus \{1\}$, combined with exactly two edges connecting vertex 1 to the spanning tree. The set of all 1-trees is denoted as $\mathcal{T}$. A TSP tour is a 1-tree where every vertex $i \in V$ has a degree equal to 2. The algorithm relaxes the degree constraints for all vertices while maintaining the requirement that the subgraph forms a 1-tree.

### Lagrangian Relaxation and the Dual Problem
The degree constraints $\sum_{j} x_{ij} = 2$ are moved into the objective function using a vector of Lagrange multipliers $\pi = (\pi_1, \pi_2, \dots, \pi_n) \in \mathbb{R}^n$.

The modified edge costs, $c'_{ij}$, are defined as:

$$
c'_{ij} = c_{ij} + \pi_i + \pi_j
$$

The Lagrangian function $L(\pi)$ is computed by finding the minimum weight 1-tree with respect to $c'_{ij}$:

$$
L(\pi) = \min_{x \in \mathcal{T}} \sum_{(i,j) \in E} (c_{ij} + \pi_i + \pi_j) x_{ij} - 2\sum_{i \in V} \pi_i
$$

$$
L(\pi) = \left( \min_{x \in \mathcal{T}} \sum_{(i,j) \in E} c'_{ij} x_{ij} \right) - 2\sum_{i \in V} \pi_i
$$

The Dual Problem maximizes this formulation:

$$
\max_{\pi \in \mathbb{R}^n} L(\pi)
$$

### Subgradient Ascent and $\pi$ Vector Computation
The dual problem is non-differentiable. Subgradient ascent is utilized to iteratively update the $\pi$ vector. Let $x^{(k)}$ be the minimum 1-tree computed at iteration $k$ using weights $c'_{ij}$ derived from $\pi^{(k)}$.

The degree of vertex $i$ in the 1-tree $x^{(k)}$ is denoted as $d_i^{(k)} = \sum_{j} x_{ij}^{(k)}$.
The subgradient vector $g^{(k)}$ has components defined as:

$$
g_i^{(k)} = d_i^{(k)} - 2
$$

The multipliers are updated using the formula:

$$
\pi_i^{(k+1)} = \pi_i^{(k)} + t_k \cdot g_i^{(k)}
$$

The step size $t_k$ is computed using the formulation:

$$
t_k = \lambda_k \frac{UB - L(\pi^{(k)})}{\sum_{i \in V} (g_i^{(k)})^2}
$$

where $UB$ is the length of a previously computed tour and $\lambda_k$ is a scalar parameter initialized at $2$ and halved if $L(\pi)$ fails to increase after a predetermined number of iterations.

### Algorithmic Evaluation of the Relaxation
This specific relaxation is evaluated algorithmically because the problem of finding a minimum weight 1-tree is solvable in $O(|E| \log |V|)$ operations. The calculation of $L(\pi)$ provides a bound for branch-and-bound processes. Furthermore, the final modified costs $c^{\prime}_ {ij}$ generated by the subgradient ascent procedure are utilized as edge weights to direct the $k\text{-opt}$ search heuristic. Edges with values of $c^{\prime}_ {ij}$ approaching zero are included in the candidate sets evaluated during the sequential edge exchange operations.

## 2. Initialization and Seeding Strategies

### 2.1 Greedy Nearest Neighbor Setup

The algorithm applies the Greedy Nearest Neighbor setup for initialization. The data in `doc/archive_notes.md` records an 800-trial experiment with parameters N=5000 and max_opt=5. The metrics from this experiment are:

*   100% Greedy Nearest Neighbor yielded a 5.71% Gap.
*   100% Hilbert yielded a 6.70% Gap.
*   The ANOVA test resulted in p < 2.07 x 10^-85.

The Greedy Nearest Neighbor algorithm was chosen based on these gap metrics. Algorithmically, Greedy Nearest Neighbor provides a different search basin than space-filling curves.

### 2.2 Continuation Seeding Setup

A trial testing re-seeding via 100% Exploitation yielded a 5.52% Gap, with an ANOVA p < 2.74 x 10^-19. 

Based on this 5.52% Gap result, the algorithm enforces 100% resets to the global minimum-distance tour for all cores. The algorithm enforces this reset rule so that every processing unit starts its search sequence from the exact vertex coordinates that produced the global minimum distance, eliminating iterations on coordinates with higher distance values.

### 2.3 Uniform Index Rotation

The algorithm maps seed index `i` to starting vertices using Uniform Index Rotation math. The equation is:

```python
shift = (i * n) // num_seeds
```

The variables are defined as:
*   `shift`: The integer offset applied to the vertex array.
*   `i`: The seed index.
*   `n`: The vertex count.
*   `num_seeds`: The seed count.

This integer division spaces the starting vertices at intervals of `n // num_seeds` across the vertices to spread search origins.

## 3. Iterated Local Search (ILS) Core Engine

### Cascading k-opt Sequence
The local search phase executes a sequence of edge-exchange operators in a specified order: 2-opt, Or-opt, 3-opt, proceeding up to the integer limit `MAX_OPT`.

#### 2-opt Operator
The 2-opt operation removes two non-adjacent edges $(u, v)$ and $(x, y)$ from a tour and replaces them with edges $(u, x)$ and $(v, y)$. Mathematically, applying this to a metric space governed by the triangle inequality resolves segment intersections.
*   **Array Manipulation:** The tour is stored as a 1D integer array. The operation reverses the sequence of vertices from index $v$ to index $x$. The algorithm performs this using a two-pointer loop, swapping array elements at the boundary indices and iterating inward.

#### Or-opt Operator
The Or-opt operation relocates a sequence of contiguous vertices (length 1, 2, or 3) to a different target index within the tour array. Mathematically, it evaluates a subset of the 3-opt neighborhood to test block translations.
*   **Array Manipulation:** The algorithm copies the source segment into a temporary buffer. Array elements between the source and target indices are shifted left or right by the segment length. The buffer contents are written to the target indices.

#### 3-opt and MAX_OPT Operators
The 3-opt operation removes three edges, generating three disconnected path segments. The algorithm evaluates the four strictly 3-opt reconnection permutations. The sequence scales to `MAX_OPT` by enumerating combinations of $k$ edge removals.
*   **Array Manipulation:** Reconnection executes multiple segment reversals and block shifts. The loop evaluates the sum of the removed edge weights minus the sum of the added edge weights. Permutations yielding a positive scalar (indicating a distance reduction) trigger the array index updates.

### Double Bridge Perturbation Operator
The engine implements the Double Bridge operator to perturb the tour state between local search phases. 

#### Fixed Kick Count
The current implementation utilizes a fixed kick count per iteration. It applies a constant integer of perturbation moves per cycle. The system does not utilize variable kicks or adaptive kick counts.

#### Mathematical Purpose
The Double Bridge is a 4-opt move that removes four edges and reconnects the resulting four segments (labeled A, B, C, D) into the sequence A, D, C, B. This operator generates a coordinate topology that cannot be reverted by sequential 2-opt operations. It alters the state coordinates beyond the boundary of the 2-opt search neighborhood.

#### Array Manipulation
The algorithm selects four array indices $i_1, i_2, i_3, i_4$ using a uniform probability distribution generator. The 1D tour array is partitioned at these indices. The vertex indices are copied into a second 1D array in the A, D, C, B sequence order.

## 4. Evolutionary Backbone Mechanism (Gamma Penalty)

### Frequency Analysis
The mechanism executes a frequency analysis operation after `N` iterations of the algorithm across `S` distinct seed executions. For a set of `S` output tours $T = \{T_1, T_2, ..., T_S\}$, where each tour $T_s$ consists of a set of undirected edges $E_s$, the mechanism computes the appearance count $C_{i,j}$ for every edge $(i, j)$ in the union of all generated tours $\bigcup_{s=1}^S E_s$.

The frequency of an edge, $f_{i,j}$, is calculated as the ratio of its appearance count to the integer count of seed executions:
$f_{i,j} = \frac{C_{i,j}}{S}$
where $C_{i,j} = \sum_{s=1}^S \mathbb{1}((i, j) \in E_s)$ and $\mathbb{1}$ is the indicator function.

### Gamma Penalty Multiplier Generation
The system utilizes the calculated frequencies $f_{i,j}$ to define a penalty term for the candidate set generation phase of iteration $N+1$ and beyond. The edge appearance frequency $f_{i,j}$ functions as the penalty multiplier. The gamma penalty multiplier $\gamma_{i,j}$ is equal to the frequency $f_{i,j}$:
$\gamma_{i,j} = f_{i,j}$

### Candidate Set Sorting Key Mutation
During candidate set generation, the system sorts edges based on an alpha value $\alpha_{i,j}$. The backbone mechanism recalculates the sort value by applying the gamma penalty multiplier.

For a candidate edge $(i, k)$ evaluated for node $i$, the mutated sorting key is calculated using the following equation:
$sort\_key_{i, k} = \alpha_{i, k} - W_{backbone} \times \gamma_{i, k}$

Where:
* $sort\_key_{i, k}$ is the mutated scalar value used to sort the candidate edges for node $i$ in ascending numerical sequence.
* $\alpha_{i, k}$ is the pre-mutation alpha measure derived from the 1-tree computations for edge $(i, k)$.
* $W_{backbone}$ is the configured scalar parameter `backbone_weight`.
* $\gamma_{i, k}$ is the gamma penalty multiplier, equal to $f_{i, k}$.

### Implementation Mechanics
1. **Tour Aggregation:** The system stores the $S$ generated tours in a 2-dimensional integer array of shape `(S, V)`, where `V` is the vertex count.
2. **Frequency Map Construction:** A hash map data structure records $C_{i,j}$ for each distinct edge observed in the $S$ tours.
3. **Multiplier Distribution:** The system maps the frequencies $f_{i,j}$ to the candidate set data structures. The candidate set for each node $i$ is stored as an array of structures containing the destination node $k$ and the pre-mutation alpha value $\alpha_{i, k}$.
4. **Key Recalculation:** The sorting routine iterates through the candidate list of each node $i$. For each candidate $k$, the operation subtracts the product of `backbone_weight` and $f_{i, k}$ from the stored $\alpha_{i, k}$ value.
5. **Sorting:** The candidate arrays are processed by a comparison sort algorithm using the computed $sort\_key_{i, k}$ values to determine the index sequence of candidates explored in operations $N+1$ and beyond.

## 5. Iterative Sequence and Orchestration & System Parameters and Configuration

### Orchestration Limit Formula
The application imports the `multiprocessing` module for task distribution. The process limit executes the formula:
`min(mp.cpu_count(), args.seeds)`

*   `mp.cpu_count()`: The operating system function returns the hardware thread integer.
*   `args.seeds`: The variable stores the integer passed via the CLI argument.

The application instantiates a `multiprocessing.Pool` object. The `processes` parameter of the pool receives the integer output from the formula. The pool partitions the seed array into chunks and maps the chunks to the processes.

### Configuration Parameters (`src/config.py`)
The engine reads integers from the `src/config.py` file to bound array sizes and loops. The file contains 7 assignments:

*   `KD_TREE_QUERY_SIZE=64`: The KD-Tree index returns an array containing 64 points per query.
*   `K_NEIGHBORS=64`: The matrix allocates rows of length 64 to store distances per vertex.
*   `K_3OPT=48`: The 3-opt function restricts the array length for permutation loops to 48 vertices.
*   `K_4OPT=32`: The 4-opt function restricts the array length for permutation loops to 32 vertices.
*   `K_5OPT=16`: The 5-opt function restricts the array length for permutation loops to 16 vertices.
*   `OR_OPT_MAX_LEN=8`: The Or-opt function executes block relocations on an array slice with a length limit of 8 elements.
*   `MAX_OPT=5`: The engine restricts the k-opt steps to a k-value limit of 5.

### Execution Termination Bounds
The engine reads the `args.iters` integer to set the loop termination bound.

1.  The process receives a coordinate matrix and a seed integer.
2.  The process instantiates a counter with a value of 0.
3.  The process executes array mutations.
4.  The process increments the counter by 1.
5.  The process evaluates the condition: `counter < args.iters`.
6.  When the condition equates to `True`, execution jumps to step 3.
7.  When the condition equates to `False`, the loop terminates.
8.  The process returns the array matrix to the `multiprocessing` pool.

## 6. Memory Allocations, Algorithmic Complexity, and I/O Mechanics

### Memory Allocations
*   **Coordinate Matrices**: Instantiated with dimensions (N, 2). The dtype `np.float64` allocates 8 bytes per scalar, resulting in N * 16 bytes per matrix.
*   **Index Arrays**: Instantiated with dimensions (N,). The dtype `np.int32` allocates 4 bytes per scalar, resulting in N * 4 bytes per array.
*   **64-byte Memory Alignment**: Memory is allocated via `np.empty(nbytes + 64, dtype=np.uint8)`. The memory address modulo 64 is computed. The value of 64 minus the remainder is added to the memory pointer to reach a multiple of 64 bytes. An array view is cast to the dtype `np.float64` or `np.int32` starting from this offset.

### Algorithmic Complexity
*   **Distance Metric**: Time complexity is O(1). Space complexity is O(1).
*   **KD-tree**: Time complexity is O(N log N) for tree construction. Space complexity is O(N).
*   **1-Tree Generation**: Time complexity is O(N * K log N), where K represents the count of neighbors evaluated per vertex. Space complexity is O(N).
*   **Lowest Common Ancestor (LCA) via Binary Lifting**: Preprocessing time complexity is O(N log N). Query time complexity is O(1) per vertex pair. Space complexity is O(N log N) for the table of ancestors stored at intervals of 2^i.

### I/O Mechanics
*   **Input Deserialization**: `np.loadtxt` reads coordinate data from text files and assigns the parsed data to the (N, 2) `np.float64` memory block.
*   **Output Serialization**: CSV serialization writes the (N,) `np.int32` vertex sequence and the `np.float64` objective sum to a text file on disk.
