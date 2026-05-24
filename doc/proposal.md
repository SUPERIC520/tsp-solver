# TSP Solver Requirements Proposal

## 1. Project Overview
Goal: Develop a high-performance Python-based solver for the Traveling Salesperson Problem (TSP) to handle approximately 115,000 cities with a target precision of <5% margin from the Held-Karp lower bound.

## 2. Technical Strategy (Confirmed)
- **Algorithm**: Full Cascading K-Opt optimization engine (2/3/4/5-opt).
- **Initial Phase**: Focus on 2-opt and 3-opt for performance and stability during early validation.
- **Candidate Sets**: Generated via Delaunay triangulation (SciPy), truncated to the top 16 neighbors per city.
- **Initialization**: 8 unique seeds using rotated Hilbert curves for spatial diversity.
- **Parallelization**: `multiprocessing.Pool` utilizing all physical CPU cores.
- **Optimization Engine**: Numba-JIT (@njit) accelerated code for cache-locality and SIMD vectorization.
- **Orchestration**: All tasks, implementation, and verification steps MUST be performed by specialized Sub-Agents.
- **Iterative Refinement**: Backbone consensus mechanism to lock edges appearing in >95% of tours.
- **Validation**: Mathematical verification against the Held-Karp lower bound.

## 3. Strict Testing Protocol
- All tests and implementations must start with sample size N=100.
- Samples scale only after success at lower N (N=100 -> N=500 -> N=1000 ...).
- Sub-agents are mandatory for every test execution.

## 4. Data Specification
- **Input File**: `data/cities.csv`
- **Scale**: 115,475 cities.
- **Format**: `Index X Y` (Space-separated).
- **Libraries**: `numpy`, `scipy`, `numba`.

## 5. Implementation Phases
1. **Global Setup**: Coordinate processing, Delaunay candidate set generation (top 16), and Hilbert seeding.
2. **Core Engine**: JIT-compiled Full Cascading K-Opt routines.
3. **Orchestration**: Sub-Agent dispatch and result collection.
4. **Refinement & Convergence**: Backbone locking and iterative solving.
5. **Validation**: Performance and accuracy reporting.
