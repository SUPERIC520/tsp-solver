# Autonomous TSP Solver Development Prompt

## 1. Objective
You are an autonomous AI engineering team tasked with implementing a high-performance Traveling Salesperson Problem (TSP) solver. 
- **Scale**: ~115,000 cities.
- **Target**: < 5% gap from the Held-Karp lower bound.
- **Tech Stack**: Python 3.12+, NumPy, SciPy, Numba (@njit), Multiprocessing.

## 2. Project Architecture
The project is divided into the following modules as defined in `doc/detailed-design.md` with codes in `src/`:
1. **Data I/O**: CSV loading/saving.
2. **Preprocessing**: Delaunay-based candidate sets (16 neighbors).
3. **Seed Generation**: Diverse Hilbert curve initializations (8 seeds).
4. **K-Opt Engine**: Numba-accelerated Full Cascading K-Opt optimization engine (2/3/4/5-opt).
5. **Orchestration**: Multiprocessing management.
6. **Backbone Consensus**: Edge locking and iterative refinement.
7. **Validation**: Held-Karp lower bound estimation.

## 3. Operational Rules (Mandatory)

### 3.1 Orchestration & Delegation
- **Main Agent**: You are the Main Agent. Your primary responsibility is to track overall progress and manage the implementation and verification lifecycles.
- **Progress Tracking**: You MUST update `doc/tasks/progress.md` after every significant module completion, test cycle, or benchmark milestone.
- **Sub-Agent Delegation for ALL Phases**: For EVERY single task—including module implementation, individual testing cycles, static analysis sweeps, and final scale benchmarking—you MUST spawn a specialized Sub-Agent. 
    - **Isolation**: Sub-agents must operate within isolated execution scopes, specializing entirely in the discrete sub-task assigned (e.g., Test-Runner Sub-Agent, Static-Analysis Sub-Agent).
    - The Sub-Agent is responsible for running the test suites, parsing errors, and ensuring the module passes all validation checks before handing back control to the Main Agent.
    - Have multiple subagents work in parallel. You are the manager responsible for efficiently assigning work to the workers.

### 3.2 Autonomous Mode
- **Zero Human Intervention**: You must solve all implementation, compilation, and runtime errors autonomously.
- **Refinement**: If a test fails or performance is sub-optimal, analyze the root cause, adjust the strategy, and re-implement until successful.

### 3.3 Quality & Verification
- **Testing**: Every module MUST have comprehensive `pytest` unit tests.
- **Static Analysis**: All code must pass `mypy --strict` and `ruff check`. No type errors or linting violations are permitted.
- **Performance**: High-performance sections (K-Opt Engine, Preprocessing) must be benchmarked against target scales.

## 4. TSP-Specific Technical Constraints

### 4.1 Numba (@njit) Guidelines
- **Type Inference**: Be explicit with Numba types where possible to avoid inference failures in complex loops.
- **Pre-compilation/Warmup**: In unit tests and the final solver, implement a "warmup" call with a tiny dataset (e.g., N=5) to trigger JIT compilation before measuring performance.
- **Unsupported Features**: Ensure code inside `@njit` blocks only uses Numba-compatible Python/NumPy features.

### 4.2 Multiprocessing & Resource Safety
- **Deadlock Prevention**: Use proper synchronization or shared memory (e.g., `multiprocessing.shared_memory` or passing read-only NumPy arrays) to avoid deadlocks.
- **Resource Cleanup**: Ensure all process pools and shared resources are explicitly closed and unlinked after use.

### 4.3 Memory Management (16GB Limit)
- **NO N*N MATRICES**: It is strictly forbidden to instantiate or calculate a full $N \times N$ distance matrix for 115k cities (~100GB).
- **Localized Computation**: Use the `candidate_set` (N x 16) to restrict distance calculations to immediate neighbors and compute distances on-the-fly where necessary.
- **Precision**: Use `float64` for coordinates and `int32` for indices to balance precision and memory usage.
- **Held-Karp Persistence**

## 5. Implementation Roadmap & Testing Protocol
Follow the sequence defined in `doc/tasks/`:
1. Data I/O & Environment Setup.
2. Preprocessing (Delaunay/Neighbors).
3. Seed Generation (Hilbert).
4. K-Opt Engine (Full Cascading JIT K-Opt).
5. Orchestration (Multiprocessing).
6. Backbone Consensus (Edge Locking).
7. Validation (Held-Karp).


## 5. Implementation Roadmap & Testing Protocol

### 5.1 Testing Protocol & Terminal Execution Guardrails (Strict)
- **Sequential Scale-up**: Every new implementation or algorithmic tuning cycle MUST strictly progress through the following dataset scales: 
  $$\text{Scale Progression: } N = 100 \rightarrow 500 \rightarrow 1,000 \rightarrow 5,000 \rightarrow 10,000 \rightarrow 115,475 \text{ (Full Scale)}$$
- **Progression Gate**: A Sub-Agent is strictly prohibited from executing a benchmark or optimization run at a higher scale if the current scale fails to pass verification or violates quality constraints.
- **Engine Move Constraints**: Use at most 3-opt for initial testing/benchmarking across the intermediate scales ($N \le 10,000$) to maintain rapid iteration speed. Advanced 4/5-opt passes are reserved exclusively for final gap closure on the Full dataset once the pipeline is proven robust.
- **Held-Karp (HK) Lower Bound Caching Constraint**: To prevent redundant, computationally expensive calculations during iterative development and automated test sweeps, the validation engine **MUST NOT** recompute the Held-Karp lower bound if it has already been computed for the target dataset scale. 
  - **Caching Mechanism**: Sub-agents must implement a robust serialization mechanism (e.g., saving to a centralized `.cache/hk_bounds.json` or `.npy` matrix directory keyed by the dataset hash/scale).
  - **Execution Rule**: Every script execution and test runner must check this cache first. If a valid pre-computed HK lower bound exists, it must be loaded immediately. Live computation is strictly reserved for the absolute first run of a unique scale.
- **Mandatory Command Timeouts**: Every terminal command executed by any Sub-Agent (including `pytest`, `mypy`, `ruff`, or running execution scripts) MUST be bound by a hard, explicitly defined timeout limit to prevent indefinite hanging from C-level JIT infinite loop regressions or multi-process deadlocks. STRICT time limits on EVERY terminal command param.
- **Dynamic Timeout & Wiggle Room Calibration**: Timeouts are dynamically scaled to reflect sample size complexity, factoring in the sub-quadratic $O(N \log N)$ performance profile of LKH Candidate Sets combined with cascading K-opt. The limits include a strict **25% wiggle room buffer** to accommodate Numba's cold compilation overhead on initial runs:

    | Task / Scale | N = 100 (Cold/Warm) | N = 500 | N = 1,000 | N = 5,000 & 10,000 | N = 115,475 (Full Scale) |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **`ruff` / `mypy --strict`** | 10s max | 10s max | 10s max | 12s max | 15s max |
    | **`pytest` Unit Tests** | 30s / 5s | 10s max | 15s max | 45s max | 90s max |
    | **Optimization Engine Run**| 15s max | **10s max** | 25s max | 150s (2.5 mins) max | **600s (10 mins) max** |

- **Timeout Enforcement & Recovery**: If a command hits its timeout ceiling, the Sub-Agent must immediately send a `SIGKILL` to terminate the process, dump the current execution stack trace, flag the component as `FAILED` due to a performance/infinite loop bottleneck, and report back to the Main Agent for autonomous architectural modification.

## 6. Initialization
Before starting, read all files in `doc/` and `doc/tasks/` to internalize the full technical specification. Begin with Task 1 in `doc/tasks/data-io.md`.
