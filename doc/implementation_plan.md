# Reach TSP Gap < 5% at Full Scale with Reasonable Runtime

We need to resolve the freezing issue in the sequential K-Opt engine, optimize Held-Karp accuracy, and scale the solver to run successfully on the full dataset of ~115,000 cities with a final gap from Held-Karp lower bound < 5% and execution time < 10 minutes.

## User Review Required

> [!IMPORTANT]
> - We will run the commands using `uv` with `BypassSandbox = true` as `uv` is installed on the host machine.
> - We will strictly spawn Sub-Agents for execution, testing, debugging, and final scaling verification to comply with the project instructions.

## Proposed Changes

---

### K-Opt Optimization Engine

We will fix the bug in the 3-opt sequential logic and prevent any potential infinite loops by verifying actual gain using exact path reconstruction.

#### [MODIFY] [kopt_engine.py](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/src/core/kopt_engine.py)

- **`_reconstruct_tour_3opt` [NEW]**: A helper function to build the new tour for all 8 3-opt segment reconnections.
- **`_optimize_3opt_sequential`**: Re-implement to search the candidate set, evaluate the cost of each reconnection, and apply the best improving 3-opt move only if the actual cost decreases.
- **`_optimize_2opt`**: Fix any indexing or logic issues that cause it to fail on simple cases.

---

### Validation & Caching

#### [MODIFY] [validation.py](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/src/core/validation.py)

- Ensure Held-Karp bounds and Pi vectors are loaded from and saved to cache files when using `compute_hk_lower_bound`.

---

## Verification Plan

### Automated Tests
- Run `uv run pytest` to verify the fixed 2-opt and 3-opt algorithms.
- Run `mypy --strict src tests` and `ruff check .` to verify type and style compliance.

### Manual Verification
- We will execute the sequential scale-up protocol using Sub-Agents:
  1. $N = 100$ scale validation: `uv run python src/scripts/run_sample.py --n 100 --iters 2`
  2. $N = 500$ scale validation: `uv run python src/scripts/run_sample.py --n 500 --iters 2`
  3. $N = 1,000$ scale validation: `uv run python src/scripts/run_sample.py --n 1000 --iters 2`
  4. $N = 5,000$ scale validation: `uv run python src/scripts/run_sample.py --n 5000 --iters 2`
  5. $N = 10,000$ scale validation: `uv run python src/scripts/run_sample.py --n 10000 --iters 2`
  6. $N = 115,475$ full-scale validation: `uv run python src/scripts/main.py`
