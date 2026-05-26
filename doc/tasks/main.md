# Main Pipeline & Integration Task List

- [x] **[T8.1]** Integrate all refactored modules inside the main entry script [main.py](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/src/scripts/main.py):
- [x] **[T8.2]** Run the progressive validation pipeline to verify correctness and convergence metrics:
  - $N=100$: `uv run python src/scripts/run_sample.py --n 100 --iters 2`
  - $N=500$: `uv run python src/scripts/run_sample.py --n 500 --iters 2`
  - $N=1000$: `uv run python src/scripts/run_sample.py --n 1000 --iters 2`
  - $N=5000$: `uv run python src/scripts/run_sample.py --n 5000 --iters 2`
  - $N=115,475$ (full scale): `uv run python src/scripts/main.py` (Verify Gap < 5% and runtime < 10 mins).
