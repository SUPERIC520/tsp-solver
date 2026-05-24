# Task: Debug and Fix Freezing Issue in Iterative Optimization

## Objective
Diagnose and fix the freezing behavior that occurs during the second iteration (when `iters > 1`) of the cascading K-Opt solver.

## Protocol & Timeouts
- **Scale**: Start with N=100. Do not scale up until N=100 passes without freezing and completes successfully.
- **Timeouts**:
  - `pytest`: 30s limit
  - Run scripts (N=100): 15s limit
- **Method**: Run pytest and debug scripts with Python's `subprocess.run(..., timeout=...)` to ensure commands terminate on hang, dumping stack traces if possible.

## Sub-Tasks
1. Run existing test suite using `pytest` to see if tests are passing/failing or hanging.
2. Run `src/scripts/run_sample.py` with N=100, seeds=2, kicks=10, iters=2, max_opt=3, and check if it hangs.
3. Diagnose the freeze location (using Python's `faulthandler` or inserting prints, or analyzing the code for JIT loops/infinite loop states).
4. Implement a fix for the freezing issue.
5. Verify the fix at N=100, then run unit tests.
