# Final Validation Subtasks

## Subtask 1: Optimize LKH Core for Speed and Quality
- **Goal:** Fix the timeout issue in `lkh_core.py` while maintaining a < 1% gap for N=5000.
- **Constraints:** Max 8 seeds, maximum execution time for N=5000 should be around 1-2 minutes.
- **Steps:**
  - Investigate why `optimize_lk` or individual `optimize_*` methods are hanging. 
  - Ensure the `candidate_set` limits (e.g., `limit1=8`, `limit2=5`) in `_run_full_local_search` are aggressive enough.
  - Implement a fast-to-slow short-circuit method or sequential method depending on what yields the best gap/time ratio.
  - Benchmark on N=1000 and N=5000 using `uv run python src/scripts/run_sample.py --n 5000 --seeds 8 --kicks 5000 --iters 2`.
  - Record the gap and time in `notes.md`.