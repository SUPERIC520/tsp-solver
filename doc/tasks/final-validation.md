# Final Validation Tasks

## Target: < 5% Gap from Held-Karp Bound

- [ ] **Task 1: Implementation of Full Cascading K-Opt Engine**
  - Use sub-agent to refactor `kopt_engine.py`.
  - Protocol: Start with N=100.
- [ ] **Task 2: Performance Benchmarking (N=500)**
  - Load first 500 cities.
  - Target: < 5% gap in < 10s.
- [ ] **Task 3: Scaling Test (N=1000, N=5000)**
  - Load subsets of the main dataset.
  - Verify that execution time scales approximately linearly.
- [ ] **Task 4: Full Dataset Execution (N=115,475)**
  - Run the solver on the complete dataset.
  - Target: < 5% gap in < 10 minutes.
- [ ] **Task 5: Final Documentation & Quality Check**
  - Ensure all code passes `ruff` and `mypy`.
  - Update `notes.md` with final metrics.
