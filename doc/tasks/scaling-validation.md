# Progressive Scaling & Final Validation Tasks

## Mandatory Testing Protocol
- **MANDATORY**: Use specialized Sub-Agents for every task and test execution.
- **Sequential Scale-up**: EVERY new implementation or test MUST begin with N=100.
- **Progression**: Only move to larger samples (N=500, 1000, etc.) if the N=100 test passes perfectly.
- **Initial Limit**: Use at most 3-opt for initial validation to ensure speed.

## Tasks
- [ ] **Task 1: Optimize Held-Karp Accuracy**
  - Increase `max_iter` and implement better step-size control in `src/core/validation.py`.
  - Add caching logic for pi-vectors and lower bounds.
- [ ] **Task 2: Update Sample Runner for Scaling**
  - Modify `src/scripts/run_sample.py` to accept sample size as an argument.
- [ ] **Task 3: Validate on 100 Cities**
  - Goal: < 5% gap.
- [ ] **Task 4: Validate on 500 Cities**
  - Goal: < 5% gap, < 10s.
- [ ] **Task 5: Validate on 1000 Cities**
  - Goal: < 5% gap.
- [ ] **Task 6: Validate on 5000 Cities**
  - Goal: < 5% gap.
- [ ] **Task 7: Final Full-Scale Run (115,475 Cities)**
  - Goal: < 5% gap, < 10m.
  - Ensure all modules pass `ruff` and `mypy`.

## Verification Checklist
- [ ] `ruff check .`
- [ ] `mypy --strict src tests`
- [ ] `pytest tests`
