# Advanced Optimization Tasks

- [x] **Task 1: Full K-Opt Engine Implementation**
  - Implement a generalized K-Opt local search engine.
  - Use `candidate_set` (Delaunay top 16) to restrict search.
- [ ] **Task 2: Recursive Move Search**
  - Implement a variable-depth search using the gain criterion and Euclidean pruning.
  - Implement pruning based on the triangle inequality.
- [x] **Task 3: Backbone Strategy**
  - Adjust backbone mechanism to lock high-confidence edges.
- [ ] **Task 4: Organization & Documentation**
  - Ensure all modules pass `mypy --strict` and `ruff check`.
- [ ] **Task 5: Final Validation Run**
  - Execute the full pipeline on 115k cities.
  - Target gap: < 5%.
  - Log results in `notes.md`.

## Mandatory Rules
- **Sub-Agent**: MUST use a specialized sub-agent for every implementation and test.
- **Testing**: Every new implementation or test MUST start with N=100.
- **Initial Limit**: Use at most 3-opt for initial testing/benchmarking.
