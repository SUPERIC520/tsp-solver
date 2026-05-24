# Advanced Full K-Opt & Candidate Refinement Tasks

- [ ] **Task 1: Alpha-Value & Euclidean Pruning**
  - Implement Alpha-value calculation if needed for pruning.
  - Prioritize Euclidean triangle inequality pruning in the Full K-Opt search.
- [ ] **Task 2: Full K-Opt Search Strategy**
  - Implement a depth-first search for improving K-edge swaps.
  - Use "Don't Look Bits" to optimize search start nodes.
- [ ] **Task 3: Backbone Iterative Refinement**
  - Refine the backbone mechanism to focus Full K-Opt on unlocked regions.
- [ ] **Task 4: Parameter Tuning**
  - Tune max K and search depth to balance quality (< 5% gap) and time (< 10m).

## Validation Requirements
- All code must pass `ruff check` and `mypy --strict`.
- `pytest` for all new core logic.
- Target gap < 5%.
