# Backbone Consensus Module Tasks

- [ ] **Task 1: Edge Counting (@njit)**
  - Implement `@njit count_edges(tours)` where `tours` is `(M, N)`.
  - For 115k cities, use a sparse approach or limited neighbor frequency check to avoid $N^2$ memory.
- [ ] **Task 2: Consensus Extraction**
  - Implement `extract_consensus_edges(tours, threshold=0.95)`.
  - Identify edges that appear in at least `0.95 * M` tours.
- [ ] **Task 3: Data Structure for Locked Edges**
  - Implement the `(N, 2)` array representation for `locked_edges`.
  - Initialize with `-1`.
- [ ] **Task 4: Backbone Logic Verification**
  - Create 3 identical tours and 1 different tour.
  - Verify that edges present in the 3 identical ones are correctly locked.
