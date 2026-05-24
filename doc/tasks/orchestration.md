# Orchestration Module Tasks

- [ ] **Task 1: Worker Wrapper**
  - Implement `_worker(args)` that unpacks `(seed, coords, candidate_set, locked_edges)`.
  - Ensure it returns `(optimized_tour, length)`.
- [ ] **Task 2: Multiprocessing Pool Setup**
  - Implement `parallel_solve` using `multiprocessing.Pool(processes=num_cores)`.
  - Use `pool.imap_unordered` for dispatching 8 seeds.
- [ ] **Task 3: IPC & Memory Efficiency**
  - Use read-only NumPy arrays or shared memory for `coords` and `candidate_set`.
  - Test memory usage with 8 workers active.
- [ ] **Task 4: Result Aggregation**
  - Collect all 8 results and identify the top-performing tours.
- [ ] **Task 5: Orchestration Test (Mock)**
  - Use a dummy worker to verify the Pool logic works and captures all 8 results.

## Mandatory Rules
- **Sub-Agent**: MUST use a specialized sub-agent for every task.
- **Testing**: Start at N=100.
- **Seeds**: Exactly 8 seeds max.
