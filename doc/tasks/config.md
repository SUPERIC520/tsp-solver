# Configuration Module Task List

- [ ] **[T1.1]** Create the configuration file at `src/config.py` defining the global constants:
  - `KD_TREE_QUERY_SIZE: int = 64` (Initial KD-Tree neighbor query count)
  - `K_NEIGHBORS: int = 40` (Final alpha-sorted candidate set size for local search width)
  - `K_3OPT: int = 15` (Dynamic funneling width limit for sequential 3-opt)
  - `K_4OPT: int = 5` (Dynamic funneling width limit for sequential 4-opt)
  - `K_5OPT: int = 3` (Dynamic funneling width limit for sequential 5-opt)
  - `OR_OPT_MAX_LEN: int = 5` (Maximum segment length for Or-opt relocations)
  - `MAX_OPT: int = 5` (Maximum local search cascade depth level)
  - `NUM_PROCESSES_SOLVER: int = -1` (CPU process count for solver concurrency, default -1 to use CPU cores)
  - `NUM_PROCESSES_SEEDING: int = -1` (CPU process count for parallel greedy NN seed tours computation)
  - `SEED_STRATEGY: str = "Greedy"` (Default seeding strategy)
- [ ] **[T1.2]** Create configuration import and runtime parameter override unit tests in `tests/test_config.py` (Verify parameter overrides function properly in a non-JIT environment).
