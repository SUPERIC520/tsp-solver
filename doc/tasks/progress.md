# TSP Solver Refactoring Progress

This document tracks the overall refactoring progress. Each module contains a checklist of atomic tasks that can be accessed in detail via their respective Markdown task files.

---

## 1. Module Tasks Summary

- [x] **Global Configuration Module** (Details: [config.md](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/doc/tasks/config.md))
  - [x] [T1.1] Create configuration file `src/config.py` with default parameters.
  - [x] [T1.2] Implement configuration override validation tests.

- [x] **Preprocessing & Alignment Module** (Details: [preprocessing.md](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/doc/tasks/preprocessing.md))
  - [x] [T2.1] Clear Delaunay filtering remnants.
  - [x] [T2.2] Implement 64-byte alignment check helper.
  - [x] [T2.3] Refine `hilbert_reorder_cities` for aligned contiguous data.
  - [x] [T2.4] Refine `build_candidate_sets` KD-Tree neighbor matrices.
  - [x] [T2.5] Refine `refine_candidate_set_with_alpha` with C-contiguity and alignment.
  - [x] [T2.6] Implement preprocessing correctness unit tests.

- [x] **Seed Generation & Rotation Module** (Details: [seed-generation.md](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/doc/tasks/seed-generation.md))
  - [x] [T3.1] Remove Hilbert and Random seeds standalone methods.
  - [x] [T3.2] Implement path cycle rotation function `rotate_tour`.
  - [x] [T3.3] Implement parallel Greedy NN seeding using `multiprocessing.Pool`.
  - [x] [T3.4] Implement seed generation unit tests.

- [ ] **K-Opt Engine Module** (Details: [kopt-engine.md](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/doc/tasks/kopt-engine.md))
  - [x] [T4.1] Delete `locked_edges` parameters from K-opt engine completely.
  - [x] [T4.2] Optimize distance precomputations with JIT-parallel SIMD directives.
  - [x] [T4.3] Refine `_optimize_2opt` using C-contiguous aligned arrays and DLB.
  - [x] [T4.4] Implement `_optimize_or_opt` relocate swaps.
  - [x] [T4.5] Refine `_optimize_3opt_sequential` implementing dynamic funneling (`K_3OPT`) and gain pruning.
  - [x] [T4.6] Refine `_optimize_4opt_sequential` implementing dynamic funneling (`K_3OPT`, `K_4OPT`).
  - [x] [T4.7] Refine `_optimize_5opt_sequential` implementing dynamic funneling (`K_3OPT`, `K_4OPT`, `K_5OPT`).
  - [x] [T4.8] Implement sequential cascade manager `_full_cascade` running up to 5-opt.
  - [x] [T4.9] Implement `_cascading_kopt_inner` ILS JIT chunk kick loops.
  - [x] [T4.10] Implement Python wrapper `cascading_kopt_optimize` asserting 64-byte alignments and return precise kick counts.
  - [x] [T4.11] Implement local search operators correctness unit tests.

- [x] **Orchestration Module** (Details: [orchestration.md](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/doc/tasks/orchestration.md))
  - [x] [T5.1] Replace Manager proxy list with multiprocessing Array shared memory.
  - [x] [T5.2] Setup worker allocations under `NUM_PROCESSES_SOLVER`.
  - [x] [T5.3] Implement pool managers catching worker JIT errors and time budgets.
  - [x] [T5.4] Implement orchestration unit tests.

- [ ] **Validation & Held-Karp Module** (Details: [validation.md](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/doc/tasks/validation.md))
  - [x] [T6.1] Clean up `locked_edges` parameters from validation subroutines.
  - [x] [T6.2] Optimize MST weight calculations and subgradients with JIT-parallel.
  - [x] [T6.3] Implement JIT-parallel Alpha value estimation.
  - [x] [T6.4] Implement validation unit tests.

- [x] **Data I/O & Persistence Module** (Details: [data-io.md](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/doc/tasks/data-io.md))
  - [x] [T7.1] Implement CSV tour length loading `load_best_length_from_csv`.
  - [x] [T7.2] Restructure `update_best_tour` with `is_full_run` conditional checks.
  - [x] [T7.3] Fix and run I/O verification unit tests.

- [x] **Main Pipeline & Integration Module** (Details: [main.md](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/doc/tasks/main.md))
  - [x] [T8.1] Connect all refactored parts inside main entry script.
  - [x] [T8.2] Execute progressive scale checks (N=100 -> 500 -> 1000 -> 5000 -> 115,475) to achieve Gap < 5% and runtime < 10 mins.
