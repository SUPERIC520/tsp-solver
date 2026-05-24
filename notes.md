# TSP Solver Testing Notes

## [2026-05-22] - Initial Implementation & Benchmarks

### Sample Run (500 Cities)
- **Tech**: Basic 2-opt + Delaunay Candidate Set.
- **Time**: ~8.12s.
- **Gap**: ~35.8%.
- **Observation**: Initial pipeline verified. Preprocessing is fast (<0.5s).

### Optimized Sample Run (500 Cities)
- **Tech**: Optimized 2-opt with Don't Look Bits + Numba Cache.
- **Time**: ~1.92s.
- **Gap**: ~43.3% (Note: Hilbert seeds transformed, causing variation).
- **Observation**: Numba cache and DLB significantly improved execution speed.

### Enhanced LKH Core (2.5-opt) - Sample (500 Cities)
- **Tech**: 2-opt + Node Moves (3-opt subset) + Merged Candidate Set (Delaunay + KDTree).
- **Time**: ~4.31s.
- **Gap**: 18.70%.
- **Observation**: 2.5-opt significantly reduces the gap compared to pure 2-opt.

### Full Scale Run (115,475 Cities) - Partial
- **Held-Karp Lower Bound**: 6,111,040.82 (Computed in 14.84s).
- **Iteration 1 Results**:
    - Best Length: 6,861,405.37
    - Gap (Iter 1): 12.28%
    - Time (Iter 1): 10.43s (Parallel solve).
- **Observation**: Pre-computed HK bound saved. Iterative refinement (Backbone) showing promise but requires full execution.

## [2026-05-22] - Continued Optimization

### Refined 2.5-opt Sample Run (500 Cities)
- **Tech**: Optimized 2.5-opt with DLB + Numba Cache.
- **Time**: ~1.89s.
- **Gap**: 18.70%.
- **Observation**: Improved efficiency while maintaining quality. Readiness for full-scale iterative run confirmed.

### Full Scale Run (115,475 Cities) - 10 Iterations
- **Held-Karp Lower Bound**: 6,111,040.82 (Cached).
- **Final Results**:
    - Best Length: 6,861,405.37 (Found in Iteration 1).
    - Final Gap: 12.28%.
    - Total Time: 89.83s.
- **Iteration Details**:
    - Iteration 1: 0 locked edges, Best Length 6,861,405.37.
    - Iteration 10: 79,676 locked edges, Best Length 6,861,405.37.
- **Observations**: 
    1. The solver reached a local optimum in Iteration 1 and failed to improve further.
    2. Backbone consensus successfully locked ~70% of edges but did not lead to improvements, suggesting the consensus was formed around a suboptimal backbone.
    3. 2.5-opt (2-opt + node moves) is insufficient for the <1% target gap at this scale.
    4. Execution speed is very high (under 10s per iteration for 115k cities), allowing for more complex optimization logic.

## [2026-05-22] - Iterated Local Search (ILS) Implementation

### ILS Sample Run (500 Cities)
- **Tech**: 2.5-opt + Double Bridge Kick (50 kicks per seed).
- **Time**: ~6.45s.
- **Gap**: 12.74% (vs 18.70% previously).
- **Observation**: ILS effectively escapes local optima, reducing the gap significantly. Execution time remains within the 10s sample limit. Ready for full scale execution.

### High-Kick ILS Sample Run (500 Cities)
- **Tech**: 2.5-opt + Double Bridge Kick (200 kicks per seed).
- **Time**: ~6.65s (Numba cache helped keep this fast).
- **Gap**: 11.43% (Improvement from 12.74%).
- **Observation**: Increasing kick count continues to improve solution quality while remaining within time constraints.

## [2026-05-22] - Advanced Optimization (3-opt & Restructuring)

### Restructuring
- Restructured `src/` into `core/`, `utils/`, `scripts/`.
- Updated all imports and verified with `pytest`.

### 3-opt Implementation - Sample (500 Cities)
- **Tech**: 2-opt + Node Moves + 3-opt + 200 Kicks.
- **Time**: ~86s.
- **Gap**: 8.97%.
- **Observation**: 3-opt provides a quality boost but is significantly slower ($O(N \cdot k^2)$). Execution time for 500 cities (86s) indicates that for 115k cities, this will be prohibitively slow unless optimized or the kick count is reduced.

### Full Scale Run (115,475 Cities) - ILS (50 kicks) - 10 Iterations
- **Held-Karp Lower Bound**: 6,111,040.82 (Cached).
- **Final Results**:
    - Best Length: 6,772,628.45 (Found in Iteration 1).
    - Final Gap: 10.83% (Improvement over 12.28%).
    - Total Time: 168.71s.
- **Observations**: 
    1. ILS with 50 kicks improved the initial local optimum by ~1.3% compared to pure 2.5-opt.
    2. Again, the best solution was found in Iteration 1. Subsequent iterations with locked edges failed to find improvements.
    3. The backbone mechanism (80% threshold) is likely locking in suboptimal edges too early, preventing ILS from exploring better regions.
    4. Execution time is still very reasonable (~15-20s per iteration), suggesting room for much more intensive ILS or higher kick counts.

## [2026-05-22 16:00] - Strategy Refinement: Full Cascading K-Opt (2/3/4/5-opt)

### New Objective
- **Target Gap**: < 5% from Held-Karp.
- **Engine**: Full Cascading K-Opt implementation (2, 3, 4, 5-opt).
- **Neighbors**: Top 16 from Delaunay triangulation.
- **Seeds**: Max 8.
- **Caching**: Held-Karp bound and Pi vector caching mandatory.

### Rationale
- "Cascading" logic means trying moves in order of complexity: if 2-opt fails, try 3-opt, then 4-opt, then 5-opt.
- This approach is more thorough than just 2-opt + kicks, and can reach better local optima.
- 5% gap is the new target, allowing more flexibility in execution time while maintaining high quality.
- Euclidean pruning remains critical for keeping the search efficient at higher K.

### Planned Implementation
1. `_optimize_2opt`: Standard 2-opt (done).
2. `_optimize_3opt`: Sequential 3-opt search.
3. `_optimize_4opt`: Sequential 4-opt search (non-reducible).
4. `_optimize_5opt`: Sequential 5-opt search.
5. `cascading_kopt_step`: Orchestrates the call to these functions.

## [2026-05-22 17:30] - Initial Performance Validation

### Cascading K-Opt Engine - Performance Validation (N=500)
- **Tech**: Cascading K-Opt (2-opt + Sequential 3-opt) + ILS (1000 kicks) + 8 Hilbert Seeds.
- **Time**: 1.24s (Solve).
- **Gap**: 3.7382%.
- **Kick Count**: 1000.
- **Result**: **SUCCESS**. Meets <10s and <5% gap requirements.

### Cascading K-Opt Engine - Baseline (N=100)
- **Time**: 1.04s (Solve).
- **Gap**: 3.2847%.
- **Observation**: N=100 remains stable with the new engine.

- Need to debug since n=100 freezes on the second iteration. we need the iterations to improve

### Cascading K-Opt Engine --n 500 --seeds 4 --kicks 200 --iters 1 --max_opt 3 
- **Time**: 1.15s (Solve).
- **Gap**: 3.3204% 
- **Observation**: N=500 remains stable with the new engine.

### Todo
- Need to debug since code freezes on the second iteration when iter > 1.
- Test integration on small sample size to fix freezing
- Estimate runtime of larger samples before running based on the time complexity of cascading k-opt.
- With each experiment, store params and runtime in a runtime.csv under docs.
- Use the expected time complexity as the max degree / terms to make a regression model to see roughly how long it should take to run for a given param.
## [2026-05-23 14:25] - Benchmark N=100 (Refined)
- **Params**: seeds=2, kicks=10, iterations=2, hk_iter=2000, max_opt=3
- **Results**: Gap=1.9226%, LB=27175.81, Best=27698.29, Time=1.70s
- **Status**: FAILURE

## [2026-05-23 15:07] - Benchmark N=100 (Refined)
- **Params**: seeds=4, kicks=100, iterations=2, hk_iter=2000, max_opt=3
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=1.84s
- **Status**: SUCCESS

## [2026-05-23 15:07] - Benchmark N=500 (Refined)
- **Params**: seeds=4, kicks=200, iterations=2, hk_iter=2000, max_opt=3
- **Results**: Gap=3.1416%, LB=54426.68, Best=56136.54, Time=1.88s
- **Status**: FAILURE

## [2026-05-23 15:08] - Benchmark N=1000 (Refined)
- **Params**: seeds=4, kicks=200, iterations=2, hk_iter=2000, max_opt=3
- **Results**: Gap=3.4066%, LB=77728.46, Best=80376.36, Time=1.92s
- **Status**: FAILURE

## [2026-05-23 15:32] - Benchmark N=100 (Refined)
- **Params**: seeds=4, kicks=100, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=1.54s
- **Status**: SUCCESS

## [2026-05-23 15:32] - Benchmark N=500 (Refined)
- **Params**: seeds=4, kicks=200, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=3.2655%, LB=54426.68, Best=56203.99, Time=1.52s
- **Status**: SUCCESS

## [2026-05-23 15:34] - Benchmark N=1000 (Refined)
- **Params**: seeds=4, kicks=200, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=4.5585%, LB=77728.46, Best=81271.75, Time=1.82s
- **Status**: SUCCESS

## [2026-05-23 15:35] - Benchmark N=5000 (Refined)
- **Params**: seeds=4, kicks=500, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=7.7270%, LB=251684.88, Best=271132.63, Time=2.85s
- **Status**: FAILURE

## [2026-05-23 15:35] - Benchmark N=10000 (Refined)
- **Params**: seeds=4, kicks=500, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=11.3525%, LB=615978.91, Best=685907.77, Time=5.30s
- **Status**: FAILURE

## [2026-05-23 15:35] - Full Scale Bench N=100
- **Params**: seeds=2, kicks=50, iterations=1, hk_iter=2000, max_opt=3, processes=2, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=1.3073%, LB=27175.81, Best=27531.07, Time=1.58s
- **Status**: SUCCESS

## [2026-05-23 15:35] - full_scale_bench.py Changes Applied
- num_kicks default: 2000 ? 3000
- num_iterations default: 2 ? 1
- Added --no_backbone flag (skips backbone consensus when set)
- Added --backbone_threshold arg (default 0.99)
- Added estimated time print before optimization loop
- N=100 sanity check: Gap=1.3073%, Time=1.58s ? SUCCESS

## [2026-05-23 15:36] - Benchmark N=5000 (Refined)
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=4.4045%, LB=251684.88, Best=262770.42, Time=9.67s
- **Status**: SUCCESS

## [2026-05-23 15:36] - Key Discovery: Kick Intensity Critical for Gap Reduction

### Observation
Low kick counts (4 seeds, 500 kicks) leave massive time budget unused:
- N=5000: 2.85s used of 240s budget → 7.73% gap
- N=10000: 5.30s used of 360s budget → 11.35% gap

Fixing with 8 seeds and 5000 kicks:
- N=5000: 9.67s used → **4.40% gap** (SUCCESS!)

### Critical Bug Fix Applied (2026-05-23 15:30)
- **Or-opt `return globally_improved` was missing** — the or-opt function never signaled improvement to the main cascade loop. This meant or-opt was running but its improvements were silently discarded.
- **Fix**: Added proper `return globally_improved` + DLB update at function end
- **Impact**: Major improvement expected at all scales since or-opt segment relocation moves are now properly enabled

### Strategy for Full Scale (N=115,475)
- Use many kicks: target 1000-3000 per seed within 600s wall time
- Use 8 seeds in parallel (8 processes)
- HK bound is cached from previous runs (6,111,040.82)
- Expected gap with fixes: 6-8% initially, targeting < 5% with enough kicks

## [2026-05-23 15:36] - Benchmark N=5000 (Refined)
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=4.5926%, LB=251684.88, Best=263243.64, Time=9.77s
- **Status**: SUCCESS


## [2026-05-23 15:37] - Benchmark N=10000 (Refined)
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=7.5925%, LB=615978.91, Best=662747.36, Time=23.27s
- **Status**: FAILURE

## [2026-05-23 15:38] - Benchmark N=10000 (Refined)
- **Params**: seeds=8, kicks=10000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=6.6166%, LB=615978.91, Best=656735.50, Time=67.04s
- **Status**: FAILURE

## [2026-05-23 15:40] - Scaling Analysis: N=10000 Bottleneck

### Gap vs. Kicks Trend (N=10000, 8 seeds)
| Kicks | Gap% | Time(s) | Budget Used |
|-------|------|---------|-------------|
| 500   | 11.35% | 5.3   | 1.5%        |
| 5000  | 7.59%  | 23.3  | 6.5%        |
| 10000 | 6.62%  | 67.0  | 18.6%       |

Sub-linear improvement: doubling kicks from 5k→10k reduced gap by only 0.97pp but time grew 2.9x.

### Critical Bug #2 Found: DLB Not Reset After Stagnation
In `cascading_kopt_optimize`, the stagnation handler resets `tour = best_tour.copy()` but does NOT clear `dlb`. After the reset, almost all nodes are still marked as "don't look", severely limiting local search quality on the fresh starting point.

**Fix applied**: Added `dlb.fill(False)` after `tour = best_tour.copy()` in stagnation handler.

### Implications for Full Scale
- Time scales ~O(N) with cities (super-linear due to candidate set density)
- N=115k is ~11.5x harder than N=10000
- 1000 kicks at N=115k ≈ 77s (fits in budget)
- Expected gap with 1000 kicks: ~8-9% before DLB fix
- With DLB fix: potentially 5-7% improvement per kick → may reach 5-6%
- Need additional algorithmic improvements to reach < 5%

## [2026-05-23 15:41] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=1000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=True
- **Results**: Gap=9.9568%, LB=6158023.27, Best=6771167.63, Time=250.63s
- **Status**: FAILURE

## [2026-05-23 15:42] - Benchmark N=100 (Refined)
- **Params**: seeds=4, kicks=100, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=1.14s
- **Status**: SUCCESS

## [2026-05-23 15:42] - Benchmark N=500 (Refined)
- **Params**: seeds=4, kicks=500, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=3.5362%, LB=54426.68, Best=56351.33, Time=1.74s
- **Status**: SUCCESS

## [2026-05-23 15:42] - Benchmark N=10000 (Refined)
- **Params**: seeds=4, kicks=2000, iterations=1, hk_iter=2000, max_opt=2
- **Results**: Gap=9.6026%, LB=615978.91, Best=675129.19, Time=13.56s
- **Status**: FAILURE

## [2026-05-23 15:43] - Benchmark N=10000 (Refined)
- **Params**: seeds=4, kicks=2000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=9.1065%, LB=615978.91, Best=672072.73, Time=13.82s
- **Status**: FAILURE

## [2026-05-23 15:45] - Benchmark N=10000 (Refined)
- **Params**: seeds=8, kicks=10000, iterations=1, hk_iter=2000, max_opt=2
- **Results**: Gap=6.8629%, LB=615978.91, Best=658253.10, Time=71.81s
- **Status**: FAILURE

## [2026-05-23 15:46] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=1000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=True
- **Results**: Gap=10.1210%, LB=6158023.27, Best=6781277.75, Time=254.78s
- **Status**: FAILURE

## [2026-05-23 17:45] - Benchmark N=100 (Refined)
- **Params**: seeds=4, kicks=100, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=1.10s
- **Status**: SUCCESS

## [2026-05-23 17:46] - Benchmark N=500 (Refined)
- **Params**: seeds=4, kicks=500, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=3.1955%, LB=54426.68, Best=56165.90, Time=1.16s
- **Status**: SUCCESS

## [2026-05-23 17:48] - Benchmark N=1000 (Refined)
- **Params**: seeds=4, kicks=500, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=3.7867%, LB=77728.46, Best=80671.77, Time=13.74s
- **Status**: SUCCESS

## [2026-05-23 17:50] - Benchmark N=5000 (Refined)
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=4.4822%, LB=251684.88, Best=262965.90, Time=21.13s
- **Status**: SUCCESS

## [2026-05-23 17:54] - Benchmark N=10000 (Refined)
- **Params**: seeds=8, kicks=20000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=6.0137%, LB=615978.91, Best=653022.16, Time=229.76s
- **Status**: FAILURE

## [2026-05-23 17:55] - Benchmark N=10000 (Refined)
- **Params**: seeds=8, kicks=10000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=6.7991%, LB=615978.91, Best=657859.85, Time=45.11s
- **Status**: FAILURE

## [2026-05-23 18:01] - Benchmark N=10000 (Refined)
- **Params**: seeds=8, kicks=20000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=6.0518%, LB=615978.91, Best=653256.71, Time=81.85s
- **Status**: FAILURE

## [2026-05-23 18:02] - Benchmark N=10000 (Refined)
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=4
- **Results**: Gap=8.0619%, LB=615978.91, Best=665638.58, Time=22.59s
- **Status**: FAILURE

## [2026-05-23 18:06] - Benchmark N=10000 (Refined)
- **Params**: seeds=12, kicks=30000, iterations=1, hk_iter=2000, max_opt=4
- **Results**: Gap=5.5775%, LB=615978.91, Best=650335.03, Time=205.50s
- **Status**: FAILURE

## [2026-05-23 18:45] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=1000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=6.4635%, LB=251684.88, Best=267952.62, Time=10.26s
- **Status**: FAILURE

## [2026-05-23 18:45] - Full Scale Bench N=10000
- **Params**: seeds=8, kicks=1000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=10.3694%, LB=615978.91, Best=679852.05, Time=10.28s
- **Status**: FAILURE

## [2026-05-23 18:45] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=2000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.5130%, LB=251684.88, Best=265560.33, Time=10.27s
- **Status**: FAILURE

## [2026-05-23 19:00] - Parameter Exploration N=5000

### Goal: Reach Gap < 5% within 300s
Explored different kick counts, max_opt levels, and candidate set (CS) sizes.

| Config | Seeds | Kicks | Max Opt | CS | Gap | Time | Status |
|--------|-------|-------|---------|----|-----|------|--------|
| 1      | 8     | 2000  | 4       | 40 | 5.8234% | 10.23s | FAILURE |
| 2      | 8     | 3000  | 3       | 40 | 5.2912% | 10.27s | FAILURE |
| 1b     | 8     | 2000  | 4       | 64 | 5.9726% | 10.27s | FAILURE |
| 2b     | 8     | 3000  | 3       | 64 | 5.2133% | 20.27s | FAILURE |
| 3      | 8     | 5000  | 3       | 64 | 4.4356% | 20.27s | **SUCCESS** |
| 4      | 8     | 4000  | 4       | 64 | 4.6279% | 10.27s | **SUCCESS** |

### Observations
1. **Kick count** is the primary driver for gap reduction. 5000 kicks consistently reaches < 5% for N=5000.
2. **Max Opt 4** shows some benefit but is not a silver bullet; it's more effective when combined with sufficient kicks.
3. **Candidate Set (CS)** increase to 64 provides a small quality boost (e.g., in config 2 vs 2b) but increases solve time per kick.
4. **Timing Granularity**: `parallel_solve` heartbeat at 10s makes small-run timing noisy. Actual work time is likely < 5s for most N=5000 seeds.

### Successful Configuration for N=5000
- **Params**: `--n 5000 --seeds 8 --kicks 5000 --max_opt 3` (or 4)
- **Result**: Gap ~4.4%, Time ~20s (well within 300s budget).

## [2026-05-23 18:46] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=2000, iterations=1, hk_iter=2000, max_opt=4, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.8234%, LB=251684.88, Best=266341.62, Time=10.23s
- **Status**: FAILURE

## [2026-05-23 18:47] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=3000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.2912%, LB=251684.88, Best=265002.07, Time=10.27s
- **Status**: FAILURE

## [2026-05-23 18:47] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=2000, iterations=1, hk_iter=2000, max_opt=4, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.9726%, LB=251684.88, Best=266717.02, Time=10.27s
- **Status**: FAILURE

## [2026-05-23 18:48] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=3000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.2133%, LB=251684.88, Best=264806.05, Time=20.27s
- **Status**: FAILURE

## [2026-05-23 18:48] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=4.4356%, LB=251684.88, Best=262848.69, Time=20.27s
- **Status**: SUCCESS

## [2026-05-23 18:48] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=4000, iterations=1, hk_iter=2000, max_opt=4, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=4.6279%, LB=251684.88, Best=263332.54, Time=10.27s
- **Status**: SUCCESS
