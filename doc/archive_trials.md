# TSP Solver Benchmark Trials

This file contains the individual benchmark trial logs detailing specific parameter lists, results (Gap, LB, Best, Time), and success/failure statuses.

## [2026-05-22] - Sample Run (500 Cities)
- **Params**: Tech: Basic 2-opt + Delaunay Candidate Set.
- **Results**: Gap=~35.8%, Time=~8.12s
- **Status**: SUCCESS

## [2026-05-22] - Optimized Sample Run (500 Cities)
- **Params**: Tech: Optimized 2-opt with Don't Look Bits + Numba Cache.
- **Results**: Gap=~43.3% (Hilbert seeds variation), Time=~1.92s
- **Status**: SUCCESS

## [2026-05-22] - Enhanced LKH Core (2.5-opt) - Sample (500 Cities)
- **Params**: Tech: 2-opt + Node Moves (3-opt subset) + Merged Candidate Set (Delaunay + KDTree).
- **Results**: Gap=18.70%, Time=~4.31s
- **Status**: SUCCESS

## [2026-05-22] - Full Scale Run (115,475 Cities) - Partial
- **Params**: Held-Karp Lower Bound: 6,111,040.82
- **Results**: Gap (Iter 1)=12.28%, Best Length=6,861,405.37, Time (Iter 1)=10.43s
- **Status**: SUCCESS

## [2026-05-22] - Refined 2.5-opt Sample Run (500 Cities)
- **Params**: Tech: Optimized 2.5-opt with DLB + Numba Cache.
- **Results**: Gap=18.70%, Time=~1.89s
- **Status**: SUCCESS

## [2026-05-22] - Full Scale Run (115,475 Cities) - 10 Iterations
- **Params**: Held-Karp Lower Bound: 6,111,040.82 (Cached), 10 Iterations
- **Results**: Final Gap=12.28%, Best Length=6,861,405.37, Total Time=89.83s
- **Status**: SUCCESS (Plateaued)

## [2026-05-22] - ILS Sample Run (500 Cities)
- **Params**: Tech: 2.5-opt + Double Bridge Kick (50 kicks per seed).
- **Results**: Gap=12.74% (vs 18.70%), Time=~6.45s
- **Status**: SUCCESS

## [2026-05-22] - High-Kick ILS Sample Run (500 Cities)
- **Params**: Tech: 2.5-opt + Double Bridge Kick (200 kicks per seed).
- **Results**: Gap=11.43%, Time=~6.65s
- **Status**: SUCCESS

## [2026-05-22] - 3-opt Implementation - Sample (500 Cities)
- **Params**: Tech: 2-opt + Node Moves + 3-opt + 200 Kicks.
- **Results**: Gap=8.97%, Time=~86s
- **Status**: SUCCESS

## [2026-05-22] - Full Scale Run (115,475 Cities) - ILS (50 kicks) - 10 Iterations
- **Params**: Held-Karp Lower Bound: 6,111,040.82 (Cached), 10 Iterations, 50 kicks
- **Results**: Final Gap=10.83%, Best Length=6,772,628.45, Total Time=168.71s
- **Status**: SUCCESS

## [2026-05-22 17:30] - Cascading K-Opt Engine - Performance Validation (N=500)
- **Params**: Tech: Cascading K-Opt (2-opt + Sequential 3-opt) + ILS (1000 kicks) + 8 Hilbert Seeds.
- **Results**: Gap=3.7382%, Time=1.24s
- **Status**: SUCCESS

## [2026-05-22 17:30] - Cascading K-Opt Engine - Baseline (N=100)
- **Params**: Cascading K-Opt Engine, N=100
- **Results**: Gap=3.2847%, Time=1.04s
- **Status**: SUCCESS (Stables)

## [2026-05-22 17:30] - Cascading K-Opt Engine --n 500 --seeds 4 --kicks 200 --iters 1 --max_opt 3
- **Params**: seeds=4, kicks=200, iterations=1, max_opt=3
- **Results**: Gap=3.3204%, Time=1.15s
- **Status**: SUCCESS

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

## [2026-05-23 15:36] - Benchmark N=5000 (Refined)
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=4.4045%, LB=251684.88, Best=262770.42, Time=9.67s
- **Status**: SUCCESS

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

## [2026-05-23 19:26] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=3000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=8.5104%, LB=6158023.27, Best=6682097.29, Time=96.22s
- **Status**: FAILURE

## [2026-05-23 19:34] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=15000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.7933%, LB=6158023.27, Best=6514774.90, Time=459.26s
- **Status**: FAILURE

## [2026-05-24 02:16] - Sequential Scale-up (Heartbeat Polling Optimized)
- **N=100**: seeds=4, kicks=100, max_opt=3 | Gap=0.7346%, Time=1.14s | SUCCESS
- **N=500**: seeds=4, kicks=500, max_opt=3 | Gap=3.0322%, Time=1.38s | SUCCESS
- **N=1000**: seeds=4, kicks=1000, max_opt=3 | Gap=2.2085%, Time=1.34s | SUCCESS
- **N=5000**: seeds=8, kicks=5000, max_opt=3 | Gap=3.4126%, Time=8.10s | SUCCESS
- **N=10000**: seeds=8, kicks=5000, max_opt=3 | Gap=6.3252%, Time=15.39s | FAILURE

## [2026-05-24 02:17] - Scale-up Benchmark N=10000 (Localized Kicks)
- **Params**: seeds=8, kicks=20000, iterations=1, max_opt=3
- **Results**: Gap=4.9055%, LB=615978.91, Best=646195.64, Time=41.48s
- **Status**: SUCCESS (Meets the < 5% gap and < 150s limit)

## [2026-05-24 02:17] - Parallel CS=64 Tuning Runs (N=10000)
- **kicks=5000, CS=64**: Gap=7.6659%, Time=37.01s | FAILURE
- **kicks=10000, CS=64**: Gap=6.9355%, Time=60.47s | FAILURE

## [2026-05-24 02:24] - Delaunay vs KDTree Neighbor Count Study (N=5000, seeds=4, kicks=2000)
- **Delaunay + KDTree, CS=40**: Gap=5.7864%, Time=5.48s (1.00x speed)
- **Delaunay + KDTree, CS=16**: Gap=6.5896%, Time=3.63s (1.51x speed)
- **Delaunay-Only, CS=16**: Gap=7.1004%, Time=2.55s (2.15x speed)
- **Delaunay-Only, CS=8**: Gap=7.9859%, Time=2.68s (2.04x speed)

## [2026-05-24 02:35] - KDTree-Only vs. Both Neighbor Study (N=5000, seeds=4, kicks=2000, max_opt=3)
- **Condition 1 (Baseline - Both, CS=40)**: Gap=5.8285%, Solve Time=3.86s, Preprocessing=0.05s
- **Condition 2 (KDTree-Only, CS=40)**: Gap=5.7991%, Solve Time=3.65s, Preprocessing=0.02s
- **Condition 3 (KDTree-Only, CS=16)**: Gap=6.2815%, Solve Time=3.15s, Preprocessing=0.01s

## [2026-05-23 20:29] - Benchmark N=100 (Refined)
- **Params**: seeds=4, kicks=500, iterations=5, hk_iter=2000, max_opt=3
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=8.59s
- **Status**: SUCCESS

## [2026-05-23 20:56] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=25000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.0523%, LB=6158023.27, Best=6469142.50, Time=637.20s
- **Status**: FAILURE

## [2026-05-23 20:59] - Benchmark N=100 (Refined)
- **Params**: seeds=4, kicks=200, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=1.33s
- **Status**: SUCCESS

## [2026-05-23 21:09] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=50000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.1444%, LB=6158023.27, Best=6474815.68, Time=586.88s
- **Status**: FAILURE

## [2026-05-23 21:14] - Full Scale Bench N=100
- **Params**: seeds=6, kicks=100, iterations=1, hk_iter=2000, max_opt=3, processes=6, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=2.72s
- **Status**: SUCCESS

## [2026-05-23 21:25] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=25000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=4.5470%, LB=6158023.27, Best=6438031.25, Time=635.86s
- **Status**: SUCCESS

## [2026-05-24 03:29] - Multi-Iteration Freeze-Fix Verification (N=100, 5 iters)
- **Params**: seeds=4, kicks=500, iterations=5, max_opt=3
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=8.59s, Locked=50 edges/iter
- **Status**: SUCCESS — No freeze observed; all 5 iterations completed; backbone consensus working correctly
- **Note**: Confirmed the _optimize_or_opt double-reversal fix resolves the multi-iteration deadlock

## [2026-05-24 03:29] - Full Scale Bench N=115475 (2 iters, 3,500 kicks) [CANCELLED]
- **Params**: seeds=8, kicks=3500, iterations=2, max_opt=3, no_backbone=False
- **Results**: N/A -- cancelled after >16 minutes (exceeds 10-min budget)
- **Status**: CANCELLED
- **Reason**: Backbone-constrained Iteration 2 ran ~2.5x slower than Iteration 1 at N=115k due to O(N*k^2) worst-case proof-of-optimality scans on locked edges. Multi-iteration backbone is not feasible at this scale within a 600s budget.

## [2026-05-24 04:27] - Full Scale Bench N=115475 (mixed seeds + time_limit=595) [DEFINITIVE SUCCESS]
- **Params**: seeds=8 (4 Hilbert + 4 greedy-NN), kicks=25000, iterations=1, max_opt=3, time_limit=595
- **Results**: Gap=**4.5632%**, LB=6158023.27, Best=6439023.63, Time=588.15s
- **Status**: **SUCCESS** (Gap < 5% AND Time < 600s both achieved!)
- **Seed results**: 6476896, 6441687, 6479273, 6483625, 6490618, 6444479, **6439023** (best), 6443814
- **Note**: Time limit cut ~3,000 kicks (from 25k to ~22k effective) but greedy NN basin is good enough that the result is still well under 5% gap


## [2026-05-23 22:03] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=3.1415%, LB=251684.88, Best=259591.56, Time=5.45s
- **Status**: SUCCESS

## [2026-05-23 22:03] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=4, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=3.2122%, LB=251684.88, Best=259769.52, Time=5.46s
- **Status**: SUCCESS

## [2026-05-23 22:03] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=5, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=3.2235%, LB=251684.88, Best=259797.94, Time=5.47s
- **Status**: SUCCESS

## [2026-05-23 22:04] - Full Scale Bench N=5000
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=3.1154%, LB=251684.88, Best=259525.87, Time=5.41s
- **Status**: SUCCESS

## [2026-05-23 22:04] - Full Scale Bench N=10000
- **Params**: seeds=4, kicks=2000, iterations=1, hk_iter=2000, max_opt=3, processes=4, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=7.7069%, LB=615978.91, Best=663451.83, Time=5.20s
- **Status**: FAILURE

## [2026-05-23 22:04] - Full Scale Bench N=10000
- **Params**: seeds=4, kicks=2000, iterations=1, hk_iter=2000, max_opt=4, processes=4, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=7.8089%, LB=615978.91, Best=664080.31, Time=4.84s
- **Status**: FAILURE

## [2026-05-23 22:04] - Full Scale Bench N=10000
- **Params**: seeds=4, kicks=2000, iterations=1, hk_iter=2000, max_opt=5, processes=4, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=7.8389%, LB=615978.91, Best=664265.06, Time=4.86s
- **Status**: FAILURE

## [2026-05-23 22:09] - Full Scale Bench N=10000
- **Params**: seeds=8, kicks=100000, iterations=1, hk_iter=2000, max_opt=5, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=4.5340%, LB=615978.91, Best=643907.22, Time=223.29s
- **Status**: SUCCESS

## [2026-05-23 22:38] - Full Scale Bench N=10000
- **Params**: seeds=8, kicks=25000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.0550%, LB=615978.91, Best=647116.44, Time=46.84s
- **Status**: FAILURE

## [2026-05-23 22:39] - Full Scale Bench N=1000
- **Params**: seeds=8, kicks=25000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=1.3632%, LB=77728.46, Best=78788.06, Time=6.68s
- **Status**: SUCCESS

## [2026-05-23 22:45] - Full Scale Bench N=1000
- **Params**: seeds=8, kicks=25000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=0.9975%, LB=77728.46, Best=78503.83, Time=5.70s
- **Status**: SUCCESS

## [2026-05-24 00:06] - Full Scale Bench N=10000
- **Params**: seeds=12, kicks=15000, iterations=1, hk_iter=2000, max_opt=3, processes=12, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.2312%, LB=615978.91, Best=648201.69, Time=35.85s
- **Status**: FAILURE

## [2026-05-24 00:07] - Full Scale Bench N=10000
- **Params**: seeds=12, kicks=15000, iterations=1, hk_iter=2000, max_opt=4, processes=12, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.1960%, LB=615978.91, Best=647985.36, Time=36.87s
- **Status**: FAILURE

## [2026-05-24 00:08] - Full Scale Bench N=5000
- **Params**: seeds=12, kicks=25000, iterations=1, hk_iter=2000, max_opt=3, processes=12, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=2.3692%, LB=251684.88, Best=257647.82, Time=22.80s
- **Status**: SUCCESS

## [2026-05-24 19:07] - CS vs Max-K Grid Search (N=500, Kicks=500)
- **Params**: seeds=1, kicks=500, n=500, comparison between standard and cascading modes across CS size [8, 16, 32, 64] and Max-K [3, 4, 5, 6, 7].
- **Results**:
| CS | Max-K | Mode | Best Len | Time (s) | Gap (%) |
|---|---|---|---|---|---|
| 8 | 3 | standard | 55345.74 | 2.68 | 1.6886% |
| 8 | 3 | cascading | 55526.89 | 1.23 | 2.0215% |
| 8 | 4 | standard | 55435.41 | 4.36 | 1.8534% |
| 8 | 4 | cascading | 55456.00 | 1.19 | 1.8912% |
| 8 | 5 | standard | 55440.93 | 6.20 | 1.8635% |
| 8 | 5 | cascading | 55770.27 | 1.40 | 2.4686% |
| 8 | 6 | standard | 55314.78 | 9.37 | 1.6317% |
| 8 | 6 | cascading | 55461.63 | 1.24 | 1.9015% |
| 8 | 7 | standard | 55289.26 | 9.33 | 1.5848% |
| 8 | 7 | cascading | 55630.27 | 1.34 | 2.2114% |
| 16 | 3 | standard | 55290.08 | 3.33 | 1.5864% |
| 16 | 3 | cascading | 55854.22 | 1.82 | 2.6229% |
| 16 | 4 | standard | 55218.41 | 5.45 | 1.4547% |
| 16 | 4 | cascading | 55629.90 | 1.77 | 2.2107% |
| 16 | 5 | standard | 55300.60 | 7.64 | 1.6057% |
| 16 | 5 | cascading | 55579.95 | 1.86 | 2.1189% |
| 16 | 6 | standard | 55297.81 | 8.52 | 1.6006% |
| 16 | 6 | cascading | 55731.13 | 1.99 | 2.3967% |
| 16 | 7 | standard | 55255.62 | 12.58 | 1.5230% |
| 16 | 7 | cascading | 55655.06 | 1.79 | 2.2569% |
| 32 | 3 | standard | 55283.34 | 3.44 | 1.5740% |
| 32 | 3 | cascading | 55654.75 | 1.84 | 2.2564% |
| 32 | 4 | standard | 55293.88 | 4.94 | 1.5933% |
| 32 | 4 | cascading | 55530.14 | 2.11 | 2.0274% |
| 32 | 5 | standard | 55289.26 | 7.02 | 1.5848% |
| 32 | 5 | cascading | 55600.01 | 2.10 | 2.1558% |
| 32 | 6 | standard | 55293.88 | 8.66 | 1.5933% |
| 32 | 6 | cascading | 55615.26 | 1.89 | 2.1838% |
| 32 | 7 | standard | 55396.89 | 13.56 | 1.7826% |
| 32 | 7 | cascading | 55364.99 | 1.64 | 1.7240% |
| 64 | 3 | standard | 55294.07 | 3.18 | 1.5937% |
| 64 | 3 | cascading | 55350.80 | 1.81 | 1.6979% |
| 64 | 4 | standard | 55416.78 | 4.82 | 1.8191% |
| 64 | 4 | cascading | 55370.36 | 1.89 | 1.7339% |
| 64 | 5 | standard | 55459.36 | 6.75 | 1.8974% |
| 64 | 5 | cascading | 55733.25 | 1.80 | 2.4006% |
| 64 | 6 | standard | 55299.79 | 7.97 | 1.6042% |
| 64 | 6 | cascading | 55329.82 | 1.73 | 1.6594% |
| 64 | 7 | standard | 55407.21 | 10.74 | 1.8016% |
| 64 | 7 | cascading | 55437.54 | 1.66 | 1.8573% |
- **Status**: SUCCESS
# TSP Solver Benchmark Trials

This file contains the individual benchmark trial logs detailing specific parameter lists, results (Gap, LB, Best, Time), and success/failure statuses.

## [2026-05-22] - Sample Run (500 Cities)
- **Params**: Tech: Basic 2-opt + Delaunay Candidate Set.
- **Results**: Gap=~35.8%, Time=~8.12s
- **Status**: SUCCESS

## [2026-05-22] - Optimized Sample Run (500 Cities)
- **Params**: Tech: Optimized 2-opt with Don't Look Bits + Numba Cache.
- **Results**: Gap=~43.3% (Hilbert seeds variation), Time=~1.92s
- **Status**: SUCCESS

## [2026-05-22] - Enhanced LKH Core (2.5-opt) - Sample (500 Cities)
- **Params**: Tech: 2-opt + Node Moves (3-opt subset) + Merged Candidate Set (Delaunay + KDTree).
- **Results**: Gap=18.70%, Time=~4.31s
- **Status**: SUCCESS

## [2026-05-22] - Full Scale Run (115,475 Cities) - Partial
- **Params**: Held-Karp Lower Bound: 6,111,040.82
- **Results**: Gap (Iter 1)=12.28%, Best Length=6,861,405.37, Time (Iter 1)=10.43s
- **Status**: SUCCESS

## [2026-05-22] - Refined 2.5-opt Sample Run (500 Cities)
- **Params**: Tech: Optimized 2.5-opt with DLB + Numba Cache.
- **Results**: Gap=18.70%, Time=~1.89s
- **Status**: SUCCESS

## [2026-05-22] - Full Scale Run (115,475 Cities) - 10 Iterations
- **Params**: Held-Karp Lower Bound: 6,111,040.82 (Cached), 10 Iterations
- **Results**: Final Gap=12.28%, Best Length=6,861,405.37, Total Time=89.83s
- **Status**: SUCCESS (Plateaued)

## [2026-05-22] - ILS Sample Run (500 Cities)
- **Params**: Tech: 2.5-opt + Double Bridge Kick (50 kicks per seed).
- **Results**: Gap=12.74% (vs 18.70%), Time=~6.45s
- **Status**: SUCCESS

## [2026-05-22] - High-Kick ILS Sample Run (500 Cities)
- **Params**: Tech: 2.5-opt + Double Bridge Kick (200 kicks per seed).
- **Results**: Gap=11.43%, Time=~6.65s
- **Status**: SUCCESS

## [2026-05-22] - 3-opt Implementation - Sample (500 Cities)
- **Params**: Tech: 2-opt + Node Moves + 3-opt + 200 Kicks.
- **Results**: Gap=8.97%, Time=~86s
- **Status**: SUCCESS

## [2026-05-22] - Full Scale Run (115,475 Cities) - ILS (50 kicks) - 10 Iterations
- **Params**: Held-Karp Lower Bound: 6,111,040.82 (Cached), 10 Iterations, 50 kicks
- **Results**: Final Gap=10.83%, Best Length=6,772,628.45, Total Time=168.71s
- **Status**: SUCCESS

## [2026-05-22 17:30] - Cascading K-Opt Engine - Performance Validation (N=500)
- **Params**: Tech: Cascading K-Opt (2-opt + Sequential 3-opt) + ILS (1000 kicks) + 8 Hilbert Seeds.
- **Results**: Gap=3.7382%, Time=1.24s
- **Status**: SUCCESS

## [2026-05-22 17:30] - Cascading K-Opt Engine - Baseline (N=100)
- **Params**: Cascading K-Opt Engine, N=100
- **Results**: Gap=3.2847%, Time=1.04s
- **Status**: SUCCESS (Stables)

## [2026-05-22 17:30] - Cascading K-Opt Engine --n 500 --seeds 4 --kicks 200 --iters 1 --max_opt 3
- **Params**: seeds=4, kicks=200, iterations=1, max_opt=3
- **Results**: Gap=3.3204%, Time=1.15s
- **Status**: SUCCESS

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

## [2026-05-23 15:36] - Benchmark N=5000 (Refined)
- **Params**: seeds=8, kicks=5000, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=4.4045%, LB=251684.88, Best=262770.42, Time=9.67s
- **Status**: SUCCESS

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

## [2026-05-23 19:26] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=3000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=8.5104%, LB=6158023.27, Best=6682097.29, Time=96.22s
- **Status**: FAILURE

## [2026-05-23 19:34] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=15000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.7933%, LB=6158023.27, Best=6514774.90, Time=459.26s
- **Status**: FAILURE

## [2026-05-24 02:16] - Sequential Scale-up (Heartbeat Polling Optimized)
- **N=100**: seeds=4, kicks=100, max_opt=3 | Gap=0.7346%, Time=1.14s | SUCCESS
- **N=500**: seeds=4, kicks=500, max_opt=3 | Gap=3.0322%, Time=1.38s | SUCCESS
- **N=1000**: seeds=4, kicks=1000, max_opt=3 | Gap=2.2085%, Time=1.34s | SUCCESS
- **N=5000**: seeds=8, kicks=5000, max_opt=3 | Gap=3.4126%, Time=8.10s | SUCCESS
- **N=10000**: seeds=8, kicks=5000, max_opt=3 | Gap=6.3252%, Time=15.39s | FAILURE

## [2026-05-24 02:17] - Scale-up Benchmark N=10000 (Localized Kicks)
- **Params**: seeds=8, kicks=20000, iterations=1, max_opt=3
- **Results**: Gap=4.9055%, LB=615978.91, Best=646195.64, Time=41.48s
- **Status**: SUCCESS (Meets the < 5% gap and < 150s limit)

## [2026-05-24 02:17] - Parallel CS=64 Tuning Runs (N=10000)
- **kicks=5000, CS=64**: Gap=7.6659%, Time=37.01s | FAILURE
- **kicks=10000, CS=64**: Gap=6.9355%, Time=60.47s | FAILURE

## [2026-05-24 02:24] - Delaunay vs KDTree Neighbor Count Study (N=5000, seeds=4, kicks=2000)
- **Delaunay + KDTree, CS=40**: Gap=5.7864%, Time=5.48s (1.00x speed)
- **Delaunay + KDTree, CS=16**: Gap=6.5896%, Time=3.63s (1.51x speed)
- **Delaunay-Only, CS=16**: Gap=7.1004%, Time=2.55s (2.15x speed)
- **Delaunay-Only, CS=8**: Gap=7.9859%, Time=2.68s (2.04x speed)

## [2026-05-24 02:35] - KDTree-Only vs. Both Neighbor Study (N=5000, seeds=4, kicks=2000, max_opt=3)
- **Condition 1 (Baseline - Both, CS=40)**: Gap=5.8285%, Solve Time=3.86s, Preprocessing=0.05s
- **Condition 2 (KDTree-Only, CS=40)**: Gap=5.7991%, Solve Time=3.65s, Preprocessing=0.02s
- **Condition 3 (KDTree-Only, CS=16)**: Gap=6.2815%, Solve Time=3.15s, Preprocessing=0.01s

## [2026-05-23 20:29] - Benchmark N=100 (Refined)
- **Params**: seeds=4, kicks=500, iterations=5, hk_iter=2000, max_opt=3
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=8.59s
- **Status**: SUCCESS

## [2026-05-23 20:56] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=25000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.0523%, LB=6158023.27, Best=6469142.50, Time=637.20s
- **Status**: FAILURE

## [2026-05-23 20:59] - Benchmark N=100 (Refined)
- **Params**: seeds=4, kicks=200, iterations=1, hk_iter=2000, max_opt=3
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=1.33s
- **Status**: SUCCESS

## [2026-05-23 21:09] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=50000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=5.1444%, LB=6158023.27, Best=6474815.68, Time=586.88s
- **Status**: FAILURE

## [2026-05-23 21:14] - Full Scale Bench N=100
- **Params**: seeds=6, kicks=100, iterations=1, hk_iter=2000, max_opt=3, processes=6, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=2.72s
- **Status**: SUCCESS

## [2026-05-23 21:25] - Full Scale Bench N=115475
- **Params**: seeds=8, kicks=25000, iterations=1, hk_iter=2000, max_opt=3, processes=8, backbone_threshold=0.99, no_backbone=False
- **Results**: Gap=4.5470%, LB=6158023.27, Best=6438031.25, Time=635.86s
- **Status**: SUCCESS

## [2026-05-24 03:29] - Multi-Iteration Freeze-Fix Verification (N=100, 5 iters)
- **Params**: seeds=4, kicks=500, iterations=5, max_opt=3
- **Results**: Gap=0.7346%, LB=27175.81, Best=27375.45, Time=8.59s, Locked=50 edges/iter
- **Status**: SUCCESS — No freeze observed; all 5 iterations completed; backbone consensus working correctly
- **Note**: Confirmed the _optimize_or_opt double-reversal fix resolves the multi-iteration deadlock

## [2026-05-24 03:29] - Full Scale Bench N=115475 (2 iters, 3,500 kicks) [CANCELLED]
- **Params**: seeds=8, kicks=3500, iterations=2, max_opt=3, no_backbone=False
- **Results**: N/A -- cancelled after >16 minutes (exceeds 10-min budget)
- **Status**: CANCELLED
- **Reason**: Backbone-constrained Iteration 2 ran ~2.5x slower than Iteration 1 at N=115k due to O(N*k^2) worst-case proof-of-optimality scans on locked edges. Multi-iteration backbone is not feasible at this scale within a 600s budget.

## [2026-05-24 04:27] - Full Scale Bench N=115475 (mixed seeds + time_limit=595) [DEFINITIVE SUCCESS]
- **Params**: seeds=8 (4 Hilbert + 4 greedy-NN), kicks=25000, iterations=1, max_opt=3, time_limit=595
- **Results**: Gap=**4.5632%**, LB=6158023.27, Best=6439023.63, Time=588.15s
- **Status**: **SUCCESS** (Gap < 5% AND Time < 600s both achieved!)
- **Seed results**: 6476896, 6441687, 6479273, 6483625, 6490618, 6444479, **6439023** (best), 6443814
- **Note**: Time limit cut ~3,000 kicks (from 25k to ~22k effective) but greedy NN basin is good enough that the result is still well under 5% gap

## [2026-05-25 19:15] - Comprehensive Seeding Strategy Experiment (N=5,000, max_opt=5)

**Held-Karp Lower Bound**: 251684.88
**Goal**: Evaluate 16 different seeding and re-seeding configurations across 800 total trials.

### Initial Seeding Comparison (N=5,000, max_opt=5, kicks=500, iters=1)
| ID | Configuration                           | Trials | Avg Length | Avg Gap (%) | Std Dev | Min       | Max       | Avg Time (s) |
|----|-----------------------------------------|--------|------------|-------------|---------|-----------|-----------|--------------|
| 1  | 100% Hilbert                            | 49     | 268553.04  | 6.7021%     | 532.23  | 266698.02 | 269686.64 | 5.8          |
| 2  | 100% Random                             | 49     | 266644.01  | 5.9436%     | 534.25  | 265351.05 | 267803.11 | 5.8          |
| 3  | 100% Greedy NN                          | 49     | 266055.68  | 5.7098%     | 463.07  | 264591.98 | 266917.69 | 5.5          |
| 4  | 50% Hilbert + 50% Random                | 49     | 267098.09  | 6.1240%     | 584.68  | 265612.26 | 268572.40 | 5.4          |
| 5  | 50% Hilbert + 50% Greedy NN             | 49     | 266622.04  | 5.9349%     | 503.20  | 265349.92 | 267793.40 | 5.5          |
| 6  | 50% Random + 50% Greedy NN              | 49     | 266452.93  | 5.8677%     | 456.13  | 265233.17 | 267436.49 | 5.4          |
| 7  | 75% Hilbert + 25% Greedy NN             | 49     | 266764.84  | 5.9916%     | 514.09  | 265731.39 | 268133.16 | 5.6          |
| 8  | Balanced Mix (4H, 4R, 4G)               | 49     | 266551.69  | 5.9069%     | 491.77  | 265627.55 | 267724.10 | 5.5          |

### Re-seeding Strategy Comparison (N=5,000, max_opt=5, kicks=100, iters=5)
| ID | Configuration                           | Trials | Avg Length | Avg Gap (%) | Std Dev | Min       | Max       | Avg Time (s) |
|----|-----------------------------------------|--------|------------|-------------|---------|-----------|-----------|--------------|
| 9  | 100% Continuation (No Resets)           | 49     | 266694.02  | 5.9635%     | 524.63  | 265316.45 | 267975.78 | 24.9         |
| 10 | 100% Best (Exploitation)                | 49     | 265569.31  | 5.5166%     | 615.70  | 264543.63 | 266991.25 | 24.9         |
| 11 | 50% Best + 50% fresh Hilbert            | 49     | 265795.73  | 5.6066%     | 633.50  | 264257.31 | 267068.59 | 24.8         |
| 12 | 75% Best + 25% fresh Hilbert            | 49     | 265625.77  | 5.5390%     | 604.90  | 264548.20 | 267022.86 | 24.7         |
| 13 | 50% Best + 50% fresh Random             | 49     | 266035.79  | 5.7019%     | 627.93  | 264775.83 | 267590.41 | 24.4         |
| 14 | 75% Best + 25% fresh Random             | 49     | 265756.51  | 5.5910%     | 623.19  | 264709.12 | 267817.91 | 24.4         |
| 15 | 50% Best + 25% Hilbert + 25% Random     | 49     | 265866.64  | 5.6347%     | 610.19  | 264644.57 | 266877.82 | 24.4         |
| 16 | 50% Best + 25% Hilbert + 25% Greedy NN  | 49     | 265773.52  | 5.5977%     | 610.34  | 264301.31 | 267131.69 | 24.3         |

### Statistical Significance Analysis (Configs 1-16)
- **ANOVA Initial (Group A)**: F=105.4702, p=2.07e-85 (Highly Significant)
- **ANOVA Re-seed (Group B)**: F=16.7329, p=2.74e-19 (Highly Significant)
- **Verdict**: Greedy NN is the superior initial seed. "100% Best" exploitation is the most robust re-seeding strategy.

### Final Long-Run Seeding Comparison (N=5,000, max_opt=5, kicks=100, iters=10)
**Strategy**: All configs reset to current global best (100% Best) after each iteration.

| ID | Configuration                           | Trials | Avg Length | Avg Gap (%) | Std Dev | Min       | Max       | Avg Time (s) |
|----|-----------------------------------------|--------|------------|-------------|---------|-----------|-----------|--------------|
| 1  | LR: 100% Hilbert                        | 50     | 264227.44  | 4.9834%     | 546.30  | 262827.19 | 265433.71 | 57.1         |
| 2  | LR: 100% Random                         | 25     | 263275.53  | 4.6052%     | 577.15  | 262270.59 | 264735.24 | 55.7         |
| 3  | LR: 100% Greedy NN                      | 9      | 262885.69  | 4.4503%     | 543.22  | 261898.73 | 263602.61 | 53.9         |

#### Statistical Significance Analysis (Long-Run)
- **One-way ANOVA p-value**: 2.1084e-12 (Highly Significant)
- **F-statistic**: 38.1597
- **Tukey HSD Insights**:
    - Hilbert (1) vs Random (2): **Significant** ($p < 2.04e-09$)
    - Hilbert (1) vs Greedy NN (3): **Significant** ($p < 8.58e-09$)
    - Random (2) vs Greedy NN (3): **Not Significant** ($p = 0.174$) at this sample size, though Greedy NN trended better.

#### Raw Long-Run Trial Data
| ID | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 | T20 | T21 | T22 | T23 | T24 | T25 | T26 | T27 | T28 | T29 | T30 | T31 | T32 | T33 | T34 | T35 | T36 | T37 | T38 | T39 | T40 | T41 | T42 | T43 | T44 | T45 | T46 | T47 | T48 | T49 | T50 |
|----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 1  | 264357.02 | 264616.57 | 263950.78 | 264480.74 | 264224.50 | 262827.19 | 264638.70 | 264415.35 | 265231.54 | 263744.22 | 264126.09 | 263759.11 | 264594.71 | 264715.13 | 264333.97 | 264192.49 | 265433.71 | 264599.81 | 264052.24 | 262927.64 | 264452.41 | 264828.58 | 263093.27 | 264886.89 | 264514.61 | 264280.60 | 264723.65 | 264529.32 | 264468.11 | 264476.85 | 264579.20 | 264175.20 | 263870.65 | 264004.70 | 264762.30 | 264453.42 | 264100.08 | 263831.15 | 263821.93 | 264137.88 | 263363.70 | 264760.08 | 264720.69 | 264091.40 | 264073.23 | 263968.43 | 263967.25 | 264426.97 | 262926.12 | 263861.78 |
| 2  | 264735.24 | 262270.59 | 263029.72 | 263169.13 | 263499.50 | 263039.90 | 263813.29 | 263855.14 | 263742.97 | 263154.21 | 263004.27 | 262842.58 | 263685.07 | 263781.57 | 262579.15 | 262840.79 | 262642.34 | 263364.93 | 263398.40 | 263641.03 | 262990.25 | 262787.45 | 264237.22 | 263317.61 | 262465.87 |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |
| 3  | 262788.10 | 263352.32 | 262747.36 | 262290.61 | 261898.73 | 262794.03 | 263241.29 | 263602.61 | 263256.12 |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |           |

**Conclusion**: The initial advantage of **Greedy NN** persists in the long run (10 iterations), yielding the lowest average gap (**4.45%**). **Random** starts are significantly better than Hilbert starts (**4.61%** vs **4.98%**), confirming that Hilbert seeds trap the search in inferior basins of attraction.

## [2026-05-25 20:37] - Long-Run Seeding Experiment (N=100, max_opt=5)
- **Held-Karp LB**: 27175.81
- **Parameters**: seeds=12, kicks_per_iter=10, iterations=2, trials=1
- **Strategy**: All workers reset to global best (100% Best) after each iteration.

### Summary Table (10 Iterations)
| ID | Configuration                           | Trials | Avg Length | Avg Gap (%) | Std Dev | Min       | Max       | Avg Time (s) |
|----|-----------------------------------------|--------|------------|-------------|---------|-----------|-----------|--------------|
| 1  | LR: 100% Hilbert                        | 1      | 27531.12   | 1.3074%     | 0.00    | 27531.12  | 27531.12  | 8.3          |
| 2  | LR: 100% Random                         | 1      | 27814.92   | 2.3517%     | 0.00    | 27814.92  | 27814.92  | 8.4          |
| 3  | LR: 100% Greedy NN                      | 1      | 27597.43   | 1.5514%     | 0.00    | 27597.43  | 27597.43  | 8.8          |
| 4  | LR: 50/50 Hil/Rand                      | 1      | 27392.17   | 0.7962%     | 0.00    | 27392.17  | 27392.17  | 8.8          |
| 5  | LR: 50/50 Hil/Greedy                    | 1      | 27578.46   | 1.4817%     | 0.00    | 27578.46  | 27578.46  | 8.3          |
| 6  | LR: 50/50 Rand/Greedy                   | 1      | 27440.92   | 0.9755%     | 0.00    | 27440.92  | 27440.92  | 7.8          |
| 7  | LR: 75/25 Hil/Greedy                    | 1      | 27375.45   | 0.7346%     | 0.00    | 27375.45  | 27375.45  | 7.7          |
| 8  | LR: Balanced Mix                        | 1      | 27650.45   | 1.7465%     | 0.00    | 27650.45  | 27650.45  | 7.8          |
