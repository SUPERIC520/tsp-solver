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
