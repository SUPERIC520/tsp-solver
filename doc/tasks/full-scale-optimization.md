# Full Scale Optimization Task

## Goal
Achieve < 5% gap from Held-Karp lower bound at N=115,475 within 600s.

## Current State (as of 2026-05-23)
- N=100: 0.73% gap (SUCCESS)
- N=500: 3.14% gap (within 5% target)
- N=1000: 3.41% gap (within 5% target)
- N=115,475: 10.83% gap (FAILURE - needs ~6% improvement)

## Known Issues to Fix
1. `run_sample.py` marks >1% as FAILURE — change threshold to 5%
2. `_optimize_or_opt` missing `return globally_improved` at end (critical bug!)
3. Stagnation limit too aggressive for large N (n//10 = 11,500 for 115k)
4. Backbone with 90% threshold may lock suboptimal edges prematurely
5. 4/5-opt move application uses sequential 2-opt flips (may not be true k-opt)

## Optimization Strategy
1. **More seeds diversity**: Use 8 seeds with diverse Hilbert orientations
2. **More kicks per seed**: 2000+ kicks at full scale within time budget
3. **Or-opt with reversal**: Try reversed segment insertion in Or-opt  
4. **Disable backbone locking** until convergence is shown
5. **3-opt focus**: Keep max_opt=3 for speed, allow more iterations within time limit

## Parameter Targets for Full Scale
- seeds: 8
- kicks: 2000 (per seed, parallel)
- iters: 2 (backbone iterations)  
- max_opt: 3
- hk_iter: 2000 (cached after first run)
- num_processes: min(cpu_count, 8)

## Expected Improvement Path
- With fixes + more kicks: expect ~8-9% → 6-7% gap
- With 4-opt enabled: expect ~5-6% gap
- With better backbone strategy: expect ~4-5% gap

## Sub-Agent Protocol
- Fix Agent: Implements all code fixes
- Bench Agent: Runs N=100→500→1000→5000→10000→115475 validation chain
