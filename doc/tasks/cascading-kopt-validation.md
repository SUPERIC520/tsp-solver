# Task: Cascading K-Opt Validation

## Objective
Reach < 5% gap from Held-Karp lower bound on the full 115k dataset using a cascading K-Opt approach.

## Constraints
- **Mechanism**: Use cascading K-Opt (2, 3, 4, 5-opt).
- **Sub-Agent**: Every step must be performed by a sub-agent.
- **Testing**: Start at N=100.
- **Initial Phase**: Use at most 3-opt for testing.

## Checklist

### Phase 1: Engine Refactoring
- [ ] Implement `_optimize_2opt` and `_optimize_3opt_sequential` in `src/core/kopt_engine.py`.
- [ ] Implement `full_cascade` logic.

### Phase 2: N=100 Validation
- [ ] Run benchmark on N=100 sample.
- [ ] Verify < 5% gap and correct tour logic.

### Phase 3: Scaling to N=500
- [ ] Run benchmark on N=500 sample.
- [ ] Optimize parameters to reach < 5% in < 10s.

### Phase 4: Scaling to Full Dataset
- [ ] Run benchmarks on N=1000, 5000, 115475.
- [ ] Target: < 5% gap on all.
