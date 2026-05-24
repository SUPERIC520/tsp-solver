# Task: Full Cascading K-Opt Engine Implementation

## Objective
Refactor the optimization core into a cascading engine that attempts moves in sequence: 2-opt -> 3-opt -> 4-opt -> 5-opt.

## Rules
- **Sub-Agent**: MUST use a specialized sub-agent for implementation and unit tests.
- **Testing**: Every test session MUST start with N=100.
- **Initial Limit**: Use at most 3-opt for early validation (N=100, N=500).

## Sub-Tasks

### 1. Sequential Move Implementation (Sub-Agent)
- [ ] Implement `_optimize_2opt` (if not already optimized).
- [ ] Implement `_optimize_3opt_sequential` using the gain criterion.
- [ ] Implement `_optimize_4opt_sequential` (non-reducible).
- [ ] Implement `_optimize_5opt_sequential`.

### 2. Cascading Logic (Sub-Agent)
- [ ] Implement `full_cascade` loop:
  - While improved:
    - While `_optimize_2opt` improved: pass
    - If `_optimize_3opt` improved: continue
    - If `_optimize_4opt` improved: continue
    - If `_optimize_5opt` improved: continue
- [ ] Ensure `dlb` (Don't Look Bits) management across all opt levels.

### 3. Verification Protocol (Sub-Agent)
- [ ] **Step 1: N=100 Validation**
  - Run solver with up to 3-opt on N=100 sample.
  - Verify length decreases and gap < 5%.
- [ ] **Step 2: N=500 Scaling** (Only if Step 1 passes)
  - Run solver on N=500.
  - Verify time < 10s and gap < 5%.
- [ ] **Step 3: Scaling to Full Dataset** (Only if Step 2 passes)
  - Re-evaluate if 4-opt/5-opt are needed for 115k cities within 10m.
