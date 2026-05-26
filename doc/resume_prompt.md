# Resume Prompt: Seeding Strategy Statistical Evidence

**Context:**
We are investigating the impact of different seeding and re-seeding strategies on TSP solver performance. Our previous N=10,000 data suggested no statistically significant difference (ANOVA p > 0.05), but we need more trials for definitive proof. Numba was freezing during import/execution, so a system restart was initiated.

**Codebase State:**
- Reverted to stable version (`commit 311546b`) while preserving current `notes.md` and `trials.md`.
- `src/scripts/experiment_runner.py` is configured for:
    - **N=5,000** cities.
    - **50 trials per configuration** (16 configs, 800 trials total).
    - **Kicks**: 500 (initial), 100 (re-seed).
    - **Iterations**: 5.
    - **Output**: Detailed trial-by-trial logs in `scratch/seeding_strategy.csv`.

**Immediate Actions for Future Gemini:**

1. **Verify Numba**: Run a minimal test to ensure JIT is working after the restart:
   ```powershell
   uv run python -c "from numba import njit; f = njit(lambda x: x + 1); print(f'Numba OK: {f(1)}')"
   ```

2. **Run Experiment**: If Numba is OK, start the 800-trial batch:
   ```powershell
   uv run src/scripts/experiment_runner.py
   ```

3. **Monitor Progress**: Periodically check the CSV to ensure data is being recorded:
   ```powershell
   Get-Content scratch/seeding_strategy.csv -Tail 10
   ```

4. **Analysis**: Once complete, perform a statistical analysis (ANOVA and mean comparisons) to determine if strategies like "75% Best + 25% Random" provide a significant advantage over the baseline.
