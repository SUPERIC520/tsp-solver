"""Pytest configuration and shared fixtures for the TSP solver tests."""

import os
import platform

# Mock platform.machine to bypass WMI query hang on Windows
platform.machine = lambda: "AMD64"

# Set thread limits for numerical libraries to prevent MemoryErrors and deadlocks
# on Windows when using multiprocessing.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def warmup_numba_jit() -> None:
    """Session-scoped fixture that runs a tiny N=5 problem ONCE before any test.

    To trigger Numba JIT compilation. All subsequent tests will use the
    cached compiled kernels and run fast.
    """
    from src.core.kopt_engine import cascading_kopt_optimize

    n = 5
    rng = np.random.default_rng(0)
    coords_x = rng.random(n).astype(np.float64)
    coords_y = rng.random(n).astype(np.float64)
    tour = np.arange(n, dtype=np.int32)

    # Full candidate set for tiny N
    candidate_set = np.full((n, n - 1), -1, dtype=np.int32)
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        candidate_set[i, : len(cands)] = cands

    # This single call compiles all JIT-decorated kernels used by tests
    cascading_kopt_optimize(
        tour, coords_x, coords_y, candidate_set, num_kicks=1
    )
