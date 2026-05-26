import numpy as np
from src.core.orchestration import parallel_solve


def test_parallel_solve_basic() -> None:
    # Small problem to test backward compatibility (with locked_edges)
    n = 20
    coords = np.random.rand(n, 2).astype(np.float64)
    seeds = np.array([np.random.permutation(n).astype(np.int32) for _ in range(4)])
    candidate_set = np.full((n, 5), -1, dtype=np.int32)
    # Fill candidate set with some indices
    for i in range(n):
        candidate_set[i, :4] = np.random.choice(n, 4, replace=False)

    locked_edges = np.full((n, 2), -1, dtype=np.int32)

    results = parallel_solve(
        seeds, coords, candidate_set, locked_edges, num_processes=2
    )

    assert len(results) == 4
    for tour, length in results:
        assert tour.shape == (n,)
        assert length > 0
        assert len(np.unique(tour)) == n


def test_parallel_solve_no_locked_edges() -> None:
    # Small problem to test forward compatibility (without locked_edges)
    n = 20
    coords = np.random.rand(n, 2).astype(np.float64)
    seeds = np.array([np.random.permutation(n).astype(np.int32) for _ in range(4)])
    candidate_set = np.full((n, 5), -1, dtype=np.int32)
    for i in range(n):
        candidate_set[i, :4] = np.random.choice(n, 4, replace=False)

    results = parallel_solve(
        seeds, coords, candidate_set, num_processes=2
    )

    assert len(results) == 4
    for tour, length in results:
        assert tour.shape == (n,)
        assert length > 0
        assert len(np.unique(tour)) == n


def test_parallel_solve_scaling() -> None:
    # Test execution with various process bounds
    n = 20
    coords = np.random.rand(n, 2).astype(np.float64)
    seeds = np.array([np.random.permutation(n).astype(np.int32) for _ in range(3)])
    candidate_set = np.full((n, 5), -1, dtype=np.int32)
    for i in range(n):
        candidate_set[i, :4] = np.random.choice(n, 4, replace=False)

    # 1 process
    res_1 = parallel_solve(seeds, coords, candidate_set, num_processes=1)
    assert len(res_1) == 3

    # 4 processes (or CPU limit)
    res_4 = parallel_solve(seeds, coords, candidate_set, num_processes=4)
    assert len(res_4) == 3


def test_parallel_solve_progress_reporting(capsys) -> None:
    # Verify that status logs and progress are printed during execution
    n = 20
    coords = np.random.rand(n, 2).astype(np.float64)
    seeds = np.array([np.random.permutation(n).astype(np.int32) for _ in range(2)])
    candidate_set = np.full((n, 5), -1, dtype=np.int32)
    for i in range(n):
        candidate_set[i, :4] = np.random.choice(n, 4, replace=False)

    results = parallel_solve(
        seeds, coords, candidate_set, num_processes=1, num_kicks=50
    )
    assert len(results) == 2

    captured = capsys.readouterr()
    # Check that progress logs were written to stdout/stderr
    assert len(captured.out) > 0 or len(captured.err) > 0


def test_parallel_solve_failure_recovery() -> None:
    # Test worker exception handling (e.g. ValueError during input parsing)
    n = 20
    coords = np.random.rand(n, 2).astype(np.float64)
    seeds = np.array([np.random.permutation(n).astype(np.int32) for _ in range(2)])
    
    # Passing an invalid candidate set with strings triggers a ValueError during parsing in the worker
    invalid_candidate_set = np.full((n, 5), "invalid", dtype=object)

    results = parallel_solve(
        seeds, coords, invalid_candidate_set, num_processes=2
    )

    # It must catch the exception, terminate the pool, and return the intermediate results
    assert isinstance(results, list)


def test_parallel_solve_timeout() -> None:
    # Test that solver terminates and returns intermediate results when exceeding time budget
    n = 20
    coords = np.random.rand(n, 2).astype(np.float64)
    seeds = np.array([np.random.permutation(n).astype(np.int32) for _ in range(4)])
    candidate_set = np.full((n, 5), -1, dtype=np.int32)
    for i in range(n):
        candidate_set[i, :4] = np.random.choice(n, 4, replace=False)

    # A very tiny timeout of 0.001 seconds should trigger the timeout manager
    results = parallel_solve(
        seeds, coords, candidate_set, num_processes=2, time_limit_s=0.001
    )

    # It must exit gracefully and return a list of intermediate results
    assert isinstance(results, list)
