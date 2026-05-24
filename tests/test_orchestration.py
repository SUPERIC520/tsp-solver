import numpy as np
from src.core.orchestration import parallel_solve


def test_parallel_solve_basic():
    # Small problem
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
