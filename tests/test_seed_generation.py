import numpy as np
from src.core.seed_generation import generate_hilbert_seeds


def test_generate_hilbert_seeds() -> None:
    # 100 random points
    coords = np.random.rand(100, 2).astype(np.float64)
    num_seeds = 8
    seeds = generate_hilbert_seeds(coords, num_seeds=num_seeds)

    assert seeds.shape == (num_seeds, 100)

    for i in range(num_seeds):
        # Check if it's a valid permutation
        assert len(np.unique(seeds[i])) == 100
        assert np.min(seeds[i]) == 0
        assert np.max(seeds[i]) == 99

    # Check for uniqueness of all 8 Hilbert seeds
    # With 100 random points, the 8 symmetries should produce different Hilbert sorts
    unique_seeds = np.unique(seeds, axis=0)
    assert len(unique_seeds) == 8
