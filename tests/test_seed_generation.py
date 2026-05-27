"""Tests for seed generation strategies, including greedy NN and rotation."""

import numba
import numpy as np
import pytest

from src.core.preprocessing import build_candidate_sets
from src.core.seed_generation import (
    ensure_alignment,
    generate_greedy_nn_seeds,
    rotate_tour,
)


def test_ensure_alignment() -> None:
    # Test alignment with normal NumPy array
    arr = np.arange(10, dtype=np.int32)
    aligned = ensure_alignment(arr, alignment=64)
    assert aligned.ctypes.data % 64 == 0
    assert aligned.flags["C_CONTIGUOUS"]
    assert np.array_equal(arr, aligned)

    # Test alignment with non-contiguous slice
    non_contig = np.arange(20, dtype=np.int32)[::2]
    assert not non_contig.flags["C_CONTIGUOUS"]
    aligned_non_contig = ensure_alignment(non_contig, alignment=64)
    assert aligned_non_contig.ctypes.data % 64 == 0
    assert aligned_non_contig.flags["C_CONTIGUOUS"]
    assert np.array_equal(non_contig, aligned_non_contig)


def test_rotate_tour() -> None:
    tour = np.array([3, 1, 4, 2, 0], dtype=np.int32)

    # Rotate to start node 4
    rotated = rotate_tour(tour, 4)
    assert rotated.shape == (5,)
    assert rotated.dtype == np.int32
    assert rotated.ctypes.data % 64 == 0
    assert rotated.flags["C_CONTIGUOUS"]
    assert np.array_equal(rotated, [4, 2, 0, 3, 1])

    # Rotate to start node 3 (already at start)
    rotated = rotate_tour(tour, 3)
    assert np.array_equal(rotated, tour)
    assert rotated.ctypes.data % 64 == 0
    assert rotated.flags["C_CONTIGUOUS"]

    # Rotate to start node 0 (at the end)
    rotated = rotate_tour(tour, 0)
    assert np.array_equal(rotated, [0, 3, 1, 4, 2])

    # Error cases
    with pytest.raises(ValueError, match="not found in tour"):
        rotate_tour(tour, 9)


def test_rotate_tour_edge_cases() -> None:
    # Single element tour
    tour = np.array([0], dtype=np.int32)
    rotated = rotate_tour(tour, 0)
    assert np.array_equal(rotated, [0])

    # Two elements
    tour = np.array([1, 0], dtype=np.int32)
    assert np.array_equal(rotate_tour(tour, 0), [0, 1])
    assert np.array_equal(rotate_tour(tour, 1), [1, 0])


def test_generate_greedy_nn_seeds() -> None:
    # Setup coordinates and candidate sets
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], dtype=np.float64)

    candidate_set = build_candidate_sets(coords, k=3)

    # 3 seeds
    num_seeds = 3
    seeds = generate_greedy_nn_seeds(coords, candidate_set, num_seeds=num_seeds)

    assert seeds.shape == (num_seeds, 5)
    assert seeds.dtype == np.int32
    assert seeds.ctypes.data % 64 == 0
    assert seeds.flags["C_CONTIGUOUS"]

    for i in range(num_seeds):
        # Verify valid permutation (non-duplication)
        assert len(np.unique(seeds[i])) == 5
        assert np.min(seeds[i]) == 0
        assert np.max(seeds[i]) == 4

        # Verify coordinate mapping correctness (every seed mapped is valid)
        mapped_coords = coords[seeds[i]]
        assert mapped_coords.shape == (5, 2)
        assert mapped_coords.dtype == np.float64

    # Verify that different start nodes work
    start_nodes = np.array([1, 3], dtype=np.int32)
    seeds_custom = generate_greedy_nn_seeds(
        coords, candidate_set, num_seeds=2, start_nodes=start_nodes
    )
    assert seeds_custom[0, 0] == 1
    assert seeds_custom[1, 0] == 3

    # Edge case: num_seeds = 0
    empty_seeds = generate_greedy_nn_seeds(coords, candidate_set, num_seeds=0)
    assert empty_seeds.shape == (0, 5)
    if empty_seeds.size > 0:
        assert empty_seeds.ctypes.data % 64 == 0

    # Error case: length of start_nodes mismatch
    with pytest.raises(ValueError, match="does not match num_seeds"):
        generate_greedy_nn_seeds(
            coords, candidate_set, num_seeds=3, start_nodes=start_nodes
        )


def test_under_disabled_jit() -> None:
    """Test functionality specifically with JIT disabled.

    Ensures the pure-Python implementation is robust.
    """
    # 1. Test rotate_tour using the uncompiled py_func internally
    from src.core.seed_generation import (
        _find_index_jit,
        _greedy_nn_tour,
        _rotate_tour_jit,
    )

    tour = np.array([5, 2, 7, 1], dtype=np.int32)
    idx = _find_index_jit.py_func(tour, 7)
    assert idx == 2

    rotated = _rotate_tour_jit.py_func(tour, idx)
    assert np.array_equal(rotated, [7, 1, 5, 2])

    # 2. Test greedy NN tour using the uncompiled py_func
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    candidate_set = np.array([
        [1, 3, -1],
        [0, 2, -1],
        [1, 3, -1],
        [0, 2, -1]
    ], dtype=np.int32)

    # Run uncompiled _greedy_nn_tour
    tour_py = _greedy_nn_tour.py_func(coords, candidate_set, 0)
    assert len(np.unique(tour_py)) == 4
    assert tour_py[0] == 0

    # 3. Verify dynamically disabling JIT runs correctly
    was_disabled = numba.config.DISABLE_JIT
    try:
        numba.config.DISABLE_JIT = True

        # Test rotate_tour runs under disabled JIT config
        rotated_jit_disabled = rotate_tour(tour, 7)
        assert np.array_equal(rotated_jit_disabled, [7, 1, 5, 2])

        # Test generate_greedy_nn_seeds runs under disabled JIT config
        seeds = generate_greedy_nn_seeds(coords, candidate_set, num_seeds=2)
        assert seeds.shape == (2, 4)
        for seed in seeds:
            assert len(np.unique(seed)) == 4
    finally:
        numba.config.DISABLE_JIT = was_disabled
