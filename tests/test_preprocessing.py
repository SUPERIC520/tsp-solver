"""Tests for src/core/preprocessing.py.

Covers:
  - ensure_alignment: alignment guarantee, C-contiguity, data preservation
  - hilbert_reorder_cities: valid permutation mapping, dtypes, C-contiguous/aligned
  - build_candidate_sets: shape, dtype, C-contiguous, 64-byte aligned
  - refine_candidate_set_with_alpha: shape, valid indices, dtype, alignment
"""

import numpy as np

from src.core.preprocessing import (
    build_candidate_sets,
    ensure_alignment,
    hilbert_reorder_cities,
    refine_candidate_set_with_alpha,
)

# ---------------------------------------------------------------------------
# ensure_alignment
# ---------------------------------------------------------------------------


def test_ensure_alignment_basic_int32() -> None:
    """Unaligned array must come back aligned, C-contiguous, values preserved."""
    arr = np.arange(10, dtype=np.int32)
    aligned = ensure_alignment(arr, alignment=64)
    assert aligned.ctypes.data % 64 == 0
    assert aligned.flags["C_CONTIGUOUS"]
    assert np.array_equal(arr, aligned)


def test_ensure_alignment_non_contiguous() -> None:
    """Non-C-contiguous slice must be made aligned and C-contiguous."""
    arr_2d = np.arange(20, dtype=np.int32).reshape(5, 4)
    non_contiguous = arr_2d[:, 1]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    aligned = ensure_alignment(non_contiguous, alignment=64)
    assert aligned.ctypes.data % 64 == 0
    assert aligned.flags["C_CONTIGUOUS"]
    assert np.array_equal(non_contiguous, aligned)


def test_ensure_alignment_already_aligned() -> None:
    """Already-aligned array should be returned without unnecessary copy."""
    raw = np.empty(100 + 64, dtype=np.uint8)
    offset = (64 - (raw.ctypes.data % 64)) % 64
    aligned_buf = raw[offset : offset + 80]
    aligned_arr = np.ndarray((10,), dtype=np.float64, buffer=aligned_buf)
    assert aligned_arr.ctypes.data % 64 == 0
    assert aligned_arr.flags["C_CONTIGUOUS"]

    returned = ensure_alignment(aligned_arr, alignment=64)
    assert returned.ctypes.data == aligned_arr.ctypes.data


def test_ensure_alignment_empty_array() -> None:
    """Empty array must not raise and must be returned as-is."""
    empty = np.array([], dtype=np.float64)
    result = ensure_alignment(empty, alignment=64)
    assert result.size == 0
    assert result.dtype == np.float64


def test_ensure_alignment_float64() -> None:
    """float64 array must also be aligned and values preserved."""
    arr = np.linspace(0, 1, 50, dtype=np.float64)
    aligned = ensure_alignment(arr, alignment=64)
    assert aligned.ctypes.data % 64 == 0
    assert aligned.flags["C_CONTIGUOUS"]
    assert np.allclose(arr, aligned)


def test_ensure_alignment_2d_array() -> None:
    """2-D array should be aligned and remain 2-D."""
    arr = np.arange(30, dtype=np.float64).reshape(5, 6)
    aligned = ensure_alignment(arr, alignment=64)
    assert aligned.ctypes.data % 64 == 0
    assert aligned.flags["C_CONTIGUOUS"]
    assert aligned.shape == (5, 6)
    assert np.array_equal(arr, aligned)


# ---------------------------------------------------------------------------
# hilbert_reorder_cities
# ---------------------------------------------------------------------------


def test_hilbert_reorder_cities_correctness() -> None:
    """Mapping must be a valid permutation and coords must round-trip."""
    rng = np.random.default_rng(42)
    coords = rng.random((20, 2)) * 100.0

    reordered_coords, original_to_new = hilbert_reorder_cities(coords)

    # --- dtypes ---
    assert reordered_coords.dtype == np.float64
    assert original_to_new.dtype == np.int32

    # --- alignment & contiguity ---
    assert reordered_coords.ctypes.data % 64 == 0
    assert reordered_coords.flags["C_CONTIGUOUS"]
    assert original_to_new.ctypes.data % 64 == 0
    assert original_to_new.flags["C_CONTIGUOUS"]

    # --- valid permutation ---
    assert set(original_to_new.tolist()) == set(range(20))

    # --- round-trip: original city i lands at position original_to_new[i] ---
    for i in range(coords.shape[0]):
        new_idx = original_to_new[i]
        assert np.allclose(reordered_coords[new_idx], coords[i])


def test_hilbert_reorder_cities_single_point() -> None:
    """Single-point input should not raise."""
    coords = np.array([[3.0, 7.0]], dtype=np.float64)
    reordered, mapping = hilbert_reorder_cities(coords)
    assert reordered.shape == (1, 2)
    assert mapping.shape == (1,)
    assert mapping[0] == 0


def test_hilbert_reorder_cities_identical_points() -> None:
    """All identical coords (max_range == 0) should return a valid permutation."""
    coords = np.ones((10, 2), dtype=np.float64)
    reordered, mapping = hilbert_reorder_cities(coords)
    assert set(mapping.tolist()) == set(range(10))
    assert reordered.shape == (10, 2)


# ---------------------------------------------------------------------------
# build_candidate_sets
# ---------------------------------------------------------------------------


def test_build_candidate_sets_shape_dtype_alignment() -> None:
    """Output must be (N, k) int32, C-contiguous, 64-byte aligned."""
    rng = np.random.default_rng(0)
    coords = rng.random((50, 2)).astype(np.float64)
    k = 8
    c_set = build_candidate_sets(coords, k=k)

    assert c_set.shape == (50, k)
    assert c_set.dtype == np.int32
    assert c_set.flags["C_CONTIGUOUS"]
    assert c_set.ctypes.data % 64 == 0


def test_build_candidate_sets_simple_square() -> None:
    """4-point square: each point's 2-NN should be its two nearest neighbours."""
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )
    c_set = build_candidate_sets(coords, k=2)

    assert c_set.shape == (4, 2)
    assert set(c_set[0]) == {1, 3}
    assert set(c_set[1]) == {0, 2}


def test_build_candidate_sets_grid_center() -> None:
    """3x3 grid: centre node's 4-NN should be its 4 cardinal neighbours."""
    x = np.linspace(0, 2, 3)
    y = np.linspace(0, 2, 3)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float64)

    c_set = build_candidate_sets(coords, k=4)

    assert c_set.shape == (9, 4)
    assert set(c_set[4].tolist()) == {1, 3, 5, 7}


def test_build_candidate_sets_10_cities() -> None:
    """2x5 grid: verify specific neighbours for two nodes."""
    coords_list = [[float(x), float(y)] for y in range(2) for x in range(5)]
    coords = np.array(coords_list, dtype=np.float64)

    c_set = build_candidate_sets(coords, k=3)
    assert c_set.shape == (10, 3)

    # City 1 (1,0): neighbours 0, 2, 6
    assert {0, 2, 6}.issubset(set(c_set[1].tolist()))
    # City 7 (2,1): neighbours 6, 8, 2
    assert {6, 8, 2}.issubset(set(c_set[7].tolist()))


def test_build_candidate_sets_fewer_cities_than_k() -> None:
    """When N < k the surplus columns must be filled with -1."""
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 10.0]], dtype=np.float64
    )
    c_set = build_candidate_sets(coords, k=64)

    assert c_set.shape == (4, 64)
    assert np.all(c_set[:, 3:] == -1)


def test_build_candidate_sets_valid_indices() -> None:
    """All non-(-1) entries must be valid node indices in [0, N)."""
    rng = np.random.default_rng(7)
    n = 30
    coords = rng.random((n, 2)).astype(np.float64)
    k = 10
    c_set = build_candidate_sets(coords, k=k)

    mask = c_set != -1
    assert np.all(c_set[mask] >= 0)
    assert np.all(c_set[mask] < n)


# ---------------------------------------------------------------------------
# refine_candidate_set_with_alpha
# ---------------------------------------------------------------------------


def test_refine_candidate_set_shape_dtype_alignment() -> None:
    """Output must be (N, top_k) int32, C-contiguous, 64-byte aligned."""
    coords = np.array([[float(i), 0.0] for i in range(15)], dtype=np.float64)
    candidate_set = build_candidate_sets(coords, k=10)
    pi = np.zeros(15, dtype=np.float64)

    refined = refine_candidate_set_with_alpha(coords, candidate_set, pi, top_k=5)

    assert refined.shape == (15, 5)
    assert refined.dtype == np.int32
    assert refined.flags["C_CONTIGUOUS"]
    assert refined.ctypes.data % 64 == 0


def test_refine_candidate_set_valid_indices() -> None:
    """All non-(-1) entries must be valid node indices in [0, N)."""
    coords = np.array([[float(i), 0.0] for i in range(15)], dtype=np.float64)
    candidate_set = build_candidate_sets(coords, k=10)
    pi = np.zeros(15, dtype=np.float64)

    refined = refine_candidate_set_with_alpha(coords, candidate_set, pi, top_k=5)

    mask = refined != -1
    assert np.all(refined[mask] >= 0)
    assert np.all(refined[mask] < 15)


def test_refine_candidate_set_correct_neighbors() -> None:
    """For collinear cities with zero pi, alpha-sort should match distance-sort."""
    coords = np.array([[float(i), 0.0] for i in range(15)], dtype=np.float64)
    candidate_set = build_candidate_sets(coords, k=10)
    pi = np.zeros(15, dtype=np.float64)

    refined = refine_candidate_set_with_alpha(coords, candidate_set, pi, top_k=5)

    # City 7's 5 closest: 6, 8, 5, 9, 4 (symmetrically equidistant pairs)
    assert set(refined[7].tolist()) == {4, 5, 6, 8, 9}


def test_refine_candidate_set_padding_when_top_k_larger() -> None:
    """When top_k > candidates per row, surplus must be padded with -1."""
    coords = np.array([[float(i), 0.0] for i in range(15)], dtype=np.float64)
    candidate_set = build_candidate_sets(coords, k=10)
    pi = np.zeros(15, dtype=np.float64)

    refined = refine_candidate_set_with_alpha(coords, candidate_set, pi, top_k=12)

    assert refined.shape == (15, 12)
    assert np.all(refined[:, 10:] == -1)


def test_refine_candidate_set_empty_input() -> None:
    """Zero-city input must return empty array without raising."""
    coords = np.empty((0, 2), dtype=np.float64)
    candidate_set = np.empty((0, 10), dtype=np.int32)
    pi = np.empty(0, dtype=np.float64)

    refined = refine_candidate_set_with_alpha(coords, candidate_set, pi, top_k=5)

    assert refined.shape == (0, 5)
    assert refined.dtype == np.int32
