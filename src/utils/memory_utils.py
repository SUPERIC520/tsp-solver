"""Memory management utilities for the TSP solver."""

import numpy as np


def ensure_alignment(arr: np.ndarray, alignment: int = 64) -> np.ndarray:
    """Ensure that the input array is C-contiguous and its data pointer is aligned.

    Data pointer is aligned to the specified byte boundary.
    Uses the over-allocate-and-slice pattern to guarantee alignment.
    If already aligned and contiguous, returns the array as-is.
    If size == 0, returns the array without asserting alignment.
    """
    arr = np.ascontiguousarray(arr)
    if arr.size == 0:
        return arr
    if arr.ctypes.data % alignment == 0:
        return arr

    nbytes = arr.nbytes
    dtype = arr.dtype
    shape = arr.shape

    # Allocate extra bytes for alignment padding
    raw = np.empty(nbytes + alignment, dtype=np.uint8)
    start_address = raw.ctypes.data
    offset = (alignment - (start_address % alignment)) % alignment

    # Slice the aligned buffer and view it with correct shape/dtype
    aligned_raw = raw[offset : offset + nbytes]
    aligned_arr = aligned_raw.view(dtype).reshape(shape)
    np.copyto(aligned_arr, arr)

    assert aligned_arr.ctypes.data % alignment == 0
    assert aligned_arr.flags["C_CONTIGUOUS"]
    return aligned_arr
