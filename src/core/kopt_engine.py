"""K-opt engine for TSP optimization.

This module provides high-performance K-opt (2-opt, 3-opt, 4-opt, 5-opt) and
Or-opt local search implementations using Numba for acceleration. It also
includes an Iterated Local Search (ILS) framework with double-bridge kicks.
"""

import time

import numpy as np
import numpy.typing as npt
from numba import njit

from src.config import (
    GAIN_EPSILON,
    K_3OPT,
    K_4OPT,
    K_5OPT,
    K_NEIGHBORS,
    LOCALIZED_KICK_THRESHOLD,
    LOCALIZED_KICK_W_DIVISOR,
    LOCALIZED_KICK_W_MAX,
    LOCALIZED_KICK_W_MIN,
    MIN_TOUR_SIZE_2OPT,
    MIN_TOUR_SIZE_3OPT,
    MIN_TOUR_SIZE_4OPT,
    MIN_TOUR_SIZE_5OPT,
    MIN_TOUR_SIZE_KICK,
    OR_OPT_MAX_LEN,
    STAGNATION_LIMIT_DIVISOR,
    STAGNATION_LIMIT_MIN,
    TIME_SAFETY_MARGIN,
)
from src.utils.memory_utils import ensure_alignment


class KOptWorkspace:
    """Pre-allocated memory buffers for zero-allocation K-opt local search."""

    def __init__(self, n: int, num_cand: int) -> None:
        """Initialize pre-allocated memory buffers for the K-opt workspace.

        Args:
            n: Number of cities in the TSP instance.
            num_cand: Number of candidate neighbors per city.
        """
        # Current and best tours
        self.tour = ensure_alignment(np.empty(n, dtype=np.int32))
        self.best_tour = ensure_alignment(np.empty(n, dtype=np.int32))
        self.kicked_tour = ensure_alignment(np.empty(n, dtype=np.int32))

        # Position maps
        self.pos = ensure_alignment(np.empty(n, dtype=np.int32))
        self.best_pos = ensure_alignment(np.empty(n, dtype=np.int32))
        self.kicked_pos = ensure_alignment(np.empty(n, dtype=np.int32))

        # Don't Look Bits
        self.dlb = ensure_alignment(np.zeros(n, dtype=np.bool_))

        # Precomputed distances
        self.candidate_dists = ensure_alignment(
            np.empty((n, num_cand), dtype=np.float64)
        )


@njit(inline="always", fastmath=True)  # type: ignore
def _dist(
    c1: int,
    c2: int,
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
) -> float:
    """Get Euclidean distance between two cities."""
    dx = coords_x[c1] - coords_x[c2]
    dy = coords_y[c1] - coords_y[c2]
    return float(np.sqrt(dx * dx + dy * dy))


@njit(fastmath=True)  # type: ignore
def compute_tour_length(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
) -> float:
    """Compute the total length of a TSP tour."""
    length = 0.0
    n = tour.shape[0]
    for i in range(n):
        length += _dist(int(tour[i]), int(tour[(i + 1) % n]), coords_x, coords_y)
    return float(length)


@njit(fastmath=True)  # type: ignore
def _update_pos(tour: npt.NDArray[np.int32], pos: npt.NDArray[np.int32]) -> None:
    """Update the city-to-index position mapping."""
    for i in range(tour.shape[0]):
        pos[int(tour[i])] = i


@njit(fastmath=True)  # type: ignore
def _reverse_segment(
    tour: npt.NDArray[np.int32], i: int, j: int, pos: npt.NDArray[np.int32]
) -> None:
    """In-place reversal of a tour segment."""
    n = tour.shape[0]
    i = i % n
    j = j % n
    size = (j - i + n) % n
    if size <= 1:
        return
    if i < j:
        count = size // 2
        for k in range(count):
            idx1 = i + k
            idx2 = j - 1 - k
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
            pos[int(tour[idx1])] = idx1
            pos[int(tour[idx2])] = idx2
    else:
        count = size // 2
        idx1 = i
        idx2 = (j - 1 + n) % n
        for _k in range(count):
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
            pos[int(tour[idx1])] = idx1
            pos[int(tour[idx2])] = idx2
            idx1 += 1
            if idx1 == n:
                idx1 = 0
            idx2 -= 1
            if idx2 == -1:
                idx2 = n - 1


@njit(fastmath=True)  # type: ignore
def _optimize_2opt(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
) -> bool:
    """Standard 2-opt local search."""
    n = tour.shape[0]
    if n < MIN_TOUR_SIZE_2OPT:
        return False
    globally_improved = False

    num_cand = min(candidate_set.shape[1], K_NEIGHBORS)

    for u_idx in range(n):
        u = int(tour[u_idx])
        if dlb[u]:
            continue
        v_idx = (u_idx + 1) % n
        v = int(tour[v_idx])

        dist_uv = _dist(u, v, coords_x, coords_y)
        found = False
        for k in range(num_cand):
            w = int(candidate_set[u, k])
            if w == -1:
                break
            if w in (u, v):
                continue
            dist_uw = float(candidate_dists[u, k])
            if dist_uw >= dist_uv:
                continue

            w_idx = int(pos[w])
            x_idx = (w_idx + 1) % n
            x = int(tour[x_idx])
            if x in (u, v):
                continue

            if (
                dist_uv + _dist(w, x, coords_x, coords_y)
                > dist_uw + _dist(v, x, coords_x, coords_y) + GAIN_EPSILON
            ):
                _reverse_segment(tour, v_idx, (w_idx + 1) % n, pos)
                dlb[u] = dlb[v] = dlb[w] = dlb[x] = False
                globally_improved = True
                found = True
                break
        if not found:
            dlb[u] = True
    return globally_improved


@njit(fastmath=True)  # type: ignore
def _get_cut_indices(pos1: int, pos2: int, n: int) -> int:
    """Get the cut index between two adjacent cities in the tour."""
    if (pos1 + 1) % n == pos2:
        return pos1
    return pos2


@njit(fastmath=True)  # type: ignore
def _reconstruct_tour_3opt(
    tour: npt.NDArray[np.int32],
    idx_1: int,
    idx_2: int,
    idx_3: int,
    case_idx: int,
    pos: npt.NDArray[np.int32],
) -> None:
    """Apply the specific 3-opt case modification."""
    i1, i2, i3 = idx_1, idx_2, idx_3
    if i1 > i2:
        i1, i2 = i2, i1
    if i2 > i3:
        i2, i3 = i3, i2
    if i1 > i2:
        i1, i2 = i2, i1

    match case_idx:
        case 2:
            _reverse_segment(tour, i1 + 1, i2 + 1, pos)
        case 3:
            _reverse_segment(tour, i2 + 1, i3 + 1, pos)
        case 4:
            _reverse_segment(tour, i3 + 1, i1 + 1, pos)
        case 5:
            _reverse_segment(tour, i2 + 1, i3 + 1, pos)
            _reverse_segment(tour, i3 + 1, i1 + 1, pos)
        case 6:
            _reverse_segment(tour, i2 + 1, i3 + 1, pos)
            _reverse_segment(tour, i3 + 1, i1 + 1, pos)
            _reverse_segment(tour, i2 + 1, i1 + 1, pos)
        case 7:
            _reverse_segment(tour, i2 + 1, i3 + 1, pos)
            _reverse_segment(tour, i2 + 1, i1 + 1, pos)
        case 8:
            _reverse_segment(tour, i3 + 1, i1 + 1, pos)
            _reverse_segment(tour, i2 + 1, i1 + 1, pos)


@njit(fastmath=True)  # type: ignore
def _optimize_3opt_sequential(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
) -> bool:
    """Sequential 3-opt local search implementation."""
    n = tour.shape[0]
    if n < MIN_TOUR_SIZE_3OPT:
        return False
    globally_improved = False

    num_cand_k3 = min(candidate_set.shape[1], K_NEIGHBORS)
    num_cand_k5 = min(candidate_set.shape[1], K_3OPT)

    for t1_idx in range(n):
        t1 = int(tour[t1_idx])
        if dlb[t1]:
            continue

        t2_idx = (t1_idx + 1) % n
        t2 = int(tour[t2_idx])
        dist_t1_t2 = _dist(t1, t2, coords_x, coords_y)

        found = False
        for k3 in range(num_cand_k3):
            t3 = int(candidate_set[t2, k3])
            if t3 == -1:
                break
            if t3 in (t1, t2):
                continue
            dist_t2_t3 = float(candidate_dists[t2, k3])
            g1 = dist_t1_t2 - dist_t2_t3
            if g1 <= GAIN_EPSILON:
                continue

            t3_idx = int(pos[t3])
            for d1 in (-1, 1):
                t4_idx = (t3_idx + d1 + n) % n
                t4 = int(tour[t4_idx])
                if t4 in (t1, t2, t3):
                    continue
                dist_t3_t4 = _dist(t3, t4, coords_x, coords_y)

                for k5 in range(num_cand_k5):
                    t5 = int(candidate_set[t4, k5])
                    if t5 == -1:
                        break
                    if t5 in (t1, t2, t3, t4):
                        continue
                    dist_t4_t5 = float(candidate_dists[t4, k5])
                    g2 = g1 + dist_t3_t4 - dist_t4_t5
                    if g2 <= GAIN_EPSILON:
                        continue

                    t5_idx = int(pos[t5])
                    for d2 in (-1, 1):
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = int(tour[t6_idx])
                        if t6 in (t1, t2, t3, t4, t5):
                            continue

                        idx_1 = _get_cut_indices(t1_idx, t2_idx, n)
                        idx_2 = _get_cut_indices(t3_idx, t4_idx, n)
                        idx_3 = _get_cut_indices(t5_idx, t6_idx, n)

                        if idx_1 in (idx_2, idx_3) or idx_2 == idx_3:
                            continue

                        # Sorted edges for length computation
                        i1, i2, i3 = idx_1, idx_2, idx_3
                        if i1 > i2:
                            i1, i2 = i2, i1
                        if i2 > i3:
                            i2, i3 = i3, i2
                        if i1 > i2:
                            i1, i2 = i2, i1

                        a, b = int(tour[i1]), int(tour[(i1 + 1) % n])
                        c, d = int(tour[i2]), int(tour[(i2 + 1) % n])
                        e, f = int(tour[i3]), int(tour[(i3 + 1) % n])

                        E_remove = (
                            _dist(a, b, coords_x, coords_y)
                            + _dist(c, d, coords_x, coords_y)
                            + _dist(e, f, coords_x, coords_y)
                        )

                        best_case = -1
                        best_delta = -GAIN_EPSILON

                        # Evaluate all possible 3-reconnections (Cases 2-8)
                        # Cases 2, 3, 4 are 2-opt sub-moves within the 3-edge selection
                        # Cases 5, 6, 7, 8 are true 3-opt moves
                        d2_val = _dist(a, c, coords_x, coords_y) + _dist(b, d, coords_x, coords_y) + _dist(e, f, coords_x, coords_y) - E_remove
                        if d2_val < best_delta:
                            best_delta, best_case = d2_val, 2

                        d3_val = _dist(a, b, coords_x, coords_y) + _dist(c, e, coords_x, coords_y) + _dist(d, f, coords_x, coords_y) - E_remove
                        if d3_val < best_delta:
                            best_delta, best_case = d3_val, 3

                        d4_val = _dist(e, a, coords_x, coords_y) + _dist(f, b, coords_x, coords_y) + _dist(c, d, coords_x, coords_y) - E_remove
                        if d4_val < best_delta:
                            best_delta, best_case = d4_val, 4

                        d5_val = _dist(c, e, coords_x, coords_y) + _dist(d, a, coords_x, coords_y) + _dist(f, b, coords_x, coords_y) - E_remove
                        if d5_val < best_delta:
                            best_delta, best_case = d5_val, 5

                        d6_val = _dist(c, f, coords_x, coords_y) + _dist(a, d, coords_x, coords_y) + _dist(e, b, coords_x, coords_y) - E_remove
                        if d6_val < best_delta:
                            best_delta, best_case = d6_val, 6

                        d7_val = _dist(c, a, coords_x, coords_y) + _dist(f, d, coords_x, coords_y) + _dist(e, b, coords_x, coords_y) - E_remove
                        if d7_val < best_delta:
                            best_delta, best_case = d7_val, 7

                        d8_val = _dist(c, f, coords_x, coords_y) + _dist(a, e, coords_x, coords_y) + _dist(d, b, coords_x, coords_y) - E_remove
                        if d8_val < best_delta:
                            best_delta, best_case = d8_val, 8

                        if best_case != -1:
                            _reconstruct_tour_3opt(tour, i1, i2, i3, best_case, pos)
                            dlb[a] = dlb[b] = dlb[c] = dlb[d] = dlb[e] = dlb[f] = False
                            globally_improved = True
                            found = True
                            break
                    if found: break
                if found: break
            if found: break
        if not found:
            dlb[t1] = True
    return globally_improved


@njit(fastmath=True)  # type: ignore
def _optimize_or_opt(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
    max_len: int = 5,
) -> bool:
    """Relocation local search (insertion moves)."""
    n = tour.shape[0]
    globally_improved = False
    num_cand = min(candidate_set.shape[1], K_NEIGHBORS)

    for i_idx in range(n):
        u = int(tour[i_idx])
        if dlb[u]:
            continue
        found = False
        for length in range(1, max_len + 1):
            if length >= n - 2: break
            j_idx = (i_idx + length - 1) % n
            v = int(tour[j_idx])

            p_u_idx = (i_idx - 1 + n) % n
            p_u = int(tour[p_u_idx])
            s_v_idx = (j_idx + 1) % n
            s_v = int(tour[s_v_idx])

            base_g = (
                _dist(p_u, u, coords_x, coords_y)
                + _dist(v, s_v, coords_x, coords_y)
                - _dist(p_u, s_v, coords_x, coords_y)
            )
            if base_g <= GAIN_EPSILON: continue

            for k in range(num_cand):
                w = int(candidate_set[u, k])
                if w == -1: break
                if w in (u, v, p_u, s_v): continue

                w_idx = int(pos[w])
                is_inside = False
                if i_idx <= j_idx:
                    if i_idx <= w_idx <= j_idx: is_inside = True
                elif w_idx >= i_idx or w_idx <= j_idx: is_inside = True
                if is_inside: continue

                s_w_idx = (w_idx + 1) % n
                s_w = int(tour[s_w_idx])
                dist_wu = float(candidate_dists[u, k])
                if base_g + _dist(w, s_w, coords_x, coords_y) > dist_wu + _dist(v, s_w, coords_x, coords_y) + GAIN_EPSILON:
                    if i_idx <= j_idx < w_idx:
                        _reverse_segment(tour, i_idx, (j_idx + 1) % n, pos)
                        _reverse_segment(tour, s_v_idx, (w_idx + 1) % n, pos)
                        _reverse_segment(tour, i_idx, (w_idx + 1) % n, pos)
                    elif w_idx < i_idx <= j_idx:
                        _reverse_segment(tour, i_idx, (j_idx + 1) % n, pos)
                        _reverse_segment(tour, s_w_idx, (p_u_idx + 1) % n, pos)
                        _reverse_segment(tour, s_w_idx, (j_idx + 1) % n, pos)
                    else: continue
                    dlb[p_u] = dlb[u] = dlb[v] = dlb[s_v] = dlb[w] = dlb[s_w] = False
                    globally_improved = True
                    found = True
                    break

            if not found:
                for k in range(num_cand):
                    w = int(candidate_set[v, k])
                    if w == -1: break
                    if w in (u, v, p_u, s_v): continue
                    w_idx = int(pos[w])
                    is_inside = False
                    if i_idx <= j_idx:
                        if i_idx <= w_idx <= j_idx: is_inside = True
                    elif w_idx >= i_idx or w_idx <= j_idx: is_inside = True
                    if is_inside: continue

                    s_w_idx = (w_idx + 1) % n
                    s_w = int(tour[s_w_idx])
                    dist_wv = float(candidate_dists[v, k])
                    if base_g + _dist(w, s_w, coords_x, coords_y) > dist_wv + _dist(u, s_w, coords_x, coords_y) + GAIN_EPSILON:
                        if i_idx <= j_idx < w_idx:
                            _reverse_segment(tour, s_v_idx, (w_idx + 1) % n, pos)
                            _reverse_segment(tour, i_idx, (w_idx + 1) % n, pos)
                        elif w_idx < i_idx <= j_idx:
                            _reverse_segment(tour, s_w_idx, (p_u_idx + 1) % n, pos)
                            _reverse_segment(tour, s_w_idx, (j_idx + 1) % n, pos)
                        else: continue
                        dlb[p_u] = dlb[u] = dlb[v] = dlb[s_v] = dlb[w] = dlb[s_w] = False
                        globally_improved = True
                        found = True
                        break
        if not found:
            dlb[u] = True
    return globally_improved


@njit(fastmath=True)  # type: ignore
def _reverse_segment_between_cuts(
    tour: npt.NDArray[np.int32],
    t1_idx: int,
    t2_idx: int,
    t3_idx: int,
    t4_idx: int,
    pos: npt.NDArray[np.int32],
) -> None:
    """Helper for 4-opt and 5-opt path reversals."""
    n = tour.shape[0]
    if (t1_idx + 1) % n == t2_idx:
        _reverse_segment(tour, t2_idx, (t3_idx + 1) % n, pos)
    else:
        _reverse_segment(tour, t1_idx, (t4_idx + 1) % n, pos)


@njit(fastmath=True)  # type: ignore
def _optimize_4opt_sequential(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
) -> bool:
    """Sequential 4-opt implementation."""
    n = tour.shape[0]
    if n < MIN_TOUR_SIZE_4OPT: return False
    globally_improved = False
    num_cand_k3 = min(candidate_set.shape[1], K_NEIGHBORS)
    num_cand_k5 = min(candidate_set.shape[1], K_3OPT)
    num_cand_k7 = min(candidate_set.shape[1], K_4OPT)

    for t1_idx in range(n):
        t1 = int(tour[t1_idx])
        if dlb[t1]: continue
        t2_idx = (t1_idx + 1) % n
        t2 = int(tour[t2_idx])
        dist_t1_t2 = _dist(t1, t2, coords_x, coords_y)
        found = False
        for k3 in range(num_cand_k3):
            t3 = int(candidate_set[t2, k3])
            if t3 == -1: break
            if t3 in (t1, t2): continue
            dist_t2_t3 = float(candidate_dists[t2, k3])
            g1 = dist_t1_t2 - dist_t2_t3
            if g1 <= GAIN_EPSILON: continue
            t3_idx = int(pos[t3])
            for d1 in [1, -1]:
                t4_idx = (t3_idx + d1 + n) % n
                t4 = int(tour[t4_idx])
                if t4 in (t1, t2): continue
                dist_t3_t4 = _dist(t3, t4, coords_x, coords_y)
                for k5 in range(num_cand_k5):
                    t5 = int(candidate_set[t4, k5])
                    if t5 == -1: break
                    if t5 in (t1, t2, t3, t4): continue
                    dist_t4_t5 = float(candidate_dists[t4, k5])
                    g2 = g1 + dist_t3_t4 - dist_t4_t5
                    if g2 <= GAIN_EPSILON: continue
                    t5_idx = int(pos[t5])
                    for d2 in [1, -1]:
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = int(tour[t6_idx])
                        if t6 in (t1, t2, t3, t4, t5): continue
                        dist_t5_t6 = _dist(t5, t6, coords_x, coords_y)
                        for k7 in range(num_cand_k7):
                            t7 = int(candidate_set[t6, k7])
                            if t7 == -1: break
                            if t7 in (t1, t2, t3, t4, t5, t6): continue
                            dist_t6_t7 = float(candidate_dists[t6, k7])
                            g3 = g2 + dist_t5_t6 - dist_t6_t7
                            if g3 <= GAIN_EPSILON: continue
                            t7_idx = int(pos[t7])
                            for d3 in [1, -1]:
                                t8_idx = (t7_idx + d3 + n) % n
                                t8 = int(tour[t8_idx])
                                if t8 in (t1, t2, t3, t4, t5, t6, t7): continue
                                dist_t7_t8 = _dist(t7, t8, coords_x, coords_y)
                                if (g3 + dist_t7_t8 - _dist(t8, t1, coords_x, coords_y)) > GAIN_EPSILON:
                                    _reverse_segment_between_cuts(tour, t1_idx, t2_idx, t3_idx, t4_idx, pos)
                                    _reverse_segment_between_cuts(tour, int(pos[t1]), int(pos[t4]), int(pos[t5]), int(pos[t6]), pos)
                                    _reverse_segment_between_cuts(tour, int(pos[t1]), int(pos[t6]), int(pos[t7]), int(pos[t8]), pos)
                                    dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = dlb[t5] = dlb[t6] = dlb[t7] = dlb[t8] = False
                                    globally_improved = True
                                    found = True
                                    break
                            if found: break
                        if found: break
                    if found: break
                if found: break
            if found: break
        if not found: dlb[t1] = True
    return globally_improved


@njit(fastmath=True)  # type: ignore
def _optimize_5opt_sequential(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
) -> bool:
    """Sequential 5-opt implementation."""
    n = tour.shape[0]
    if n < MIN_TOUR_SIZE_5OPT: return False
    globally_improved = False
    num_cand_k3, num_cand_k5 = min(candidate_set.shape[1], K_NEIGHBORS), min(candidate_set.shape[1], K_3OPT)
    num_cand_k7, num_cand_k9 = min(candidate_set.shape[1], K_4OPT), min(candidate_set.shape[1], K_5OPT)

    for t1_idx in range(n):
        t1 = int(tour[t1_idx])
        if dlb[t1]: continue
        t2_idx = (t1_idx + 1) % n
        t2 = int(tour[t2_idx])
        dist_t1_t2 = _dist(t1, t2, coords_x, coords_y)
        found = False
        for k3 in range(num_cand_k3):
            t3 = int(candidate_set[t2, k3])
            if t3 == -1: break
            if t3 in (t1, t2): continue
            dist_t2_t3 = float(candidate_dists[t2, k3])
            g1 = dist_t1_t2 - dist_t2_t3
            if g1 <= GAIN_EPSILON: continue
            t3_idx = int(pos[t3])
            for d1 in [1, -1]:
                t4_idx = (t3_idx + d1 + n) % n
                t4 = int(tour[t4_idx])
                if t4 in (t1, t2): continue
                dist_t3_t4 = _dist(t3, t4, coords_x, coords_y)
                for k5 in range(num_cand_k5):
                    t5 = int(candidate_set[t4, k5])
                    if t5 == -1: break
                    if t5 in (t1, t2, t3, t4): continue
                    dist_t4_t5 = float(candidate_dists[t4, k5])
                    g2 = g1 + dist_t3_t4 - dist_t4_t5
                    if g2 <= GAIN_EPSILON: continue
                    t5_idx = int(pos[t5])
                    for d2 in [1, -1]:
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = int(tour[t6_idx])
                        if t6 in (t1, t2, t3, t4, t5): continue
                        dist_t5_t6 = _dist(t5, t6, coords_x, coords_y)
                        for k7 in range(num_cand_k7):
                            t7 = int(candidate_set[t6, k7])
                            if t7 == -1: break
                            if t7 in (t1, t2, t3, t4, t5, t6): continue
                            dist_t6_t7 = float(candidate_dists[t6, k7])
                            g3 = g2 + dist_t5_t6 - dist_t6_t7
                            if g3 <= GAIN_EPSILON: continue
                            t7_idx = int(pos[t7])
                            for d3 in [1, -1]:
                                t8_idx = (t7_idx + d3 + n) % n
                                t8 = int(tour[t8_idx])
                                if t8 in (t1, t2, t3, t4, t5, t6, t7): continue
                                dist_t7_t8 = _dist(t7, t8, coords_x, coords_y)
                                for k9 in range(num_cand_k9):
                                    t9 = int(candidate_set[t8, k9])
                                    if t9 == -1: break
                                    if t9 in (t1, t2, t3, t4, t5, t6, t7, t8): continue
                                    dist_t8_t9 = float(candidate_dists[t8, k9])
                                    g4 = g3 + dist_t7_t8 - dist_t8_t9
                                    if g4 <= GAIN_EPSILON: continue
                                    t9_idx = int(pos[t9])
                                    for d4 in [1, -1]:
                                        t10_idx = (t9_idx + d4 + n) % n
                                        t10 = int(tour[t10_idx])
                                        if t10 in (t1, t2, t3, t4, t5, t6, t7, t8, t9): continue
                                        dist_t9_t10 = _dist(t9, t10, coords_x, coords_y)
                                        if (g4 + dist_t9_t10 - _dist(t10, t1, coords_x, coords_y)) > GAIN_EPSILON:
                                            _reverse_segment_between_cuts(tour, t1_idx, t2_idx, t3_idx, t4_idx, pos)
                                            _reverse_segment_between_cuts(tour, int(pos[t1]), int(pos[t4]), int(pos[t5]), int(pos[t6]), pos)
                                            _reverse_segment_between_cuts(tour, int(pos[t1]), int(pos[t6]), int(pos[t7]), int(pos[t8]), pos)
                                            _reverse_segment_between_cuts(tour, int(pos[t1]), int(pos[t8]), int(pos[t9]), int(pos[t10]), pos)
                                            dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = dlb[t5] = dlb[t6] = dlb[t7] = dlb[t8] = dlb[t9] = dlb[t10] = False
                                            globally_improved = True
                                            found = True
                                            break
                                    if found: break
                                if found: break
                            if found: break
                        if found: break
                    if found: break
                if found: break
            if found: break
        if not found: dlb[t1] = True
    return globally_improved


@njit(fastmath=True)  # type: ignore
def _full_cascade(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
    max_opt: int = 3,
) -> bool:
    """Full cascading local search engine. Returns True if improved."""
    globally_improved = False
    improved = True
    while improved:
        improved = False
        if _optimize_2opt(tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb):
            improved = True
            globally_improved = True
            while _optimize_2opt(tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb): pass
        if _optimize_or_opt(tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb, max_len=OR_OPT_MAX_LEN):
            improved = True
            globally_improved = True
            continue
        if max_opt >= 3 and _optimize_3opt_sequential(tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb):
            improved = True
            globally_improved = True
            continue
        if max_opt >= 4 and _optimize_4opt_sequential(tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb):
            improved = True
            globally_improved = True
            continue
        if max_opt >= 5 and _optimize_5opt_sequential(tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb):
            improved = True
            globally_improved = True
            continue
    return globally_improved


@njit(fastmath=True)  # type: ignore
def _double_bridge_kick(
    tour: npt.NDArray[np.int32], pos: npt.NDArray[np.int32]
) -> None:
    """Apply a double-bridge kick to escape local optima."""
    n = tour.shape[0]
    if n < MIN_TOUR_SIZE_KICK: return
    if n <= LOCALIZED_KICK_THRESHOLD:
        indices = np.random.choice(n, 4, replace=False)
        indices.sort()
        p1, p2, p3, p4 = int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])
    else:
        W = min(LOCALIZED_KICK_W_MAX, max(LOCALIZED_KICK_W_MIN, n // LOCALIZED_KICK_W_DIVISOR))
        start = np.random.randint(0, n - W)
        offsets = np.random.choice(W, 4, replace=False)
        offsets.sort()
        p1, p2, p3, p4 = int(start + offsets[0]), int(start + offsets[1]), int(start + offsets[2]), int(start + offsets[3])

    _reverse_segment(tour, p1 + 1, p2 + 1, pos)
    _reverse_segment(tour, p2 + 1, p3 + 1, pos)
    _reverse_segment(tour, p3 + 1, p4 + 1, pos)
    _reverse_segment(tour, p1 + 1, p4 + 1, pos)


@njit(fastmath=True)  # type: ignore
def _cascading_kopt_inner(
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    chunk_size: int,
    max_opt: int,
    # Injected workspace state
    tour: npt.NDArray[np.int32],
    best_tour: npt.NDArray[np.int32],
    best_length: float,
    stagnation_count: int,
    kicked_tour: npt.NDArray[np.int32],
    pos: npt.NDArray[np.int32],
    best_pos: npt.NDArray[np.int32],
    kicked_pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], float, int]:
    """ILS kick-and-optimize loop using injected workspace."""
    n = tour.shape[0]
    stagnation_limit = max(STAGNATION_LIMIT_MIN, n // STAGNATION_LIMIT_DIVISOR)

    for _i in range(chunk_size):
        kicked_tour[:] = tour[:]
        kicked_pos[:] = pos[:]
        _double_bridge_kick(kicked_tour, kicked_pos)
        dlb.fill(False)
        _full_cascade(kicked_tour, coords_x, coords_y, candidate_set, candidate_dists, kicked_pos, dlb, max_opt=max_opt)
        length = compute_tour_length(kicked_tour, coords_x, coords_y)
        if length < best_length - GAIN_EPSILON:
            best_length, stagnation_count = length, 0
            best_tour[:], best_pos[:] = kicked_tour[:], kicked_pos[:]
            tour[:], pos[:] = kicked_tour[:], kicked_pos[:]
        else:
            stagnation_count += 1
            if stagnation_count > stagnation_limit:
                tour[:], pos[:] = best_tour[:], best_pos[:]
                dlb.fill(False)
                stagnation_count = 0
    return tour, best_tour, best_length, stagnation_count


@njit(fastmath=True) # type: ignore
def _precompute_candidate_dists_opt(
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
) -> None:
    """Fill pre-allocated candidate distance buffer (Serial)."""
    n, num_cand = candidate_set.shape
    for i in range(n):
        for k in range(num_cand):
            c_idx = int(candidate_set[i, k])
            if c_idx != -1:
                dx, dy = coords_x[i] - coords_x[c_idx], coords_y[i] - coords_y[c_idx]
                candidate_dists[i, k] = np.sqrt(dx * dx + dy * dy)


def cascading_kopt_optimize(
    initial_tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    num_kicks: int = 500,
    max_opt: int = 5,
    time_limit_s: float = -1.0,
    chunk_size: int = 1,
    progress_array: npt.NDArray[np.int32] | None = None,
    seed_idx: int = 0,
) -> tuple[npt.NDArray[np.int32], float, int]:
    """ILS engine with workspace pre-allocation."""
    n = initial_tour.shape[0]
    num_cand = candidate_set.shape[1]
    ws = KOptWorkspace(n, num_cand)

    # Precompute distances into workspace
    _precompute_candidate_dists_opt(coords_x, coords_y, candidate_set, ws.candidate_dists)

    # Initialize workspace tour and pos
    ws.tour[:] = initial_tour[:]
    _update_pos(ws.tour, ws.pos)
    ws.dlb.fill(False)

    # Initial search
    _full_cascade(ws.tour, coords_x, coords_y, candidate_set, ws.candidate_dists, ws.pos, ws.dlb, max_opt=max_opt)

    ws.best_tour[:] = ws.tour[:]
    ws.best_pos[:] = ws.pos[:]
    best_length = float(compute_tour_length(ws.tour, coords_x, coords_y))
    stagnation_count, kicks_done = 0, 0

    t_start = time.monotonic()
    deadline = t_start + time_limit_s * TIME_SAFETY_MARGIN if time_limit_s > 0 else float("inf")

    while kicks_done < num_kicks:
        if time.monotonic() >= deadline:
            break
        this_chunk = min(chunk_size, num_kicks - kicks_done)
        ws.tour, ws.best_tour, best_length, stagnation_count = _cascading_kopt_inner(
            coords_x,
            coords_y,
            candidate_set,
            ws.candidate_dists,
            this_chunk,
            max_opt,
            ws.tour,
            ws.best_tour,
            best_length,
            stagnation_count,
            ws.kicked_tour,
            ws.pos,
            ws.best_pos,
            ws.kicked_pos,
            ws.dlb,
        )
        kicks_done += this_chunk
        if progress_array is not None:
            progress_array[seed_idx] = kicks_done

    return ws.best_tour.copy(), best_length, kicks_done
