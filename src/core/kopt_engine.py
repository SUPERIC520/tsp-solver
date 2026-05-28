"""K-opt engine for TSP optimization.

This module provides high-performance K-opt (2-opt, 3-opt, 4-opt, 5-opt) and
Or-opt local search implementations using Numba for acceleration. It also
includes an Iterated Local Search (ILS) framework with double-bridge kicks.
"""

import time
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit, prange

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


@njit(inline="always", fastmath=True)  # type: ignore
def _dist(
    c1: int,
    c2: int,
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
) -> float:
    dx = coords_x[c1] - coords_x[c2]
    dy = coords_y[c1] - coords_y[c2]
    return float(np.sqrt(dx * dx + dy * dy))


@njit(fastmath=True, cache=True)  # type: ignore
def compute_tour_length(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
) -> float:
    """Compute the total length of a TSP tour.

    Args:
        tour: Array of city indices representing the tour.
        coords_x: X-coordinates of the cities.
        coords_y: Y-coordinates of the cities.

    Returns:
        The total Euclidean length of the tour.
    """
    length = 0.0
    n = tour.shape[0]
    for i in range(n):
        length += _dist(int(tour[i]), int(tour[(i + 1) % n]), coords_x, coords_y)
    return float(length)


@njit(fastmath=True, cache=True)  # type: ignore
def _update_pos(tour: npt.NDArray[np.int32], pos: npt.NDArray[np.int32]) -> None:
    for i in range(tour.shape[0]):
        pos[int(tour[i])] = i


@njit(fastmath=True, cache=True)  # type: ignore
def _reverse_segment(tour: npt.NDArray[np.int32], i: int, j: int) -> None:
    n = tour.shape[0]
    i = i % n
    j = j % n
    # Reverse segment from i (inclusive) to j (exclusive)
    size = (j - i + n) % n
    if size <= 1:
        return
    if i < j:
        count = size // 2
        for k in range(count):
            idx1 = i + k
            idx2 = j - 1 - k
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
    else:
        count = size // 2
        idx1 = i
        idx2 = (j - 1 + n) % n
        for _k in range(count):
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
            idx1 += 1
            if idx1 == n:
                idx1 = 0
            idx2 -= 1
            if idx2 == -1:
                idx2 = n - 1


@njit(fastmath=True, cache=True)  # type: ignore
def _optimize_2opt(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
) -> bool:
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
                _reverse_segment(tour, v_idx, (w_idx + 1) % n)
                _update_pos(tour, pos)
                dlb[u] = dlb[v] = dlb[w] = dlb[x] = False
                globally_improved = True
                found = True
                break
        if not found:
            dlb[u] = True
    return globally_improved


@njit(fastmath=True, cache=True)  # type: ignore
def _get_cut_indices(pos1: int, pos2: int, n: int) -> int:
    if (pos1 + 1) % n == pos2:
        return pos1
    return pos2


@njit(fastmath=True, cache=True)  # type: ignore
def _reconstruct_tour_3opt(
    tour: npt.NDArray[np.int32],
    idx_1: int,
    idx_2: int,
    idx_3: int,
    case_idx: int,
) -> None:
    # Sort indices
    i1, i2, i3 = idx_1, idx_2, idx_3
    if i1 > i2:
        i1, i2 = i2, i1
    if i2 > i3:
        i2, i3 = i3, i2
    if i1 > i2:
        i1, i2 = i2, i1

    # Segments:
    # S1: from i1 + 1 to i2 (inclusive)
    # S2: from i2 + 1 to i3 (inclusive)
    # S3: from i3 + 1 to i1 (inclusive, wrapping around)

    match case_idx:
        case 1:
            pass
        case 2:
            _reverse_segment(tour, i1 + 1, i2 + 1)
        case 3:
            _reverse_segment(tour, i2 + 1, i3 + 1)
        case 4:
            _reverse_segment(tour, i3 + 1, i1 + 1)
        case 5:
            _reverse_segment(tour, i2 + 1, i3 + 1)
            _reverse_segment(tour, i3 + 1, i1 + 1)
        case 6:
            _reverse_segment(tour, i2 + 1, i3 + 1)
            _reverse_segment(tour, i3 + 1, i1 + 1)
            _reverse_segment(tour, i2 + 1, i1 + 1)
        case 7:
            _reverse_segment(tour, i2 + 1, i3 + 1)
            _reverse_segment(tour, i2 + 1, i1 + 1)
        case 8:
            _reverse_segment(tour, i3 + 1, i1 + 1)
            _reverse_segment(tour, i2 + 1, i1 + 1)




@njit(fastmath=True, cache=True)  # type: ignore
def _optimize_3opt_sequential(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
) -> bool:
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
            # Prune early if first cut has negative gain
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

                        # Sort indices
                        i1, i2, i3 = idx_1, idx_2, idx_3
                        if i1 > i2:
                            i1, i2 = i2, i1
                        if i2 > i3:
                            i2, i3 = i3, i2
                        if i1 > i2:
                            i1, i2 = i2, i1

                        # Edges
                        a = int(tour[i1])
                        b = int(tour[(i1 + 1) % n])
                        c = int(tour[i2])
                        d = int(tour[(i2 + 1) % n])
                        e = int(tour[i3])
                        f = int(tour[(i3 + 1) % n])

                        E_remove = (
                            _dist(a, b, coords_x, coords_y)
                            + _dist(c, d, coords_x, coords_y)
                            + _dist(e, f, coords_x, coords_y)
                        )

                        best_case = -1
                        best_delta = -GAIN_EPSILON

                        # Case 2 (2-opt reversing S1)
                        delta2 = (
                            _dist(a, c, coords_x, coords_y)
                            + _dist(b, d, coords_x, coords_y)
                            - _dist(a, b, coords_x, coords_y)
                            - _dist(c, d, coords_x, coords_y)
                        )
                        if delta2 < best_delta:
                            best_delta = delta2
                            best_case = 2

                        # Case 3 (2-opt reversing S2)
                        delta3 = (
                            _dist(c, e, coords_x, coords_y)
                            + _dist(d, f, coords_x, coords_y)
                            - _dist(c, d, coords_x, coords_y)
                            - _dist(e, f, coords_x, coords_y)
                        )
                        if delta3 < best_delta:
                            best_delta = delta3
                            best_case = 3

                        # Case 4 (2-opt reversing S3)
                        delta4 = (
                            _dist(e, a, coords_x, coords_y)
                            + _dist(f, b, coords_x, coords_y)
                            - _dist(e, f, coords_x, coords_y)
                            - _dist(a, b, coords_x, coords_y)
                        )
                        if delta4 < best_delta:
                            best_delta = delta4
                            best_case = 4

                        # Case 5 (3-opt S1 + S2_rev + S3_rev)
                        delta5 = (
                            _dist(c, e, coords_x, coords_y)
                            + _dist(d, a, coords_x, coords_y)
                            + _dist(f, b, coords_x, coords_y)
                            - E_remove
                        )
                        if delta5 < best_delta:
                            best_delta = delta5
                            best_case = 5

                        # Case 6 (3-opt S1 + S3 + S2)
                        delta6 = (
                            _dist(c, f, coords_x, coords_y)
                            + _dist(a, d, coords_x, coords_y)
                            + _dist(e, b, coords_x, coords_y)
                            - E_remove
                        )
                        if delta6 < best_delta:
                            best_delta = delta6
                            best_case = 6

                        # Case 7 (3-opt S1 + S3_rev + S2)
                        delta7 = (
                            _dist(c, a, coords_x, coords_y)
                            + _dist(f, d, coords_x, coords_y)
                            + _dist(e, b, coords_x, coords_y)
                            - E_remove
                        )
                        if delta7 < best_delta:
                            best_delta = delta7
                            best_case = 7

                        # Case 8 (3-opt S1 + S3 + S2_rev)
                        delta8 = (
                            _dist(c, f, coords_x, coords_y)
                            + _dist(a, e, coords_x, coords_y)
                            + _dist(d, b, coords_x, coords_y)
                            - E_remove
                        )
                        if delta8 < best_delta:
                            best_delta = delta8
                            best_case = 8

                        if best_case != -1:
                            _reconstruct_tour_3opt(tour, i1, i2, i3, best_case)
                            _update_pos(tour, pos)

                            dlb[a] = dlb[b] = dlb[c] = dlb[d] = dlb[e] = dlb[f] = False
                            globally_improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            dlb[t1] = True
    return globally_improved


@njit(fastmath=True, cache=True)  # type: ignore
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
    n = tour.shape[0]
    globally_improved = False

    num_cand = min(candidate_set.shape[1], K_NEIGHBORS)

    for i_idx in range(n):
        u = int(tour[i_idx])
        if dlb[u]:
            continue
        found = False
        for length in range(1, max_len + 1):
            if length >= n - 2:
                break
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
            if base_g <= GAIN_EPSILON:
                continue

            for k in range(num_cand):
                w = int(candidate_set[u, k])
                if w == -1:
                    break
                if w in (u, v, p_u, s_v):
                    continue

                w_idx = int(pos[w])
                # Check if w is inside the segment [u, v]
                is_inside = False
                if i_idx <= j_idx:
                    if i_idx <= w_idx <= j_idx:
                        is_inside = True
                elif w_idx >= i_idx or w_idx <= j_idx:
                    is_inside = True
                if is_inside:
                    continue

                s_w_idx = (w_idx + 1) % n
                s_w = int(tour[s_w_idx])

                dist_wu = float(candidate_dists[u, k])
                if (
                    base_g + _dist(w, s_w, coords_x, coords_y)
                    > dist_wu + _dist(v, s_w, coords_x, coords_y) + GAIN_EPSILON
                ):
                    # Relocate segment [u, v] (indices i_idx to j_idx) to after w
                    if i_idx <= j_idx < w_idx:
                        _reverse_segment(tour, i_idx, (j_idx + 1) % n)
                        _reverse_segment(tour, s_v_idx, (w_idx + 1) % n)
                        _reverse_segment(tour, i_idx, (w_idx + 1) % n)
                    elif w_idx < i_idx <= j_idx:
                        _reverse_segment(tour, i_idx, (j_idx + 1) % n)
                        _reverse_segment(tour, s_w_idx, (p_u_idx + 1) % n)
                        _reverse_segment(tour, s_w_idx, (j_idx + 1) % n)
                    else:
                        continue
                    _update_pos(tour, pos)
                    dlb[p_u] = dlb[u] = dlb[v] = dlb[s_v] = dlb[w] = dlb[s_w] = False
                    globally_improved = True
                    found = True
                    break

            # Also try inserting the segment in reversed order
            if not found:
                for k in range(num_cand):
                    w = int(candidate_set[v, k])
                    if w == -1:
                        break
                    if w in (u, v, p_u, s_v):
                        continue

                    w_idx = int(pos[w])
                    # Check if w is inside the segment [u, v]
                    is_inside = False
                    if i_idx <= j_idx:
                        if i_idx <= w_idx <= j_idx:
                            is_inside = True
                    elif w_idx >= i_idx or w_idx <= j_idx:
                        is_inside = True
                    if is_inside:
                        continue

                    s_w_idx = (w_idx + 1) % n
                    s_w = int(tour[s_w_idx])

                    dist_wv = float(candidate_dists[v, k])
                    if (
                        base_g + _dist(w, s_w, coords_x, coords_y)
                        > dist_wv + _dist(u, s_w, coords_x, coords_y) + GAIN_EPSILON
                    ):
                        if i_idx <= j_idx < w_idx:
                            _reverse_segment(tour, s_v_idx, (w_idx + 1) % n)
                            _reverse_segment(tour, i_idx, (w_idx + 1) % n)
                        elif w_idx < i_idx <= j_idx:
                            _reverse_segment(tour, s_w_idx, (p_u_idx + 1) % n)
                            _reverse_segment(tour, s_w_idx, (j_idx + 1) % n)
                        else:
                            continue
                        _update_pos(tour, pos)
                        dlb[p_u] = dlb[u] = dlb[v] = dlb[s_v] = dlb[w] = dlb[s_w] = (
                            False
                        )
                        globally_improved = True
                        found = True
                        break

        if not found:
            dlb[u] = True
    return globally_improved


@njit(fastmath=True, cache=True)  # type: ignore
def _reverse_segment_between_cuts(
    tour: npt.NDArray[np.int32],
    t1_idx: int,
    t2_idx: int,
    t3_idx: int,
    t4_idx: int,
) -> None:
    n = tour.shape[0]
    if (t1_idx + 1) % n == t2_idx:
        _reverse_segment(tour, t2_idx, (t3_idx + 1) % n)
    else:
        _reverse_segment(tour, t1_idx, (t4_idx + 1) % n)


@njit(fastmath=True, cache=True)  # type: ignore
def _optimize_4opt_sequential(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
) -> bool:
    n = tour.shape[0]
    if n < MIN_TOUR_SIZE_4OPT:
        return False
    globally_improved = False

    num_cand_k3 = min(candidate_set.shape[1], K_NEIGHBORS)
    num_cand_k5 = min(candidate_set.shape[1], K_3OPT)
    num_cand_k7 = min(candidate_set.shape[1], K_4OPT)

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
            for d1 in [1, -1]:
                t4_idx = (t3_idx + d1 + n) % n
                t4 = int(tour[t4_idx])
                if t4 in (t1, t2):
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
                    for d2 in [1, -1]:
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = int(tour[t6_idx])
                        if t6 in (t1, t2, t3, t4, t5):
                            continue
                        dist_t5_t6 = _dist(t5, t6, coords_x, coords_y)

                        for k7 in range(num_cand_k7):
                            t7 = int(candidate_set[t6, k7])
                            if t7 == -1:
                                break
                            if t7 in (t1, t2, t3, t4, t5, t6):
                                continue
                            dist_t6_t7 = float(candidate_dists[t6, k7])
                            g3 = g2 + dist_t5_t6 - dist_t6_t7
                            if g3 <= GAIN_EPSILON:
                                continue

                            t7_idx = int(pos[t7])
                            for d3 in [1, -1]:
                                t8_idx = (t7_idx + d3 + n) % n
                                t8 = int(tour[t8_idx])
                                if t8 in (t1, t2, t3, t4, t5, t6, t7):
                                    continue
                                dist_t7_t8 = _dist(t7, t8, coords_x, coords_y)

                                g4_part = g3 + dist_t7_t8
                                gain = g4_part - _dist(t8, t1, coords_x, coords_y)
                                if gain > GAIN_EPSILON:
                                    _reverse_segment_between_cuts(
                                        tour,
                                        t1_idx,
                                        t2_idx,
                                        t3_idx,
                                        t4_idx,
                                    )
                                    _reverse_segment_between_cuts(
                                        tour,
                                        int(pos[t1]),
                                        int(pos[t4]),
                                        int(pos[t5]),
                                        int(pos[t6]),
                                    )
                                    _reverse_segment_between_cuts(
                                        tour,
                                        int(pos[t1]),
                                        int(pos[t6]),
                                        int(pos[t7]),
                                        int(pos[t8]),
                                    )
                                    _update_pos(tour, pos)

                                    dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = dlb[
                                        t5
                                    ] = dlb[t6] = dlb[t7] = dlb[t8] = False
                                    globally_improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            dlb[t1] = True
    return globally_improved


@njit(fastmath=True, cache=True)  # type: ignore
def _optimize_5opt_sequential(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
) -> bool:
    n = tour.shape[0]
    if n < MIN_TOUR_SIZE_5OPT:
        return False
    globally_improved = False

    num_cand_k3 = min(candidate_set.shape[1], K_NEIGHBORS)
    num_cand_k5 = min(candidate_set.shape[1], K_3OPT)
    num_cand_k7 = min(candidate_set.shape[1], K_4OPT)
    num_cand_k9 = min(candidate_set.shape[1], K_5OPT)

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
            for d1 in [1, -1]:
                t4_idx = (t3_idx + d1 + n) % n
                t4 = int(tour[t4_idx])
                if t4 in (t1, t2):
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
                    for d2 in [1, -1]:
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = int(tour[t6_idx])
                        if t6 in (t1, t2, t3, t4, t5):
                            continue
                        dist_t5_t6 = _dist(t5, t6, coords_x, coords_y)

                        for k7 in range(num_cand_k7):
                            t7 = int(candidate_set[t6, k7])
                            if t7 == -1:
                                break
                            if t7 in (t1, t2, t3, t4, t5, t6):
                                continue
                            dist_t6_t7 = float(candidate_dists[t6, k7])
                            g3 = g2 + dist_t5_t6 - dist_t6_t7
                            if g3 <= GAIN_EPSILON:
                                continue

                            t7_idx = int(pos[t7])
                            for d3 in [1, -1]:
                                t8_idx = (t7_idx + d3 + n) % n
                                t8 = int(tour[t8_idx])
                                if t8 in (t1, t2, t3, t4, t5, t6, t7):
                                    continue
                                dist_t7_t8 = _dist(t7, t8, coords_x, coords_y)

                                for k9 in range(num_cand_k9):
                                    t9 = int(candidate_set[t8, k9])
                                    if t9 == -1:
                                        break
                                    if t9 in (t1, t2, t3, t4, t5, t6, t7, t8):
                                        continue
                                    dist_t8_t9 = float(candidate_dists[t8, k9])
                                    g4 = g3 + dist_t7_t8 - dist_t8_t9
                                    if g4 <= GAIN_EPSILON:
                                        continue

                                    t9_idx = int(pos[t9])
                                    for d4 in [1, -1]:
                                        t10_idx = (t9_idx + d4 + n) % n
                                        t10 = int(tour[t10_idx])
                                        if t10 in (t1, t2, t3, t4, t5, t6, t7, t8, t9):
                                            continue
                                        dist_t9_t10 = _dist(t9, t10, coords_x, coords_y)

                                        g5_part = g4 + dist_t9_t10
                                        gain = g5_part - _dist(
                                            t10, t1, coords_x, coords_y
                                        )
                                        if gain > GAIN_EPSILON:
                                            _reverse_segment_between_cuts(
                                                tour,
                                                t1_idx,
                                                t2_idx,
                                                t3_idx,
                                                t4_idx,
                                            )
                                            _reverse_segment_between_cuts(
                                                tour,
                                                int(pos[t1]),
                                                int(pos[t4]),
                                                int(pos[t5]),
                                                int(pos[t6]),
                                            )
                                            _reverse_segment_between_cuts(
                                                tour,
                                                int(pos[t1]),
                                                int(pos[t6]),
                                                int(pos[t7]),
                                                int(pos[t8]),
                                            )
                                            _reverse_segment_between_cuts(
                                                tour,
                                                int(pos[t1]),
                                                int(pos[t8]),
                                                int(pos[t9]),
                                                int(pos[t10]),
                                            )
                                            _update_pos(tour, pos)

                                            dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = dlb[
                                                t5
                                            ] = dlb[t6] = dlb[t7] = dlb[t8] = dlb[
                                                t9
                                            ] = dlb[t10] = False
                                            globally_improved = True
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            dlb[t1] = True
    return globally_improved


@njit(fastmath=True, cache=True)  # type: ignore
def _full_cascade(
    tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    pos: npt.NDArray[np.int32],
    dlb: npt.NDArray[np.bool_],
    max_opt: int = 3,
) -> None:
    improved = True
    while improved:
        improved = False

        # 2-opt search
        if _optimize_2opt(
            tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb
        ):
            improved = True
            while _optimize_2opt(
                tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb
            ):
                pass

        # Or-opt search
        if _optimize_or_opt(
            tour,
            coords_x,
            coords_y,
            candidate_set,
            candidate_dists,
            pos,
            dlb,
            max_len=OR_OPT_MAX_LEN,
        ):
            improved = True
            continue

        # Or-opt and 3-opt sequential
        if max_opt >= 3 and _optimize_3opt_sequential(  # noqa: PLR2004
            tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb
        ):
            improved = True
            continue

        if max_opt >= 4 and _optimize_4opt_sequential(  # noqa: PLR2004
            tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb
        ):
            improved = True
            continue

        if max_opt >= 5 and _optimize_5opt_sequential(  # noqa: PLR2004
            tour, coords_x, coords_y, candidate_set, candidate_dists, pos, dlb
        ):
            improved = True
            continue


@njit(fastmath=True, cache=True)  # type: ignore
def _double_bridge_kick(
    tour: npt.NDArray[np.int32], pos: npt.NDArray[np.int32]
) -> None:
    n = tour.shape[0]
    if n < MIN_TOUR_SIZE_KICK:
        return

    # Choose 4 positions to cut the tour.
    # For large tours, use a localized window to avoid creating long,
    # unrepairable edges.
    if n <= LOCALIZED_KICK_THRESHOLD:
        indices = np.random.choice(n, 4, replace=False)  # noqa: NPY002
        indices.sort()
        p1, p2, p3, p4 = (
            int(indices[0]),
            int(indices[1]),
            int(indices[2]),
            int(indices[3]),
        )
    else:
        # Localized kick
        W = min(
            LOCALIZED_KICK_W_MAX,
            max(LOCALIZED_KICK_W_MIN, n // LOCALIZED_KICK_W_DIVISOR),
        )
        start = np.random.randint(0, n - W)  # noqa: NPY002
        offsets = np.random.choice(W, 4, replace=False)  # noqa: NPY002
        offsets.sort()
        p1 = int(start + offsets[0])
        p2 = int(start + offsets[1])
        p3 = int(start + offsets[2])
        p4 = int(start + offsets[3])

    # Reconnect as: [0, p1], [p3+1, p4], [p2+1, p3], [p1+1, p2], [p4+1, n-1]
    # This is equivalent to reversing the segments B, C, D and then B+C+D in-place:
    # B: p1 + 1 to p2 (inclusive) -> reverse_segment(tour, p1 + 1, p2 + 1)
    # C: p2 + 1 to p3 (inclusive) -> reverse_segment(tour, p2 + 1, p3 + 1)
    # D: p3 + 1 to p4 (inclusive) -> reverse_segment(tour, p3 + 1, p4 + 1)
    # B+C+D: p1 + 1 to p4 (inclusive) -> reverse_segment(tour, p1 + 1, p4 + 1)
    _reverse_segment(tour, p1 + 1, p2 + 1)
    _reverse_segment(tour, p2 + 1, p3 + 1)
    _reverse_segment(tour, p3 + 1, p4 + 1)
    _reverse_segment(tour, p1 + 1, p4 + 1)
    _update_pos(tour, pos)



@njit(fastmath=True, cache=True)  # type: ignore
def _cascading_kopt_inner(
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    candidate_dists: npt.NDArray[np.float64],
    chunk_size: int,
    max_opt: int,
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
    """Inner @njit kick loop.

    Accepts externally-supplied state so the Python
    wrapper can resume across chunk boundaries. Returns updated
    (tour, best_tour, best_length, stagnation_count).
    """
    n = tour.shape[0]
    stagnation_limit = max(STAGNATION_LIMIT_MIN, n // STAGNATION_LIMIT_DIVISOR)

    for _i in range(chunk_size):
        kicked_tour[:] = tour[:]
        kicked_pos[:] = pos[:]
        _double_bridge_kick(kicked_tour, kicked_pos)
        dlb.fill(False)  # noqa: FBT003
        _full_cascade(
            kicked_tour,
            coords_x,
            coords_y,
            candidate_set,
            candidate_dists,
            kicked_pos,
            dlb,
            max_opt=max_opt,
        )
        length = compute_tour_length(kicked_tour, coords_x, coords_y)
        if length < best_length - GAIN_EPSILON:
            best_length = length
            best_tour[:] = kicked_tour[:]
            best_pos[:] = kicked_pos[:]
            tour[:] = kicked_tour[:]
            pos[:] = kicked_pos[:]
            stagnation_count = 0
        else:
            stagnation_count += 1
            if stagnation_count > stagnation_limit:
                tour[:] = best_tour[:]
                pos[:] = best_pos[:]
                dlb.fill(False)  # noqa: FBT003  # CRITICAL: clear DLB after reset!
                stagnation_count = 0

    return tour, best_tour, best_length, stagnation_count


@njit(fastmath=True, parallel=True)  # type: ignore
def _precompute_candidate_dists(
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    n, num_cand = candidate_set.shape
    candidate_dists = np.zeros((n, num_cand), dtype=np.float64)
    for i in prange(n):
        for k in range(num_cand):
            c_idx = int(candidate_set[i, k])
            if c_idx != -1:
                dx = coords_x[i] - coords_x[c_idx]
                dy = coords_y[i] - coords_y[c_idx]
                candidate_dists[i, k] = np.sqrt(dx * dx + dy * dy)
    return candidate_dists


def cascading_kopt_optimize(
    initial_tour: npt.NDArray[np.int32],
    coords_x: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    candidate_set: npt.NDArray[np.int32],
    num_kicks: int = 500,
    max_opt: int = 5,
    time_limit_s: float = -1.0,
    chunk_size: int = 1,
    progress_array: Any = None,  # noqa: ANN401
    seed_idx: int = 0,
) -> tuple[npt.NDArray[np.int32], float, int]:
    """Python-level wrapper around _cascading_kopt_inner.

    Runs ILS in `chunk_size` kick batches. If `time_limit_s > 0`, stops
    before starting the next chunk when elapsed wall time exceeds
    `time_limit_s * 0.97` (3% safety margin), ensuring we never go over
    budget.
    """
    # Force C-contiguity and 64-byte alignment on input arrays
    coords_x = ensure_alignment(np.ascontiguousarray(coords_x, dtype=np.float64))
    coords_y = ensure_alignment(np.ascontiguousarray(coords_y, dtype=np.float64))
    candidate_set = ensure_alignment(
        np.ascontiguousarray(candidate_set, dtype=np.int32)
    )
    initial_tour = ensure_alignment(np.ascontiguousarray(initial_tour, dtype=np.int32))

    # Assert alignments and layouts
    assert coords_x.flags["C_CONTIGUOUS"]
    assert coords_y.flags["C_CONTIGUOUS"]
    assert candidate_set.flags["C_CONTIGUOUS"]
    assert initial_tour.flags["C_CONTIGUOUS"]

    assert coords_x.ctypes.data % 64 == 0
    assert coords_y.ctypes.data % 64 == 0
    assert candidate_set.ctypes.data % 64 == 0
    assert initial_tour.ctypes.data % 64 == 0

    n = initial_tour.shape[0]

    # Precompute candidate distances using parallel JIT
    candidate_dists = _precompute_candidate_dists(coords_x, coords_y, candidate_set)
    candidate_dists = ensure_alignment(
        np.ascontiguousarray(candidate_dists, dtype=np.float64)
    )
    assert candidate_dists.flags["C_CONTIGUOUS"]
    assert candidate_dists.ctypes.data % 64 == 0

    tour = ensure_alignment(initial_tour.copy())
    pos = ensure_alignment(np.empty(n, dtype=np.int32))
    for i in range(n):
        pos[int(tour[i])] = i
    dlb = ensure_alignment(np.zeros(n, dtype=np.bool_))

    # Initial local search before kicks
    _full_cascade(
        tour,
        coords_x,
        coords_y,
        candidate_set,
        candidate_dists,
        pos,
        dlb,
        max_opt=max_opt,
    )

    best_tour = ensure_alignment(tour.copy())
    best_pos = ensure_alignment(pos.copy())
    best_length = float(compute_tour_length(tour, coords_x, coords_y))
    stagnation_count = 0
    kicked_tour = ensure_alignment(np.empty(n, dtype=np.int32))
    kicked_pos = ensure_alignment(np.empty(n, dtype=np.int32))

    t_start = time.monotonic()
    deadline = (
        t_start + time_limit_s * TIME_SAFETY_MARGIN
        if time_limit_s > 0
        else float("inf")
    )

    kicks_done = 0
    while kicks_done < num_kicks:
        # Check time budget before starting next chunk
        if time.monotonic() >= deadline:
            break
        this_chunk = min(chunk_size, num_kicks - kicks_done)
        tour, best_tour, best_length, stagnation_count = _cascading_kopt_inner(
            coords_x,
            coords_y,
            candidate_set,
            candidate_dists,
            this_chunk,
            max_opt,
            tour,
            best_tour,
            best_length,
            stagnation_count,
            kicked_tour,
            pos,
            best_pos,
            kicked_pos,
            dlb,
        )
        kicks_done += this_chunk

        # Update shared memory
        if progress_array is not None:
            progress_array[seed_idx] = kicks_done

    return best_tour, best_length, kicks_done
