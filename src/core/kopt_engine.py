import numpy as np
import time
import multiprocessing as mp
from typing import Tuple, Optional
from numba import njit, prange


@njit(inline="always", fastmath=True)  # type: ignore
def _dist(c1: int, c2: int, coords_x: np.ndarray, coords_y: np.ndarray) -> float:
    dx = coords_x[c1] - coords_x[c2]
    dy = coords_y[c1] - coords_y[c2]
    return float(np.sqrt(dx * dx + dy * dy))


@njit(fastmath=True, cache=True)  # type: ignore
def compute_tour_length(
    tour: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
) -> float:
    length = 0.0
    n = tour.shape[0]
    for i in range(n):
        length += _dist(tour[i], tour[(i + 1) % n], coords_x, coords_y)
    return float(length)


@njit(fastmath=True, cache=True)  # type: ignore
def _apply_2opt(tour: np.ndarray, i: int, j: int) -> None:
    n = tour.shape[0]
    if i == j:
        return
    # Reverse segment from i to j (inclusive)
    size = (j - i + n) % n + 1
    count = size // 2
    for k in range(count):
        idx1 = (i + k) % n
        idx2 = (j - k + n) % n
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]


@njit(fastmath=True, cache=True)  # type: ignore
def _update_pos(tour: np.ndarray, pos: np.ndarray) -> None:
    for i in range(tour.shape[0]):
        pos[tour[i]] = i


@njit(fastmath=True, cache=True)  # type: ignore
def _optimize_2opt(
    tour: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    candidate_set: np.ndarray,
    candidate_dists: np.ndarray,
    locked_edges: np.ndarray,
    pos: np.ndarray,
    dlb: np.ndarray,
) -> bool:
    n = tour.shape[0]
    if n < 4:
        return False
    globally_improved = False
    for u_idx in range(n):
        u = tour[u_idx]
        if dlb[u]:
            continue
        v_idx = (u_idx + 1) % n
        v = tour[v_idx]
        if locked_edges[u, 0] == v or locked_edges[u, 1] == v:
            continue

        dist_uv = _dist(u, v, coords_x, coords_y)
        found = False
        for k in range(candidate_set.shape[1]):
            w = candidate_set[u, k]
            if w == -1:
                break
            if w == u or w == v:
                continue
            dist_uw = candidate_dists[u, k]
            if dist_uw >= dist_uv:
                continue

            w_idx = pos[w]
            x_idx = (w_idx + 1) % n
            x = tour[x_idx]
            if x == u or x == v:
                continue
            if locked_edges[w, 0] == x or locked_edges[w, 1] == x:
                continue

            if (
                dist_uv + _dist(w, x, coords_x, coords_y)
                > dist_uw + _dist(v, x, coords_x, coords_y) + 1e-9
            ):
                _apply_2opt(tour, v_idx, w_idx)
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
    else:
        return pos2


@njit(fastmath=True, cache=True)  # type: ignore
def _reconstruct_tour_3opt(
    tour: np.ndarray,
    idx_1: int,
    idx_2: int,
    idx_3: int,
    case_idx: int,
) -> np.ndarray:
    n = tour.shape[0]
    new_tour = np.empty_like(tour)
    curr = 0

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

    if case_idx == 4:
        # S1 + S2_rev + S3_rev
        for i in range(i1 + 1, i2 + 1):
            new_tour[curr] = tour[i]
            curr += 1
        for i in range(i2 + 1, i3 + 1):
            idx = i3 - (i - (i2 + 1))
            new_tour[curr] = tour[idx]
            curr += 1
        L3 = (i1 - i3 + n) % n
        for k in range(L3):
            idx = (i1 - k + n) % n
            new_tour[curr] = tour[idx]
            curr += 1

    elif case_idx == 5:
        # S1 + S3 + S2
        for i in range(i1 + 1, i2 + 1):
            new_tour[curr] = tour[i]
            curr += 1
        L3 = (i1 - i3 + n) % n
        for k in range(L3):
            idx = (i3 + 1 + k) % n
            new_tour[curr] = tour[idx]
            curr += 1
        for i in range(i2 + 1, i3 + 1):
            new_tour[curr] = tour[i]
            curr += 1

    elif case_idx == 6:
        # S1 + S3_rev + S2
        for i in range(i1 + 1, i2 + 1):
            new_tour[curr] = tour[i]
            curr += 1
        L3 = (i1 - i3 + n) % n
        for k in range(L3):
            idx = (i1 - k + n) % n
            new_tour[curr] = tour[idx]
            curr += 1
        for i in range(i2 + 1, i3 + 1):
            new_tour[curr] = tour[i]
            curr += 1

    elif case_idx == 7:
        # S1 + S3 + S2_rev
        for i in range(i1 + 1, i2 + 1):
            new_tour[curr] = tour[i]
            curr += 1
        L3 = (i1 - i3 + n) % n
        for k in range(L3):
            idx = (i3 + 1 + k) % n
            new_tour[curr] = tour[idx]
            curr += 1
        for i in range(i2 + 1, i3 + 1):
            idx = i3 - (i - (i2 + 1))
            new_tour[curr] = tour[idx]
            curr += 1

    elif case_idx == 8:
        # S1 + S3_rev + S2_rev
        for i in range(i1 + 1, i2 + 1):
            new_tour[curr] = tour[i]
            curr += 1
        L3 = (i1 - i3 + n) % n
        for k in range(L3):
            idx = (i1 - k + n) % n
            new_tour[curr] = tour[idx]
            curr += 1
        for i in range(i2 + 1, i3 + 1):
            idx = i3 - (i - (i2 + 1))
            new_tour[curr] = tour[idx]
            curr += 1

    return new_tour


@njit(fastmath=True, cache=True)  # type: ignore
def _optimize_3opt_sequential(
    tour: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    candidate_set: np.ndarray,
    candidate_dists: np.ndarray,
    locked_edges: np.ndarray,
    pos: np.ndarray,
    dlb: np.ndarray,
) -> bool:
    n = tour.shape[0]
    if n < 6:
        return False
    globally_improved = False
    num_cand = candidate_set.shape[1]

    for t1_idx in range(n):
        t1 = tour[t1_idx]
        if dlb[t1]:
            continue

        t2_idx = (t1_idx + 1) % n
        t2 = tour[t2_idx]
        if locked_edges[t1, 0] == t2 or locked_edges[t1, 1] == t2:
            continue
        dist_t1_t2 = _dist(t1, t2, coords_x, coords_y)

        found = False
        for k3 in range(num_cand):
            t3 = candidate_set[t2, k3]
            if t3 == -1:
                break
            if t3 == t1 or t3 == t2:
                continue
            dist_t2_t3 = candidate_dists[t2, k3]
            g1 = dist_t1_t2 - dist_t2_t3
            if g1 <= 1e-9:
                continue

            t3_idx = pos[t3]
            for d1 in (-1, 1):
                t4_idx = (t3_idx + d1 + n) % n
                t4 = tour[t4_idx]
                if t4 == t1 or t4 == t2 or t4 == t3:
                    continue
                if locked_edges[t3, 0] == t4 or locked_edges[t3, 1] == t4:
                    continue
                dist_t3_t4 = _dist(t3, t4, coords_x, coords_y)

                for k5 in range(num_cand):
                    t5 = candidate_set[t4, k5]
                    if t5 == -1:
                        break
                    if t5 == t1 or t5 == t2 or t5 == t3 or t5 == t4:
                        continue
                    dist_t4_t5 = candidate_dists[t4, k5]
                    g2 = g1 + dist_t3_t4 - dist_t4_t5
                    if g2 <= 1e-9:
                        continue

                    t5_idx = pos[t5]
                    for d2 in (-1, 1):
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = tour[t6_idx]
                        if t6 == t1 or t6 == t2 or t6 == t3 or t6 == t4 or t6 == t5:
                            continue
                        if locked_edges[t5, 0] == t6 or locked_edges[t5, 1] == t6:
                            continue

                        idx_1 = _get_cut_indices(t1_idx, t2_idx, n)
                        idx_2 = _get_cut_indices(t3_idx, t4_idx, n)
                        idx_3 = _get_cut_indices(t5_idx, t6_idx, n)

                        if idx_1 == idx_2 or idx_2 == idx_3 or idx_1 == idx_3:
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
                        a = tour[i1]
                        b = tour[(i1 + 1) % n]
                        c = tour[i2]
                        d = tour[(i2 + 1) % n]
                        e = tour[i3]
                        f = tour[(i3 + 1) % n]

                        # Double check locked edges
                        if (
                            locked_edges[a, 0] == b
                            or locked_edges[a, 1] == b
                            or locked_edges[c, 0] == d
                            or locked_edges[c, 1] == d
                            or locked_edges[e, 0] == f
                            or locked_edges[e, 1] == f
                        ):
                            continue

                        E_remove = (
                            _dist(a, b, coords_x, coords_y)
                            + _dist(c, d, coords_x, coords_y)
                            + _dist(e, f, coords_x, coords_y)
                        )

                        best_case = -1
                        best_delta = -1e-9

                        # Case 4
                        d4 = (
                            _dist(c, e, coords_x, coords_y)
                            + _dist(d, a, coords_x, coords_y)
                            + _dist(f, b, coords_x, coords_y)
                            - E_remove
                        )
                        if d4 < best_delta:
                            best_delta = d4
                            best_case = 4

                        # Case 5
                        d5 = (
                            _dist(c, f, coords_x, coords_y)
                            + _dist(a, d, coords_x, coords_y)
                            + _dist(e, b, coords_x, coords_y)
                            - E_remove
                        )
                        if d5 < best_delta:
                            best_delta = d5
                            best_case = 5

                        # Case 6
                        d6 = (
                            _dist(c, a, coords_x, coords_y)
                            + _dist(f, d, coords_x, coords_y)
                            + _dist(e, b, coords_x, coords_y)
                            - E_remove
                        )
                        if d6 < best_delta:
                            best_delta = d6
                            best_case = 6

                        # Case 7
                        d7 = (
                            _dist(c, f, coords_x, coords_y)
                            + _dist(a, e, coords_x, coords_y)
                            + _dist(d, b, coords_x, coords_y)
                            - E_remove
                        )
                        if d7 < best_delta:
                            best_delta = d7
                            best_case = 7

                        # Case 8
                        d8 = (
                            _dist(c, a, coords_x, coords_y)
                            + _dist(f, e, coords_x, coords_y)
                            + _dist(d, b, coords_x, coords_y)
                            - E_remove
                        )
                        if d8 < best_delta:
                            best_delta = d8
                            best_case = 8

                        if best_case != -1:
                            new_tour = _reconstruct_tour_3opt(
                                tour, i1, i2, i3, best_case
                            )
                            tour[:] = new_tour[:]
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
    tour: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    candidate_set: np.ndarray,
    candidate_dists: np.ndarray,
    locked_edges: np.ndarray,
    pos: np.ndarray,
    dlb: np.ndarray,
    max_len: int = 5,
) -> bool:
    n = tour.shape[0]
    globally_improved = False
    for i_idx in range(n):
        u = tour[i_idx]
        if dlb[u]:
            continue
        found = False
        for length in range(1, max_len + 1):
            if length >= n - 2:
                break
            j_idx = (i_idx + length - 1) % n
            v = tour[j_idx]

            p_u_idx = (i_idx - 1 + n) % n
            p_u = tour[p_u_idx]
            s_v_idx = (j_idx + 1) % n
            s_v = tour[s_v_idx]

            if (
                locked_edges[p_u, 0] == u
                or locked_edges[p_u, 1] == u
                or locked_edges[v, 0] == s_v
                or locked_edges[v, 1] == s_v
            ):
                continue

            base_g = (
                _dist(p_u, u, coords_x, coords_y)
                + _dist(v, s_v, coords_x, coords_y)
                - _dist(p_u, s_v, coords_x, coords_y)
            )
            if base_g <= 1e-9:
                continue

            for k in range(candidate_set.shape[1]):
                w = candidate_set[u, k]
                if w == -1:
                    break
                if w == u or w == v or w == p_u or w == s_v:
                    continue

                w_idx = pos[w]
                # Check if w is inside the segment [u, v]
                is_inside = False
                if i_idx <= j_idx:
                    if i_idx <= w_idx <= j_idx:
                        is_inside = True
                else:
                    if w_idx >= i_idx or w_idx <= j_idx:
                        is_inside = True
                if is_inside:
                    continue

                s_w_idx = (w_idx + 1) % n
                s_w = tour[s_w_idx]
                if locked_edges[w, 0] == s_w or locked_edges[w, 1] == s_w:
                    continue

                dist_wu = candidate_dists[u, k]
                if (
                    base_g + _dist(w, s_w, coords_x, coords_y)
                    > dist_wu + _dist(v, s_w, coords_x, coords_y) + 1e-9
                ):
                    # Relocate segment [u, v] (indices i_idx to j_idx) to after w
                    if i_idx <= j_idx < w_idx:
                        _apply_2opt(tour, i_idx, j_idx)
                        _apply_2opt(tour, s_v_idx, w_idx)
                        _apply_2opt(tour, i_idx, w_idx)
                    elif w_idx < i_idx <= j_idx:
                        _apply_2opt(tour, i_idx, j_idx)
                        _apply_2opt(tour, s_w_idx, p_u_idx)
                        _apply_2opt(tour, s_w_idx, j_idx)
                    else:
                        continue

                    _update_pos(tour, pos)
                    dlb[p_u] = dlb[u] = dlb[v] = dlb[s_v] = dlb[w] = dlb[s_w] = False
                    globally_improved = True
                    found = True
                    break

            # Also try inserting the segment in reversed order
            if not found:
                for k in range(candidate_set.shape[1]):
                    w = candidate_set[v, k]
                    if w == -1:
                        break
                    if w == u or w == v or w == p_u or w == s_v:
                        continue

                    w_idx = pos[w]
                    # Check if w is inside the segment [u, v]
                    is_inside = False
                    if i_idx <= j_idx:
                        if i_idx <= w_idx <= j_idx:
                            is_inside = True
                    else:
                        if w_idx >= i_idx or w_idx <= j_idx:
                            is_inside = True
                    if is_inside:
                        continue

                    s_w_idx = (w_idx + 1) % n
                    s_w = tour[s_w_idx]
                    if locked_edges[w, 0] == s_w or locked_edges[w, 1] == s_w:
                        continue

                    dist_wv = candidate_dists[v, k]
                    # Reversed insertion: connect w->v and u->s_w instead of w->u and v->s_w
                    if (
                        base_g + _dist(w, s_w, coords_x, coords_y)
                        > dist_wv + _dist(u, s_w, coords_x, coords_y) + 1e-9
                    ):
                        if i_idx <= j_idx < w_idx:
                            _apply_2opt(tour, s_v_idx, w_idx)
                            _apply_2opt(tour, i_idx, w_idx)
                        elif w_idx < i_idx <= j_idx:
                            _apply_2opt(tour, s_w_idx, p_u_idx)
                            _apply_2opt(tour, s_w_idx, j_idx)
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
def _apply_2opt_indices(
    tour: np.ndarray,
    t1_idx: int,
    t2_idx: int,
    t3_idx: int,
    t4_idx: int,
) -> None:
    n = tour.shape[0]
    # Standard 2-opt flip to replace (t1,t2) and (t3,t4) with (t2,t3) and (t4,t1)
    # The edges are tour[t1_idx], tour[t2_idx] and tour[t3_idx], tour[t4_idx]
    # We must reverse the segment between t2 and t3 (if they are in that order)
    if (t1_idx + 1) % n == t2_idx:
        _apply_2opt(tour, t2_idx, t3_idx)
    else:
        _apply_2opt(tour, t1_idx, t4_idx)


@njit(fastmath=True, cache=True)  # type: ignore
def _optimize_4opt_sequential(
    tour: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    candidate_set: np.ndarray,
    candidate_dists: np.ndarray,
    locked_edges: np.ndarray,
    pos: np.ndarray,
    dlb: np.ndarray,
) -> bool:
    n = tour.shape[0]
    if n < 8:
        return False
    globally_improved = False
    num_cand = candidate_set.shape[1]

    for t1_idx in range(n):
        t1 = tour[t1_idx]
        if dlb[t1]:
            continue

        t2_idx = (t1_idx + 1) % n
        t2 = tour[t2_idx]
        if locked_edges[t1, 0] == t2 or locked_edges[t1, 1] == t2:
            continue
        dist_t1_t2 = _dist(t1, t2, coords_x, coords_y)

        found = False
        for k3 in range(num_cand):
            t3 = candidate_set[t2, k3]
            if t3 == -1:
                break
            if t3 == t1 or t3 == t2:
                continue
            dist_t2_t3 = candidate_dists[t2, k3]
            g1 = dist_t1_t2 - dist_t2_t3
            if g1 <= 1e-9:
                continue

            t3_idx = pos[t3]
            for d1 in [1, -1]:
                t4_idx = (t3_idx + d1 + n) % n
                t4 = tour[t4_idx]
                if t4 == t1 or t4 == t2:
                    continue
                if locked_edges[t3, 0] == t4 or locked_edges[t3, 1] == t4:
                    continue
                dist_t3_t4 = _dist(t3, t4, coords_x, coords_y)

                for k5 in range(num_cand):
                    t5 = candidate_set[t4, k5]
                    if t5 == -1:
                        break
                    if t5 == t1 or t5 == t2 or t5 == t3 or t5 == t4:
                        continue
                    dist_t4_t5 = candidate_dists[t4, k5]
                    g2 = g1 + dist_t3_t4 - dist_t4_t5
                    if g2 <= 1e-9:
                        continue

                    t5_idx = pos[t5]
                    for d2 in [1, -1]:
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = tour[t6_idx]
                        if t6 == t1 or t6 == t2 or t6 == t3 or t6 == t4 or t6 == t5:
                            continue
                        if locked_edges[t5, 0] == t6 or locked_edges[t5, 1] == t6:
                            continue
                        dist_t5_t6 = _dist(t5, t6, coords_x, coords_y)

                        for k7 in range(num_cand):
                            t7 = candidate_set[t6, k7]
                            if t7 == -1:
                                break
                            if (
                                t7 == t1
                                or t7 == t2
                                or t7 == t3
                                or t7 == t4
                                or t7 == t5
                                or t7 == t6
                            ):
                                continue
                            dist_t6_t7 = candidate_dists[t6, k7]
                            g3 = g2 + dist_t5_t6 - dist_t6_t7
                            if g3 <= 1e-9:
                                continue

                            t7_idx = pos[t7]
                            for d3 in [1, -1]:
                                t8_idx = (t7_idx + d3 + n) % n
                                t8 = tour[t8_idx]
                                if (
                                    t8 == t1
                                    or t8 == t2
                                    or t8 == t3
                                    or t8 == t4
                                    or t8 == t5
                                    or t8 == t6
                                    or t8 == t7
                                ):
                                    continue
                                if (
                                    locked_edges[t7, 0] == t8
                                    or locked_edges[t7, 1] == t8
                                ):
                                    continue
                                dist_t7_t8 = _dist(t7, t8, coords_x, coords_y)

                                gain = g3 + dist_t7_t8 - _dist(t8, t1, coords_x, coords_y)
                                if gain > 1e-9:
                                    _apply_2opt_indices(
                                        tour, t1_idx, t2_idx, t3_idx, t4_idx
                                    )
                                    _update_pos(tour, pos)
                                    _apply_2opt_indices(
                                        tour, pos[t1], pos[t4], pos[t5], pos[t6]
                                    )
                                    _update_pos(tour, pos)
                                    _apply_2opt_indices(
                                        tour, pos[t1], pos[t6], pos[t7], pos[t8]
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
    tour: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    candidate_set: np.ndarray,
    candidate_dists: np.ndarray,
    locked_edges: np.ndarray,
    pos: np.ndarray,
    dlb: np.ndarray,
) -> bool:
    n = tour.shape[0]
    if n < 10:
        return False
    globally_improved = False
    num_cand = candidate_set.shape[1]

    for t1_idx in range(n):
        t1 = tour[t1_idx]
        if dlb[t1]:
            continue

        t2_idx = (t1_idx + 1) % n
        t2 = tour[t2_idx]
        if locked_edges[t1, 0] == t2 or locked_edges[t1, 1] == t2:
            continue
        dist_t1_t2 = _dist(t1, t2, coords_x, coords_y)

        found = False
        for k3 in range(num_cand):
            t3 = candidate_set[t2, k3]
            if t3 == -1:
                break
            if t3 == t1 or t3 == t2:
                continue
            dist_t2_t3 = candidate_dists[t2, k3]
            g1 = dist_t1_t2 - dist_t2_t3
            if g1 <= 1e-9:
                continue

            t3_idx = pos[t3]
            for d1 in [1, -1]:
                t4_idx = (t3_idx + d1 + n) % n
                t4 = tour[t4_idx]
                if t4 == t1 or t4 == t2:
                    continue
                if locked_edges[t3, 0] == t4 or locked_edges[t3, 1] == t4:
                    continue
                dist_t3_t4 = _dist(t3, t4, coords_x, coords_y)

                for k5 in range(num_cand):
                    t5 = candidate_set[t4, k5]
                    if t5 == -1:
                        break
                    if t5 == t1 or t5 == t2 or t5 == t3 or t5 == t4:
                        continue
                    dist_t4_t5 = candidate_dists[t4, k5]
                    g2 = g1 + dist_t3_t4 - dist_t4_t5
                    if g2 <= 1e-9:
                        continue

                    t5_idx = pos[t5]
                    for d2 in [1, -1]:
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = tour[t6_idx]
                        if t6 == t1 or t6 == t2 or t6 == t3 or t6 == t4 or t6 == t5:
                            continue
                        if locked_edges[t5, 0] == t6 or locked_edges[t5, 1] == t6:
                            continue
                        dist_t5_t6 = _dist(t5, t6, coords_x, coords_y)

                        for k7 in range(num_cand):
                            t7 = candidate_set[t6, k7]
                            if t7 == -1:
                                break
                            if (
                                t7 == t1
                                or t7 == t2
                                or t7 == t3
                                or t7 == t4
                                or t7 == t5
                                or t7 == t6
                            ):
                                continue
                            dist_t6_t7 = candidate_dists[t6, k7]
                            g3 = g2 + dist_t5_t6 - dist_t6_t7
                            if g3 <= 1e-9:
                                continue

                            t7_idx = pos[t7]
                            for d3 in [1, -1]:
                                t8_idx = (t7_idx + d3 + n) % n
                                t8 = tour[t8_idx]
                                if (
                                    t8 == t1
                                    or t8 == t2
                                    or t8 == t3
                                    or t8 == t4
                                    or t8 == t5
                                    or t8 == t6
                                    or t8 == t7
                                ):
                                    continue
                                if (
                                    locked_edges[t7, 0] == t8
                                    or locked_edges[t7, 1] == t8
                                ):
                                    continue
                                dist_t7_t8 = _dist(t7, t8, coords_x, coords_y)

                                for k9 in range(num_cand):
                                    t9 = candidate_set[t8, k9]
                                    if t9 == -1:
                                        break
                                    if (
                                        t9 == t1
                                        or t9 == t2
                                        or t9 == t3
                                        or t9 == t4
                                        or t9 == t5
                                        or t9 == t6
                                        or t9 == t7
                                        or t9 == t8
                                    ):
                                        continue
                                    dist_t8_t9 = candidate_dists[t8, k9]
                                    g4 = g3 + dist_t7_t8 - dist_t8_t9
                                    if g4 <= 1e-9:
                                        continue

                                    t9_idx = pos[t9]
                                    for d4 in [1, -1]:
                                        t10_idx = (t9_idx + d4 + n) % n
                                        t10 = tour[t10_idx]
                                        if (
                                            t10 == t1
                                            or t10 == t2
                                            or t10 == t3
                                            or t10 == t4
                                            or t10 == t5
                                            or t10 == t6
                                            or t10 == t7
                                            or t10 == t8
                                            or t10 == t9
                                        ):
                                            continue
                                        if (
                                            locked_edges[t9, 0] == t10
                                            or locked_edges[t9, 1] == t10
                                        ):
                                            continue
                                        dist_t9_t10 = _dist(t9, t10, coords_x, coords_y)

                                        gain = g4 + dist_t9_t10 - _dist(t10, t1, coords_x, coords_y)
                                        if gain > 1e-9:
                                            _apply_2opt_indices(
                                                tour, t1_idx, t2_idx, t3_idx, t4_idx
                                            )
                                            _update_pos(tour, pos)
                                            _apply_2opt_indices(
                                                tour, pos[t1], pos[t4], pos[t5], pos[t6]
                                            )
                                            _update_pos(tour, pos)
                                            _apply_2opt_indices(
                                                tour, pos[t1], pos[t6], pos[t7], pos[t8]
                                            )
                                            _update_pos(tour, pos)
                                            _apply_2opt_indices(
                                                tour,
                                                pos[t1],
                                                pos[t8],
                                                pos[t9],
                                                pos[t10],
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
    tour: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    candidate_set: np.ndarray,
    candidate_dists: np.ndarray,
    locked_edges: np.ndarray,
    pos: np.ndarray,
    dlb: np.ndarray,
    max_opt: int = 3,
) -> None:
    improved = True
    while improved:
        improved = False
        if _optimize_2opt(
            tour, coords_x, coords_y, candidate_set, candidate_dists, locked_edges, pos, dlb
        ):
            improved = True
            while _optimize_2opt(
                tour, coords_x, coords_y, candidate_set, candidate_dists, locked_edges, pos, dlb
            ):
                pass

        # Or-opt search
        if _optimize_or_opt(
            tour, coords_x, coords_y, candidate_set, candidate_dists, locked_edges, pos, dlb, max_len=5
        ):
            improved = True
            continue

        # Or-opt and 3-opt sequential
        if max_opt >= 3:
            if _optimize_3opt_sequential(
                tour, coords_x, coords_y, candidate_set, candidate_dists, locked_edges, pos, dlb
            ):
                improved = True
                continue

        if max_opt >= 4:
            if _optimize_4opt_sequential(
                tour, coords_x, coords_y, candidate_set, candidate_dists, locked_edges, pos, dlb
            ):
                improved = True
                continue

        if max_opt >= 5:
            if _optimize_5opt_sequential(
                tour, coords_x, coords_y, candidate_set, candidate_dists, locked_edges, pos, dlb
            ):
                improved = True
                continue


@njit(fastmath=True, cache=True)  # type: ignore
def _double_bridge_kick(tour: np.ndarray) -> None:
    n = tour.shape[0]
    if n < 8:
        return
    
    # Choose 4 positions to cut the tour.
    # For large tours, use a localized window to avoid creating long, unrepairable edges.
    if n <= 1000:
        indices = np.random.choice(n, 4, replace=False)
        indices.sort()
        p1, p2, p3, p4 = indices[0], indices[1], indices[2], indices[3]
    else:
        # Localized kick
        W = min(500, max(100, n // 20))
        start = np.random.randint(0, n - W)
        offsets = np.random.choice(W, 4, replace=False)
        offsets.sort()
        p1 = start + offsets[0]
        p2 = start + offsets[1]
        p3 = start + offsets[2]
        p4 = start + offsets[3]

    # Tour segments: [0, p1], [p1+1, p2], [p2+1, p3], [p3+1, p4], [p4+1, n-1]
    # Reconnect as: [0, p1], [p3+1, p4], [p2+1, p3], [p1+1, p2], [p4+1, n-1]
    new_tour = np.empty_like(tour)
    curr = 0
    for i in range(0, p1 + 1):
        new_tour[curr] = tour[i]
        curr += 1
    for i in range(p3 + 1, p4 + 1):
        new_tour[curr] = tour[i]
        curr += 1
    for i in range(p2 + 1, p3 + 1):
        new_tour[curr] = tour[i]
        curr += 1
    for i in range(p1 + 1, p2 + 1):
        new_tour[curr] = tour[i]
        curr += 1
    for i in range(p4 + 1, n):
        new_tour[curr] = tour[i]
        curr += 1
    tour[:] = new_tour[:]


@njit(fastmath=True, cache=True)  # type: ignore
def _cascading_kopt_inner(
    initial_tour: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    candidate_set: np.ndarray,
    candidate_dists: np.ndarray,
    locked_edges: np.ndarray,
    num_kicks: int,
    max_opt: int,
    in_tour: np.ndarray,
    in_best_tour: np.ndarray,
    in_best_length: float,
    in_stagnation_count: int,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Inner @njit kick loop. Accepts externally-supplied state so the Python
    wrapper can resume across chunk boundaries. Returns updated
    (tour, best_tour, best_length, stagnation_count).
    """
    n = in_tour.shape[0]
    tour = in_tour.copy()
    best_tour = in_best_tour.copy()
    best_length = in_best_length
    stagnation_count = in_stagnation_count
    stagnation_limit = max(50, n // 100)

    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    for i in range(num_kicks):
        kicked_tour = tour.copy()
        _double_bridge_kick(kicked_tour)
        _update_pos(kicked_tour, pos)
        dlb.fill(False)
        _full_cascade(
            kicked_tour,
            coords_x,
            coords_y,
            candidate_set,
            candidate_dists,
            locked_edges,
            pos,
            dlb,
            max_opt=max_opt,
        )
        length = compute_tour_length(kicked_tour, coords_x, coords_y)
        if length < best_length - 1e-9:
            best_length = length
            best_tour = kicked_tour.copy()
            tour = kicked_tour
            stagnation_count = 0
        else:
            stagnation_count += 1
            if stagnation_count > stagnation_limit:
                tour = best_tour.copy()
                dlb.fill(False)  # CRITICAL: clear DLB after reset!
                stagnation_count = 0

    return tour, best_tour, best_length, stagnation_count


@njit(fastmath=True, cache=True, parallel=True)  # type: ignore
def _precompute_candidate_dists(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    candidate_set: np.ndarray,
) -> np.ndarray:
    n, num_cand = candidate_set.shape
    candidate_dists = np.zeros((n, num_cand), dtype=np.float64)
    for i in prange(n):
        for k in range(num_cand):
            c_idx = candidate_set[i, k]
            if c_idx != -1:
                dx = coords_x[i] - coords_x[c_idx]
                dy = coords_y[i] - coords_y[c_idx]
                candidate_dists[i, k] = np.sqrt(dx * dx + dy * dy)
    return candidate_dists
from typing import Tuple, Optional, Any
...
def cascading_kopt_optimize(
    initial_tour: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    candidate_set: np.ndarray,
    locked_edges: np.ndarray,
    num_kicks: int = 500,
    max_opt: int = 3,
    time_limit_s: float = -1.0,
    chunk_size: int = 1,
    progress_array: Any = None,
    seed_idx: int = 0,
) -> Tuple[np.ndarray, float, int]:

    """
    Python-level wrapper around _cascading_kopt_inner.
    Supports real-time progress reporting via progress_array.
    """
    # ... (byte alignment code)
    # Byte alignment
    coords_x = np.ascontiguousarray(coords_x, dtype=np.float64)
    coords_y = np.ascontiguousarray(coords_y, dtype=np.float64)
    candidate_set = np.ascontiguousarray(candidate_set, dtype=np.int32)
    locked_edges = np.ascontiguousarray(locked_edges, dtype=np.int32)
    initial_tour = np.ascontiguousarray(initial_tour, dtype=np.int32)

    n = initial_tour.shape[0]

    # Precompute candidate distances using parallel JIT
    candidate_dists = _precompute_candidate_dists(coords_x, coords_y, candidate_set)
    candidate_dists = np.ascontiguousarray(candidate_dists)

    tour = initial_tour.copy()
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)

    # Initial local search before kicks
    _full_cascade(
        tour,
        coords_x,
        coords_y,
        candidate_set,
        candidate_dists,
        locked_edges,
        pos,
        dlb,
        max_opt=max_opt,
    )

    best_tour = tour.copy()
    best_length = float(compute_tour_length(tour, coords_x, coords_y))
    stagnation_count = 0

    t_start = time.monotonic()
    deadline = t_start + time_limit_s * 0.97 if time_limit_s > 0 else float("inf")

    kicks_done = 0
    while kicks_done < num_kicks:
        # Check time budget before starting next chunk
        if time.monotonic() >= deadline:
            break
        this_chunk = min(chunk_size, num_kicks - kicks_done)
        tour, best_tour, best_length, stagnation_count = _cascading_kopt_inner(
            initial_tour,
            coords_x,
            coords_y,
            candidate_set,
            candidate_dists,
            locked_edges,
            this_chunk,
            max_opt,
            tour,
            best_tour,
            best_length,
            stagnation_count,
        )
        kicks_done += this_chunk
        
        # Update shared memory
        if progress_array is not None:
            progress_array[seed_idx] = kicks_done

    return best_tour, best_length, kicks_done
