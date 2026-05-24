import numpy as np
from numba import njit
from typing import Tuple


@njit(inline="always")  # type: ignore
def _dist(c1: int, c2: int, coords: np.ndarray) -> float:
    dx = coords[c1, 0] - coords[c2, 0]
    dy = coords[c1, 1] - coords[c2, 1]
    return float(np.sqrt(dx * dx + dy * dy))


@njit(fastmath=True, cache=True)  # type: ignore
def compute_tour_length(
    tour: np.ndarray,
    coords: np.ndarray,
) -> float:
    length = 0.0
    n = tour.shape[0]
    for i in range(n):
        length += _dist(tour[i], tour[(i + 1) % n], coords)
    return float(length)


@njit(fastmath=True, cache=True)  # type: ignore
def _apply_2opt(tour: np.ndarray, i: int, j: int) -> None:
    n = tour.shape[0]
    if i == j:
        return
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
def optimize_2opt(
    tour: np.ndarray,
    coords: np.ndarray,
    candidate_set: np.ndarray,
    locked_edges: np.ndarray,
    pos: np.ndarray,
    dlb: np.ndarray,
) -> bool:
    n = tour.shape[0]
    globally_improved = False
    for u_idx in range(n):
        u = tour[u_idx]
        if dlb[u]:
            continue
        v_idx = (u_idx + 1) % n
        v = tour[v_idx]
        if locked_edges[u, 0] == v or locked_edges[u, 1] == v:
            continue
        dist_uv = _dist(u, v, coords)
        found = False
        for k in range(candidate_set.shape[1]):
            w = candidate_set[u, k]
            if w == -1:
                break
            if w == u or w == v:
                continue
            dist_uw = _dist(u, w, coords)
            if dist_uw >= dist_uv:
                continue
            w_idx = pos[w]
            x_idx = (w_idx + 1) % n
            x = tour[x_idx]
            if x == u or x == v:
                continue
            if locked_edges[w, 0] == x or locked_edges[w, 1] == x:
                continue
            if dist_uv + _dist(w, x, coords) > dist_uw + _dist(v, x, coords) + 1e-9:
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
def optimize_3opt_sequential(
    tour: np.ndarray,
    coords: np.ndarray,
    candidate_set: np.ndarray,
    locked_edges: np.ndarray,
    pos: np.ndarray,
    dlb: np.ndarray,
) -> bool:
    n = tour.shape[0]
    if n < 6:
        return False
    globally_improved = False
    for t1_idx in range(n):
        t1 = tour[t1_idx]
        if dlb[t1]:
            continue

        # We try to remove (t1, t2), (t3, t4), (t5, t6)
        # and add (t2, t3), (t4, t5), (t6, t1)
        # For simplicity, we only consider t2 as successor of t1
        t2_idx = (t1_idx + 1) % n
        t2 = tour[t2_idx]
        if locked_edges[t1, 0] == t2 or locked_edges[t1, 1] == t2:
            continue
        dist_t1_t2 = _dist(t1, t2, coords)

        found = False
        num_cand = candidate_set.shape[1]
        for k3 in range(num_cand):
            t3 = candidate_set[t2, k3]
            if t3 == -1:
                break
            if t3 == t1 or t3 == t2:
                continue
            dist_t2_t3 = _dist(t2, t3, coords)
            g1 = dist_t1_t2 - dist_t2_t3
            if g1 <= 1e-9:
                continue

            t3_idx = pos[t3]
            # t4 is neighbor of t3. To be sequential, t4 must be successor or predecessor.
            # We try successor first
            for d in [1, -1]:
                t4_idx = (t3_idx + d + n) % n
                t4 = tour[t4_idx]
                if t4 == t1 or t4 == t2:
                    continue
                if locked_edges[t3, 0] == t4 or locked_edges[t3, 1] == t4:
                    continue
                dist_t3_t4 = _dist(t3, t4, coords)

                # Try depth 3
                for k5 in range(num_cand):
                    t5 = candidate_set[t4, k5]
                    if t5 == -1:
                        break
                    if t5 == t1 or t5 == t2 or t5 == t3 or t5 == t4:
                        continue
                    dist_t4_t5 = _dist(t4, t5, coords)
                    g2 = g1 + dist_t3_t4 - dist_t4_t5
                    if g2 <= 1e-9:
                        continue

                    t5_idx = pos[t5]
                    for d2 in [1, -1]:
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = tour[t6_idx]
                        if t6 in (t1, t2, t3, t4, t5):
                            continue
                        if locked_edges[t5, 0] == t6 or locked_edges[t5, 1] == t6:
                            continue

                        dist_t5_t6 = _dist(t5, t6, coords)

                        # Euclidean metric pruning: if current gain + tour edge
                        # cannot possibly overcome the distance back to t1, skip.
                        if g2 + dist_t5_t6 <= 1e-9:
                            continue

                        gain = g2 + dist_t5_t6 - _dist(t6, t1, coords)
                        if gain > 1e-9:
                            # Apply 3-opt sequential move
                            # This is two 2-opt flips.
                            # Flip 1: (t1,t2) and (t3,t4)
                            _apply_2opt_indices(tour, t1_idx, t2_idx, t3_idx, t4_idx)
                            _update_pos(tour, pos)
                            # Flip 2: (t4,t1) and (t5,t6)
                            _apply_2opt_indices(
                                tour, pos[t1], pos[t4], pos[t5], pos[t6]
                            )
                            _update_pos(tour, pos)

                            dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = dlb[t5] = dlb[
                                t6
                            ] = False
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
def _apply_2opt_indices(
    tour: np.ndarray, t1_idx: int, t2_idx: int, t3_idx: int, t4_idx: int
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
def optimize_or_opt(
    tour: np.ndarray,
    coords: np.ndarray,
    candidate_set: np.ndarray,
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
                _dist(p_u, u, coords) + _dist(v, s_v, coords) - _dist(p_u, s_v, coords)
            )
            if base_g <= 1e-9:
                continue
            num_cand = candidate_set.shape[1]
            for k in range(num_cand):
                w = candidate_set[u, k]
                if w == -1:
                    break
                w_idx = pos[w]
                if i_idx <= j_idx:
                    if i_idx <= w_idx <= j_idx:
                        continue
                else:
                    if w_idx >= i_idx or w_idx <= j_idx:
                        continue
                s_w_idx = (w_idx + 1) % n
                s_w = tour[s_w_idx]
                if locked_edges[w, 0] == s_w or locked_edges[w, 1] == s_w:
                    continue
                if (
                    base_g + _dist(w, s_w, coords)
                    > _dist(w, u, coords) + _dist(v, s_w, coords) + 1e-9
                ):
                    _apply_2opt(tour, i_idx, j_idx)
                    _apply_2opt(tour, (j_idx + 1) % n, w_idx)
                    _apply_2opt(tour, i_idx, w_idx)
                    _update_pos(tour, pos)
                    dlb[p_u] = dlb[u] = dlb[v] = dlb[s_v] = dlb[w] = dlb[s_w] = False
                    globally_improved = True
                    found = True
                    break
            if found:
                break
        if not found:
            dlb[u] = True
    return globally_improved


@njit(fastmath=True, cache=True)  # type: ignore
def optimize_4opt_sequential(
    tour: np.ndarray,
    coords: np.ndarray,
    candidate_set: np.ndarray,
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
        dist_t1_t2 = _dist(t1, t2, coords)

        found = False
        for k3 in range(num_cand):
            t3 = candidate_set[t2, k3]
            if t3 == -1:
                break
            if t3 == t1 or t3 == t2:
                continue
            dist_t2_t3 = _dist(t2, t3, coords)
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
                dist_t3_t4 = _dist(t3, t4, coords)

                for k5 in range(num_cand):
                    t5 = candidate_set[t4, k5]
                    if t5 == -1:
                        break
                    if t5 in (t1, t2, t3, t4):
                        continue
                    dist_t4_t5 = _dist(t4, t5, coords)
                    g2 = g1 + dist_t3_t4 - dist_t4_t5
                    if g2 <= 1e-9:
                        continue

                    t5_idx = pos[t5]
                    for d2 in [1, -1]:
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = tour[t6_idx]
                        if t6 in (t1, t2, t3, t4, t5):
                            continue
                        if locked_edges[t5, 0] == t6 or locked_edges[t5, 1] == t6:
                            continue
                        dist_t5_t6 = _dist(t5, t6, coords)

                        for k7 in range(num_cand):
                            t7 = candidate_set[t6, k7]
                            if t7 == -1:
                                break
                            if t7 in (t1, t2, t3, t4, t5, t6):
                                continue
                            dist_t6_t7 = _dist(t6, t7, coords)
                            g3 = g2 + dist_t5_t6 - dist_t6_t7
                            if g3 <= 1e-9:
                                continue

                            t7_idx = pos[t7]
                            for d3 in [1, -1]:
                                t8_idx = (t7_idx + d3 + n) % n
                                t8 = tour[t8_idx]
                                if t8 in (t1, t2, t3, t4, t5, t6, t7):
                                    continue
                                if (
                                    locked_edges[t7, 0] == t8
                                    or locked_edges[t7, 1] == t8
                                ):
                                    continue
                                dist_t7_t8 = _dist(t7, t8, coords)

                                gain = g3 + dist_t7_t8 - _dist(t8, t1, coords)
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
def optimize_5opt_sequential(
    tour: np.ndarray,
    coords: np.ndarray,
    candidate_set: np.ndarray,
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
        dist_t1_t2 = _dist(t1, t2, coords)

        found = False
        for k3 in range(num_cand):
            t3 = candidate_set[t2, k3]
            if t3 == -1:
                break
            if t3 == t1 or t3 == t2:
                continue
            dist_t2_t3 = _dist(t2, t3, coords)
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
                dist_t3_t4 = _dist(t3, t4, coords)

                for k5 in range(num_cand):
                    t5 = candidate_set[t4, k5]
                    if t5 == -1:
                        break
                    if t5 in (t1, t2, t3, t4):
                        continue
                    dist_t4_t5 = _dist(t4, t5, coords)
                    g2 = g1 + dist_t3_t4 - dist_t4_t5
                    if g2 <= 1e-9:
                        continue

                    t5_idx = pos[t5]
                    for d2 in [1, -1]:
                        t6_idx = (t5_idx + d2 + n) % n
                        t6 = tour[t6_idx]
                        if t6 in (t1, t2, t3, t4, t5):
                            continue
                        if locked_edges[t5, 0] == t6 or locked_edges[t5, 1] == t6:
                            continue
                        dist_t5_t6 = _dist(t5, t6, coords)

                        for k7 in range(num_cand):
                            t7 = candidate_set[t6, k7]
                            if t7 == -1:
                                break
                            if t7 in (t1, t2, t3, t4, t5, t6):
                                continue
                            dist_t6_t7 = _dist(t6, t7, coords)
                            g3 = g2 + dist_t5_t6 - dist_t6_t7
                            if g3 <= 1e-9:
                                continue

                            t7_idx = pos[t7]
                            for d3 in [1, -1]:
                                t8_idx = (t7_idx + d3 + n) % n
                                t8 = tour[t8_idx]
                                if t8 in (t1, t2, t3, t4, t5, t6, t7):
                                    continue
                                if (
                                    locked_edges[t7, 0] == t8
                                    or locked_edges[t7, 1] == t8
                                ):
                                    continue
                                dist_t7_t8 = _dist(t7, t8, coords)

                                for k9 in range(num_cand):
                                    t9 = candidate_set[t8, k9]
                                    if t9 == -1:
                                        break
                                    if t9 in (t1, t2, t3, t4, t5, t6, t7, t8):
                                        continue
                                    dist_t8_t9 = _dist(t8, t9, coords)
                                    g4 = g3 + dist_t7_t8 - dist_t8_t9
                                    if g4 <= 1e-9:
                                        continue

                                    t9_idx = pos[t9]
                                    for d4 in [1, -1]:
                                        t10_idx = (t9_idx + d4 + n) % n
                                        t10 = tour[t10_idx]
                                        if t10 in (t1, t2, t3, t4, t5, t6, t7, t8, t9):
                                            continue
                                        if (
                                            locked_edges[t9, 0] == t10
                                            or locked_edges[t9, 1] == t10
                                        ):
                                            continue
                                        dist_t9_t10 = _dist(t9, t10, coords)

                                        gain = g4 + dist_t9_t10 - _dist(t10, t1, coords)
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
def _run_full_local_search(
    t: np.ndarray,
    coords: np.ndarray,
    candidate_set: np.ndarray,
    locked_edges: np.ndarray,
    p: np.ndarray,
    d: np.ndarray,
) -> None:
    improved = True
    while improved:
        improved = False
        if optimize_2opt(t, coords, candidate_set, locked_edges, p, d):
            improved = True
            while optimize_2opt(t, coords, candidate_set, locked_edges, p, d):
                pass
        if optimize_or_opt(t, coords, candidate_set, locked_edges, p, d, max_len=5):
            improved = True
            continue
        if optimize_3opt_sequential(t, coords, candidate_set, locked_edges, p, d):
            improved = True
            continue
        if optimize_4opt_sequential(t, coords, candidate_set, locked_edges, p, d):
            improved = True
            continue
        if optimize_5opt_sequential(t, coords, candidate_set, locked_edges, p, d):
            improved = True
            continue


@njit(fastmath=True, cache=True)  # type: ignore
def _double_bridge_kick(tour: np.ndarray) -> None:
    n = tour.shape[0]
    if n < 8:
        return
    pos = np.sort(np.random.choice(n, 4, replace=False))
    p1, p2, p3, p4 = pos[0], pos[1], pos[2], pos[3]
    temp = np.empty_like(tour)
    curr = 0
    for i in range(0, p1 + 1):
        temp[curr] = tour[i]
        curr += 1
    for i in range(p3 + 1, p4 + 1):
        temp[curr] = tour[i]
        curr += 1
    for i in range(p2 + 1, p3 + 1):
        temp[curr] = tour[i]
        curr += 1
    for i in range(p1 + 1, p2 + 1):
        temp[curr] = tour[i]
        curr += 1
    for i in range(p4 + 1, n):
        temp[curr] = tour[i]
        curr += 1
    tour[:] = temp[:]


@njit(fastmath=True, cache=True)  # type: ignore
def cascading_kopt_optimize_tour(
    initial_tour: np.ndarray,
    coords: np.ndarray,
    candidate_set: np.ndarray,
    locked_edges: np.ndarray,
    num_kicks: int = 500,
) -> Tuple[np.ndarray, float]:
    n = initial_tour.shape[0]
    tour = initial_tour.copy()
    pos = np.empty(n, dtype=np.int32)
    _update_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.bool_)
    _run_full_local_search(tour, coords, candidate_set, locked_edges, pos, dlb)
    best_tour = tour.copy()
    best_length = compute_tour_length(tour, coords)

    stagnation_count = 0
    stagnation_limit = max(2000, n // 2)
    for i in range(num_kicks):
        kicked_tour = tour.copy()
        _double_bridge_kick(kicked_tour)
        _update_pos(kicked_tour, pos)
        dlb.fill(False)
        _run_full_local_search(
            kicked_tour, coords, candidate_set, locked_edges, pos, dlb
        )
        length = compute_tour_length(kicked_tour, coords)
        if length < best_length - 1e-9:
            best_length = length
            best_tour = kicked_tour.copy()
            tour = kicked_tour
            stagnation_count = 0
        else:
            stagnation_count += 1
            if stagnation_count > stagnation_limit:
                tour = best_tour.copy()
                stagnation_count = 0
    return best_tour, float(best_length)
