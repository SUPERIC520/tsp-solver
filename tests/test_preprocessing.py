import numpy as np
from src.core.preprocessing import build_candidate_sets


def test_build_candidate_sets_simple() -> None:
    # 4 points in a square
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )

    # Each point should have the other 3 as neighbors (since k=16 and there are only 4 points)
    c_set = build_candidate_sets(coords, k=2)

    assert c_set.shape == (4, 2)
    # Point 0 (0,0) should have neighbors 1 (1,0) and 3 (0,1) as closest
    assert set(c_set[0]) == {1, 3}
    # Point 1 (1,0) should have neighbors 0 (0,0) and 2 (1,1) as closest
    assert set(c_set[1]) == {0, 2}


def test_build_candidate_sets_grid() -> None:
    # 3x3 grid
    x = np.linspace(0, 2, 3)
    y = np.linspace(0, 2, 3)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float64)

    # Center point is index 4 (1.0, 1.0)
    # Neighbors: (0.0, 1.0) [index 3], (1.0, 0.0) [index 1], (1.0, 2.0) [index 7], (2.0, 1.0) [index 5]
    # These are distance 1.0 away.
    # Diagonals are distance sqrt(2).

    c_set = build_candidate_sets(coords, k=4)

    assert c_set.shape == (9, 4)
    center_neighbors = set(c_set[4])
    assert center_neighbors == {1, 3, 5, 7}


def test_build_candidate_sets_10_cities() -> None:
    # 10 cities in a 2x5 grid
    # (0,0) (1,0) (2,0) (3,0) (4,0)
    # (0,1) (1,1) (2,1) (3,1) (4,1)
    coords_list = []
    for y in range(2):
        for x in range(5):
            coords_list.append([float(x), float(y)])
    coords = np.array(coords_list, dtype=np.float64)

    k = 3
    c_set = build_candidate_sets(coords, k=k)

    assert c_set.shape == (10, k)

    # City at (1,0) is index 1
    # Neighbors: (0,0) [index 0], (2,0) [index 2], (1,1) [index 6]
    # Distances: 1.0, 1.0, 1.0
    # Diagonals: (0,1) [index 5], (2,1) [index 7] are distance sqrt(2)

    neighbors_1 = set(c_set[1])
    assert 0 in neighbors_1
    assert 2 in neighbors_1
    assert 6 in neighbors_1

    # City at (2,1) is index 7
    # Neighbors: (1,1) [index 6], (3,1) [index 8], (2,0) [index 2]
    neighbors_7 = set(c_set[7])
    assert 6 in neighbors_7
    assert 8 in neighbors_7
    assert 2 in neighbors_7
