import numpy as np
from src.core.backbone import extract_consensus_edges


def test_extract_consensus_edges():
    # 4 points, 3 identical tours and 1 different
    tours = np.array(
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 1, 3]], dtype=np.int32
    )

    # Threshold 0.75 means 3/4 tours must have the edge.
    # Tours 0, 1, 2 have edges (0,1), (1,2), (2,3), (3,0).
    # Tour 3 has edges (0,2), (2,1), (1,3), (3,0).
    # Common edges (appear >= 3 times):
    # (3,0) appears 4 times.
    # (1,2) appears 3 times (in tours 0,1,2).
    # (0,1) appears 3 times (in tours 0,1,2). Actually in tour 0,1,2 (0,1) is an edge. In tour 3 (0,1) is NOT an edge.
    # (2,3) appears 3 times (in tours 0,1,2).

    locked_edges = extract_consensus_edges(tours, threshold=0.75)

    # Edge (3,0) should be locked
    assert locked_edges[3, 0] == 0 or locked_edges[3, 1] == 0
    assert locked_edges[0, 0] == 3 or locked_edges[0, 1] == 3

    # Edge (1,2) should be locked
    assert locked_edges[1, 0] == 2 or locked_edges[1, 1] == 2
    assert locked_edges[2, 0] == 1 or locked_edges[2, 1] == 1
