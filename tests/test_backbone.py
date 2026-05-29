"""Tests for edge frequency and soft-backbone bias logic."""

import numpy as np

from src.core.backbone import compute_edge_frequencies
from src.core.preprocessing import apply_backbone_bias


def test_compute_edge_frequencies() -> None:
    """Test that edge frequencies are correctly calculated from a tour population."""
    # 4 cities, 2 tours
    # Tour 1: 0-1-2-3-0 (Edges: (0,1), (1,2), (2,3), (3,0))
    # Tour 2: 0-2-1-3-0 (Edges: (0,2), (2,1), (1,3), (3,0))
    tours = np.array([
        [0, 1, 2, 3],
        [0, 2, 1, 3]
    ], dtype=np.int32)

    # Candidate set: everyone is everyone's neighbor for simplicity
    candidate_set = np.array([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2]
    ], dtype=np.int32)

    freq = compute_edge_frequencies(tours, candidate_set)

    # Check edge (0, 3) - appears in both tours
    # Index of 3 in node 0's candidate list is 2
    assert freq[0, 2] == 1.0

    # Check edge (0, 1) - appears only in tour 1
    # Index of 1 in node 0's candidate list is 0
    assert freq[0, 0] == 0.5

    # Check edge (1, 2) - appears in both (as (2,1) in tour 2)
    # Index of 2 in node 1's candidate list is 1
    assert freq[1, 1] == 1.0


def test_apply_backbone_bias() -> None:
    """Test that backbone bias correctly re-sorts candidates."""
    # 3 cities
    candidate_set = np.array([
        [1, 2],
        [0, 2],
        [0, 1]
    ], dtype=np.int32)

    # Alpha values: prefer edge (0, 1) [alpha=0.1] over (0, 2) [alpha=0.5]
    alpha_values = np.array([
        [0.1, 0.5],
        [0.1, 0.5],
        [0.5, 0.5]
    ], dtype=np.float64)

    # Frequencies: edge (0, 2) is very frequent [1.0], (0, 1) is not [0.0]
    freq_matrix = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0]
    ], dtype=np.float64)

    # Without bias, order should be [1, 2] for node 0
    biased_cs = apply_backbone_bias(candidate_set, alpha_values, freq_matrix, backbone_weight=1.0)

    # Node 0 keys:
    # (0, 1): 0.1 - 1.0 * 0.0 = 0.1
    # (0, 2): 0.5 - 1.0 * 1.0 = -0.5
    # Since -0.5 < 0.1, edge (0, 2) should now be first
    assert biased_cs[0, 0] == 2
    assert biased_cs[0, 1] == 1
