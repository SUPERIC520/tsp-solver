import importlib
import pytest
from src import config


def test_config_default_values() -> None:
    """
    Verify that all config constants are defined with the correct default values.
    """
    assert config.KD_TREE_QUERY_SIZE == 64
    assert config.K_NEIGHBORS == 40
    assert config.K_3OPT == 15
    assert config.K_4OPT == 5
    assert config.K_5OPT == 3
    assert config.OR_OPT_MAX_LEN == 5
    assert config.MAX_OPT == 5
    assert config.NUM_PROCESSES_SOLVER == -1
    assert config.NUM_PROCESSES_SEEDING == -1
    assert config.SEED_STRATEGY == "Greedy"


def test_config_parameter_override_via_argument() -> None:
    """
    Verify that parameters passed as function arguments with default values
    mapped to global configs can be overridden at runtime.
    """
    def mock_optimize_or_opt(max_len: int = config.OR_OPT_MAX_LEN) -> int:
        return max_len

    # Default argument should match config value
    assert mock_optimize_or_opt() == 5

    # Passing explicit argument overrides the default config value
    assert mock_optimize_or_opt(max_len=3) == 3
    assert mock_optimize_or_opt(max_len=10) == 10


def test_config_module_attribute_mutation() -> None:
    """
    Verify that config module attributes can be modified at runtime in a non-JIT
    environment and successfully restored.
    """
    original_k_neighbors = config.K_NEIGHBORS
    original_seed_strategy = config.SEED_STRATEGY

    try:
        # Mutate the configuration parameters
        config.K_NEIGHBORS = 20
        config.SEED_STRATEGY = "Hybrid"

        # Verify mutated values are visible in the config module
        assert config.K_NEIGHBORS == 20
        assert config.SEED_STRATEGY == "Hybrid"

        # Verify that a function using these mutated attributes dynamically picks them up
        def get_current_seeding_strategy() -> str:
            return config.SEED_STRATEGY

        assert get_current_seeding_strategy() == "Hybrid"

    finally:
        # Restore original values to prevent side effects in other tests
        config.K_NEIGHBORS = original_k_neighbors
        config.SEED_STRATEGY = original_seed_strategy

    # Verify restoration
    assert config.K_NEIGHBORS == original_k_neighbors
    assert config.SEED_STRATEGY == original_seed_strategy
