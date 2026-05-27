"""Global configuration constants for the TSP solver."""

from pathlib import Path


def _get_cache_version() -> str:
    git_head = Path(".git/HEAD")
    if not git_head.exists():
        return "default"
    try:
        with git_head.open("r", encoding="utf-8") as f:
            ref = f.read().strip()
            if ref.startswith("ref: "):
                branch = ref.split("refs/heads/")[-1]
                return branch.replace("/", "_")
            return ref[:7]
    except (OSError, IndexError):
        return "default"


# Path to the primary city coordinates dataset
DATA_PATH: Path = Path("data/cities.csv")

# Path to the sample city coordinates dataset for testing
SAMPLE_DATA_PATH: Path = Path("tests/data/sample_cities.csv")

# Path to the results file for the current run
SOLUTIONS_PATH: Path = Path("data/solutions.csv")

# Path to the global best tour file
BEST_TOUR_PATH: Path = Path("data/best_tour.csv")

# Path to the research notes file
NOTES_PATH: Path = Path("notes.md")

# Directory for generated artifacts (e.g. .npy bounds)
CACHE_DIR: Path = Path(".cache")

# Cache version for generated artifacts
CACHE_VERSION: str = _get_cache_version()

# Initial KD-Tree neighbor query count
KD_TREE_QUERY_SIZE: int = 64

# Final alpha-sorted candidate set size for local search width
K_NEIGHBORS: int = 64

# Maximum local search cascade depth level
MAX_OPT: int = 5

# Gain tolerance epsilon for pruning and validation
GAIN_EPSILON: float = 1e-9

# Dynamic funneling width limits
K_3OPT: int = 48
K_4OPT: int = 32
K_5OPT: int = 16

# Maximum segment length for Or-opt relocations
OR_OPT_MAX_LEN: int = 8

# Localized kick window threshold for double-bridge perturbation
LOCALIZED_KICK_THRESHOLD: int = 1000

# Minimum tour size requirements for sequential optimizations
MIN_TOUR_SIZE_2OPT: int = 4
MIN_TOUR_SIZE_3OPT: int = 6
MIN_TOUR_SIZE_4OPT: int = 8
MIN_TOUR_SIZE_5OPT: int = 10
MIN_TOUR_SIZE_KICK: int = 8

# CPU process count for solver concurrency, default -1 to use CPU cores
NUM_PROCESSES_SOLVER: int = -1

# CPU process count for parallel greedy NN seed tours computation
NUM_PROCESSES_SEEDING: int = -1

# Default seeding strategy
SEED_STRATEGY: str = "Greedy"
