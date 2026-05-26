"""
Global configuration constants for the TSP solver.
"""

# Initial KD-Tree neighbor query count
KD_TREE_QUERY_SIZE: int = 64

# Final alpha-sorted candidate set size for local search width
K_NEIGHBORS: int = 64

# Dynamic funneling width limit for sequential 3-opt
K_3OPT: int = 48

# Dynamic funneling width limit for sequential 4-opt
K_4OPT: int = 32

# Dynamic funneling width limit for sequential 5-opt
K_5OPT: int = 16

# Maximum segment length for Or-opt relocations
OR_OPT_MAX_LEN: int = 5

# Maximum local search cascade depth level
MAX_OPT: int = 5

# CPU process count for solver concurrency, default -1 to use CPU cores
NUM_PROCESSES_SOLVER: int = -1

# CPU process count for parallel greedy NN seed tours computation
NUM_PROCESSES_SEEDING: int = -1

# Default seeding strategy
SEED_STRATEGY: str = "Greedy"
