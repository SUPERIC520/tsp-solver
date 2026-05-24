import time
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets


def main() -> None:
    print("Loading cities...")
    coords = load_cities("data/cities.csv")
    print(f"Loaded {len(coords)} cities.")

    # Warmup
    _ = build_candidate_sets(coords[:100], k=16)

    print("Building candidate sets...")
    start = time.time()
    candidate_set = build_candidate_sets(coords, k=16)
    end = time.time()

    print(f"Preprocessing completed in {end - start:.2f} seconds.")
    print(f"Candidate set shape: {candidate_set.shape}")
    print(f"Memory usage: {candidate_set.nbytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
