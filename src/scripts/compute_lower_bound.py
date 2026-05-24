import time
from src.utils.data_io import load_cities
from src.core.preprocessing import build_candidate_sets, refine_candidate_set_with_alpha
from src.core.validation import compute_hk_lower_bound


def main() -> None:
    print("Step 1: Loading cities...")
    coords = load_cities("data/cities.csv")
    n = coords.shape[0]
    print(f"Loaded {n} cities.")

    print("Step 2: Building candidate sets...")
    candidate_set = build_candidate_sets(coords, k=16)

    print("Step 3: Estimating Held-Karp lower bound and refining candidate set...")
    print("This is a one-time intensive calculation.")
    start_val = time.time()
    # Increase max_iter for a more stable bound since it's only run once.
    lower_bound, pi = compute_hk_lower_bound(coords, candidate_set, max_iter=200)
    candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)
    duration = time.time() - start_val
    print(f"Calculation done in {duration:.2f}s.")
    print(f"Estimated Lower Bound: {lower_bound:.2f}")

    # Save to file
    output_path = "data/lower_bound.txt"
    with open(output_path, "w") as f:
        f.write(str(lower_bound))
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
