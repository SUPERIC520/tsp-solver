# Data I/O Module Tasks

- [x] **Task 1: Basic CSV Loading**
  - Implement `load_cities(filepath: str) -> np.ndarray` using `np.loadtxt` or `pandas`.
  - Ensure coordinates are loaded as `float64`.
  - Handle the "Index X Y" format specifically.
- [x] **Task 2: Verification of Loading**
  - Create `tests/data/sample_cities.csv` (small set of 5-10 cities).
  - Write a test script to assert shape and values of loaded coordinates.
- [x] **Task 3: Tour Export Functionality**
  - Implement `save_tour(filepath: str, tour: np.ndarray, length: float)`.
  - Output should include the ordered indices and the final total length.
- [x] **Task 4: Result Validation**
  - Verify that a saved tour can be re-loaded and its length recalculated correctly.
