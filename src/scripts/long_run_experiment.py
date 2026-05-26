"""
TSP Long-Run Seeding Experiment (Optimized)
===========================================
Focuses only on 100% Random vs 100% Greedy NN.
10 iterations per trial, 100 kicks per iteration.
100% Best (Exploitation) re-seeding.
"""

import argparse
import csv
import os
import time
import threading
import numpy as np
import scipy.stats as stats
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

from src.utils.data_io import load_cities
from src.core.preprocessing import hilbert_reorder_cities, build_candidate_sets, refine_candidate_set_with_alpha
from src.core.orchestration import parallel_solve
from src.core.seed_generation import (
    generate_random_seeds,
    generate_greedy_nn_seeds,
)

# ──────────────────────────────────────────────────────────────────────────────
# Global experiment parameters
# ──────────────────────────────────────────────────────────────────────────────

NUM_SEEDS     = 12       # 12 seeds per trial
NUM_WORKERS   = 12       # 1 worker per seed
CONCURRENT    = 2        # 2 trials at once (24 cores)
CSV_PATH      = "scratch/long_run_strategy.csv"


def get_hk_cache(n: int, new_to_orig: np.ndarray) -> Tuple[float, np.ndarray]:
    hk_npy = f"data/sample_{n}_hk.npy"
    pi_npy = f"data/sample_{n}_pi.npy"
    if os.path.exists(hk_npy) and os.path.exists(pi_npy):
        lb_val  = float(np.load(hk_npy)[0])
        pi_orig = np.load(pi_npy)
        pi      = pi_orig[new_to_orig]
        return lb_val, pi
    else:
        raise FileNotFoundError(f"HK cache for N={n} not found.")


def run_experiment(
    config: Dict[str, Any],
    coords: np.ndarray,
    candidate_set: np.ndarray,
    n: int,
    num_seeds: int,
    num_processes: int,
) -> float:
    kicks   = config["kicks"]
    iters   = config["iters"]
    max_opt = config["max_opt"]

    # Initial seeds
    if config["type"] == "random":
        seeds = generate_random_seeds(n, num_seeds=num_seeds)
    else:
        seeds = generate_greedy_nn_seeds(coords, candidate_set, num_seeds=num_seeds)

    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    global_best_tour: Optional[np.ndarray] = None
    global_best_length = np.inf
    start_opt = time.time()

    for current_iter in range(1, iters + 1):
        iter_start = time.time()
        results = parallel_solve(
            seeds, coords, candidate_set, locked_edges,
            num_processes=num_processes,
            num_kicks=kicks,
            max_opt=max_opt,
            iteration_start_time=iter_start,
            total_start_time=start_opt,
        )

        for tour, length in results:
            if length < global_best_length:
                global_best_length = length
                global_best_tour = tour.copy()

        # Always use 100% Best reset
        if current_iter < iters:
            assert global_best_tour is not None
            seeds = np.tile(global_best_tour, (num_seeds, 1))

    return global_best_length


def build_markdown_report(
    configs: Dict[int, Dict[str, Any]],
    results_dict: Dict[int, List[float]],
    runtimes_dict: Dict[int, List[float]],
    lb_val: float,
    n: int,
    kicks_val: int,
    trials_val: int,
    iters_val: int,
) -> str:
    lines = [
        f"\n## [{datetime.now().strftime('%Y-%m-%d %H:%M')}] - Optimized Long-Run Comparison (N={n})",
        f"- **Held-Karp LB**: {lb_val:.2f}",
        f"- **Parameters**: kicks={kicks_val}, iterations={iters_val}, target_trials={trials_val}",
        "\n### Summary Table",
        "| ID | Configuration                           | Trials | Avg Length | Avg Gap (%) | Std Dev | Min       | Max       |",
        "|----|-----------------------------------------|--------|------------|-------------|---------|-----------|-----------|",
    ]

    for c_id in sorted(results_dict.keys()):
        name = configs[c_id]["name"]
        res  = results_dict[c_id]
        if not res: continue
        avg_l = np.mean(res); std_l = np.std(res, ddof=1) if len(res)>1 else 0.0
        gap = (avg_l - lb_val)/lb_val*100; min_l = np.min(res); max_l = np.max(res)
        lines.append(f"| {c_id:<2} | {name:<39} | {len(res):<6} | {avg_l:<10.2f} | {gap:<10.4f}% | {std_l:<7.2f} | {min_l:<9.2f} | {max_l:<9.2f} |")

    # t-test for Random vs Greedy if both have enough data
    if 2 in results_dict and 3 in results_dict:
        r_data = results_dict[2]
        g_data = results_dict[3]
        if len(r_data) >= 5 and len(g_data) >= 5:
            t_stat, p_val = stats.ttest_ind(r_data, g_data)
            lines.append("\n### Random vs Greedy NN (t-test)")
            lines.append(f"- **p-value**: {p_val:.4e}")
            lines.append(f"- **Significant**: {'YES' if p_val < 0.05 else 'NO'}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()

    n=5000; kicks=100; iters=10; trials=args.trials

    # Setup
    coords_orig = load_cities("data/cities.csv")[:n]
    coords, o2n = hilbert_reorder_cities(coords_orig)
    n2orig = np.empty(n, dtype=np.int32); n2orig[o2n] = np.arange(n)
    lb_val, pi = get_hk_cache(n, n2orig)
    candidate_set = build_candidate_sets(coords, k=64)
    candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)[:, :40]

    configs = {
        2: {"name": "LR: 100% Random",  "type": "random", "kicks": kicks, "iters": iters, "max_opt": 5},
        3: {"name": "LR: 100% Greedy NN","type": "greedy", "kicks": kicks, "iters": iters, "max_opt": 5},
    }

    # Load current progress from CSV
    results_dict = {2: [], 3: []}
    runtimes_dict = {2: [], 3: []}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                c_id = int(row['config_id'])
                if c_id in configs:
                    results_dict[c_id].append(float(row['length']))
                    runtimes_dict[c_id].append(float(row['runtime']))

    work_queue = []
    for c_id in [2, 3]:
        done = len(results_dict[c_id])
        for t in range(done, trials):
            work_queue.append((c_id, t))

    print(f"Starting optimized run: {len(work_queue)} trials to complete.")

    _lock = threading.Lock()

    def run_one(c_id, t_idx):
        t0 = time.time()
        best = run_experiment(configs[c_id], coords, candidate_set, n, NUM_SEEDS, NUM_WORKERS)
        elapsed = time.time() - t0
        return c_id, t_idx+1, best, elapsed

    with ThreadPoolExecutor(max_workers=CONCURRENT) as executor:
        futures = [executor.submit(run_one, c, t) for c, t in work_queue]
        for fut in futures:
            c_id_res, t_num, best_len, elapsed = fut.result()
            with _lock:
                results_dict[c_id_res].append(best_len)
                gap = (best_len / lb_val - 1.0) * 100
                with open(CSV_PATH, "a", newline="") as f:
                    csv.writer(f).writerow([c_id_res, configs[c_id_res]["name"], t_num, best_len, elapsed, gap])
                print(f"  [DONE] {configs[c_id_res]['name']} Trial {t_num} | Gap: {gap:.2f}% | Time: {elapsed:.1f}s")

    report = build_markdown_report(configs, results_dict, runtimes_dict, lb_val, n, kicks, trials, iters)
    with open("trials.md", "a") as f: f.write(report)
    print("\n=== Optimized Experiment Complete! ===")

if __name__ == "__main__":
    main()
