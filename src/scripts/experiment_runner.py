"""
TSP Seeding Strategy Experiment Runner
=======================================
Parallelism design (24-core machine):
  - 12 seeds per trial, 12 worker processes per trial  (1 core/seed)
  - 2 configs run concurrently via ThreadPoolExecutor
    → 24 cores fully occupied at all times
"""

import time
import os
import csv
import json
import argparse
import threading
import numpy as np
import scipy.stats as stats
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict, Any, Optional

from src.utils.data_io import load_cities
from src.core.preprocessing import (
    build_candidate_sets,
    refine_candidate_set_with_alpha,
    hilbert_reorder_cities,
)
from src.core.seed_generation import (
    generate_hilbert_seeds,
    generate_greedy_nn_seeds,
    generate_random_seeds,
)
from src.core.orchestration import parallel_solve

PROGRESS_FILE = "scratch/experiment_progress.json"
NUM_SEEDS = 12          # seeds per trial
NUM_WORKERS = 12        # worker processes per trial (1 per seed)
CONCURRENT_CONFIGS = 2  # number of configs running simultaneously


# ──────────────────────────────────────────────────────────────────────────────
# HK Cache
# ──────────────────────────────────────────────────────────────────────────────

def get_hk_cache(n: int, new_to_orig: np.ndarray) -> Tuple[float, np.ndarray]:
    hk_npy = f"data/sample_{n}_hk.npy"
    pi_npy  = f"data/sample_{n}_pi.npy"
    if os.path.exists(hk_npy) and os.path.exists(pi_npy):
        lb_val  = float(np.load(hk_npy)[0])
        pi_orig = np.load(pi_npy)
        pi      = pi_orig[new_to_orig]
        return lb_val, pi
    elif os.path.exists(hk_npy):
        lb_val = float(np.load(hk_npy)[0])
        pi     = np.zeros(n, dtype=np.float64)
        return lb_val, pi
    else:
        raise FileNotFoundError(
            f"HK cache for N={n} not found at '{hk_npy}'. "
            f"Run: python -m src.scripts.main --n {n}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Single Trial
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    config: Dict[str, Any],
    coords: np.ndarray,
    candidate_set: np.ndarray,
    n: int,
    num_seeds: int = NUM_SEEDS,
    num_processes: int = NUM_WORKERS,
) -> float:
    """Run one trial of a config. Returns best tour length."""
    is_initial_test = config["is_initial_test"]
    kicks   = config["kicks"]
    iters   = config["iters"]
    max_opt = config["max_opt"]

    # ── build initial seeds ────────────────────────────────────────────────────
    if is_initial_test:
        h_cnt = config["hilbert_cnt"]
        r_cnt = config["random_cnt"]
        g_cnt = config["greedy_cnt"]
        # Scale counts proportionally to num_seeds (configs defined for 12)
        total = h_cnt + r_cnt + g_cnt   # always == 12 in config definitions
        assert total == num_seeds, f"Seed counts must sum to {num_seeds}"
        seeds_list: List[np.ndarray] = []
        if h_cnt > 0:
            seeds_list.append(generate_hilbert_seeds(coords, num_seeds=h_cnt))
        if r_cnt > 0:
            seeds_list.append(generate_random_seeds(n, num_seeds=r_cnt))
        if g_cnt > 0:
            seeds_list.append(generate_greedy_nn_seeds(coords, candidate_set, num_seeds=g_cnt))
        seeds = np.vstack(seeds_list) if len(seeds_list) > 1 else seeds_list[0]
    else:
        # Standard initial mix: 9 Hilbert + 3 Greedy NN = 12 seeds
        seeds = np.vstack([
            generate_hilbert_seeds(coords, num_seeds=9),
            generate_greedy_nn_seeds(coords, candidate_set, num_seeds=3),
        ])

    locked_edges = np.full((n, 2), -1, dtype=np.int32)

    global_best_tour: Optional[np.ndarray] = None
    global_best_length = np.inf
    start_opt = time.time()

    for current_iter in range(1, iters + 1):
        iter_start = time.time()
        print(f"      [ITER] Iteration {current_iter}/{iters} starting (kicks={kicks}, max_opt={max_opt})...", flush=True)
        results = parallel_solve(
            seeds, coords, candidate_set, locked_edges,
            num_processes=num_processes,
            num_kicks=kicks,
            max_opt=max_opt,
            iteration_start_time=iter_start,
            total_start_time=start_opt,
        )
        print(f"      [ITER] Iteration {current_iter}/{iters} completed.", flush=True)

        for tour, length in results:
            if length < global_best_length:
                global_best_length = length
                global_best_tour = tour.copy()

        # Re-seed for next iteration
        if current_iter < iters:
            assert global_best_tour is not None
            strategy = config["strategy"]
            h = num_seeds // 2   # half
            q = num_seeds // 4   # quarter
            r = num_seeds - h    # remainder for exploitation

            if strategy == "continuation":
                seeds = np.empty((num_seeds, n), dtype=np.int32)
                for i, (tour, _) in enumerate(results[:num_seeds]):
                    seeds[i] = tour.copy()
            elif strategy == "pure_best":
                seeds = np.tile(global_best_tour, (num_seeds, 1))
            elif strategy == "50_best_50_hilbert":
                seeds = np.vstack([
                    np.tile(global_best_tour, (h, 1)),
                    generate_hilbert_seeds(coords, num_seeds=num_seeds - h),
                ])
            elif strategy == "75_best_25_hilbert":
                seeds = np.vstack([
                    np.tile(global_best_tour, (num_seeds - q, 1)),
                    generate_hilbert_seeds(coords, num_seeds=q),
                ])
            elif strategy == "50_best_50_random":
                seeds = np.vstack([
                    np.tile(global_best_tour, (h, 1)),
                    generate_random_seeds(n, num_seeds=num_seeds - h),
                ])
            elif strategy == "75_best_25_random":
                seeds = np.vstack([
                    np.tile(global_best_tour, (num_seeds - q, 1)),
                    generate_random_seeds(n, num_seeds=q),
                ])
            elif strategy == "50_best_25_hilbert_25_random":
                seeds = np.vstack([
                    np.tile(global_best_tour, (h, 1)),
                    generate_hilbert_seeds(coords, num_seeds=q),
                    generate_random_seeds(n, num_seeds=num_seeds - h - q),
                ])
            elif strategy == "50_best_25_hilbert_25_greedy":
                seeds = np.vstack([
                    np.tile(global_best_tour, (h, 1)),
                    generate_hilbert_seeds(coords, num_seeds=q),
                    generate_greedy_nn_seeds(coords, candidate_set, num_seeds=num_seeds - h - q),
                ])

    return global_best_length


# ──────────────────────────────────────────────────────────────────────────────
# Statistical Analysis
# ──────────────────────────────────────────────────────────────────────────────

def perform_statistical_analysis(
    results_dict: Dict[str, List[float]]
) -> Tuple[float, float, Any, float, float, Any]:
    group_a = [results_dict[f"Config {i}"] for i in range(1, 9)]
    group_b = [results_dict[f"Config {i}"] for i in range(9, 17)]

    f_a, p_a = stats.f_oneway(*group_a)
    f_b, p_b = stats.f_oneway(*group_b)

    try:
        tukey_a = stats.tukey_hsd(*group_a)
    except Exception as e:
        tukey_a = str(e)
    try:
        tukey_b = stats.tukey_hsd(*group_b)
    except Exception as e:
        tukey_b = str(e)

    return f_a, p_a, tukey_a, f_b, p_b, tukey_b


# ──────────────────────────────────────────────────────────────────────────────
# Markdown Report
# ──────────────────────────────────────────────────────────────────────────────

def build_markdown_report(
    configs: Dict[int, Dict[str, Any]],
    results_dict: Dict[str, List[float]],
    runtimes_dict: Dict[str, List[float]],
    lb_val: float,
    n: int,
    kicks_init: int,
    kicks_reseed: int,
    trials_init: int,
    trials_reseed: int,
    iters_reseed: int,
    num_seeds: int,
) -> str:
    lines: List[str] = []
    ts = time.strftime("%Y-%m-%d %H:%M")

    lines.append(f"\n## [{ts}] - Seeding & Re-seeding Comprehensive Comparison (N={n}, max_opt=5)")
    lines.append(f"- **Held-Karp LB**: {lb_val:.2f}")
    lines.append(f"- **Seeds per trial**: {num_seeds}")
    lines.append(f"- **Initial Seed Test**: kicks={kicks_init}, iters=1, trials={trials_init}")
    lines.append(f"- **Re-seeding Test**: kicks={kicks_reseed}, iters={iters_reseed}, trials={trials_reseed}")
    lines.append("")

    # Summary table
    lines.append("### Summary of Configuration Results")
    lines.append("| ID | Configuration | Trials | Avg Length | Avg Gap (%) | Std Dev | Min | Max | Avg Time (s) | Std Time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for c_id in range(1, 17):
        cfg  = configs[c_id]
        key  = f"Config {c_id}"
        data = np.array(results_dict[key])
        tdata = np.array(runtimes_dict[key])
        nt    = len(data)
        avg_l = np.mean(data)
        std_l = float(np.std(data, ddof=1)) if nt > 1 else 0.0
        avg_t = np.mean(tdata)
        std_t = float(np.std(tdata, ddof=1)) if nt > 1 else 0.0
        gap   = (avg_l - lb_val) / lb_val * 100
        lines.append(
            f"| {c_id} | {cfg['name']} | {nt} | "
            f"{avg_l:.2f} | {gap:.4f}% | {std_l:.2f} | "
            f"{np.min(data):.2f} | {np.max(data):.2f} | "
            f"{avg_t:.1f} | {std_t:.1f} |"
        )

    # Raw data table
    max_t = max(trials_init, trials_reseed)
    lines.append("")
    lines.append("### Raw Trial Data (Tour Lengths)")
    lines.append("| ID | " + " | ".join(f"T{t+1}" for t in range(max_t)) + " |")
    lines.append("|---|" + "|".join("---" for _ in range(max_t)) + "|")
    for c_id in range(1, 17):
        key  = f"Config {c_id}"
        vals = results_dict[key]
        cells = [f"{v:.2f}" for v in vals] + ["--"] * (max_t - len(vals))
        lines.append(f"| {c_id} | " + " | ".join(cells) + " |")

    # Statistical analysis
    f_a, p_a, tukey_a, f_b, p_b, tukey_b = perform_statistical_analysis(results_dict)
    group_a = [results_dict[f"Config {i}"] for i in range(1, 9)]
    group_b = [results_dict[f"Config {i}"] for i in range(9, 17)]

    lines.append("")
    lines.append("### Statistical Significance Analysis")

    for label, f_val, p_val, tukey, grp_data, id_start, n_grp in [
        ("Initial Seeding Mix (Configs 1-8)",    f_a, p_a, tukey_a, group_a, 1, 8),
        ("Re-seeding Strategies (Configs 9-16)", f_b, p_b, tukey_b, group_b, 9, 8),
    ]:
        lines.append("")
        lines.append(f"#### {label}")
        n_trials_grp = trials_init if id_start == 1 else trials_reseed
        lines.append(f"- **Sample size per config**: {n_trials_grp} trials")
        if np.isnan(f_val):
            lines.append("- **One-way ANOVA**: N/A (insufficient samples)")
        else:
            lines.append(f"- **One-way ANOVA F-statistic**: {f_val:.4f}")
            lines.append(f"- **p-value**: {p_val:.4e}")
            sig = p_val < 0.05
            lines.append(f"- **Statistically Significant**: **{'YES' if sig else 'NO'}** (alpha=0.05)")
            if sig and not isinstance(tukey, str):
                lines.append("")
                lines.append("**Pairwise Tukey HSD (significant pairs only):**")
                lines.append("| Config i | Config j | Mean Diff | p-value | Reject H0? |")
                lines.append("|---|---|---|---|---|")
                for i in range(n_grp):
                    for j in range(i + 1, n_grp):
                        pv   = tukey.pvalue[i, j]
                        diff = np.mean(grp_data[i]) - np.mean(grp_data[j])
                        if pv < 0.05:
                            lines.append(
                                f"| Config {i+id_start} | Config {j+id_start} "
                                f"| {diff:.2f} | {pv:.4e} | Yes |"
                            )
            elif sig and isinstance(tukey, str):
                lines.append(f"- Tukey HSD error: {tukey}")

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Progress / checkpoint helpers (thread-safe)
# ──────────────────────────────────────────────────────────────────────────────

_ckpt_lock = threading.Lock()

def save_checkpoint(
    results_dict: Dict[str, List[float]],
    runtimes_dict: Dict[str, List[float]],
) -> None:
    with _ckpt_lock:
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"lengths": results_dict, "runtimes": runtimes_dict}, f, indent=2)


def print_config_report(
    c_id: int, name: str,
    l_arr: np.ndarray, t_arr: np.ndarray,
) -> None:
    nt    = len(l_arr)
    std_l = float(np.std(l_arr, ddof=1)) if nt > 1 else 0.0
    std_t = float(np.std(t_arr, ddof=1)) if nt > 1 else 0.0
    print(f"\n[REPORT] Config {c_id} ({name}) -- {nt} trials done")
    print(f"  Length : avg={np.mean(l_arr):.2f}  std={std_l:.2f}  "
          f"min={np.min(l_arr):.2f}  max={np.max(l_arr):.2f}")
    print(f"  Runtime: avg={np.mean(t_arr):.2f}s  std={std_t:.2f}s")
    print("-" * 60, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TSP Seeding Strategy Experiment Runner")
    parser.add_argument("--dry-run", action="store_true", help="Quick dry-run at N=100")
    args = parser.parse_args()

    if args.dry_run:
        n            = 100
        kicks_init   = 10
        kicks_reseed = 10
        trials_init  = 1
        trials_reseed = 1
        iters_reseed = 2
        num_seeds    = 12
        num_workers  = 12
        concurrent   = 1   # dry-run: serial
    else:
        n            = 5000
        kicks_init   = 500
        kicks_reseed = 100
        trials_init  = 50
        trials_reseed = 50
        iters_reseed = 5
        num_seeds    = 12       # 12 seeds per trial
        num_workers  = 12       # 12 worker processes per trial
        concurrent   = 2        # 2 configs run in parallel → 24 cores total

    CSV_PATH = "scratch/seeding_strategy.csv"
    # Clear the file for a fresh start if not dry-run
    if not args.dry_run:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["config_id", "config_name", "trial_idx", "length", "runtime", "gap_pct"])
    elif not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["config_id", "config_name", "trial_idx", "length", "runtime", "gap_pct"])

    print(f"TSP Experiment Runner")
    print(f"  N={n} | seeds={num_seeds} | workers/trial={num_workers} | concurrent configs={concurrent}")
    print(f"  Initial Seed Test : {trials_init} trials, kicks={kicks_init}, iters=1")
    print(f"  Re-seeding Test   : {trials_reseed} trials, kicks={kicks_reseed}, iters={iters_reseed}")

    # ── Setup ─────────────────────────────────────────────────────────────────
    print("  [STEP] Loading cities...", flush=True)
    coords_orig = load_cities("data/cities.csv")[:n]
    print("  [STEP] Hilbert reordering...", flush=True)
    coords, orig_to_new = hilbert_reorder_cities(coords_orig)
    new_to_orig = np.empty(n, dtype=np.int32)
    new_to_orig[orig_to_new] = np.arange(n, dtype=np.int32)

    print("  [STEP] Building initial candidate sets...", flush=True)
    candidate_set = build_candidate_sets(coords, k=64)

    if args.dry_run:
        lb_val = 27175.81
        pi = np.zeros(n, dtype=np.float64)
    else:
        print("  [STEP] Getting HK cache...", flush=True)
        lb_val, pi = get_hk_cache(n, new_to_orig)

    print(f"  HK Lower Bound: {lb_val:.2f}")

    print("  [STEP] Refining candidate sets with alpha...", flush=True)
    candidate_set = refine_candidate_set_with_alpha(coords, candidate_set, pi)
    candidate_set = candidate_set[:, :40]

    # ── Configs ───────────────────────────────────────────────────────────────
    # All seed counts sum to num_seeds (12)
    configs: Dict[int, Dict[str, Any]] = {
        # Initial Seed Tests
        1:  {"name": "100% Hilbert",                 "is_initial_test": True,  "kicks": kicks_init,   "iters": 1,           "max_opt": 5, "hilbert_cnt": 12, "random_cnt": 0,  "greedy_cnt": 0},
        2:  {"name": "100% Random",                  "is_initial_test": True,  "kicks": kicks_init,   "iters": 1,           "max_opt": 5, "hilbert_cnt": 0,  "random_cnt": 12, "greedy_cnt": 0},
        3:  {"name": "100% Greedy NN",               "is_initial_test": True,  "kicks": kicks_init,   "iters": 1,           "max_opt": 5, "hilbert_cnt": 0,  "random_cnt": 0,  "greedy_cnt": 12},
        4:  {"name": "50% Hilbert + 50% Random",     "is_initial_test": True,  "kicks": kicks_init,   "iters": 1,           "max_opt": 5, "hilbert_cnt": 6,  "random_cnt": 6,  "greedy_cnt": 0},
        5:  {"name": "50% Hilbert + 50% Greedy NN",  "is_initial_test": True,  "kicks": kicks_init,   "iters": 1,           "max_opt": 5, "hilbert_cnt": 6,  "random_cnt": 0,  "greedy_cnt": 6},
        6:  {"name": "50% Random + 50% Greedy NN",   "is_initial_test": True,  "kicks": kicks_init,   "iters": 1,           "max_opt": 5, "hilbert_cnt": 0,  "random_cnt": 6,  "greedy_cnt": 6},
        7:  {"name": "75% Hilbert + 25% Greedy NN",  "is_initial_test": True,  "kicks": kicks_init,   "iters": 1,           "max_opt": 5, "hilbert_cnt": 9,  "random_cnt": 0,  "greedy_cnt": 3},
        8:  {"name": "Balanced Mix (4H, 4R, 4G)",    "is_initial_test": True,  "kicks": kicks_init,   "iters": 1,           "max_opt": 5, "hilbert_cnt": 4,  "random_cnt": 4,  "greedy_cnt": 4},

        # Re-seeding Tests (strategy key required, seed counts handled in run_experiment)
        9:  {"name": "100% Continuation (No Resets)",               "is_initial_test": False, "kicks": kicks_reseed, "iters": iters_reseed, "max_opt": 5, "strategy": "continuation"},
        10: {"name": "100% Best (Exploitation)",                    "is_initial_test": False, "kicks": kicks_reseed, "iters": iters_reseed, "max_opt": 5, "strategy": "pure_best"},
        11: {"name": "50% Best + 50% fresh Hilbert",                "is_initial_test": False, "kicks": kicks_reseed, "iters": iters_reseed, "max_opt": 5, "strategy": "50_best_50_hilbert"},
        12: {"name": "75% Best + 25% fresh Hilbert",                "is_initial_test": False, "kicks": kicks_reseed, "iters": iters_reseed, "max_opt": 5, "strategy": "75_best_25_hilbert"},
        13: {"name": "50% Best + 50% fresh Random",                 "is_initial_test": False, "kicks": kicks_reseed, "iters": iters_reseed, "max_opt": 5, "strategy": "50_best_50_random"},
        14: {"name": "75% Best + 25% fresh Random",                 "is_initial_test": False, "kicks": kicks_reseed, "iters": iters_reseed, "max_opt": 5, "strategy": "75_best_25_random"},
        15: {"name": "50% Best + 25% Hilbert + 25% Random",         "is_initial_test": False, "kicks": kicks_reseed, "iters": iters_reseed, "max_opt": 5, "strategy": "50_best_25_hilbert_25_random"},
        16: {"name": "50% Best + 25% Hilbert + 25% Greedy NN",      "is_initial_test": False, "kicks": kicks_reseed, "iters": iters_reseed, "max_opt": 5, "strategy": "50_best_25_hilbert_25_greedy"},
    }

    # ── Load checkpoint ───────────────────────────────────────────────────────
    progress: Dict[str, Any] = {}
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                progress = json.load(f)
            print(f"Loaded checkpoint from {PROGRESS_FILE}")
        except Exception:
            print("Failed to load checkpoint, starting fresh.")

    results_dict: Dict[str, List[float]]  = {}
    runtimes_dict: Dict[str, List[float]] = {}
    lengths_ckpt  = progress.get("lengths", {})
    runtimes_ckpt = progress.get("runtimes", {})
    for c_id in range(1, 17):
        key = f"Config {c_id}"
        results_dict[key]  = lengths_ckpt.get(key, [])
        runtimes_dict[key] = runtimes_ckpt.get(key, [])

    # ── Build queue of (config_id, trial_idx) work items ─────────────────────
    work_queue: List[Tuple[int, int]] = []
    for c_id in range(1, 17):
        cfg = configs[c_id]
        target = trials_init if cfg["is_initial_test"] else trials_reseed
        key = f"Config {c_id}"
        already_done = len(results_dict[key])
        for t in range(already_done, target):
            work_queue.append((c_id, t))

    print(f"\n{len(work_queue)} trials remaining across 16 configs.")
    print(f"Running {concurrent} config trials in parallel (each using {num_workers} workers).\n")

    # ── Execute with ThreadPoolExecutor (threads dispatch separate mp.Pools) ──
    # Using threads here avoids nesting mp.Pool inside mp.Pool.
    # Each thread calls parallel_solve which spawns its own mp.Pool of num_workers.
    _dict_lock = threading.Lock()

    def run_one(c_id: int, trial_num: int) -> Tuple[int, int, float, float]:
        """Run a single trial and return (c_id, trial_num, length, elapsed)."""
        cfg = configs[c_id]
        print(f"    [TRIAL] Starting Config {c_id} Trial {trial_num}...", flush=True)
        t0 = time.time()
        best_len = run_experiment(
            cfg, coords, candidate_set, n,
            num_seeds=num_seeds,
            num_processes=num_workers,
        )
        elapsed = time.time() - t0
        print(f"    [TRIAL] Finished Config {c_id} Trial {trial_num} in {elapsed:.1f}s", flush=True)
        return c_id, trial_num, best_len, elapsed

    # Group work items by config to respect ordering per config
    # Submit one trial per config at a time, up to `concurrent` at once
    config_queues: Dict[int, List[int]] = {c: [] for c in range(1, 17)}
    for c_id, t_idx in work_queue:
        config_queues[c_id].append(t_idx)

    active_futures: Dict[Any, Tuple[int, int]] = {}

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        # Seed initial futures — one per config, up to `concurrent` at once
        pending_configs = [c for c in range(1, 17) if config_queues[c]]

        def submit_next_for_config(c_id: int) -> Optional[Any]:
            if not config_queues[c_id]:
                return None
            t_idx = config_queues[c_id].pop(0)
            trial_num = t_idx + 1
            cfg = configs[c_id]
            print(f"  [START] Config {c_id} ({cfg['name']}) — Trial {trial_num}", flush=True)
            fut = executor.submit(run_one, c_id, trial_num)
            return fut

        # Fill up to `concurrent` slots
        config_iter = iter(pending_configs)
        for _ in range(concurrent):
            try:
                c = next(config_iter)
                fut = submit_next_for_config(c)
                if fut:
                    active_futures[fut] = (c, 0)
            except StopIteration:
                break

        next_config_idx = concurrent  # index into pending_configs for next config to start

        while active_futures:
            done_futs = [f for f in active_futures if f.done()]
            for fut in done_futs:
                c_id, _ = active_futures.pop(fut)
                c_id_res, trial_num, best_len, elapsed = fut.result()

                with _dict_lock:
                    results_dict[f"Config {c_id_res}"].append(best_len)
                    runtimes_dict[f"Config {c_id_res}"].append(elapsed)

                gap_pct = ((best_len / lb_val) - 1.0) * 100.0
                with open(CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([c_id_res, configs[c_id_res]["name"], trial_num, best_len, elapsed, gap_pct])

                print(
                    f"  [DONE ] Config {c_id_res} ({configs[c_id_res]['name']}) "
                    f"Trial {trial_num} | Length: {best_len:.2f} | Time: {elapsed:.1f}s",
                    flush=True
                )
                save_checkpoint(results_dict, runtimes_dict)

                cfg = configs[c_id_res]
                target = trials_init if cfg["is_initial_test"] else trials_reseed
                n_done = len(results_dict[f"Config {c_id_res}"])

                if n_done >= target:
                    # Config fully finished — print aggregate
                    l_arr = np.array(results_dict[f"Config {c_id_res}"])
                    t_arr = np.array(runtimes_dict[f"Config {c_id_res}"])
                    print_config_report(c_id_res, cfg["name"], l_arr, t_arr)
                else:
                    # Submit next trial for this same config
                    fut2 = submit_next_for_config(c_id_res)
                    if fut2:
                        active_futures[fut2] = (c_id_res, 0)

                # If a slot freed up and there's a new config to start, add it
                if len(active_futures) < concurrent and next_config_idx < len(pending_configs):
                    nc = pending_configs[next_config_idx]
                    next_config_idx += 1
                    fut3 = submit_next_for_config(nc)
                    if fut3:
                        active_futures[fut3] = (nc, 0)

            if active_futures:
                time.sleep(1)

    print("\n=== All trials completed! Writing results to trials.md ===")

    report_md = build_markdown_report(
        configs, results_dict, runtimes_dict,
        lb_val, n,
        kicks_init, kicks_reseed,
        trials_init, trials_reseed, iters_reseed,
        num_seeds,
    )
    with open("trials.md", "a", encoding="utf-8") as f:
        f.write(report_md)
    print("Results appended to trials.md.")

    if not args.dry_run and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Checkpoint removed.")


if __name__ == "__main__":
    main()
