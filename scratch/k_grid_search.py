import platform
# Mock platform.machine to bypass WMI query hang on Windows
platform.machine = lambda: "AMD64"

import subprocess
import re
import csv
import statistics
import concurrent.futures
import time
import os
import sys
import argparse

def run_trial(n, kicks, iters, max_k, trial_idx, no_warmup=True):
    cmd = [
        sys.executable, "src/scripts/full_scale_bench.py",
        "--n", str(n),
        "--seeds", "1",
        "--kicks", str(kicks),
        "--iters", str(iters),
        "--max_opt", str(max_k),
        "--no_backbone",
        "--processes", "1",
        "--no_log"
    ]
    if no_warmup:
        cmd.append("--no_warmup")
    
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
    output = proc.stdout
    
    if proc.returncode != 0:
        print(f"Warning: Trial {trial_idx} config (N={n}, K={max_k}) process exited with code {proc.returncode}")
        print(f"stderr: {proc.stderr}")
    
    gap_match = re.search(r"Gap:\s+([0-9.]+)%", output)
    time_match = re.search(r"Total execution time:\s+([0-9.]+)s", output)
    length_match = re.search(r"Final Best Length:\s+([0-9.]+)", output)
    
    if gap_match and time_match and length_match:
        return {
            "n": n,
            "kicks": kicks,
            "iters": iters,
            "max_k": max_k,
            "trial": trial_idx,
            "gap": float(gap_match.group(1)),
            "time": float(time_match.group(1)),
            "length": float(length_match.group(1))
        }
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate grid search trials.")
    parser.add_argument("--sizes", type=str, default="500", help="Comma-separated list of N (e.g. 500,1000)")
    parser.add_argument("--kicks", type=str, default="2000", help="Comma-separated list of kicks (e.g. 2000)")
    parser.add_argument("--iters", type=str, default="1", help="Comma-separated list of iterations (e.g. 1)")
    parser.add_argument("--ks", type=str, default="2,3,4,5", help="Comma-separated list of K values (e.g. 2,3,4,5)")
    parser.add_argument("--trials", type=int, default=8, help="Number of trials per configuration")
    parser.add_argument("--workers", type=int, default=16, help="Maximum number of parallel workers")
    args = parser.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    kicks_list = [int(x) for x in args.kicks.split(",")]
    iters_list = [int(x) for x in args.iters.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    NUM_TRIALS = args.trials
    MAX_WORKERS = args.workers

    # Generate tasks: list of (N, kicks, iters, K)
    param_configs = []
    for n in sizes:
        for kicks in kicks_list:
            for iters in iters_list:
                for k in ks:
                    param_configs.append((n, kicks, iters, k))

    print(f"Starting Flexible Grid Search. Total configurations: {len(param_configs)}")
    print(f"Using {MAX_WORKERS} parallel workers, {NUM_TRIALS} trials per config.")
    
    all_summaries = []
    start_total = time.time()
    
    for n, kicks, iters, max_k in param_configs:
        print(f"\n>>> Configuration: N={n}, Kicks={kicks}, Iters={iters}, K={max_k}")
        
        # Warmup/Cache check for this N (sequential, once)
        print(f"Performing warmup and cache check for N={n}...")
        warmup_res = run_trial(n, kicks, iters, max_k, -1, no_warmup=False)
        if warmup_res:
            print(f"Warmup successful. Gap: {warmup_res['gap']:.4f}%, Time: {warmup_res['time']:.2f}s, Length: {warmup_res['length']:.2f}")
        else:
            print(f"Warmup failed or returned no results. Proceeding anyway...")

        print(f"Running {NUM_TRIALS} parallel trials...")
        k_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(run_trial, n, kicks, iters, max_k, i) for i in range(NUM_TRIALS)]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    k_results.append(res)
        
        if k_results:
            gaps = [r["gap"] for r in k_results]
            times = [r["time"] for r in k_results]
            lengths = [r["length"] for r in k_results]
            
            avg_gap = statistics.mean(gaps)
            std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            avg_length = statistics.mean(lengths)
            std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0
            
            summary = {
                "N": n,
                "Kicks": kicks,
                "Iters": iters,
                "K": max_k,
                "avg_gap": avg_gap,
                "std_gap": std_gap,
                "avg_time": avg_time,
                "std_time": std_time,
                "avg_length": avg_length,
                "std_length": std_length,
                "trials": len(k_results)
            }
            all_summaries.append(summary)
            
            print(f"Summary for K={max_k}: Gap={avg_gap:.4f}% (±{std_gap:.4f}), Time={avg_time:.2f}s (±{std_time:.2f}), Length={avg_length:.2f} (±{std_length:.2f})")

    end_total = time.time()
    print(f"\nGrid Search completed in {end_total - start_total:.2f}s")
    
    # --- Logging to trials.md ---
    with open("trials.md", "a") as f:
        f.write(f"\n## Grid Search Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("| N | Kicks | Iters | K | Avg Gap (%) | Std Gap (%) | Avg Time (s) | Std Time (s) | Avg Length | Std Length | Trials |\n")
        f.write("|---|-------|-------|---|-------------|-------------|--------------|--------------|------------|------------|--------|\n")
        for s in all_summaries:
            f.write(f"| {s['N']} | {s['Kicks']} | {s['Iters']} | {s['K']} | {s['avg_gap']:.4f} | {s['std_gap']:.4f} | {s['avg_time']:.2f} | {s['std_time']:.2f} | {s['avg_length']:.2f} | {s['std_length']:.2f} | {s['trials']} |\n")

    # --- Updating notes.md ---
    with open("notes.md", "a") as f:
        f.write(f"\n## Grid Search Summary - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if all_summaries:
            best_gap_cfg = min(all_summaries, key=lambda x: x["avg_gap"])
            f.write(f"- **Best Quality Config**: N={best_gap_cfg['N']}, Kicks={best_gap_cfg['Kicks']}, Iters={best_gap_cfg['Iters']}, K={best_gap_cfg['K']} (Gap: {best_gap_cfg['avg_gap']:.4f}%, Length: {best_gap_cfg['avg_length']:.2f})\n")
            fastest_cfg = min(all_summaries, key=lambda x: x["avg_time"])
            f.write(f"- **Fastest Config**: N={fastest_cfg['N']}, Kicks={fastest_cfg['Kicks']}, Iters={fastest_cfg['Iters']}, K={fastest_cfg['K']} (Time: {fastest_cfg['avg_time']:.2f}s)\n")
            shortest_cfg = min(all_summaries, key=lambda x: x["avg_length"])
            f.write(f"- **Shortest Tour Config**: N={shortest_cfg['N']}, Kicks={shortest_cfg['Kicks']}, Iters={shortest_cfg['Iters']}, K={shortest_cfg['K']} (Length: {shortest_cfg['avg_length']:.2f}, Gap: {shortest_cfg['avg_gap']:.4f}%)\n")
