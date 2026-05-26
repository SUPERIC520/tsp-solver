import subprocess
import re
import csv
import itertools
import concurrent.futures
import multiprocessing as mp
import os

# Definitive Grid
sizes = [100, 200, 500, 1000, 2000, 5000]
max_ks = [3, 4, 5]
kicks_list = [100, 200, 500, 1000, 2000, 5000]
iters_list = [1, 2, 5, 10]
num_trials = 10

def run_single_trial(n, max_k, kicks, iters, trial):
    backbone_flag = "" if iters > 1 else "--no_backbone"
    # To avoid oversubscription, we limit processes per trial.
    # Total cores assumed: 16. If we run 4 trials in parallel, each gets 4 cores.
    procs_per_trial = 4 
    cmd = [
        "python", "src/scripts/full_scale_bench.py",
        "--n", str(n),
        "--seeds", "16",
        "--kicks", str(kicks),
        "--iters", str(iters),
        "--max_opt", str(max_k),
        "--processes", str(procs_per_trial),
        "--time_limit", "180" # 3 mins limit per trial
    ]
    if backbone_flag:
        cmd.append(backbone_flag)
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
        output = proc.stdout
        gap_match = re.search(r"Gap:\s+([0-9.]+)%", output)
        time_match = re.search(r"Total execution time:\s+([0-9.]+)s", output)
        if gap_match and time_match:
            return {
                "n": n, "max_k": max_k, "kicks": kicks, "iters": iters, "trial": trial,
                "gap": float(gap_match.group(1)), "time": float(time_match.group(1))
            }
    except Exception as e:
        print(f"Error in trial N={n}, K={max_k}, Trial={trial}: {e}")
    return None

if __name__ == "__main__":
    results_file = "scratch/definitive_trends_parallel.csv"
    
    # Generate all tasks
    tasks = []
    for n in sizes:
        for max_k in max_ks:
            for kicks in kicks_list:
                for iters in iters_list:
                    for trial in range(1, num_trials + 1):
                        tasks.append((n, max_k, kicks, iters, trial))
    
    print(f"Starting {len(tasks)} trials in parallel...")
    
    # With 16 cores and 4 processes per trial, we can run 4 trials concurrently safely.
    max_concurrent_trials = 4
    
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "max_k", "kicks", "iters", "trial", "gap", "time"])
        writer.writeheader()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_trials) as executor:
            future_to_task = {executor.submit(run_single_trial, *task): task for task in tasks}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                if result:
                    writer.writerow(result)
                    f.flush()
                
                completed += 1
                if completed % 10 == 0:
                    print(f"Progress: {completed}/{len(tasks)} trials completed.")
