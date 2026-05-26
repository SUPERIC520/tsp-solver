import subprocess
import re
import csv
import itertools
import sys

# Definitive Grid
sizes = [100, 200, 500, 1000, 2000, 5000]
max_ks = [3, 4, 5]
kicks_list = [100, 200, 500, 1000, 2000, 5000]
iters_list = [1, 2, 5, 10]

def run_trial(n, max_k, kicks, iters):
    backbone_flag = "" if iters > 1 else "--no_backbone"
    # Set time limit to avoid extremely long runs for large N
    time_limit = 120 
    cmd = f"python src/scripts/full_scale_bench.py --n {n} --seeds 16 --kicks {kicks} --iters {iters} --max_opt {max_k} {backbone_flag} --time_limit {time_limit}"
    
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = proc.stdout
    gap_match = re.search(r"Gap:\s+([0-9.]+)%", output)
    time_match = re.search(r"Total execution time:\s+([0-9.]+)s", output)
    
    if gap_match and time_match:
        return float(gap_match.group(1)), float(time_match.group(1))
    return None, None

if __name__ == "__main__":
    results_file = "scratch/definitive_trends.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "max_k", "kicks_per_iter", "iters", "trial", "gap", "time"])
        writer.writeheader()
        
        # Iterating through all sizes
        for n in sizes:
            configs = list(itertools.product(max_ks, kicks_list, iters_list))
            for max_k, kicks, iters in configs:
                for trial in range(1, 11):
                    print(f"Testing N={n}, K={max_k}, Kicks={kicks}, Iters={iters}, Trial={trial}...", end=" ", flush=True)
                    gap, run_time = run_trial(n, max_k, kicks, iters)
                    if gap is not None:
                        writer.writerow({
                            "n": n,
                            "max_k": max_k,
                            "kicks_per_iter": kicks,
                            "iters": iters,
                            "trial": trial,
                            "gap": gap,
                            "time": run_time
                        })
                        f.flush()
                        print(f"Gap={gap:.4f}%")
                    else:
                        print("Failed/Timeout.")
