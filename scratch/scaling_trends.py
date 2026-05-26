import subprocess
import re
import csv
import itertools
import sys

# Sizes to test
sizes = [100, 200, 500, 1000, 2000]

# High-Resolution Grid subset
max_ks = [3, 4]
kicks_per_iter = [200, 500, 1000]
iters = [1, 5, 10]

def run_trial(n, max_k, kicks, iters):
    backbone_flag = "" if iters > 1 else "--no_backbone"
    cmd = f"python src/scripts/full_scale_bench.py --n {n} --seeds 16 --kicks {kicks} --iters {iters} --max_opt {max_k} {backbone_flag} --time_limit 60"
    
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = proc.stdout
    gap_match = re.search(r"Gap:\s+([0-9.]+)%", output)
    time_match = re.search(r"Total execution time:\s+([0-9.]+)s", output)
    
    if gap_match and time_match:
        return float(gap_match.group(1)), float(time_match.group(1))
    return None, None

if __name__ == "__main__":
    results_file = "scratch/scaling_trends.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "max_k", "kicks_per_iter", "iters", "gap", "time"])
        writer.writeheader()
        configs = list(itertools.product(max_ks, kicks_per_iter, iters))
        for n in sizes:
            for max_k, kicks, iters in configs:
                print(f"Testing N={n}, K={max_k}, Kicks={kicks}, Iters={iters}...", end=" ", flush=True)
                gap, run_time = run_trial(n, max_k, kicks, iters)
                if gap is not None:
                    writer.writerow({
                        "n": n,
                        "max_k": max_k,
                        "kicks_per_iter": kicks,
                        "iters": iters,
                        "gap": gap,
                        "time": run_time
                    })
                    f.flush()
                    print(f"Gap={gap:.4f}%")
                else:
                    print("Failed.")
