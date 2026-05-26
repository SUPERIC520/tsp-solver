import subprocess
import re
import csv
import statistics
import itertools

# High-Resolution Grid for N=5000
max_ks = [3, 4, 5]
kicks_per_iter = [200, 500, 1000, 2000, 5000]
iters = [1, 2, 5, 10, 20]

def run_trial(max_k, kicks, iters):
    backbone_flag = "" if iters > 1 else "--no_backbone"
    # Added --time_limit to prevent excessive runtime per trial
    cmd = f"python src/scripts/full_scale_bench.py --n 5000 --seeds 16 --kicks {kicks} --iters {iters} --max_opt {max_k} {backbone_flag} --time_limit 60"
    
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = proc.stdout
    gap_match = re.search(r"Gap:\s+([0-9.]+)%", output)
    time_match = re.search(r"Total execution time:\s+([0-9.]+)s", output)
    
    if gap_match and time_match:
        return float(gap_match.group(1)), float(time_match.group(1))
    return None, None

if __name__ == "__main__":
    results = []
    configs = list(itertools.product(max_ks, kicks_per_iter, iters))
    
    # Random sample of grid to remain efficient given total combinations (75)
    # Testing 20 key configs from the grid
    sampled_configs = configs[::4] 
    
    for max_k, kicks, iters in sampled_configs:
        gaps = []
        times = []
        print(f"Testing N=5000, K={max_k}, Kicks={kicks}, Iters={iters}...", end=" ", flush=True)
        # 1 trial due to high grid resolution
        gap, run_time = run_trial(max_k, kicks, iters)
        if gap is not None:
            results.append({
                "max_k": max_k,
                "kicks_per_iter": kicks,
                "iters": iters,
                "gap": gap,
                "time": run_time
            })
            print(f"Gap={gap:.4f}%")
        else:
            print("Failed.")
            
    with open("scratch/hr_grid_n5000.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["max_k", "kicks_per_iter", "iters", "gap", "time"])
        writer.writeheader()
        writer.writerows(results)
