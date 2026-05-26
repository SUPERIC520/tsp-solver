import subprocess
import re
import csv
import statistics
import itertools

# Expanded Grid for N=5000
max_ks = [3, 4]
kicks_per_iter = [250, 500, 1000, 2000]
iters = [1, 2, 4, 8]

def run_trial(max_k, kicks, iters):
    backbone_flag = "" if iters > 1 else "--no_backbone"
    cmd = f"python src/scripts/full_scale_bench.py --n 5000 --seeds 16 --kicks {kicks} --iters {iters} --max_opt {max_k} {backbone_flag}"
    
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
    
    for max_k, kicks, iters in configs:
        gaps = []
        times = []
        print(f"Testing N=5000, K={max_k}, Kicks={kicks}, Iters={iters}...", end=" ", flush=True)
        # Using 2 trials 
        for i in range(2): 
            gap, run_time = run_trial(max_k, kicks, iters)
            if gap is not None:
                gaps.append(gap)
                times.append(run_time)
        
        if gaps:
            avg_gap = statistics.mean(gaps)
            results.append({
                "max_k": max_k,
                "kicks_per_iter": kicks,
                "iters": iters,
                "avg_gap": avg_gap,
                "avg_time": statistics.mean(times),
                "std_gap": statistics.stdev(gaps) if len(gaps) > 1 else 0
            })
            print(f"Avg Gap={avg_gap:.4f}%")
        else:
            print("Failed.")
            
    with open("scratch/expanded_grid_n5000.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["max_k", "kicks_per_iter", "iters", "avg_gap", "avg_time", "std_gap"])
        writer.writeheader()
        writer.writerows(results)
