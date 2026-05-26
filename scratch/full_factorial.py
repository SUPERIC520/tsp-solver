import subprocess
import re
import csv
import statistics
import itertools

# Full Factorial Grid
max_ks = [3, 4, 5]
kicks_per_iter = [500, 1000, 2000]
iters = [1, 2, 4]

def run_trial(max_k, kicks, iters):
    cmd = f"python src/scripts/full_scale_bench.py --n 500 --seeds 1 --kicks {kicks} --iters {iters} --max_opt {max_k} --no_backbone"
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
        print(f"Testing K={max_k}, Kicks={kicks}, Iters={iters}...", end=" ", flush=True)
        for i in range(10):
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
            
    with open("scratch/full_factorial_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["max_k", "kicks_per_iter", "iters", "avg_gap", "avg_time", "std_gap"])
        writer.writeheader()
        writer.writerows(results)
