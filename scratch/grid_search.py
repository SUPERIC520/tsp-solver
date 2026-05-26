import subprocess
import re
import csv
import statistics

# Configurations: (Max-K, Kicks_per_iter, Iters)
# Target total kicks is ~2000 per seed
configs = [
    # Max-K study (Fixed 2000 total kicks)
    (3, 2000, 1),
    (4, 2000, 1),
    (5, 2000, 1),
    
    # Strategy study (Max-K=3, total kicks = 2000)
    (3, 2000, 1),
    (3, 1000, 2),
    (3, 500, 4),
]

def run_trial(max_k, kicks, iters):
    # Running full_scale_bench.py
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
    for max_k, kicks, iters in configs:
        gaps = []
        times = []
        for i in range(10):
            gap, run_time = run_trial(max_k, kicks, iters)
            if gap is not None:
                gaps.append(gap)
                times.append(run_time)
        
        if gaps:
            results.append({
                "max_k": max_k,
                "kicks": kicks,
                "iters": iters,
                "avg_gap": statistics.mean(gaps),
                "avg_time": statistics.mean(times),
                "std_gap": statistics.stdev(gaps) if len(gaps) > 1 else 0
            })
            print(f"Config (K={max_k}, Kicks={kicks}, Iters={iters}): Avg Gap={statistics.mean(gaps):.4f}%")
            
    with open("scratch/grid_search_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["max_k", "kicks", "iters", "avg_gap", "avg_time", "std_gap"])
        writer.writeheader()
        writer.writerows(results)
