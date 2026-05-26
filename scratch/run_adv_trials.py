import subprocess
import re
import csv

# We compare:
# 1. Max-K = 3, 4, 5 (Fixed CS=32)
# 2. Strategy: 
#    - Single iteration, 2000 kicks
#    - Two iterations, 1000 kicks per iteration
#    - Four iterations, 500 kicks per iteration

# Test N=500
configs = [
    (3, 500, 1),
    (4, 500, 1),
    (5, 500, 1),
    (3, 2000, 1),
    (3, 1000, 2),
    (3, 500, 4),
]

def run_trial(max_k, kicks, iters):
    # main.py arguments are slightly different:
    # --kicks is kicks PER SEED, and it is a single iteration by default.
    # To test iterations, we need to ensure the script supports --iters.
    # The previous full_scale_bench.py supports --iters, main.py does not.
    # Let's use full_scale_bench.py instead, as it supports iterations.
    
    cmd = f"python src/scripts/full_scale_bench.py --n 500 --seeds 1 --kicks {kicks} --iters {iters} --max_opt {max_k} --no_backbone"
    
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    output = proc.stdout
    # Regex for Gap and Total Time
    gap_match = re.search(r"Gap:\s+([0-9.]+)%", output)
    time_match = re.search(r"Total execution time:\s+([0-9.]+)s", output)
    
    if gap_match and time_match:
        return float(gap_match.group(1)), float(time_match.group(1))
    return None, None

if __name__ == "__main__":
    results = []
    for max_k, kicks, iters in configs:
        for i in range(5): # 5 trials
            gap, run_time = run_trial(max_k, kicks, iters)
            results.append({"max_k": max_k, "kicks": kicks, "iters": iters, "gap": gap, "time": run_time})
            print(f"Trial {i} (MaxK={max_k}, Kicks={kicks}, Iters={iters}): Gap={gap}, Time={run_time}")
            
    with open("scratch/adv_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["max_k", "kicks", "iters", "gap", "time"])
        writer.writeheader()
        writer.writerows(results)
