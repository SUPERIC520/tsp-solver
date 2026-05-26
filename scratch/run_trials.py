import subprocess
import re
import csv
import numpy as np

# Configurations to test
# Mode is implicit in standard or cascading, here we just test max_opt and CS (via hardcoded candidate_set filtering in main.py)
# Actually, the grid search would require modifying src/core/kopt_engine.py to switch between standard/cascading.
# Since we are using the production main.py, it uses standard.

configs = [
    # (Max-K, CS)
    (3, 16),
    (4, 16),
    (3, 32),
    (4, 32),
]

def run_trial(max_k, cs, trial_id):
    # This assumes we have a way to set CS in main.py, but it currently hardcodes it to 40.
    # To test CS, we must patch main.py.
    # Let's perform the testing by patching the candidate set size in main.py dynamically.
    
    cmd = f"python src/scripts/main.py --n 500 --seeds 1 --kicks 500 --max_opt {max_k} --no_cache"
    
    # We will just run the trials as is for now and focus on max_opt first,
    # as changing CS requires changing the core or main.py.
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    output = proc.stdout
    gap_match = re.search(r"Solution Gap:\s+([0-9.]+)%", output)
    time_match = re.search(r"Optimization completed in\s+([0-9.]+)s", output)
    
    return float(gap_match.group(1)), float(time_match.group(1))

if __name__ == "__main__":
    results = []
    for max_k, cs in configs:
        for i in range(10):
            gap, run_time = run_trial(max_k, cs, i)
            results.append({"max_k": max_k, "cs": cs, "gap": gap, "time": run_time})
            print(f"Trial {i} (MaxK={max_k}, CS={cs}): Gap={gap}, Time={run_time}")
            
    with open("scratch/trial_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["max_k", "cs", "gap", "time"])
        writer.writeheader()
        writer.writerows(results)
