
import csv
import numpy as np
import scipy.stats as stats
from collections import defaultdict

# Load the data from the LONG RUN experiment
data_by_config = defaultdict(list)
runtimes_by_config = defaultdict(list)
config_names = {
    1: "LR: 100% Hilbert",
    2: "LR: 100% Random",
    3: "LR: 100% Greedy NN",
    4: "LR: 50/50 Hil/Rand",
    5: "LR: 50/50 Hil/Greedy",
    6: "LR: 50/50 Rand/Greedy",
    7: "LR: 75/25 Hil/Greedy",
    8: "LR: Balanced Mix"
}

CSV_PATH = 'scratch/long_run_strategy.csv'

with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        c_id = int(row['config_id'])
        length = float(row['length'])
        # Filter for N=5,000 runs
        if length > 200000:
            data_by_config[c_id].append(length)
            runtimes_by_config[c_id].append(float(row['runtime']))

lb_val = 251684.88

def get_stats(vals):
    if not vals: return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
    return {
        'mean': np.mean(vals),
        'std': np.std(vals, ddof=1) if len(vals) > 1 else 0,
        'min': np.min(vals),
        'max': np.max(vals),
        'count': len(vals)
    }

print("### Preliminary Long-Run Seeding Comparison (N=5,000, max_opt=5, kicks=100, iters=10)")
print("| ID | Configuration                           | Trials | Avg Length | Avg Gap (%) | Std Dev | Min       | Max       | Avg Time (s) |")
print("|----|-----------------------------------------|--------|------------|-------------|---------|-----------|-----------|--------------|")

for c_id in range(1, 9):
    name = config_names[c_id]
    s = get_stats(data_by_config[c_id])
    t = get_stats(runtimes_by_config[c_id])
    
    if s['count'] > 0:
        gap = (s['mean'] - lb_val) / lb_val * 100
        print(f"| {c_id:<2} | {name:<39} | {s['count']:<6} | {s['mean']:<10.2f} | {gap:<10.4f}% | {s['std']:<7.2f} | {s['min']:<9.2f} | {s['max']:<9.2f} | {t['mean']:<12.1f} |")
    else:
        print(f"| {c_id:<2} | {name:<39} | 0      | N/A        | N/A         | N/A     | N/A       | N/A       | N/A          |")
