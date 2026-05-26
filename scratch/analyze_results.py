
import csv
import numpy as np
import scipy.stats as stats
from collections import defaultdict

# Load the data
data_by_config = defaultdict(list)
runtimes_by_config = defaultdict(list)
config_names = {}

with open('scratch/seeding_strategy.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        c_id = int(row['config_id'])
        length = float(row['length'])
        # STRICT FILTER: Only include valid N=5,000 trials (exclude N=100 dry run)
        if length > 200000:
            data_by_config[c_id].append(length)
            runtimes_by_config[c_id].append(float(row['runtime']))
            config_names[c_id] = row['config_name']

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

print("### Initial Seeding Comparison (N=5,000, max_opt=5, kicks=500, iters=1)")
print("| ID | Configuration                           | Trials | Avg Length | Avg Gap (%) | Std Dev | Min       | Max       | Avg Time (s) |")
print("|----|-----------------------------------------|--------|------------|-------------|---------|-----------|-----------|--------------|")

group_a = []
for c_id in range(1, 9):
    name = config_names.get(c_id, f"Config {c_id}")
    vals = data_by_config[c_id]
    group_a.append(vals)
    s = get_stats(vals)
    t = get_stats(runtimes_by_config[c_id])
    gap = (s['mean'] - lb_val) / lb_val * 100
    print(f"| {c_id:<2} | {name:<39} | {s['count']:<6} | {s['mean']:<10.2f} | {gap:<10.4f}% | {s['std']:<7.2f} | {s['min']:<9.2f} | {s['max']:<9.2f} | {t['mean']:<12.1f} |")

print("\n### Re-seeding Strategy Comparison (N=5,000, max_opt=5, kicks=100, iters=5)")
print("| ID | Configuration                           | Trials | Avg Length | Avg Gap (%) | Std Dev | Min       | Max       | Avg Time (s) |")
print("|----|-----------------------------------------|--------|------------|-------------|---------|-----------|-----------|--------------|")

group_b = []
for c_id in range(9, 17):
    name = config_names.get(c_id, f"Config {c_id}")
    vals = data_by_config[c_id]
    group_b.append(vals)
    s = get_stats(vals)
    t = get_stats(runtimes_by_config[c_id])
    gap = (s['mean'] - lb_val) / lb_val * 100
    print(f"| {c_id:<2} | {name:<39} | {s['count']:<6} | {s['mean']:<10.2f} | {gap:<10.4f}% | {s['std']:<7.2f} | {s['min']:<9.2f} | {s['max']:<9.2f} | {t['mean']:<12.1f} |")

# ANOVA
f_a, p_a = stats.f_oneway(*group_a)
f_b, p_b = stats.f_oneway(*group_b)

print(f"\nANOVA Initial (Group A): F={f_a:.4f}, p={p_a:.4e}")
print(f"ANOVA Re-seed (Group B): F={f_b:.4f}, p={p_b:.4e}")
