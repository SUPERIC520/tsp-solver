
import csv
import numpy as np
import scipy.stats as stats
from collections import defaultdict

# Load the data from the LONG RUN experiment
data_by_config = defaultdict(list)
config_names = {
    1: "LR: 100% Hilbert",
    2: "LR: 100% Random",
    3: "LR: 100% Greedy NN"
}

CSV_PATH = 'scratch/long_run_strategy.csv'

with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        c_id = int(row['config_id'])
        if c_id in config_names:
            length = float(row['length'])
            if length > 200000:
                data_by_config[c_id].append(length)

# ANOVA
groups = [data_by_config[i] for i in [1, 2, 3] if len(data_by_config[i]) >= 2]
if len(groups) == 3:
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"ANOVA p-value: {p_val:.4e}")
    print(f"ANOVA F-statistic: {f_stat:.4f}")
    
    # Tukey HSD
    res = stats.tukey_hsd(*groups)
    print("\nTukey HSD p-values:")
    print(res.pvalue)
else:
    print("Not enough data for ANOVA.")

# Raw Data Table
max_trials = max(len(v) for v in data_by_config.values())
header = "| ID | " + " | ".join([f"T{i+1}" for i in range(max_trials)]) + " |"
print("\n" + header)
print("|" + "---| " * (max_trials + 1))

for c_id in [1, 2, 3]:
    vals = data_by_config[c_id]
    row = f"| {c_id} | " + " | ".join([f"{v:.2f}" for v in vals])
    needed = max_trials - len(vals)
    if needed > 0:
        row += " | " * needed
    print(row + " |")
