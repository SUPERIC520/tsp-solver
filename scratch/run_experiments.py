import subprocess
import re
import sys
import time

commands = [
    # max_opt comparison
    "--n 10000 --seeds 16 --kicks 2000 --iters 1 --max_opt 3",
    "--n 10000 --seeds 16 --kicks 2000 --iters 1 --max_opt 4",
    "--n 10000 --seeds 16 --kicks 2000 --iters 1 --max_opt 5",
    
    # kicks comparison
    "--n 10000 --seeds 16 --kicks 5000 --iters 1 --max_opt 3",
    "--n 10000 --seeds 16 --kicks 10000 --iters 1 --max_opt 3",
    
    # iterations comparison (N=10000 iterations=2 requires backbone so no --no_backbone)
    "--n 10000 --seeds 16 --kicks 2000 --iters 2 --max_opt 3",
]

def run():
    results = []
    for params in commands:
        cmd = f'python src/scripts/full_scale_bench.py {params}'
        print(f"Running: {cmd}")
        try:
            # We want to use the current virtual environment or set PYTHONPATH
            env = {"PYTHONPATH": "."}
            # Copy other env vars if necessary or just run directly
            proc = subprocess.run(cmd, shell=True, env=None, capture_output=True, text=True)
            output = proc.stdout
            
            # Parse output
            # HK Lower Bound: 615978.91
            # Final Best Length: 647985.36
            # Gap:               5.1960%
            # Total execution time: 36.87s
            
            hk_match = re.search(r"HK Lower Bound:\s+([0-9.]+)", output)
            best_match = re.search(r"Final Best Length:\s+([0-9.]+)", output)
            gap_match = re.search(r"Gap:\s+([0-9.]+%?)", output)
            time_match = re.search(r"Total execution time:\s+([0-9.]+s)", output)
            
            if not (hk_match and best_match and gap_match and time_match):
                # fallback parsing
                hk = "N/A"
                best = "N/A"
                gap = "N/A"
                run_time = "N/A"
                print(f"Error parsing output for {params}:\n{output[:500]}")
            else:
                hk = hk_match.group(1)
                best = best_match.group(1)
                gap = gap_match.group(1)
                run_time = time_match.group(1)
            
            print(f"Result -> HK: {hk}, Best: {best}, Gap: {gap}, Time: {run_time}")
            results.append({
                "params": params,
                "hk": hk,
                "best": best,
                "gap": gap,
                "time": run_time,
                "status": "SUCCESS" if float(gap.strip('%')) < 5.0 else "FAILURE" if gap != "N/A" else "ERROR"
            })
            
        except Exception as e:
            print(f"Failed {params}: {e}")
            
    # Write to a file
    with open("scratch/results.txt", "w") as f:
        for r in results:
            f.write(f"- **Params**: {r['params']}\n")
            f.write(f"- **Results**: Gap={r['gap']}, LB={r['hk']}, Best={r['best']}, Time={r['time']}\n")
            f.write(f"- **Status**: {r['status']}\n\n")

if __name__ == '__main__':
    run()