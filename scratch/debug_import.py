
import sys
import time

def test_import(module_name):
    print(f"Testing import of {module_name}...", flush=True)
    t0 = time.time()
    try:
        __import__(module_name)
        print(f"  Imported {module_name} in {time.time() - t0:.2f}s", flush=True)
    except Exception as e:
        print(f"  Failed to import {module_name}: {e}", flush=True)

modules = [
    "src.utils.data_io",
    "src.core.preprocessing",
    "src.core.seed_generation",
    "src.core.orchestration",
]

for m in modules:
    test_import(m)
