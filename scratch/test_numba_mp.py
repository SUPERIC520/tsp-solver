
import numpy as np
from numba import njit, prange
import multiprocessing as mp
import time

@njit(parallel=False)
def test_parallel(n):
    res = np.zeros(n)
    for i in prange(n):
        res[i] = np.sqrt(i)
    return res

def worker(q):
    print(f"Worker starting", flush=True)
    res = test_parallel(1000)
    q.put(res.sum())
    print(f"Worker finished", flush=True)

if __name__ == "__main__":
    print("Starting test...", flush=True)
    q = mp.Queue()
    p = mp.Process(target=worker, args=(q,))
    p.start()
    print("Process started, waiting...", flush=True)
    try:
        val = q.get(timeout=10)
        print(f"Result: {val}", flush=True)
    except Exception as e:
        print(f"Error or timeout: {e}", flush=True)
    p.join()
    print("Done.", flush=True)
