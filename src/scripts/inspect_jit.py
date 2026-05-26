import numpy as np
import os
# Disable caching for inspection
os.environ['NUMBA_CACHE_DIR'] = ''

from src.core.kopt_engine import (
    _optimize_2opt,
    compute_tour_length,
    _dist
)

def inspect_jit():
    print("--- Numba JIT Inspection ---")
    
    # Dummy data for signature triggering
    n = 100
    coords_x = np.random.rand(n).astype(np.float64)
    coords_y = np.random.rand(n).astype(np.float64)
    tour = np.arange(n).astype(np.int32)
    candidate_set = np.random.randint(0, n, (n, 16)).astype(np.int32)
    candidate_dists = np.random.rand(n, 16).astype(np.float64)
    locked_edges = np.full((n, 2), -1, dtype=np.int32)
    pos = np.arange(n).astype(np.int32)
    dlb = np.zeros(n, dtype=np.bool_)
    
    # Trigger JIT
    compute_tour_length(tour, coords_x, coords_y)
    _optimize_2opt(tour, coords_x, coords_y, candidate_set, candidate_dists, locked_edges, pos, dlb)
    
    def analyze_asm(func, name):
        print(f"\n[{name}] Assembly Analysis:")
        asm = func.inspect_asm()
        sig = list(asm.keys())[0]
        asm_text = asm[sig]
        
        # Look for packed instructions (SIMD)
        simd_packed = ["addpd", "subpd", "mulpd", "divpd", "sqrtpd", "maxpd", "minpd", "vaddpd", "vsubpd", "vmulpd", "vsqrtpd"]
        simd_scalar = ["addsd", "subsd", "mulsd", "divsd", "sqrtsd", "vaddsd", "vsubsd", "vmulsd", "vsqrtsd"]
        
        found_packed = [instr for instr in simd_packed if instr in asm_text.lower()]
        found_scalar = [instr for instr in simd_scalar if instr in asm_text.lower()]
        
        print(f"  - Signature: {sig}")
        if found_packed:
            print(f"  - SIMD PACKED (Vector) found: {set(found_packed)}")
        if found_scalar:
            print(f"  - SIMD SCALAR (Non-vector) found: {set(found_scalar)}")
            
        if not found_packed and not found_scalar:
            print("  - No obvious floating point math found in assembly.")
        
        # Check for unrolling hints
        if "unroll" in asm_text.lower():
            print("  - Potential loop unrolling detected.")

    analyze_asm(compute_tour_length, "compute_tour_length")
    analyze_asm(_optimize_2opt, "_optimize_2opt")
    analyze_asm(_dist, "_dist")

if __name__ == "__main__":
    inspect_jit()

if __name__ == "__main__":
    inspect_jit()
