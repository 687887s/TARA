#!/usr/bin/env python3
"""Benchmark the kNN-based adaptive-kernel implementation on synthetic data.

Generates a synthetic companion catalog, selects TEST_N random targets,
and measures runtime for different k_nn choices. Outputs simple summary
to stdout and a JSON results file.
"""
import time
import json
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
# ensure workspace code/ is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ak_density_knn import compute_density_for_targets_knn


def make_synthetic_catalog(N=20000, boxsize=50.0, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-boxsize / 2.0, boxsize / 2.0, size=N).astype(np.float32)
    y = rng.uniform(-boxsize / 2.0, boxsize / 2.0, size=N).astype(np.float32)
    w = rng.uniform(0.1, 1.0, size=N).astype(np.float32)
    return x, y, w


def benchmark(TEST_N=50, k_list=(64, 128, 256), N=20000, h0=0.5):
    x, y, w = make_synthetic_catalog(N=N)
    rng = np.random.RandomState(123)
    targets_idx = rng.choice(np.arange(N), size=TEST_N, replace=False)
    tx = x[targets_idx]
    ty = y[targets_idx]

    results = {"N": N, "TEST_N": TEST_N, "h0": h0, "runs": []}
    for k in k_list:
        t0 = time.perf_counter()
        dens, diags = compute_density_for_targets_knn(x, y, w, tx, ty, h0=h0, k_nn=k)
        dt = time.perf_counter() - t0
        results["runs"].append({"k_nn": int(k), "time_s": float(dt), "mean_density": float(np.mean(dens)), "max_density": float(np.max(dens))})
        print(f"k_nn={k:4d}  time={dt:.3f}s  mean={np.mean(dens):.3e}  max={np.max(dens):.3e}")

    out = Path("benchmark_knn_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    benchmark()
