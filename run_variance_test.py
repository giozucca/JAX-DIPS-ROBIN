import subprocess
import re
import numpy as np
import os
import sys

def run_experiment(nx):
    cmd = [
        "env_jax_dips/bin/python", 
        "tests/test_poisson.py", 
        f"solver.Nx_tr={nx}", 
        f"solver.Ny_tr={nx}", 
        f"solver.Nz_tr={nx}", 
        "experiment.sphere_Robin=true", 
        "experiment.sphere=false"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    # Extract RMSE_m from the logs
    match = re.search(r"RMSE_m:\s*([0-9\.]+)", result.stderr)
    if not match:
        match = re.search(r"RMSE_m:\s*([0-9\.]+)", result.stdout)
    
    if match:
        return float(match.group(1))
    else:
        print("Failed to find RMSE_m in output")
        # Try to find normal RMSE just in case
        match = re.search(r"RMSE:\s*([0-9\.]+)", result.stderr)
        if match:
            return float(match.group(1))
        return None

if __name__ == "__main__":
    for nx in [8, 16]:
        rmses = []
        print(f"Running experiments for Nx={nx}")
        for i in range(5):
            rmse = run_experiment(nx)
            if rmse is not None:
                rmses.append(rmse)
                print(f"  Run {i+1}: RMSE_m = {rmse}")
            else:
                sys.exit(1)
        avg = np.mean(rmses)
        std = np.std(rmses)
        print(f"Nx={nx} -> Avg RMSE_m: {avg:.6f}, Std: {std:.6f}")
