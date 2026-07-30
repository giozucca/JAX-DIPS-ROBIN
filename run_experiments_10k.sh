#!/bin/bash
export PYTHONPATH=$(pwd)
source env_jax_dips/bin/activate

# Run Sphere (Robin) for 10k epochs
for nx in 8 16 32 64; do
  python3 tests/test_poisson.py experiment.sphere_Robin=true experiment.star_Robin3=false solver.num_epochs=10000 solver.Nx_tr=$nx solver.Ny_tr=$nx solver.Nz_tr=$nx > run_sphere_10k_${nx}.log 2>&1 &
done

echo "Waiting for all 10k background processes to finish..."
wait
echo "All 10k experiments completed!"
