#!/bin/bash
export PYTHONPATH=$(pwd)
source env_jax_dips/bin/activate

# Run Sphere (Robin)
for nx in 8 16 32 64; do
  python tests/test_poisson.py experiment.sphere_Robin=true experiment.star_Robin3=false solver.num_epochs=5000 solver.Nx_tr=$nx solver.Ny_tr=$nx solver.Nz_tr=$nx > run_sphere_${nx}.log 2>&1 &
done

# Run Double Star (Robin3)
for nx in 8 16 32 64; do
  python tests/test_poisson.py experiment.sphere_Robin=false experiment.star_Robin3=true solver.num_epochs=5000 solver.Nx_tr=$nx solver.Ny_tr=$nx solver.Nz_tr=$nx > run_double_star_${nx}.log 2>&1 &
done

echo "Waiting for all background processes to finish..."
wait
echo "All experiments completed!"
