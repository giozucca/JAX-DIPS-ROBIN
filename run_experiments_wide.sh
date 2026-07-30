#!/bin/bash
export PYTHONPATH=$(pwd)
source env_jax_dips/bin/activate

# Run Sphere (Robin) for 15k epochs with a wider and deeper MLP
for nx in 8 16 32 64; do
  python3 tests/test_poisson.py \
    experiment.sphere_Robin=true experiment.star_Robin3=false \
    solver.num_epochs=15000 \
    solver.Nx_tr=$nx solver.Ny_tr=$nx solver.Nz_tr=$nx \
    model.mlp.hidden_layers_m=4 model.mlp.hidden_dim_m=128 \
    model.mlp.hidden_layers_p=4 model.mlp.hidden_dim_p=128 \
    > run_sphere_wide_${nx}.log 2>&1 &
done

echo "Waiting for all wide background processes to finish..."
wait
echo "All wide experiments completed!"
