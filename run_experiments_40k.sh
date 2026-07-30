#!/bin/bash
export PYTHONPATH=$(pwd)
source env_jax_dips/bin/activate

# Common arguments
EPOCHS=40000
DEEP_MODEL="model.mlp.hidden_layers_m=4 model.mlp.hidden_dim_m=128 model.mlp.hidden_layers_p=4 model.mlp.hidden_dim_p=128"
COMMON_FLAGS="solver.num_epochs=$EPOCHS $DEEP_MODEL"

echo "Starting Sphere Robin experiments..."
for nx in 8 16 32 64; do
  echo "Running Sphere Robin Nx=$nx"
  python tests/test_poisson.py \
    experiment.sphere_Robin=true experiment.star_Robin=false experiment.star_Robin3=false \
    solver.Nx_tr=$nx solver.Ny_tr=$nx solver.Nz_tr=$nx \
    $COMMON_FLAGS \
    > run_sphere_robin_40k_${nx}.log 2>&1 &
done
wait

echo "Starting Star Robin experiments..."
for nx in 8 16 32 64; do
  echo "Running Star Robin Nx=$nx"
  python tests/test_poisson.py \
    experiment.sphere_Robin=false experiment.star_Robin=true experiment.star_Robin3=false \
    solver.Nx_tr=$nx solver.Ny_tr=$nx solver.Nz_tr=$nx \
    $COMMON_FLAGS \
    > run_star_robin_40k_${nx}.log 2>&1 &
done
wait

echo "Starting Star Robin 3 experiments..."
for nx in 8 16 32 64; do
  echo "Running Star Robin 3 Nx=$nx"
  python tests/test_poisson.py \
    experiment.sphere_Robin=false experiment.star_Robin=false experiment.star_Robin3=true \
    solver.Nx_tr=$nx solver.Ny_tr=$nx solver.Nz_tr=$nx \
    $COMMON_FLAGS \
    > run_star_robin3_40k_${nx}.log 2>&1 &
done
wait

echo "All experiments completed!"
