#!/bin/bash
set -e

ENV_NAME="jax_dips_linux_env"

echo "============================================================"
echo " Setting up JAX-DIPS Linux GPU Environment"
echo "============================================================"

# Ensure Python 3.10 is available
PYTHON_CMD="python3"
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
fi

echo "Using Python command: $PYTHON_CMD"
$PYTHON_CMD --version

if [ -d "$ENV_NAME" ]; then
    echo "Environment $ENV_NAME already exists. Removing it to start fresh..."
    rm -rf "$ENV_NAME"
fi

echo "Creating virtual environment..."
$PYTHON_CMD -m venv $ENV_NAME

echo "Activating virtual environment..."
source $ENV_NAME/bin/activate

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing JAX 0.4.13 with CUDA 12 support..."
# The original repo targets JAX 0.4.13. 
# Depending on the python version (assuming 3.10 here since it's standard for 2023 environments), 
# we use the specific CUDA 12 jaxlib release index.
pip install "jax[cuda12_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "Installing exact package versions for compatibility..."
# Extract the requirements from requirements.txt, skipping the first two lines 
# which contain broken/hardcoded JAX wheel links
tail -n +3 requirements.txt > linux_requirements_temp.txt

# Install the rest of the dependencies
pip install -r linux_requirements_temp.txt

# PyEVTK is required in setup.py but not listed in requirements.txt directly (or cloned in Dockerfile)
echo "Installing pyevtk..."
pip install pyevtk

echo "Installing the JAX-DIPS package in editable mode without re-installing dependencies..."
pip install --no-deps -e .

rm linux_requirements_temp.txt

echo "============================================================"
echo " Setup complete! "
echo " Run 'source $ENV_NAME/bin/activate' to activate the environment."
echo "============================================================"
