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

echo "Installing exact package versions for compatibility..."
# Extract the requirements from requirements.txt, skipping the first two lines 
# which contain broken/hardcoded JAX wheel links
tail -n +3 requirements.txt > linux_requirements_temp.txt

# Force pip to resolve everything at once by appending JAX to the requirements list.
# This prevents unpinned packages (like equinox/diffrax) from silently upgrading JAX
# and breaking CUDA support.
echo "jax==0.4.13" >> linux_requirements_temp.txt
echo "https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.13+cuda12.cudnn89-cp310-cp310-manylinux2014_x86_64.whl" >> linux_requirements_temp.txt

# Install all dependencies together
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
