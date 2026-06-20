#!/bin/bash
# Setup script to ensure correct Python environment

# Initialize conda (native Apple silicon Miniconda)
eval "$(/Users/dgonzalez/miniconda3/bin/conda shell.zsh hook)"

# Activate stockinsights environment
conda activate stockinsights

# Export environment variables
export CONDA_DEFAULT_ENV=stockinsights
export PATH="/Users/dgonzalez/miniconda3/envs/stockinsights/bin:$PATH"

echo "Environment setup complete!"
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
echo "Active conda environment: $CONDA_DEFAULT_ENV"
