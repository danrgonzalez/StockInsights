#!/bin/bash
# Setup script to ensure the correct Python environment.
#
# Usage:  source setup_env.sh
#
# Locates conda rather than assuming an install path. Set CONDA_ROOT to point
# at a specific installation if you have more than one.

ENV_NAME="${STOCKINSIGHTS_ENV:-stockinsights}"

find_conda_root() {
    if [ -n "$CONDA_ROOT" ]; then
        echo "$CONDA_ROOT"
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        conda info --base
        return
    fi
    for candidate in \
        "$HOME/miniconda3" \
        "$HOME/anaconda3" \
        "$HOME/opt/miniconda3" \
        "$HOME/opt/anaconda3" \
        "/opt/homebrew/Caskroom/miniconda/base" \
        "/opt/conda"; do
        if [ -x "$candidate/bin/conda" ]; then
            echo "$candidate"
            return
        fi
    done
}

CONDA_BASE="$(find_conda_root)"

if [ -z "$CONDA_BASE" ] || [ ! -x "$CONDA_BASE/bin/conda" ]; then
    echo "Could not find a conda installation." >&2
    echo "Set CONDA_ROOT to your conda base directory and re-run." >&2
    return 1 2>/dev/null || exit 1
fi

eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"

if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo "conda environment '$ENV_NAME' not found in $CONDA_BASE." >&2
    echo "Create it with: conda create -n $ENV_NAME python=3.11" >&2
    echo "then: pip install -r requirements.txt -r requirements-dev.txt" >&2
    return 1 2>/dev/null || exit 1
fi

echo "Environment setup complete!"
echo "Python version: $(python --version 2>&1)"
echo "Python location: $(which python)"
echo "Active conda environment: $CONDA_DEFAULT_ENV"
