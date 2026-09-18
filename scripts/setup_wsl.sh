#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_EXE="${CONDA_EXE:-$HOME/anaconda3/bin/conda}"
if ! "$CONDA_EXE" env list --json | grep -q '/predict-search-ca"'; then
  "$CONDA_EXE" env create -f environment-wsl.yml
fi
"$CONDA_EXE" run -n predict-search-ca python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$CONDA_EXE" run -n predict-search-ca python -m pip install torch-geometric==2.6.1 gurobipy==13.0.3 pytest==8.3.5
"$CONDA_EXE" run -n predict-search-ca python -m pip check
echo 'Ready: source ~/anaconda3/etc/profile.d/conda.sh && conda activate predict-search-ca'
