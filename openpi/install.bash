#!/usr/bin/env bash
set -euo pipefail

conda env create -f environment-openpi.yaml
# there is a conflict between lerobot and openpi, however we only need the lerobot-dataset,
# so `--no-deps` is used.
conda run --no-capture-output -n openpi pip install lerobot --no-deps
conda run --no-capture-output -n openpi pip install -e .
conda run --no-capture-output -n openpi pip install -e ./packages/openpi-client
