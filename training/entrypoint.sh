#!/bin/sh
# download dataset + model (skipped when already cached) -> train -> plot curves.
# Extra arguments go to src/train.py, e.g. --epochs 1 --limit 200
set -eu
mkdir -p /app/outputs
python src/download_data.py
python src/train.py "$@" 2>&1 | tee /app/outputs/train.log
python src/plot_training.py
