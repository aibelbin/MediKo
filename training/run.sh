#!/bin/sh
# Train in Docker. Uses the NVIDIA GPU when Docker has the nvidia runtime, otherwise CPU.
# Usage:  ./run.sh [--epochs 4] [--limit 800] ...   (arguments go to src/train.py)
set -eu
cd "$(dirname "$0")"

runtimes=$(docker info --format '{{json .Runtimes}}') || {
  echo "Docker is not running." >&2
  exit 1
}

files="-f docker-compose.yml"
case "$runtimes" in
  *nvidia*)
    echo ">> NVIDIA runtime found: training on the GPU"
    files="$files -f docker-compose.gpu.yml" ;;
  *)
    echo ">> No NVIDIA runtime: training on the CPU (slow, about an hour for 3 epochs)" ;;
esac

# shellcheck disable=SC2086
docker compose $files build
# shellcheck disable=SC2086
docker compose $files run --rm train "$@"
