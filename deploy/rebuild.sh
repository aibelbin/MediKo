#!/bin/bash
# Pull main and rebuild the MediKo container. Invoked by the deploy webhook.
set -euo pipefail

cd /home/aibel/mediko
echo "---- $(date -u +%Y-%m-%dT%H:%M:%SZ) rebuild ----"
git pull --ff-only origin main
docker compose up -d --build
echo "---- $(date -u +%Y-%m-%dT%H:%M:%SZ) done ----"
