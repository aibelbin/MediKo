#!/usr/bin/env bash
# Copy the working tree to Fox over SSH and build the image on Fox.
set -euo pipefail

: "${FOX_HOST:?Set the FOX_HOST secret to the Fox hostname or IP.}"
: "${FOX_USER:?Set the FOX_USER secret to the SSH user on Fox.}"
: "${FOX_SSH_KEY:?Set the FOX_SSH_KEY secret to the private key for that user.}"

PORT="${FOX_PORT:-22}"
DEST="${FOX_DEPLOY_PATH:-mediko}"
DEST="${DEST%/}"
KEY_PATH="${HOME}/.ssh/fox_deploy_key"

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"
printf '%s\n' "${FOX_SSH_KEY}" | tr -d '\r' > "${KEY_PATH}"
chmod 600 "${KEY_PATH}"

SSH_OPTS=(-i "${KEY_PATH}" -p "${PORT}" -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=120)
if [[ -n "${FOX_KNOWN_HOSTS:-}" ]]; then
  printf '%s\n' "${FOX_KNOWN_HOSTS}" > "${HOME}/.ssh/known_hosts"
  chmod 644 "${HOME}/.ssh/known_hosts"
  SSH_OPTS+=(-o StrictHostKeyChecking=yes)
else
  SSH_OPTS+=(-o StrictHostKeyChecking=accept-new)
fi

TARGET="${FOX_USER}@${FOX_HOST}"
REMOTE_DIR="$(printf '%q' "${DEST}")"

ssh "${SSH_OPTS[@]}" "${TARGET}" "mkdir -p ${REMOTE_DIR}"

RSYNC_RSH="ssh -i ${KEY_PATH} -p ${PORT} -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=120"
if [[ -n "${FOX_KNOWN_HOSTS:-}" ]]; then
  RSYNC_RSH="${RSYNC_RSH} -o StrictHostKeyChecking=yes"
else
  RSYNC_RSH="${RSYNC_RSH} -o StrictHostKeyChecking=accept-new"
fi

rsync -az --delete \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  --exclude .mypy_cache \
  --exclude data \
  --exclude output \
  --exclude .env \
  -e "${RSYNC_RSH}" \
  ./ "${TARGET}:${DEST}/"

ssh "${SSH_OPTS[@]}" "${TARGET}" "set -euo pipefail; cd ${REMOTE_DIR}; docker compose up -d --build"
