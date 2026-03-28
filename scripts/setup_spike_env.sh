#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${1:-iris-langchain-spike}"
NUMPY_VERSION="1.26.4"
TARGET_PATH="/usr/irissys/mgr/python"

if ! docker ps --filter "name=${CONTAINER}" --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
  echo "ERROR: Container '${CONTAINER}' is not running" >&2
  exit 1
fi

if docker exec "${CONTAINER}" python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null | grep -q "^${NUMPY_VERSION}"; then
  echo "numpy==${NUMPY_VERSION} already installed in ${CONTAINER}"
  exit 0
fi

echo "Installing numpy==${NUMPY_VERSION} into ${CONTAINER}:${TARGET_PATH} ..."
docker exec "${CONTAINER}" pip3 install --target "${TARGET_PATH}" "numpy==${NUMPY_VERSION}" --quiet

if docker exec "${CONTAINER}" python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null | grep -q "^${NUMPY_VERSION}"; then
  echo "numpy==${NUMPY_VERSION} installed successfully"
else
  echo "ERROR: numpy install verification failed" >&2
  exit 1
fi
