#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${1:-iris-langchain-spike}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLS_FILE="${REPO_ROOT}/iris_vector_rag/pipelines/colbert_iris/sp/ColBERTSearch.cls"
CONTAINER_PATH="/tmp/ColBERTSearch.cls"

if [ ! -f "${CLS_FILE}" ]; then
  echo "ERROR: .cls file not found at ${CLS_FILE}" >&2
  exit 1
fi

if ! docker ps --filter "name=${CONTAINER}" --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
  echo "ERROR: Container '${CONTAINER}' is not running" >&2
  exit 1
fi

echo "Deploying ColBERTSearch.cls to ${CONTAINER} ..."
docker cp "${CLS_FILE}" "${CONTAINER}:${CONTAINER_PATH}"

RESULT=$(docker exec "${CONTAINER}" bash -c \
  "echo 'set sc=\$SYSTEM.OBJ.Load(\"${CONTAINER_PATH}\",\"ck\") write sc,! halt' | \
   /usr/irissys/bin/irissession IRIS -U USER 2>&1"
)

if echo "${RESULT}" | grep -q "Load finished successfully"; then
  echo "RAG.ColBERTSearch deployed successfully"
elif echo "${RESULT}" | grep -q "^1$"; then
  echo "RAG.ColBERTSearch deployed successfully"
else
  echo "ERROR: Deployment may have failed. Output:" >&2
  echo "${RESULT}" >&2
  exit 1
fi
