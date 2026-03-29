#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${1:-iris-langchain-spike}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SP_DIR="${REPO_ROOT}/iris_vector_rag/pipelines/colbert_iris/sp"

if [ ! -f "${SP_DIR}/PLAIDSearch.cls" ]; then
    echo "ERROR: ${SP_DIR}/PLAIDSearch.cls not found" >&2
    exit 1
fi

if ! docker ps --filter "name=${CONTAINER}" --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
    echo "ERROR: Container '${CONTAINER}' is not running" >&2
    exit 1
fi

echo "Deploying PLAIDSearch.cls to ${CONTAINER}..."

docker cp "${SP_DIR}/PLAIDSearch.cls" "${CONTAINER}:/tmp/PLAIDSearch.cls"
RESULT=$(docker exec "${CONTAINER}" bash -c \
    "echo 'set sc=\$SYSTEM.OBJ.Load(\"/tmp/PLAIDSearch.cls\",\"ck\") write sc,! halt' | \
     /usr/irissys/bin/irissession IRIS -U USER 2>&1")
if echo "${RESULT}" | grep -qE "^1$|finished successfully"; then
    echo "  PLAIDSearch.cls: deployed OK"
else
    echo "ERROR: Failed to deploy PLAIDSearch.cls:" >&2
    echo "${RESULT}" >&2
    exit 1
fi

VERIFY=$(docker exec "${CONTAINER}" bash -c \
    "echo 'set r=##class(Graph.KG.PLAIDSearch).Info(\"test\") write r,! halt' | \
     /usr/irissys/bin/irissession IRIS -U USER 2>&1")
if echo "${VERIFY}" | grep -q '"name"'; then
    echo "  Verification: Graph.KG.PLAIDSearch callable OK"
else
    echo "WARNING: Verification returned unexpected output: ${VERIFY}" >&2
fi

echo "PLAID deployment complete."
