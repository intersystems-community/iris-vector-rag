#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${1:-iris-langchain-spike}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SP_DIR="${REPO_ROOT}/iris_vector_rag/pipelines/colbert_iris/sp"

for f in VecIndex.cls UserExec.cls; do
    if [ ! -f "${SP_DIR}/${f}" ]; then
        echo "ERROR: ${SP_DIR}/${f} not found" >&2
        exit 1
    fi
done

if ! docker ps --filter "name=${CONTAINER}" --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
    echo "ERROR: Container '${CONTAINER}' is not running" >&2
    exit 1
fi

echo "Deploying VecIndex.cls and UserExec.cls to ${CONTAINER}..."

for f in VecIndex.cls UserExec.cls; do
    docker cp "${SP_DIR}/${f}" "${CONTAINER}:/tmp/${f}"
    RESULT=$(docker exec "${CONTAINER}" bash -c \
        "echo 'set sc=\$SYSTEM.OBJ.Load(\"/tmp/${f}\",\"ck\") write sc,! halt' | \
         /usr/irissys/bin/irissession IRIS -U USER 2>&1")
    if echo "${RESULT}" | grep -qE "^1$|finished successfully"; then
        echo "  ${f}: deployed OK"
    else
        echo "ERROR: Failed to deploy ${f}:" >&2
        echo "${RESULT}" >&2
        exit 1
    fi
done

# Verify VecIndex is callable
VERIFY=$(docker exec "${CONTAINER}" bash -c \
    "echo 'set r=##class(Graph.KG.VecIndex).Info(\"test\") write r,! halt' | \
     /usr/irissys/bin/irissession IRIS -U USER 2>&1")
if echo "${VERIFY}" | grep -q '"name"'; then
    echo "  Verification: Graph.KG.VecIndex callable OK"
else
    echo "WARNING: Verification call returned unexpected output: ${VERIFY}" >&2
fi

echo "VecIndex deployment complete."
