#!/usr/bin/env bash
set -euo pipefail
CONTAINER="${1:-kg-iris}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/iris_src/src/RAG/SDK"

for cls in Schema Pipeline Search Bridge Evaluate; do
    if [ ! -f "${SRC_DIR}/${cls}.cls" ]; then
        echo "ERROR: ${SRC_DIR}/${cls}.cls not found" >&2; exit 1
    fi
    docker cp "${SRC_DIR}/${cls}.cls" "${CONTAINER}:/tmp/RAG_SDK_${cls}.cls"
    RESULT=$(docker exec "${CONTAINER}" bash -c \
        "echo 'set sc=\$SYSTEM.OBJ.Load(\"/tmp/RAG_SDK_${cls}.cls\",\"ck\") write sc,! halt' | \
         /usr/irissys/bin/irissession IRIS -U USER 2>&1")
    if echo "${RESULT}" | grep -qE "^1$|finished successfully"; then
        echo "  RAG.SDK.${cls}: OK"
    else
        echo "ERROR: Failed to load RAG.SDK.${cls}:" >&2
        echo "${RESULT}" >&2; exit 1
    fi
done
echo "RAG SDK deployment complete."
