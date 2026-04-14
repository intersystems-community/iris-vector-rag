#!/usr/bin/env bash
set -euo pipefail
CONTAINER="${1:-kg-iris}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DIR="${REPO_ROOT}/tests/objectscript/RAG/SDK/Test"

for cls in SchemaTest PipelineTest SearchTest BridgeTest EvaluateTest PerfTest; do
    [ -f "${TEST_DIR}/${cls}.cls" ] || continue
    docker cp "${TEST_DIR}/${cls}.cls" "${CONTAINER}:/tmp/RAG_SDK_${cls}.cls"
    RESULT=$(docker exec "${CONTAINER}" bash -c \
        "echo 'set sc=\$SYSTEM.OBJ.Load(\"/tmp/RAG_SDK_${cls}.cls\",\"ck\") write sc,! halt' | \
         /usr/irissys/bin/irissession IRIS -U USER 2>&1")
    if echo "${RESULT}" | grep -qE "^1$|finished successfully"; then
        echo "  RAG.SDK.Test.${cls}: OK"
    else
        echo "ERROR: ${cls}:" >&2; echo "${RESULT}" >&2; exit 1
    fi
done
echo "RAG SDK test deployment complete."
