#!/usr/bin/env bash
set -euo pipefail
# The default is the container this SDK was first deployed to, not this project's own
# (`iris-vector-rag-iris`, declared in CLAUDE.md) — pass the container name as $1.
CONTAINER="${1:-kg-iris}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/iris_src/src/RAG/SDK"

# Load one class and report whether it compiled. Returns non-zero on failure; the caller
# decides whether that ends the deployment.
load_class() {
    local cls="$1"
    if [ ! -f "${SRC_DIR}/${cls}.cls" ]; then
        echo "ERROR: ${SRC_DIR}/${cls}.cls not found" >&2; return 1
    fi
    docker cp "${SRC_DIR}/${cls}.cls" "${CONTAINER}:/tmp/RAG_SDK_${cls}.cls"
    LOAD_OUTPUT=$(docker exec "${CONTAINER}" bash -c \
        "echo 'set sc=\$SYSTEM.OBJ.Load(\"/tmp/RAG_SDK_${cls}.cls\",\"ck\") write sc,! halt' | \
         /usr/irissys/bin/irissession IRIS -U USER 2>&1")
    if echo "${LOAD_OUTPUT}" | grep -qE "^1$|finished successfully"; then
        echo "  RAG.SDK.${cls}: OK"
        return 0
    fi
    return 1
}

for cls in Identifier Schema Pipeline Search Bridge Evaluate; do
    if ! load_class "${cls}"; then
        echo "ERROR: Failed to load RAG.SDK.${cls}:" >&2
        echo "${LOAD_OUTPUT:-}" >&2; exit 1
    fi
done

# Embed carries an embedded-Python method, which some instances cannot compile at all: on
# iris-vector-rag-iris the load fails with ERROR #7802 and takes the compiler's worker jobs
# down. That is a reason to skip the embedding step, not to fail the deployment — every class
# above is pure ObjectScript, and RAG.SDK.Pipeline.AddDocumentWithEmbed reports the absence.
if ! load_class Embed; then
    echo "  RAG.SDK.Embed: SKIPPED (embedded Python did not compile on ${CONTAINER})" >&2
    echo "${LOAD_OUTPUT:-}" >&2
fi

echo "RAG SDK deployment complete."
