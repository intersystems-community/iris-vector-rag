#!/usr/bin/env bash
# run-objectscript-tests.sh — compile the SDK's ObjectScript into this project's IRIS
# and run its %UnitTest classes against it.
#
# Usage: bash scripts/dev/run-objectscript-tests.sh [class ...]
#        with no argument, runs every RAG.SDK.Test.* class in the repo
#
# The container has no source mounted, and compiling on the server only recompiles
# what the server already stores, so "run the tests" means: copy the source in,
# compile it, then run. Skipping the copy is how a green run comes from a stale
# namespace copy of code that no longer exists on disk.
#
# Each class is compiled in its own session, one at a time. This container's
# compiler segfaults its worker jobs (ERROR #7802, signal 11 in messages.log) when
# asked to compile the whole directory in one call, and segfaults the session
# process itself under /multicompile=0. One class per session is the shape that
# survives, which is also what scripts/deploy_sdk.sh settled on.
#
# The container name comes from CLAUDE.md via check-container.sh, so this script
# cannot reach another project's IRIS.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="/tmp/sdk-objectscript"
SRC_DIR="$REPO_ROOT/iris_src/src/RAG/SDK"
TEST_DIR="$REPO_ROOT/tests/objectscript/RAG/SDK/Test"

if [ "$#" -gt 0 ]; then
    CLASSES=("$@")
else
    CLASSES=()
    for f in "$TEST_DIR"/*.cls; do
        [ -f "$f" ] || continue
        CLASSES+=("RAG.SDK.Test.$(basename "$f" .cls)")
    done
fi
if [ "${#CLASSES[@]}" -eq 0 ]; then
    echo "No test classes found in $TEST_DIR" >&2
    exit 1
fi

bash "$REPO_ROOT/.specify/scripts/bash/check-container.sh" || exit 1
CONTAINER="$(grep -E '^IRIS_CONTAINER:' "$REPO_ROOT/CLAUDE.md" | head -1 | sed 's/IRIS_CONTAINER:[[:space:]]*//' | tr -d '[:space:]')"

session() { # ObjectScript lines on stdin
    docker exec -i "$CONTAINER" iris session IRIS 2>&1
}

# docker cp writes as root while docker exec runs as irisowner, so the staging
# directory has to be cleaned and made readable as root or the next run cannot
# replace last run's files.
docker exec -u root "$CONTAINER" bash -lc "rm -rf $STAGE && mkdir -p $STAGE" || exit 1
docker cp "$SRC_DIR/." "$CONTAINER:$STAGE/" || exit 1
docker cp "$TEST_DIR/." "$CONTAINER:$STAGE/" || exit 1
docker exec -u root "$CONTAINER" bash -lc "chmod -R a+rX $STAGE" || exit 1

compile_one() { # <file-basename>
    local name="$1" out
    for _ in 1 2; do
        # The format string is ObjectScript, not shell: $SYSTEM must reach IRIS unexpanded.
        # shellcheck disable=SC2016
        out="$(printf 'zn "USER"\nset sc=$SYSTEM.OBJ.Load("%s/%s","ck-d")\nwrite "RC:",$SYSTEM.Status.GetErrorText(sc),!\nhalt\n' \
            "$STAGE" "$name" | session)"
        if echo "$out" | grep -q "^RC:$"; then
            return 0
        fi
        echo "  retrying $name after: $(echo "$out" | grep '^RC:' | head -1)" >&2
    done
    echo "FAILED to compile $name:" >&2
    echo "$out" >&2
    return 1
}

# Compiling a method marked [ Language = python ] does not work on this container:
# Evaluate.cls and Pipeline.cls both fail with ERROR #7802 (signal 11 in
# /usr/irissys/mgr/messages.log) and a three-line class with a single
# `return "pong"` python method hangs the load past two minutes. It is the embedded
# Python compile step itself, not any one class's imports. So any class carrying a
# python method is skipped, loudly, and the classes that need it are noted as
# unrunnable rather than reported green off a stale namespace copy.
echo "Compiling SDK classes..."
for f in "$SRC_DIR"/*.cls "$TEST_DIR"/*.cls; do
    [ -f "$f" ] || continue
    name="$(basename "$f")"
    if grep -q "Language = python" "$f"; then
        echo "  skipping $name (embedded Python does not compile on this container)"
        continue
    fi
    echo "  $name"
    compile_one "$name" || exit 1
done

SCRIPT='zn "USER"\nset ^UnitTestRoot="'"$STAGE"'"\n'
for cls in "${CLASSES[@]}"; do
    SCRIPT+='do ##class(%%UnitTest.Manager).DebugRunTestCase(,"'"$cls"'","/noload/nodelete")\n'
done
SCRIPT+='halt\n'

# shellcheck disable=SC2059  # SCRIPT is the format string by construction
OUTPUT="$(printf "$SCRIPT" | session)"
echo "$OUTPUT"

# %UnitTest reports results in its own summary rather than through an exit code, so
# the caller's exit status has to be derived from what it printed.
STATUS=0
if echo "$OUTPUT" | grep -qE "\*\*FAILED\*\*|Some tests FAILED"; then
    echo "FAILED: at least one test failed." >&2
    STATUS=1
fi
if ! echo "$OUTPUT" | grep -q "All PASSED"; then
    echo "FAILED: no passing suite reported -- did the class names match?" >&2
    STATUS=1
fi
exit $STATUS
