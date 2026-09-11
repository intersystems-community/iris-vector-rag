#!/usr/bin/env bash
# check-container.sh — IRIS container pre-flight for /speckit.implement and doctor.
#
# Reads the machine-readable declaration from CLAUDE.md in the repo root:
#
#   IRIS_CONTAINER: iris-vector-rag-iris
#   IRIS_PORT: 51972
#
# Exits 0 when the declared container is running (and, if a port is declared,
# publishes it). Exits 1 with a BLOCKED message otherwise.
#
# A project that declares NOTHING is blocked, not skipped: a pre-flight that
# passes on an absent declaration is a pre-flight nobody notices is absent, which
# is exactly how this repo ran a whole feature with no container check at all. A
# project that genuinely needs no container declares `IRIS_CONTAINER: none`.
#
# The port check exists because the container-crossing hazard is a right name on
# the wrong port: every project here owns one container on one port, and reaching
# another project's IRIS corrupts its data silently.
#
# Usage: bash check-container.sh [--json]

set -uo pipefail

REPO_ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CLAUDE_MD="$REPO_ROOT/CLAUDE.md"
JSON_MODE=false
[[ "${1:-}" == "--json" ]] && JSON_MODE=true

declaration() { # <key>
    [[ -f "$CLAUDE_MD" ]] || return 0
    grep -E "^$1:" "$CLAUDE_MD" 2>/dev/null \
        | head -1 \
        | sed "s/$1:[[:space:]]*//" \
        | tr -d '[:space:]'
}

blocked() { # <json-payload> <heading> <line>...
    if $JSON_MODE; then
        echo "$1"
        exit 1
    fi
    shift
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    for line in "$@"; do
        echo "║  $line"
    done
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    exit 1
}

IRIS_CONTAINER="$(declaration IRIS_CONTAINER)"
IRIS_PORT="$(declaration IRIS_PORT)"

# --- no declaration: the check has nothing to enforce, so say so loudly ------

if [[ -z "$IRIS_CONTAINER" ]]; then
    blocked \
        '{"status":"blocked","reason":"no IRIS_CONTAINER declared in CLAUDE.md"}' \
        "BLOCKED: no IRIS_CONTAINER declared in CLAUDE.md." \
        "" \
        "This project owns exactly one IRIS container. Declare it so every" \
        "script and agent reads the same one, on its own line in CLAUDE.md:" \
        "" \
        "    IRIS_CONTAINER: <name>" \
        "    IRIS_PORT: <port>" \
        "" \
        "A project that needs no container declares: IRIS_CONTAINER: none"
fi

if [[ "$IRIS_CONTAINER" == "none" ]]; then
    if $JSON_MODE; then
        echo '{"status":"skipped","reason":"IRIS_CONTAINER: none"}'
    else
        echo "✓ No IRIS container required for this project (IRIS_CONTAINER: none)."
    fi
    exit 0
fi

# --- the container must be running under exactly that name ------------------

START_HINT="docker start $IRIS_CONTAINER"
if [[ -f "$REPO_ROOT/docker-compose.yml" ]]; then
    START_HINT="docker start $IRIS_CONTAINER   # or: docker compose up -d"
fi

if ! docker ps --format "{{.Names}}" 2>/dev/null | grep -qx "$IRIS_CONTAINER"; then
    blocked \
        "{\"status\":\"blocked\",\"container\":\"$IRIS_CONTAINER\",\"hint\":\"$START_HINT\"}" \
        "BLOCKED: IRIS container '$IRIS_CONTAINER' is not running." \
        "" \
        "Integration and E2E tests cannot be validated without it, and no" \
        "other container is a substitute -- each one owns its own data." \
        "" \
        "    $START_HINT"
fi

# --- and publish the declared port, if one is declared ----------------------

if [[ -n "$IRIS_PORT" ]]; then
    PUBLISHED="$(docker port "$IRIS_CONTAINER" 2>/dev/null)"
    if ! printf '%s\n' "$PUBLISHED" | grep -qE ":$IRIS_PORT([[:space:]]|$)"; then
        blocked \
            "{\"status\":\"blocked\",\"container\":\"$IRIS_CONTAINER\",\"port\":\"$IRIS_PORT\",\"hint\":\"$START_HINT\"}" \
            "BLOCKED: '$IRIS_CONTAINER' does not publish port $IRIS_PORT." \
            "" \
            "CLAUDE.md declares IRIS_PORT: $IRIS_PORT. Published now:" \
            "    ${PUBLISHED:-<none>}" \
            "" \
            "A right name on the wrong port reaches another project's IRIS," \
            "which corrupts its data without reporting anything."
    fi
fi

if $JSON_MODE; then
    echo "{\"status\":\"ok\",\"container\":\"$IRIS_CONTAINER\",\"port\":\"$IRIS_PORT\"}"
else
    echo "✓ IRIS container '$IRIS_CONTAINER' is running${IRIS_PORT:+ on port $IRIS_PORT}."
fi
exit 0
