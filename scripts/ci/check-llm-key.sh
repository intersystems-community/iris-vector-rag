#!/usr/bin/env bash
# Fail the integration job at once when the repository's OPENAI_API_KEY secret is
# rejected. Without this, a dead key makes the batch extraction tests retry until
# the per-test timeout, which ends the whole run with no summary.
#
# HAS_LLM_KEY is "true" when the secret exists. Without it there is nothing to
# check: the job deselects the tests in tests/contract/ci_needs_llm_key.txt.
set -euo pipefail

if [ "${HAS_LLM_KEY:-}" != "true" ]; then
  echo "No OPENAI_API_KEY secret: nothing to check."
  exit 0
fi

status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  https://api.openai.com/v1/models) || status=000

case "$status" in
  200)
    echo "The OPENAI_API_KEY secret is accepted."
    ;;
  000)
    echo "::error::Could not reach api.openai.com to check the OPENAI_API_KEY secret."
    exit 1
    ;;
  *)
    echo "::error::The OPENAI_API_KEY secret was rejected (HTTP ${status}). Reset it: gh secret set OPENAI_API_KEY -R ${GITHUB_REPOSITORY:-<owner/repo>}"
    exit 1
    ;;
esac
