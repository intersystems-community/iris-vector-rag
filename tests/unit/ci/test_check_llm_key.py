"""scripts/ci/check-llm-key.sh: a rejected LLM API key secret fails the job at once.

Without it, a dead key makes the batch extraction tests retry until the per-test
timeout, which with --timeout-method=thread ends the whole run with no summary.
`curl` is replaced by a stub that answers with the HTTP status the test chooses.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "check-llm-key.sh"


def run(tmp_path, has_key, status="200"):
    calls = tmp_path / "curl-calls"
    stub = tmp_path / "curl"
    stub.write_text(
        '#!/bin/sh\necho "$@" >> "$CURL_CALLS"\nprintf "%s" "$STUB_STATUS"\n'
        '[ "$STUB_STATUS" = "000" ] && exit 7\nexit 0\n'
    )
    stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CURL_CALLS": str(calls),
            "STUB_STATUS": status,
            "HAS_LLM_KEY": has_key,
            "OPENAI_API_KEY": "sk-test-key",
            "GITHUB_REPOSITORY": "owner/repo",
        },
        capture_output=True,
        text=True,
    )
    return result, calls.read_text() if calls.exists() else ""


def test_no_secret_means_nothing_to_check(tmp_path):
    result, calls = run(tmp_path, "false")
    assert result.returncode == 0
    assert calls == ""


def test_an_accepted_key_passes(tmp_path):
    result, calls = run(tmp_path, "true", "200")
    assert result.returncode == 0
    assert "Bearer sk-test-key" in calls
    assert calls.rstrip().endswith("/models")


@pytest.mark.parametrize("status", ["401", "429"])
def test_a_rejected_key_fails_and_says_how_to_reset_it(tmp_path, status):
    result, _ = run(tmp_path, "true", status)
    assert result.returncode == 1
    assert f"::error::The OPENAI_API_KEY secret was rejected (HTTP {status})" in (
        result.stdout
    )
    assert "gh secret set OPENAI_API_KEY -R owner/repo" in result.stdout


def test_an_unreachable_api_fails_without_blaming_the_key(tmp_path):
    result, _ = run(tmp_path, "true", "000")
    assert result.returncode == 1
    assert "::error::Could not reach" in result.stdout
    assert "rejected" not in result.stdout
