"""Every place that names the iris-vector-graph requirement says the same thing.

The pin lives in pyproject.toml (core dependency and the extras), the two workflows that
install the package by hand, and the prose that tells operators what to install. A bump
that misses one leaves CI testing a different ivg than the package declares.

The floor is 4.2.0 because IVR calls ``engine.bm25_upgrade`` and passes ``max_df_fraction``
to ``bm25_search``; both first exist in ivg 4.2.0.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Files that carry the requirement as ``iris-vector-graph>=X.Y.Z,<5``.
PIN_FILES = [
    "pyproject.toml",
    ".github/workflows/pr-integration.yml",
    ".github/workflows/release.yml",
    "AGENTS.md",
    "docs/IVG_4X_UPGRADE.md",
]

PIN = re.compile(r"iris-vector-graph>=(\d+)\.(\d+)\.(\d+),<5")
FLOOR = (4, 2, 0)


def _pins(relative: str):
    text = (ROOT / relative).read_text(encoding="utf-8")
    return [tuple(int(part) for part in match.groups()) for match in PIN.finditer(text)]


@pytest.mark.parametrize("relative", PIN_FILES)
def test_file_names_the_requirement(relative):
    assert _pins(relative), f"{relative} no longer names iris-vector-graph>=X.Y.Z,<5"


@pytest.mark.parametrize("relative", PIN_FILES)
def test_every_pin_is_at_the_floor_the_code_needs(relative):
    assert all(
        pin == FLOOR for pin in _pins(relative)
    ), f"{relative}: pins {_pins(relative)} must all be {FLOOR}"


def test_pyproject_pins_core_and_both_extras():
    assert (
        len(_pins("pyproject.toml")) == 3
    ), "core dependency, embedding and hybrid-graphrag"


def test_no_other_tracked_file_pins_an_older_ivg():
    tracked = subprocess.run(
        ["git", "grep", "-l", "-E", r"iris-vector-graph>=[0-9]+\.[0-9]+\.[0-9]+,<5"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.split()
    allowed = set(PIN_FILES)
    # History, plans and the changelog describe earlier pins on purpose.
    historical = ("CHANGELOG.md", "docs/remediation/", "specs/")
    strays = [f for f in tracked if f not in allowed and not f.startswith(historical)]
    for stray in strays:
        assert all(
            pin == FLOOR for pin in _pins(stray)
        ), f"{stray} still pins an older ivg: {_pins(stray)}"
