#!/usr/bin/env python3
"""Detectors for the bug classes that have actually shipped in this repo.

Same idea as iris-agentic-dev's scripts/gates/antipatterns.py: a bug fixed after
a release leaves behind a detector here, so the next instance of the same class
fails a gate instead of reaching a user. Each detector names the shipped
instance it was written for.

## Why there is a baseline

A gate that fails hundreds of times is a gate everyone learns to bypass, so the
gate enforces *no new instances* instead of *zero instances*.
`antipatterns-baseline.txt` lists the findings that existed when each detector
was written. A finding absent from the baseline fails the gate; a baseline entry
that no longer fires also fails it, which keeps the baseline shrinking instead of
rotting. Adding a line to the baseline is a tracked edit that shows up in review.

Only git-tracked files are scanned: a detector that reads gitignored files would
pass locally and fail in CI (the `untracked-module-import` class below is exactly
that mistake, made by tests).

Usage:
    scripts/gates/antipatterns.py                     # every detector, against the baseline
    scripts/gates/antipatterns.py legacy-fk-target    # one detector
    scripts/gates/antipatterns.py --all-findings      # ignore the baseline, print everything
    scripts/gates/antipatterns.py --write-baseline    # record current findings as the baseline

Exit: 0 = no new findings, 2 = at least one new finding (or a stale baseline entry).
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE = ROOT / "scripts/gates/antipatterns-baseline.txt"


@dataclass(frozen=True)
class Finding:
    check: str
    location: str  # "path:line" or "path"
    message: str

    def key(self) -> str:
        return f"{self.check}\t{self.location}"


# ---------------------------------------------------------------------------
# File sets (git-tracked only)
# ---------------------------------------------------------------------------


def tracked_files() -> list[pathlib.Path]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return [ROOT / p for p in out.decode().split("\0") if p]


TRACKED = tracked_files()
TRACKED_REL = {p.relative_to(ROOT).as_posix() for p in TRACKED}


def files(prefix: str = "", suffix: str = "") -> Iterable[pathlib.Path]:
    for p in TRACKED:
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(prefix) and rel.endswith(suffix) and p.is_file():
            yield p


def read(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def rel(p: pathlib.Path) -> str:
    return p.relative_to(ROOT).as_posix()


def grep(p: pathlib.Path, pattern: re.Pattern[str]) -> Iterable[tuple[int, str]]:
    for i, line in enumerate(read(p).splitlines(), start=1):
        if pattern.search(line):
            yield i, line


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

PKG = "iris_vector_rag/"


def legacy_fk_target() -> list[Finding]:
    """Shipped: DocumentChunks FKs pointed at SourceDocuments(id), a column the
    canonical DDL never creates. auto_setup on a fresh IRIS then failed with
    SQLCODE -316 (v0.13.x)."""
    pat = re.compile(r"REFERENCES\s+[^(\n]*SourceDocuments[^(\n]*\(\s*id\s*\)", re.I)
    out = []
    for p in list(files(PKG, ".py")) + list(files("sql/", ".sql")):
        for n, _ in grep(p, pat):
            out.append(
                Finding(
                    "legacy-fk-target",
                    f"{rel(p)}:{n}",
                    "FK references SourceDocuments(id); primary key is doc_id",
                )
            )
    return out


STANDARD_TABLES = (
    "SourceDocuments",
    "DocumentChunks",
    "Entities",
    "EntityRelationships",
    "Communities",
)
DDL_OWNER = {"iris_vector_rag/storage/schema_manager.py", "sql/schema.sql"}


def inline_standard_ddl() -> list[Finding]:
    """Shipped: SetupOrchestrator carried its own CREATE TABLE RAG.SourceDocuments
    (id INTEGER IDENTITY, filename, ...) alongside SchemaManager's canonical DDL.
    First-run ingest failed with SQLCODE -104 and vector search with -254."""
    pat = re.compile(
        r"CREATE\s+TABLE\s+(IF\s+NOT\s+EXISTS\s+)?(RAG\.|\{[^}]*\})?\s*("
        + "|".join(STANDARD_TABLES)
        + r")\b",
        re.I,
    )
    out = []
    for p in files(PKG, ".py"):
        if rel(p) in DDL_OWNER:
            continue
        for n, line in grep(p, pat):
            out.append(
                Finding(
                    "inline-standard-ddl",
                    f"{rel(p)}:{n}",
                    f"DDL for a standard RAG table outside SchemaManager: {line.strip()[:60]}",
                )
            )
    return out


def unguarded_dbapi_connect() -> list[Finding]:
    """Shipped: the native DBAPI handshake can block forever. get_iris_connection()
    bounds it (IRIS_CONNECT_TIMEOUT) and applies backoff; a direct connect
    elsewhere bypasses both and hangs CI (v0.14.0 release runs)."""
    pat = re.compile(r"\.dbapi\.connect\(|\bcreateConnection\(|\biris\.connect\(")
    allowed = {"iris_vector_rag/common/iris_connection.py"}
    out = []
    for p in files(PKG, ".py"):
        if rel(p) in allowed:
            continue
        for n, line in grep(p, pat):
            if "#" in line and line.index("#") < line.find("connect("):
                continue
            out.append(
                Finding(
                    "unguarded-dbapi-connect",
                    f"{rel(p)}:{n}",
                    "direct DBAPI connect bypasses get_iris_connection() timeout/backoff",
                )
            )
    return out


def ignore_collect_returns_false() -> list[Finding]:
    """Shipped: a conftest pytest_ignore_collect returned False for every path.
    The hook is firstresult, so that False short-circuited pytest's own --ignore
    handling and ignored files were collected anyway."""
    out = []
    for p in files("", "conftest.py"):
        src = read(p)
        m = re.search(r"def pytest_ignore_collect\b.*?(?=\ndef |\n@|\Z)", src, re.S)
        if not m:
            continue
        body = m.group(0)
        start_line = src[: m.start()].count("\n") + 1
        for i, line in enumerate(body.splitlines()):
            if (
                re.match(r"\s+return\s+False\b", line)
                or re.match(r"\s+return\s+[\w.]+\(.*\)\s*$", line)
                and "endswith" in line
            ):
                out.append(
                    Finding(
                        "ignore-collect-returns-false",
                        f"{rel(p)}:{start_line + i}",
                        "pytest_ignore_collect must return None (not False) when it has no opinion",
                    )
                )
    return out


def untracked_module_import() -> list[Finding]:
    """Shipped: three contract tests imported iris_vector_rag.config.config_manager,
    a compatibility shim that exists only in the developer's tree because
    .gitignore ignores any config/ directory. Green locally, ModuleNotFoundError
    in CI."""
    pat = re.compile(
        r"^\s*(?:from\s+(iris_vector_rag(?:\.[\w]+)*)\s+import|import\s+(iris_vector_rag(?:\.[\w]+)+))"
    )
    out = []
    for p in list(files("tests/", ".py")) + list(files(PKG, ".py")):
        # conftest prunes this directory from collection; its imports are dead.
        if rel(p).startswith("tests/future_tests_not_ready/"):
            continue
        for n, line in grep(p, pat):
            m = pat.match(line)
            mod = (m.group(1) or m.group(2)) if m else None
            if not mod:
                continue
            path = mod.replace(".", "/")
            if f"{path}.py" in TRACKED_REL or f"{path}/__init__.py" in TRACKED_REL:
                continue
            out.append(
                Finding(
                    "untracked-module-import",
                    f"{rel(p)}:{n}",
                    f"imports {mod}, which is not a git-tracked module",
                )
            )
    return out


def inert_pytest_ini() -> list[Finding]:
    """Shipped: pytest.ini used a [tool:pytest] header (setup.cfg syntax). pytest
    reads only [pytest] from pytest.ini, so every setting in it — strict
    markers, timeouts, coverage floor — was silently inert."""
    p = ROOT / "pytest.ini"
    if p.is_file() and re.search(r"^\[tool:pytest\]", read(p), re.M):
        return [
            Finding(
                "inert-pytest-ini",
                "pytest.ini:1",
                "[tool:pytest] header is ignored in pytest.ini; use [pytest]",
            )
        ]
    return []


def positional_load_documents() -> list[Finding]:
    """Shipped: the smoke test called pipeline.load_documents([doc]); since v0.12.1
    the first positional parameter is documents_path, so the list hit os.stat."""
    pat = re.compile(r"\.load_documents\(\s*\[")
    out = []
    for p in files("tests/", ".py"):
        for n, _ in grep(p, pat):
            out.append(
                Finding(
                    "positional-load-documents",
                    f"{rel(p)}:{n}",
                    "pass documents=[...]; the first positional argument is documents_path",
                )
            )
    return out


def services_iris_container() -> list[Finding]:
    """Shipped: a GitHub Actions `services:` IRIS container. The 2026.1 image's
    post-start init crashes on amd64 and takes IRIS down; the services health
    check then kills the job. scripts/ci/start-iris.sh is the working path."""
    out = []
    for p in files(".github/workflows/", ".yml"):
        src = read(p)
        if re.search(r"^\s*services:\s*$", src, re.M) and "iris-community" in src:
            n = next(
                (i for i, l in enumerate(src.splitlines(), 1) if "iris-community" in l),
                1,
            )
            out.append(
                Finding(
                    "services-iris-container",
                    f"{rel(p)}:{n}",
                    "IRIS as a GitHub services: container; use scripts/ci/start-iris.sh",
                )
            )
    return out


def stale_deselect_entry() -> list[Finding]:
    """Guards the release gate itself: tests/contract/ci_known_failures.txt is a
    --deselect list. An entry whose file or test function no longer exists is
    silently ignored by pytest, so the list would rot without anyone noticing."""
    p = ROOT / "tests/contract/ci_known_failures.txt"
    if not p.is_file():
        return []
    out = []
    for n, line in enumerate(read(p).splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("::")
        path, func = parts[0], parts[-1]
        target = ROOT / path
        if not target.is_file():
            out.append(
                Finding(
                    "stale-deselect-entry", f"{rel(p)}:{n}", f"{path} does not exist"
                )
            )
        elif not re.search(rf"def {re.escape(func)}\b", read(target)):
            out.append(
                Finding(
                    "stale-deselect-entry",
                    f"{rel(p)}:{n}",
                    f"{func} not defined in {path}",
                )
            )
    return out


CHECKS: dict[str, Callable[[], list[Finding]]] = {
    "legacy-fk-target": legacy_fk_target,
    "inline-standard-ddl": inline_standard_ddl,
    "unguarded-dbapi-connect": unguarded_dbapi_connect,
    "ignore-collect-returns-false": ignore_collect_returns_false,
    "untracked-module-import": untracked_module_import,
    "inert-pytest-ini": inert_pytest_ini,
    "positional-load-documents": positional_load_documents,
    "services-iris-container": services_iris_container,
    "stale-deselect-entry": stale_deselect_entry,
}

# Checks that carry no baseline and must stay at zero: each fails open when
# violated (a gate list that matches nothing, a test that cannot import).
NO_BASELINE = {
    "untracked-module-import",
    "positional-load-documents",
    "stale-deselect-entry",
    "legacy-fk-target",
}


# ---------------------------------------------------------------------------
# Baseline + main
# ---------------------------------------------------------------------------


def load_baseline() -> set[str]:
    if not BASELINE.is_file():
        return set()
    return {
        l.rstrip("\n")
        for l in read(BASELINE).splitlines()
        if l.strip() and not l.startswith("#")
    }


def main(argv: list[str]) -> int:
    all_findings = "--all-findings" in argv
    write_baseline = "--write-baseline" in argv
    selected = [a for a in argv if not a.startswith("--")]
    for s in selected:
        if s not in CHECKS:
            print(
                f"antipatterns: unknown check {s!r}; known: {', '.join(CHECKS)}",
                file=sys.stderr,
            )
            return 1
    checks = selected or list(CHECKS)

    findings: list[Finding] = []
    for name in checks:
        findings.extend(CHECKS[name]())
    findings.sort(key=lambda f: f.key())

    if write_baseline:
        lines = [
            "# antipatterns baseline: findings that existed when each detector was written.",
            "# A line here silences ONE known instance. Remove lines as instances are fixed;",
            "# the gate fails on stale entries so this list only shrinks. Format: <check>\\t<path:line>",
        ]
        lines += [f.key() for f in findings if f.check not in NO_BASELINE]
        BASELINE.write_text("\n".join(lines) + "\n")
        print(
            f"antipatterns: wrote {len(lines) - 3} baseline entries to {BASELINE.relative_to(ROOT)}"
        )
        return 0

    baseline = load_baseline() if not all_findings else set()
    keys = {f.key() for f in findings}
    new = [f for f in findings if f.key() not in baseline]
    stale = sorted(
        k
        for k in baseline
        if k not in keys and (not selected or k.split("\t")[0] in checks)
    )

    for f in new:
        print(f"{f.location}: [{f.check}] {f.message}")
    for k in stale:
        chk, loc = k.split("\t", 1)
        print(
            f"{loc}: [{chk}] STALE baseline entry — finding no longer fires; remove the line"
        )

    baselined = len(findings) - len(new)
    print(
        f"antipatterns: {len(checks)} checks, {len(new)} new finding(s), {baselined} baselined, {len(stale)} stale baseline entr{'y' if len(stale) == 1 else 'ies'}"
    )
    return 2 if (new or stale) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
