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

import ast
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
#: The files exempt from `inline-standard-ddl`, because owning this DDL is their job.
#: `schema_manager.py` is the only `.py` owner — every other module that creates one of
#: the standard tables is a second account of its layout. `sql/schema.sql` is the DDL the
#: installation docs hand to a DBA; it was listed here while `.sql` files were not walked
#: at all, so the exemption read as scoping and did nothing. Every path in this set has to
#: be a file the detector actually reaches — `inline_standard_ddl_files` is that set, and
#: a test asserts the containment.
DDL_OWNER = {"iris_vector_rag/storage/schema_manager.py", "sql/schema.sql"}

#: `CREATE TABLE [IF NOT EXISTS] <target>`, where the target is read as far as the
#: whitespace or `(` that ends it. Reading the target rather than matching a table name
#: outright is what makes an interpolated name visible instead of invisible.
DDL_TARGET = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<target>[^\s(]+)", re.I
)

#: A name assembled at runtime: an f-string field, a `%s`, or ObjectScript's `_`
#: concatenation around a string literal. The detector cannot say which table this is.
INTERPOLATED_NAME = re.compile(r"\{[^}]*\}|%s|\"_|_\"|\'_|_\'")


def _opens_a_column_list(rest: str, following: list[str]) -> bool:
    """Whether a `(` follows the table name, on this line or the next non-blank one.

    A statement that creates a table has a column list; `f"Failed to create table
    {name}: {err}"` has a colon. Without this the widened detector reported three log
    messages — `vector_store_iris.py:426`, `orchestrator.py:339`, `validator.py:175` —
    as inline DDL, which is the kind of noise that gets a gate turned off. A column list
    opening two or more lines below the name is knowingly missed.
    """
    if "(" in rest:
        return True
    for line in following:
        if line.strip():
            return line.lstrip().startswith("(")
    return False


def _standard_table(target: str) -> str | None:
    """The standard table `target` names, or None. The schema half is ignored: it may be
    `RAG.`, `{self.schema}.` or `"_schema_".`, and the table is standard either way."""
    name = target.rstrip("(\"'`,;").split(".")[-1].strip("\"'`")
    for table in STANDARD_TABLES:
        if name.lower() == table.lower():
            return table
    return None


def standard_ddl(path: str, source: str) -> list[Finding]:
    """The pure half of `inline-standard-ddl`: one file's source, its findings.

    Two kinds of finding, because an honest report of what the detector cannot resolve is
    worth more than silence: DDL naming one of the standard tables, and DDL whose table
    name is assembled at runtime. The second says so — the name may be configured to
    something else entirely, and claiming it is standard would be a guess.
    """
    if path in DDL_OWNER:
        return []
    out = []
    lines = source.splitlines()
    for number, line in enumerate(lines, start=1):
        for match in DDL_TARGET.finditer(line):
            if not _opens_a_column_list(line[match.end() :], lines[number:]):
                continue
            target = match.group("target")
            table = _standard_table(target)
            if table:
                out.append(
                    Finding(
                        "inline-standard-ddl",
                        f"{path}:{number}",
                        f"DDL for the standard RAG table {table} outside SchemaManager: "
                        f"{line.strip()[:60]}",
                    )
                )
            elif INTERPOLATED_NAME.search(target):
                out.append(
                    Finding(
                        "inline-standard-ddl",
                        f"{path}:{number}",
                        f"DDL whose table name the detector cannot resolve ({target}); "
                        "it may or may not be a standard RAG table",
                    )
                )
    return out


def inline_standard_ddl_files() -> list[pathlib.Path]:
    """Every file that can carry DDL: the package's `.py` and `.sql`, the `sql/`
    directory, and the ObjectScript classes that build their DDL as strings. The detector
    shipped reading `.py` alone, which is why `sql/schema.sql` and `RAG.SDK.Schema` — the
    two largest pieces of DDL in the repo — were invisible to it."""
    return (
        list(files(PKG, ".py"))
        + list(files(PKG, ".sql"))
        + list(files("sql/", ".sql"))
        + list(files("iris_src/", ".cls"))
    )


def inline_standard_ddl() -> list[Finding]:
    """Shipped: SetupOrchestrator carried its own CREATE TABLE RAG.SourceDocuments
    (id INTEGER IDENTITY, filename, ...) alongside SchemaManager's canonical DDL.
    First-run ingest failed with SQLCODE -104 and vector search with -254."""
    out = []
    for p in inline_standard_ddl_files():
        out.extend(standard_ddl(rel(p), read(p)))
    return out


DRIVER_MODULE = "iris_vector_rag/session/drivers.py"
_CONNECT_PATTERN = re.compile(
    r"\.dbapi\.connect\(|\bcreateConnection\(|\biris\.connect\("
)


def driver_connects(path: str, source: str) -> list[Finding]:
    """Every driver connect in `source`, unless `path` is the module that owns them."""
    if path == DRIVER_MODULE:
        return []
    out = []
    for n, line in enumerate(source.splitlines(), start=1):
        if not _CONNECT_PATTERN.search(line):
            continue
        if "#" in line and line.index("#") < line.find("connect("):
            continue
        out.append(
            Finding(
                "unguarded-dbapi-connect",
                f"{path}:{n}",
                f"only {DRIVER_MODULE} may call a driver's connect; every other "
                "caller goes through the session, which bounds the handshake, opens "
                "the back-off window and recovers an expired password",
            )
        )
    return out


def unguarded_dbapi_connect() -> list[Finding]:
    """Shipped: the native DBAPI handshake can block forever, and four modules called
    it themselves, so which protections applied depended on which one a caller
    reached (v0.14.0 release runs hung in CI). Since 083-iris-session the session's
    driver module is the only place that connects."""
    out = []
    for p in files(PKG, ".py"):
        out.extend(driver_connects(rel(p), read(p)))
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


CATALOG_OWNER = "iris_vector_rag/storage/catalog.py"

#: A query, not a mention. The package explains the varchar-vs-VECTOR reporting
#: quirk in several docstrings, so the table name alone is not evidence; a `FROM`
#: or `JOIN` in front of it is.
CATALOG_READ = re.compile(
    r"\b(?:FROM|JOIN)\s+(INFORMATION_SCHEMA\.\w+|%Dictionary\.\w+)", re.I
)


def catalog_reads(path: str, source: str) -> list[Finding]:
    """The pure half of `catalog-read-outside-shape`: one file's source, its findings.

    Split out so the detector can be tested against source it is handed
    (tests/unit/test_antipatterns_detectors.py) rather than against whatever
    happens to be checked in.
    """
    if path == CATALOG_OWNER:
        return []
    out = []
    for match in CATALOG_READ.finditer(source):
        line = source.count("\n", 0, match.start(1)) + 1
        out.append(
            Finding(
                "catalog-read-outside-shape",
                f"{path}:{line}",
                f"reads {match.group(1)} outside {CATALOG_OWNER}; "
                "resolve a TableShape instead",
            )
        )
    return out


def catalog_read_outside_shape() -> list[Finding]:
    """Shipped: eleven modules each asked INFORMATION_SCHEMA their own question
    about RAG.SourceDocuments, and disagreed. The store wrote `text_content` while
    the reader selected `content` on a 0.13 table, vector search retried a second
    TO_VECTOR type after the first failed, and a `LIKE` filter on a LONGVARCHAR
    column matched nothing (feature 082, FR-013, SC-003).

    One module owns the catalog SQL now, and a shape resolved from it answers the
    questions the scattered queries used to. A new query here means a second
    account of the same table."""
    out = []
    for p in files(PKG, ".py"):
        out.extend(catalog_reads(rel(p), read(p)))
    return out


#: Where the type is declared once, named in every finding so the fix is not a guess.
CANONICAL_TYPE_OWNER = "CANONICAL_VECTOR_TYPE in iris_vector_rag/storage/table_shape.py"


#: A literal element type inside a vector construction: `VECTOR(FLOAT, n)` as a column
#: is declared, `TO_VECTOR(?, FLOAT, n)` as a value is converted. The element types IRIS
#: accepts, not every SQL type — `VECTOR(VARCHAR, n)` is not a thing this repo creates.
VECTOR_TYPE_LITERAL = re.compile(
    r"\b(?:TO_VECTOR\s*\(\s*[^,()]*,\s*|VECTOR\s*\(\s*)"
    r"(?P<type>FLOAT|DOUBLE|DECIMAL|INTEGER|BIGINT|SMALLINT|TINYINT)\b",
    re.I,
)

#: `embedding VECTOR(FLOAT, 384),` — a column declaration, where the name in front of
#: the type is what distinguishes DDL from a sentence about DDL.
DECLARED_VECTOR_COLUMN = re.compile(r"\w\s+VECTOR\s*\(", re.I)

#: Comment introducers for the three languages walked. A comment that quotes DDL is
#: prose: `validation/orchestrator.py:276` explains the canonical layout that way.
COMMENT_MARKERS = ("#", "--", "//")


def hardcoded_vector_types(path: str, source: str) -> list[Finding]:
    """The pure half of `hardcoded-vector-type`: one file's source, its findings.

    A match counts only with SQL around it on the same line — `CREATE TABLE`, `ALTER
    TABLE`, `TO_VECTOR(`, or a column name in front of the type on a line that
    continues (`,`) or constrains (`NULL`) like DDL does. The package describes vector
    columns in prose in several docstrings, and a detector that reported those would be
    switched off within a day; the line-local rule is what separates the two.

    Line-local also means a declaration wrapped onto its own line under an `ALTER TABLE`
    is missed. No such statement is in the tree, and the alternative — looking back for a
    `CREATE TABLE` several lines up — reports the docstrings that quote one.
    """
    out = []
    for number, line in enumerate(source.splitlines(), start=1):
        for match in VECTOR_TYPE_LITERAL.finditer(line):
            before = line[: match.start()]
            if any(marker in before for marker in COMMENT_MARKERS):
                continue
            upper = line.upper()
            is_sql = (
                "CREATE TABLE" in upper
                or "ALTER TABLE" in upper
                or "TO_VECTOR" in upper
                or (
                    DECLARED_VECTOR_COLUMN.search(line)
                    and (
                        line.rstrip().endswith(",")
                        or line.rstrip().upper().endswith("NULL")
                    )
                )
            )
            if not is_sql:
                continue
            out.append(
                Finding(
                    "hardcoded-vector-type",
                    f"{path}:{number}",
                    f"names the vector element type {match.group('type').upper()} in SQL; "
                    f"declare it once ({CANONICAL_TYPE_OWNER}) or resolve it from the column",
                )
            )
    return out


def hardcoded_vector_type() -> list[Finding]:
    """Shipped: `RAG.SDK.Schema` created `VECTOR(DOUBLE, n)` while every Python path
    created `VECTOR(FLOAT, n)`, and `RAG.SDK.Pipeline` wrote `TO_VECTOR(?,DOUBLE,384)`
    into whichever of the two it was given. IRIS refuses to compare vectors of different
    element types (SQLCODE -259), so a row written by one path was a row the other could
    not search — and the width was wrong too, because 384 was a literal (feature 086).

    Each site here decides on its own which tables the rest of the repo can search. The
    baseline holds the known ones; a new one has to be argued for in review."""
    out = []
    for p in (
        list(files(PKG, ".py"))
        + list(files(PKG, ".sql"))
        + list(files("sql/", ".sql"))
        + list(files("iris_src/", ".cls"))
    ):
        out.extend(hardcoded_vector_types(rel(p), read(p)))
    return out


BROAD_EXCEPTIONS = {"Exception", "BaseException"}


def _is_broad(handler_type: ast.expr | None) -> bool:
    """A bare `except:`, `Exception`, `BaseException`, or a tuple naming one."""
    if handler_type is None:
        return True
    if isinstance(handler_type, ast.Name):
        return handler_type.id in BROAD_EXCEPTIONS
    if isinstance(handler_type, ast.Attribute):
        return handler_type.attr in BROAD_EXCEPTIONS
    if isinstance(handler_type, ast.Tuple):
        return any(_is_broad(e) for e in handler_type.elts)
    return False


def _handler_raises(handler: ast.ExceptHandler) -> bool:
    """Whether the handler body can raise. A `raise` inside a nested function or
    class is a definition, not a re-raise, so it does not count."""
    todo = list(handler.body)
    while todo:
        node = todo.pop()
        if isinstance(node, ast.Raise):
            return True
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        todo.extend(ast.iter_child_nodes(node))
    return False


def swallowed_errors(path: str, source: str) -> list[Finding]:
    """The pure half of `swallowed-error`: one file's source, its findings.

    A finding is a broad `except` whose body never raises: whatever it does
    instead (`pass`, return a default, log and carry on) hides the failure from the
    caller. Handlers that catch a named, narrow error are out of scope; so are
    handlers that re-raise or raise a typed error.

    The location is `path::<qualified function>#<n>`, the n-th swallow in that
    function, not a line number: the baseline must not go stale every time another
    branch adds an import above it. A file that does not parse raises, so the gate
    cannot pass by skipping it.
    """
    tree = ast.parse(source, filename=path)
    out: list[Finding] = []
    counts: dict[str, int] = {}

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, scope + (child.name,))
                continue
            if (
                isinstance(child, ast.ExceptHandler)
                and _is_broad(child.type)
                and not _handler_raises(child)
            ):
                name = ".".join(scope) or "<module>"
                counts[name] = counts.get(name, 0) + 1
                caught = ast.unparse(child.type) if child.type else "bare except"
                out.append(
                    Finding(
                        "swallowed-error",
                        f"{path}::{name}#{counts[name]}",
                        f"line {child.lineno}: `except {caught}` never raises; "
                        "raise a typed error, or degrade only through a named "
                        "parameter that defaults to raising",
                    )
                )
            visit(child, scope)

    visit(tree, ())
    return out


def swallowed_error() -> list[Finding]:
    """Shipped: broad handlers that returned an empty result or logged and carried
    on hid a failed insert, a model that did not load and a search that found
    nothing because it had errored (findings PL-01, PL-06, PL-08..PL-11, ST-06,
    ST-07, CR-10, VM-02, VM-11, VM-12, OS-02, OS-16). Each owning feature fixes its
    own sites and removes their baseline lines; this detector stops a new one."""
    out: list[Finding] = []
    for p in files(PKG, ".py"):
        out.extend(swallowed_errors(rel(p), read(p)))
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
    "catalog-read-outside-shape": catalog_read_outside_shape,
    "hardcoded-vector-type": hardcoded_vector_type,
    "swallowed-error": swallowed_error,
}

# Checks that carry no baseline and must stay at zero: each fails open when
# violated (a gate list that matches nothing, a test that cannot import).
NO_BASELINE = {
    "unguarded-dbapi-connect",
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
