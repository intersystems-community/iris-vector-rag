"""`RAG.SDK` reports real status, validates identifiers and follows the ObjectScript rules.

Covers OS-01 to OS-05, OS-15 and OS-16 of the 2026-10 review (feature
`objectscript-sdk-fix`). The classes are compiled for real: the fixture copies the SDK's
`.cls` files into a throwaway container of its own, loads and
compiles them in USER with `$SYSTEM.OBJ.LoadDir`, and every test calls the compiled classes
through `iris session`. Nothing is mocked except `Graph.KG.BM25Index`, which belongs to
iris-vector-graph (not in the test image): a recording stand-in is compiled only when the
image has no such class, so the BM25 build can be observed.

`RAG.SDK.Embed` and `RAG.SDK.Evaluate` carry embedded-Python methods, which the container
cannot compile (DEBT entries 1 and 10), so they are neither loaded nor exercised here.

Each test runs in a scratch schema of its own (`SDKFix...`), so none reaches `RAG`.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_docker,
    pytest.mark.requires_database,
]

REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_DIR = REPO_ROOT / "iris_src" / "src" / "RAG" / "SDK"
COMPILED = tuple(
    p.stem for p in sorted(SDK_DIR.glob("*.cls")) if p.stem not in ("Embed", "Evaluate")
)
SCHEMA = "SDKFix"
DIM = 8

BM25_STUB = """Class Graph.KG.BM25Index
{

/// Test stand-in: records how much text the SDK handed over for each document.
ClassMethod Insert(pIndex As %String, pId As %String, pText As %String) As %String
{
    Set ^SDKFixBM25(pIndex, pId) = $Length(pText)
    Quit 1
}

}
"""


def _vec(seed: int, dim: int = DIM) -> str:
    return ",".join(f"{((seed * 7 + i * 3) % 11 + 1) / 11:.4f}" for i in range(dim))


class Server:
    """ObjectScript access to the ephemeral container."""

    def __init__(self, container: str):
        self.container = container

    def session(self, script: str) -> str:
        done = subprocess.run(
            ["docker", "exec", "-i", self.container, "iris", "session", "IRIS"],
            input=f'zn "USER"\n{script}\nhalt\n',
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        return done.stdout

    def run(self, code: str) -> str:
        """Run ObjectScript lines; return what the lines wrote after the `R:` marker."""
        out = self.session(code)
        for line in out.splitlines():
            if "R:" in line:
                return line.split("R:", 1)[1].strip()
        raise AssertionError(f"no R: marker in session output:\n{out}")

    def call(self, expr: str) -> str:
        return self.run(f'write "R:",{expr},!')

    def json(self, expr: str) -> dict | list:
        import json

        text = self.call(expr)
        try:
            return json.loads(text)
        except ValueError as exc:  # pragma: no cover - the message is the point
            raise AssertionError(f"{expr} did not return JSON: {text!r}") from exc

    def sql_scalar(self, sql: str) -> str:
        return self.run(
            f'set rs=##class(%SQL.Statement).%ExecDirect(,"{sql}")\n'
            'if rs.%Next() { write "R:",rs.%GetData(1),! } else { write "R:none",! }'
        )


def _load(srv: Server) -> Server:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "BM25Index.cls").write_text(BM25_STUB)
        subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "0",
                srv.container,
                "sh",
                "-c",
                "rm -rf /tmp/sdk /tmp/stub && mkdir -p /tmp/sdk /tmp/stub",
            ],
            check=True,
        )
        for name in COMPILED:
            subprocess.run(
                [
                    "docker",
                    "cp",
                    str(SDK_DIR / f"{name}.cls"),
                    f"{srv.container}:/tmp/sdk/{name}.cls",
                ],
                check=True,
            )
        subprocess.run(
            [
                "docker",
                "cp",
                str(Path(tmp) / "BM25Index.cls"),
                f"{srv.container}:/tmp/stub/BM25Index.cls",
            ],
            check=True,
        )
    subprocess.run(
        [
            "docker",
            "exec",
            "-u",
            "0",
            srv.container,
            "chmod",
            "-R",
            "a+rX",
            "/tmp/sdk",
            "/tmp/stub",
        ],
        check=True,
    )
    out = srv.session(
        'set sc=$system.OBJ.LoadDir("/tmp/sdk","ck",.err,1)\n'
        'write "LOAD:",sc,!\n'
        'if \'##class(%Dictionary.CompiledClass).%ExistsId("Graph.KG.BM25Index") '
        '{ set sc=$system.OBJ.LoadDir("/tmp/stub","ck",.err2,1) write "STUB:",sc,! }'
    )
    assert "LOAD:1" in out, f"SDK classes did not compile:\n{out}"
    assert "ERROR" not in out, out
    return srv


@pytest.fixture(scope="module")
def server() -> Server:
    """A throwaway container of its own, whichever IRIS the session pins.

    The tests stub Graph.KG.BM25Index, which the dev container has for real, and compile
    the SDK classes into USER; neither belongs in a shared database.
    """
    from tests.support.iris_environment import start_ephemeral

    container, target = start_ephemeral()
    try:
        yield _load(Server(target.container))
    finally:
        container.stop()


@pytest.fixture
def scratch(server: Server):
    """SDKFix tables, empty before and dropped after; default-table config cleared."""

    def clean():
        server.run(
            "kill ^RAG.SDK.Config, ^SDKFixBM25\n"
            + "".join(
                f'do ##class(%SQL.Statement).%ExecDirect(,"DROP TABLE IF EXISTS {t}")\n'
                for t in (
                    f"{SCHEMA}.SourceDocuments",
                    f"{SCHEMA}.DocumentChunks",
                    f"{SCHEMA}.Entities",
                    f"{SCHEMA}.Custom",
                    "GOLDEN.SourceDocuments",
                    "GOLDEN.DocumentChunks",
                    "GOLDEN.Entities",
                )
            )
            + 'write "R:clean",!'
        )

    clean()
    server.call(f'##class(RAG.SDK.Schema).Initialize({DIM},"{SCHEMA}")')
    server.call(
        f'##class(RAG.SDK.Bridge).SetDefaultTable("{SCHEMA}.SourceDocuments",'
        '"doc_id","text_content","embedding","metadata")'
    )
    yield server
    clean()


TABLE = f"{SCHEMA}.SourceDocuments"


def add(server: Server, doc_id: str, text: str, vec: str = "", table: str = TABLE):
    return server.json(
        f'##class(RAG.SDK.Pipeline).AddDocument("{doc_id}","{text}","{{}}","{vec}","{table}")'
    )


# ---------------------------------------------------------------- OS-01 / OS-02 (Pipeline)


class TestPipelineStatus:
    def test_good_insert_is_ok_and_takes_dimension_from_table(self, scratch):
        assert add(scratch, "a", "text", _vec(1))["status"] == "ok"
        assert scratch.sql_scalar(f"SELECT COUNT(*) FROM {TABLE}") == "1"

    def test_wrong_dimension_is_an_error_not_ok(self, scratch):
        result = add(scratch, "bad", "text", _vec(1, dim=3))
        assert result["status"] == "error", result
        assert scratch.sql_scalar(f"SELECT COUNT(*) FROM {TABLE}") == "0"

    def test_constraint_violation_is_an_error_not_ok(self, scratch):
        result = add(scratch, "x" * 100, "text")  # doc_id is VARCHAR(64)
        assert result["status"] == "error", result

    def test_batch_counts_real_errors(self, scratch):
        import json

        batch = json.dumps(
            [
                {"id": "ok1", "text": "t", "embedding": _vec(1)},
                {"id": "bad1", "text": "t", "embedding": _vec(2, dim=3)},
                {"id": "x" * 100, "text": "t"},
            ]
        )
        quoted = batch.replace('"', '""')
        result = scratch.json(
            f'##class(RAG.SDK.Pipeline).AddDocumentBatch("{quoted}","{TABLE}")'
        )
        assert result == {"inserted": 1, "errors": 2}
        assert scratch.sql_scalar(f"SELECT COUNT(*) FROM {TABLE}") == "1"

    def test_existing_id_is_updated(self, scratch):
        assert add(scratch, "u", "first", _vec(1))["status"] == "ok"
        assert add(scratch, "u", "second", _vec(2))["status"] == "ok"
        assert (
            scratch.sql_scalar(
                f"SELECT SUBSTRING(text_content,1,20) FROM {TABLE} WHERE doc_id='u'"
            )
            == "second"
        )

    @pytest.mark.parametrize(
        "table",
        [
            f"{SCHEMA}.SourceDocuments; DROP TABLE {SCHEMA}.SourceDocuments",
            f"{SCHEMA}.SourceDocuments (doc_id) SELECT 1 --",
            "",
            "a.b.c.d",
        ],
    )
    def test_table_argument_is_validated(self, scratch, table):
        result = add(scratch, "inj", "text", _vec(1), table=table)
        assert result["status"] == "error", result
        assert (
            "identifier" in result["message"].lower()
            or "table" in result["message"].lower()
        )
        assert scratch.sql_scalar(f"SELECT COUNT(*) FROM {TABLE}") == "0"


# ---------------------------------------------------------------- OS-03 (full text)


class TestFullText:
    def _long(self, scratch, n=5000):
        scratch.run(
            f'set t=$translate($justify("",{n})," ","x")\n'
            f'set j=##class(RAG.SDK.Pipeline).AddDocument("long",t,"{{}}","{_vec(1)}","{TABLE}")\n'
            'write "R:",j,!'
        )

    def test_text_search_returns_whole_stream(self, scratch):
        self._long(scratch)
        hits = scratch.json('##class(RAG.SDK.Search).TextSearch("xxxx",5)')
        assert [len(h["text"]) for h in hits] == [5000]
        assert "truncated" not in hits[0]

    def test_vector_search_returns_whole_stream(self, scratch):
        self._long(scratch)
        hits = scratch.json(f'##class(RAG.SDK.Search).VectorSearch("{_vec(1)}",5)')
        assert [len(h["text"]) for h in hits] == [5000]

    def test_read_text_signals_truncation(self, scratch):
        out = scratch.run(
            "set s=##class(%Stream.GlobalCharacter).%New()\n"
            'do s.Write($translate($justify("",50),$char(32),"y"))\n'
            "do s.%Save()\n"
            "set oid=s.%Oid()\n"
            "set t=##class(RAG.SDK.Search).ReadText(oid,.trunc,10)\n"
            'write "R:",$length(t),"|",trunc,!'
        )
        assert out == "10|1"

    def test_bm25_index_sees_whole_document(self, scratch):
        self._long(scratch, 5000)
        result = scratch.json('##class(RAG.SDK.Bridge).BuildBM25Index("idx")')
        assert result["status"] == "ok" and result["documents"] == 1
        assert scratch.run('write "R:",$get(^SDKFixBM25("idx","long")),!') == "5000"


# ---------------------------------------------------------------- OS-04 (BM25 build)


class TestBuildBM25Index:
    def _custom(self, scratch):
        scratch.run(
            f'do ##class(%SQL.Statement).%ExecDirect(,"CREATE TABLE {SCHEMA}.Custom '
            '(pk VARCHAR(64) NOT NULL, body VARCHAR(2000), vec VECTOR(FLOAT, 8), PRIMARY KEY (pk))")\n'
            f'do ##class(%SQL.Statement).%ExecDirect(,"INSERT INTO {SCHEMA}.Custom (pk, body) '
            "VALUES ('c1','hello world')\")\n"
            f'do ##class(RAG.SDK.Bridge).SetDefaultTable("{SCHEMA}.Custom","pk","body","vec")\n'
            'write "R:done",!'
        )

    def test_default_text_column_is_the_configured_one(self, scratch):
        self._custom(scratch)
        result = scratch.json('##class(RAG.SDK.Bridge).BuildBM25Index("idx")')
        assert result["status"] == "ok", result
        assert result["documents"] == 1
        assert scratch.run('write "R:",$get(^SDKFixBM25("idx","c1")),!') == "11"

    def test_unprepared_statement_is_an_error(self, scratch):
        result = scratch.json(
            '##class(RAG.SDK.Bridge).BuildBM25Index("idx","no_such_column")'
        )
        assert result["status"] == "error", result
        assert (
            scratch.run('write "R:",$data(^RAG.SDK.Config("bm25_index_name")),!') == "0"
        )

    def test_missing_table_is_an_error(self, scratch):
        scratch.run(
            f'do ##class(%SQL.Statement).%ExecDirect(,"DROP TABLE {TABLE}")\nwrite "R:ok",!'
        )
        result = scratch.json('##class(RAG.SDK.Bridge).BuildBM25Index("idx")')
        assert result["status"] == "error", result

    def test_text_column_is_validated(self, scratch):
        result = scratch.json(
            '##class(RAG.SDK.Bridge).BuildBM25Index("idx","text_content FROM x; DROP TABLE y --")'
        )
        assert result["status"] == "error", result
        assert "identifier" in result["message"].lower()


# ---------------------------------------------------------------- OS-05 (identifiers, topK)


class TestIdentifiersAndTopK:
    def test_set_default_table_rejects_bad_identifiers(self, scratch):
        for args in (
            '"x; DROP TABLE y","doc_id","text_content","embedding"',
            f'"{TABLE}","doc_id --","text_content","embedding"',
            f'"{TABLE}","doc_id","text_content","embedding","meta data"',
            f'"{TABLE}","doc_id","a b","embedding"',
        ):
            result = scratch.json(f"##class(RAG.SDK.Bridge).SetDefaultTable({args})")
            assert result["status"] == "error", (args, result)
        cfg = scratch.json("##class(RAG.SDK.Bridge).GetDefaultTable()")
        assert cfg["table"] == TABLE

    def test_poisoned_config_is_refused_at_search_time(self, scratch):
        scratch.run(
            'set ^RAG.SDK.Config("default_id_col")="doc_id, 1 FROM x --"\nwrite "R:set",!'
        )
        for expr in (
            f'##class(RAG.SDK.Search).VectorSearch("{_vec(1)}",3)',
            '##class(RAG.SDK.Search).TextSearch("a",3)',
        ):
            assert "error" in scratch.json(expr)

    def test_top_k_string_is_not_injectable(self, scratch):
        add(scratch, "a", "alpha", _vec(1))
        for bad in ("1; DROP TABLE " + TABLE, "0", "-3", "abc", "1.5"):
            out = scratch.json(
                f'##class(RAG.SDK.Search).VectorSearch("{_vec(1)}","{bad}")'
            )
            assert isinstance(out, dict) and "error" in out, (bad, out)
            out = scratch.json(f'##class(RAG.SDK.Search).TextSearch("alpha","{bad}")')
            assert isinstance(out, dict) and "error" in out, (bad, out)
        assert scratch.sql_scalar(f"SELECT COUNT(*) FROM {TABLE}") == "1"

    def test_numeric_string_top_k_is_coerced(self, scratch):
        for i in range(4):
            add(scratch, f"d{i}", "alpha", _vec(i))
        hits = scratch.json(f'##class(RAG.SDK.Search).VectorSearch("{_vec(1)}","2")')
        assert len(hits) == 2
        hits = scratch.json('##class(RAG.SDK.Search).TextSearch("alpha","3")')
        assert len(hits) == 3

    def test_vector_search_is_ordered_and_capped_without_top_order_by(self, scratch):
        for i in range(5):
            add(scratch, f"d{i}", "t", _vec(i))
        hits = scratch.json(f'##class(RAG.SDK.Search).VectorSearch("{_vec(2)}",3)')
        assert len(hits) == 3
        assert hits[0]["id"] == "d2"
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_valid_names_pass(self, scratch):
        for name in ("doc_id", "%ID", "Col1", "a_b_c"):
            assert (
                scratch.call(f'##class(RAG.SDK.Identifier).IsName("{name}")') == "1"
            ), name
        for name in ("", "1a", "a-b", "a b", "a;b", "a.b", 'a"b'):
            literal = name.replace(
                '"', '""'
            )  # ObjectScript doubles a quote inside a string
            assert (
                scratch.call(f'##class(RAG.SDK.Identifier).IsName("{literal}")') == "0"
            ), name
        assert (
            scratch.call('##class(RAG.SDK.Identifier).IsTable("RAG.SourceDocuments")')
            == "1"
        )
        assert scratch.call('##class(RAG.SDK.Identifier).IsTable("Table1")') == "1"
        assert scratch.call('##class(RAG.SDK.Identifier).IsTable("a.b.c.d")') == "0"


# ---------------------------------------------------------------- OS-16 (Schema)


class TestSchema:
    def test_schema_name_containing_go_is_not_split(self, scratch):
        result = scratch.json(f'##class(RAG.SDK.Schema).Initialize({DIM},"GOLDEN")')
        assert result["status"] == "ok", result
        assert (
            scratch.sql_scalar(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='GOLDEN'"
            )
            == "3"
        )

    def test_failed_ddl_is_reported(self, scratch):
        result = scratch.json('##class(RAG.SDK.Schema).Initialize(0,"SDKFix")')
        assert result["status"] == "error", result
        result = scratch.json('##class(RAG.SDK.Schema).Initialize(8,"a;b")')
        assert result["status"] == "error", result

    def test_initialize_is_idempotent(self, scratch):
        assert (
            scratch.json(f'##class(RAG.SDK.Schema).Initialize({DIM},"{SCHEMA}")')[
                "status"
            ]
            == "ok"
        )

    def test_install_command_is_not_pinned_to_a_stale_version(self, scratch):
        assert (
            scratch.call("##class(RAG.SDK.Schema).InstallCommand()")
            == "pip install iris-vector-rag"
        )
        assert (
            scratch.call('##class(RAG.SDK.Schema).InstallCommand("1.2.3")')
            == "pip install iris-vector-rag==1.2.3"
        )
        assert "0.11.4" not in (SDK_DIR / "Schema.cls").read_text()

    def test_status_reads_the_declared_dimension(self, scratch):
        add(scratch, "a", "t", _vec(1))
        status = scratch.json(f'##class(RAG.SDK.Schema).Status("{SCHEMA}")')
        assert status["embeddingDimension"] == DIM
        assert status["documentCount"] == 1
        assert f"{SCHEMA}.SourceDocuments" in status["tables"]

    def test_status_without_tables_reports_zero_dimension(self, scratch):
        status = scratch.json('##class(RAG.SDK.Schema).Status("NoSuchSchema")')
        assert status["embeddingDimension"] == 0 and status["tables"] == []

    def test_drop_removes_tables_and_reports_errors(self, scratch):
        assert (
            scratch.json(f'##class(RAG.SDK.Schema).Drop("{SCHEMA}")')["status"] == "ok"
        )
        assert (
            scratch.sql_scalar(
                f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='{SCHEMA}'"
            )
            == "0"
        )
        assert (
            scratch.json(f'##class(RAG.SDK.Schema).Drop("{SCHEMA}")')["status"] == "ok"
        )
        assert scratch.json('##class(RAG.SDK.Schema).Drop("a;b")')["status"] == "error"


# ---------------------------------------------------------------- OS-15 (conformance, static)

SDK_FILES = sorted(SDK_DIR.glob("*.cls"))
METHOD = re.compile(
    r"^(?:ClassMethod|Method)\s+(\w+)\((.*?)\)\s*(?:As\s+[\w.%]+)?\s*(\[.*\])?\s*$"
)


def _objectscript_methods(path: Path):
    """Yield (name, params, body_lines) for each non-Python method of a class file."""
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        m = (
            METHOD.match(lines[i].strip())
            if lines[i].startswith(("ClassMethod", "Method"))
            else None
        )
        if m and lines[i + 1].strip() == "{":
            j = i + 2
            while lines[j] != "}":
                j += 1
            if "python" not in (m.group(3) or "").lower():
                yield m.group(1), m.group(2), lines[i + 2 : j]
            i = j
        i += 1


OS_FILES = [p for p in SDK_FILES if p.stem not in ("Embed", "Evaluate")]


@pytest.mark.parametrize("path", OS_FILES, ids=lambda p: p.stem)
def test_methods_follow_the_naming_and_scaffold_rules(path):
    problems = []
    seen = 0
    for name, params, body in _objectscript_methods(path):
        seen += 1
        for param in re.findall(
            r"(?:ByRef\s+|Output\s+)?(\w+)\s+As\s", params + " As "
        ):
            pass
        for p in [x.strip() for x in params.split(",") if x.strip()]:
            pname = re.sub(r"^(ByRef|Output)\s+", "", p).split()[0]
            if not re.fullmatch(r"p[A-Z]\w*", pname):
                problems.append(f"{name}: parameter {pname}")
        code = [l.strip() for l in body if l.strip() and not l.strip().startswith("//")]
        if not code or code[0] != "Set tSC = $$$OK":
            problems.append(f"{name}: first line is {code[:1]}")
        if not code or not code[-1].startswith("Quit "):
            problems.append(f"{name}: last line is {code[-1:]}")
        for line in code:
            for target in re.findall(r"\b(?:Set|For|Catch)\s+(\w+)\s*(?:=|\()", line):
                # Output and ByRef parameters are assigned by name; everything else is a t-local.
                if not re.fullmatch(r"[tp][A-Z]\w*", target):
                    problems.append(f"{name}: local {target}")
            if re.search(r"\$Get\(\s*\w+\.\w+", line):
                problems.append(f"{name}: $Get on an object property")
    assert seen, f"{path.name} has no ObjectScript methods"
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", OS_FILES, ids=lambda p: p.stem)
def test_no_select_top_with_order_by(path):
    text = path.read_text()
    for stmt in re.findall(r"SELECT TOP.*", text):
        assert "ORDER BY" not in stmt, stmt
    assert not re.search(r"SELECT TOP[^\n]*\n[^\n]*ORDER BY", text)
