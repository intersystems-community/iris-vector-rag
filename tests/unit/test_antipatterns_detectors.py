"""The `catalog-read-outside-shape` and `hardcoded-vector-type` detectors.

The first is feature 082 (US4, SC-003), the second feature 086 (US3). Each detector is
what keeps its feature true after the feature ships: every `INFORMATION_SCHEMA` and
`%Dictionary` query in the package lives in `iris_vector_rag/storage/catalog.py`, and no
new statement names a vector element type. A gate is the only way to keep either from
drifting back, and a gate needs its own test -- the detectors this file exercises replaced
hand-checking `grep` output.

`scripts/gates/` is not an importable package, so the module is loaded from its
path the same way the gate runner does.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts/gates/antipatterns.py"

PACKAGE_FILE = "iris_vector_rag/pipelines/hybrid_graphrag.py"
CATALOG_FILE = "iris_vector_rag/storage/catalog.py"

COLUMN_QUERY = '''
def column_names(cursor, schema, table):
    cursor.execute(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """,
        [schema, table],
    )
    return [row[0] for row in cursor.fetchall()]
'''

CLASS_QUERY = """
def class_name(cursor, schema, table):
    cursor.execute(
        "SELECT ID FROM %Dictionary.CompiledClass "
        "WHERE SqlSchemaName = ? AND SqlTableName = ?",
        [schema, table],
    )
"""

PROSE = '''
"""INFORMATION_SCHEMA reports a VECTOR column as varchar (a driver limitation).

The class view -- %Dictionary.CompiledProperty -- is what says it is a vector.
"""
'''


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("antipatterns_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the module defines a dataclass, and dataclasses
    # resolves a field's annotation through `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_catalog_read_in_a_package_file_is_reported(gate):
    """A column query anywhere but the catalog module is a finding, with its line."""
    findings = gate.catalog_reads(PACKAGE_FILE, COLUMN_QUERY)

    assert len(findings) == 1, findings
    (finding,) = findings
    assert finding.check == "catalog-read-outside-shape"
    # Line 6 is the `FROM INFORMATION_SCHEMA.COLUMNS` line: the location has to be
    # the query, not the top of the file, or the finding is unfixable.
    assert finding.location == f"{PACKAGE_FILE}:6"
    assert "INFORMATION_SCHEMA" in finding.message


def test_dictionary_read_in_a_package_file_is_reported(gate):
    """`%Dictionary` counts too: it is the other half of what a snapshot reads."""
    findings = gate.catalog_reads(PACKAGE_FILE, CLASS_QUERY)

    assert [f.location for f in findings] == [f"{PACKAGE_FILE}:4"]


def test_the_same_code_inside_the_catalog_module_is_not_reported(gate):
    """The owner is exempt: that module is where the queries are supposed to be."""
    assert gate.catalog_reads(CATALOG_FILE, COLUMN_QUERY) == []
    assert gate.catalog_reads(CATALOG_FILE, CLASS_QUERY) == []


def test_prose_about_the_catalog_is_not_reported(gate):
    """Naming `INFORMATION_SCHEMA` in a docstring is not reading it.

    The package explains the varchar-vs-VECTOR reporting quirk in several
    docstrings; a detector that flagged those would be turned off within a day.
    """
    assert gate.catalog_reads(PACKAGE_FILE, PROSE) == []


def test_the_detector_scans_the_package_and_exempts_the_owner(gate):
    """The registered check reads real files, and only package ones."""
    assert "catalog-read-outside-shape" in gate.CHECKS

    findings = gate.CHECKS["catalog-read-outside-shape"]()
    locations = [f.location for f in findings]

    assert locations, "the package still has known catalog reads to hold at zero-new"
    assert all(loc.startswith(gate.PKG) for loc in locations), locations
    assert not [loc for loc in locations if loc.startswith(CATALOG_FILE)], locations
    # The catalog module does read the catalog -- if it were not exempt it would be
    # the largest cluster of findings in the package.
    assert gate.catalog_reads(CATALOG_FILE, gate.read(ROOT / CATALOG_FILE)) == []


# ---------------------------------------------------------------------------
# hardcoded-vector-type (feature 086, US3)
#
# IRIS refuses to compare vectors of different element types (SQLCODE -259), so a
# statement that names one is a statement that decides, on its own, which tables the
# rest of the repo can search. The type is declared once -- CANONICAL_VECTOR_TYPE in
# iris_vector_rag/storage/table_shape.py, and Parameter VECTORTYPE in the SDK classes
# -- and resolved from the column when reading an existing one. This detector holds
# the known literal sites at their current count so a new one cannot ship unnoticed.
# ---------------------------------------------------------------------------

SDK_CLASS = "iris_src/src/RAG/SDK/Pipeline.cls"
SCHEMA_SQL = "sql/schema.sql"

PY_TO_VECTOR = """
def store(cursor, table, vector):
    cursor.execute(
        f"INSERT INTO {table} (doc_id, embedding) VALUES (?, TO_VECTOR(?, FLOAT, 384))",
        [doc_id, vector],
    )
"""

CLS_TO_VECTOR = """
ClassMethod AddDocument(id As %String, embedding As %String) As %String
{
    Set sql = "INSERT INTO RAG.SourceDocuments (doc_id, embedding) VALUES (?,TO_VECTOR(?,DOUBLE,"_dim_"))"
    Return sql
}
"""

SQL_CREATE_TABLE = """CREATE TABLE IF NOT EXISTS RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    text_content LONGVARCHAR,
    embedding VECTOR(FLOAT, 384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

SQL_ALTER_TABLE = """
ALTER TABLE RAG.EntityRelationships ADD COLUMN relation_embedding VECTOR(FLOAT, 384)
"""

# The three prose forms that really are in the tree, one per shape a naive detector
# gets wrong. Reporting any of these is how a gate gets switched off.
DOCSTRING_BACKTICKED = '''
"""Two IRIS facts shape the design.

* `INFORMATION_SCHEMA` reports a `VECTOR(FLOAT, 384)` column as `varchar`. That is
  a driver limitation, never a schema defect.
"""
'''

DOCSTRING_BARE = '''
class RelationEmbeddingStore:
    """Manages relation embeddings in RAG.EntityRelationships.

    Extends the existing KG table with a VECTOR(FLOAT, 384) column and HNSW
    index so that relationship descriptions can be retrieved by semantic
    similarity at query time.
    """
'''

HASH_COMMENT = """
    # Tables whose canonical DDL lives in SchemaManager (doc_id VARCHAR primary key,
    # embedding VECTOR(FLOAT, n), ...). The inline DDL further down predates it.
    _SCHEMA_MANAGER_TABLES = ("SourceDocuments",)
"""


def test_to_vector_in_python_is_reported(gate):
    """The commonest form: a writer naming the element type it converts with."""
    findings = gate.hardcoded_vector_types(PACKAGE_FILE, PY_TO_VECTOR)

    assert len(findings) == 1, findings
    (finding,) = findings
    assert finding.check == "hardcoded-vector-type"
    assert finding.location == f"{PACKAGE_FILE}:4"
    assert "FLOAT" in finding.message


def test_to_vector_in_objectscript_is_reported(gate):
    """ObjectScript builds its SQL by concatenation; the literal is still a literal."""
    findings = gate.hardcoded_vector_types(SDK_CLASS, CLS_TO_VECTOR)

    assert [f.location for f in findings] == [f"{SDK_CLASS}:4"], findings
    assert "DOUBLE" in findings[0].message


def test_create_table_column_declaration_is_reported(gate):
    """A creation path is the other half: the column's own declared type."""
    findings = gate.hardcoded_vector_types(SCHEMA_SQL, SQL_CREATE_TABLE)

    assert [f.location for f in findings] == [f"{SCHEMA_SQL}:4"], findings


def test_alter_table_add_column_is_reported(gate):
    """ALTER TABLE creates a vector column too, and its line ends in no comma."""
    findings = gate.hardcoded_vector_types(SCHEMA_SQL, SQL_ALTER_TABLE)

    assert [f.location for f in findings] == [f"{SCHEMA_SQL}:2"], findings


def test_backticked_mention_in_a_docstring_is_not_reported(gate):
    """`iris_vector_rag/storage/table_shape.py:16` — the varchar-reporting note."""
    assert gate.hardcoded_vector_types(PACKAGE_FILE, DOCSTRING_BACKTICKED) == []


def test_bare_prose_in_a_docstring_is_not_reported(gate):
    """`iris_vector_rag/storage/relation_embedding_store.py:35` — no backticks at all."""
    assert gate.hardcoded_vector_types(PACKAGE_FILE, DOCSTRING_BARE) == []


def test_hash_comment_describing_ddl_is_not_reported(gate):
    """`iris_vector_rag/validation/orchestrator.py:276` — a comment quoting the DDL.

    This one has the shape of a column declaration, `embedding VECTOR(FLOAT, n)`, and
    is still only a comment about one.
    """
    assert gate.hardcoded_vector_types(PACKAGE_FILE, HASH_COMMENT) == []


def test_the_canonical_declaration_itself_is_not_reported(gate):
    """Declaring the type once is the fix, not the bug.

    Neither `CANONICAL_VECTOR_TYPE = "FLOAT"` nor the SDK's `Parameter VECTORTYPE`
    may be a finding, or the detector would argue against its own remedy.
    """
    assert (
        gate.hardcoded_vector_types(PACKAGE_FILE, 'CANONICAL_VECTOR_TYPE = "FLOAT"\n')
        == []
    )
    assert (
        gate.hardcoded_vector_types(SDK_CLASS, 'Parameter VECTORTYPE = "FLOAT";\n')
        == []
    )


def test_the_detector_is_registered_and_scans_shipped_code(gate):
    """Registered, baselined, and scoped to the files that create or write vectors."""
    assert "hardcoded-vector-type" in gate.CHECKS
    # Its purpose is to hold the known sites, so it must NOT be a zero-tolerance check.
    assert "hardcoded-vector-type" not in gate.NO_BASELINE

    locations = [f.location for f in gate.CHECKS["hardcoded-vector-type"]()]

    assert (
        locations
    ), "the repo still has known literal element types to hold at zero-new"
    assert all(
        loc.startswith((gate.PKG, "sql/", "iris_src/")) for loc in locations
    ), locations
    # Not tests/ (fixtures create legacy shapes on purpose) and not .md (the
    # constitution's own example would become a finding against the constitution).
    assert not [loc for loc in locations if loc.startswith("tests/")], locations
    assert not [loc for loc in locations if loc.endswith(".md")], locations


# ---------------------------------------------------------------------------
# inline-standard-ddl (feature 086, US4)
#
# The detector was written against `.py` only, so the two files that hold the most DDL
# in the repo -- `sql/schema.sql` and the SDK's `Schema.cls` -- were never walked, and
# an f-string table name made a CREATE TABLE invisible. A gate nobody can see through
# is worse than no gate: it reads as coverage.
# ---------------------------------------------------------------------------

SDK_SCHEMA_CLASS = "iris_src/src/RAG/SDK/Schema.cls"
NON_OWNER_SQL = "iris_vector_rag/common/db_init_complete.sql"

SQL_FILE_DDL = """CREATE TABLE IF NOT EXISTS RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    text_content LONGVARCHAR
)
"""

CLS_FILE_DDL = """
ClassMethod GetDefaultDDL(dim As %Integer) As %String
{
    Set d1 = "CREATE TABLE IF NOT EXISTS RAG.SourceDocuments (doc_id VARCHAR(64) NOT NULL, PRIMARY KEY (doc_id))"
    Return d1
}
"""

CLS_FILE_DDL_VARIABLE_SCHEMA = """
ClassMethod GetDefaultDDL(dim As %Integer, schema As %String) As %String
{
    Set d1 = "CREATE TABLE IF NOT EXISTS "_schema_".SourceDocuments (doc_id VARCHAR(64) NOT NULL)"
    Return d1
}
"""

PY_VARIABLE_TABLE_DDL = '''
def ensure(self, cursor):
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {self.entities_table} (
            entity_id VARCHAR(255) PRIMARY KEY
        )
        """
    )
'''

PY_BESPOKE_TABLE_DDL = """
def ensure_cache(cursor):
    cursor.execute("CREATE TABLE IF NOT EXISTS RAG.LLMCacheEntries (key VARCHAR(64))")
"""

# The three forms really in the tree that say "create table" without creating one:
# vector_store_iris.py:426, orchestrator.py:339 and validator.py:175 are all messages
# about a failure. Reporting a log line as DDL is how the widened detector would have
# earned its own suppression.
PY_LOG_MESSAGES = """
def create(self, cursor, name):
    try:
        cursor.execute(self.ddl)
    except Exception as create_error:
        self.logger.error(f"Failed to create table {self.table_name}: {create_error}")
        raise SchemaError(f"Create table {name}", create_error)
"""

PY_DDL_COLUMN_LIST_NEXT_LINE = '''
def ensure(cursor):
    cursor.execute(
        """
        CREATE TABLE RAG.Communities
        (
            community_id VARCHAR(255) PRIMARY KEY
        )
        """
    )
'''


def test_ddl_in_a_sql_file_is_reported(gate):
    """`sql/**/*.sql` is where DDL lives; not walking it was the first blind spot."""
    findings = gate.standard_ddl(NON_OWNER_SQL, SQL_FILE_DDL)

    assert len(findings) == 1, findings
    (finding,) = findings
    assert finding.check == "inline-standard-ddl"
    assert finding.location == f"{NON_OWNER_SQL}:1"
    assert "SourceDocuments" in finding.message


def test_ddl_in_an_objectscript_class_is_reported(gate):
    """A `.cls` builds its DDL as a string; it still creates the table."""
    findings = gate.standard_ddl(SDK_SCHEMA_CLASS, CLS_FILE_DDL)

    assert [f.location for f in findings] == [f"{SDK_SCHEMA_CLASS}:4"], findings


def test_objectscript_ddl_with_a_concatenated_schema_is_reported(gate):
    """`"_schema_".SourceDocuments` — the schema varies, the table is still standard."""
    findings = gate.standard_ddl(SDK_SCHEMA_CLASS, CLS_FILE_DDL_VARIABLE_SCHEMA)

    assert [f.location for f in findings] == [f"{SDK_SCHEMA_CLASS}:4"], findings
    assert "SourceDocuments" in findings[0].message


def test_ddl_with_an_interpolated_table_name_is_reported_as_unresolved(gate):
    """The second blind spot: `{self.entities_table}` hid the CREATE TABLE entirely.

    The finding has to say the detector cannot resolve the name rather than claim the
    table is standard — the name may well be configured to something else.
    """
    findings = gate.standard_ddl(PACKAGE_FILE, PY_VARIABLE_TABLE_DDL)

    assert len(findings) == 1, findings
    (finding,) = findings
    assert finding.location == f"{PACKAGE_FILE}:5"
    assert "cannot" in finding.message.lower(), finding.message
    assert "self.entities_table" in finding.message


def test_a_message_about_creating_a_table_is_not_ddl(gate):
    """ "Failed to create table X" creates nothing; a column list is what DDL has."""
    assert gate.standard_ddl(PACKAGE_FILE, PY_LOG_MESSAGES) == []


def test_ddl_with_the_column_list_on_the_next_line_is_reported(gate):
    """The column list may open on the following line — still one statement."""
    findings = gate.standard_ddl(PACKAGE_FILE, PY_DDL_COLUMN_LIST_NEXT_LINE)

    assert [f.location for f in findings] == [f"{PACKAGE_FILE}:5"], findings
    assert "Communities" in findings[0].message


def test_ddl_for_a_table_that_is_not_standard_is_not_reported(gate):
    """A module's own table is its own business; only the shared five are owned."""
    assert gate.standard_ddl(PACKAGE_FILE, PY_BESPOKE_TABLE_DDL) == []


def test_the_owner_files_are_exempt_and_reachable(gate):
    """Every exemption in `DDL_OWNER` must be a file the detector actually walks.

    `sql/schema.sql` was exempt while `.sql` files were not scanned at all — an
    exemption for a file that was never a candidate, which read as deliberate scoping
    and was in fact a no-op.
    """
    walked = {gate.rel(p) for p in gate.inline_standard_ddl_files()}

    assert gate.DDL_OWNER <= walked, gate.DDL_OWNER - walked
    for owner in gate.DDL_OWNER:
        assert gate.standard_ddl(owner, SQL_FILE_DDL) == [], owner


def test_the_widened_detector_sees_sql_and_cls_in_the_tree(gate):
    """The registered check reports the DDL the repo really carries outside SchemaManager."""
    # A location is `path:line`, so the suffix test is on the path half.
    paths = [f.location.split(":")[0] for f in gate.CHECKS["inline-standard-ddl"]()]

    assert any(p.endswith(".sql") for p in paths), paths
    assert any(p.endswith(".cls") for p in paths), paths
    assert not [p for p in paths if p in gate.DDL_OWNER], paths
