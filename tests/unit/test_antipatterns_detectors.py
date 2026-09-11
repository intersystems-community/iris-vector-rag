"""The `catalog-read-outside-shape` detector (feature 082, US4, SC-003).

The detector is the thing that keeps FR-013 true after the feature ships: every
`INFORMATION_SCHEMA` and `%Dictionary` query in the package lives in
`iris_vector_rag/storage/catalog.py`, so a shape is described in one place and a
test can describe a shape without scripting a cursor. A gate is the only way to
keep that from drifting back, and a gate needs its own test -- the detector this
file exercises replaced hand-checking `grep` output.

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


DRIVER_CONNECT = """
def _connect(host, port):
    import iris

    return iris.dbapi.connect(hostname=host, port=port)
"""

DRIVERS_FILE = "iris_vector_rag/session/drivers.py"


def test_a_driver_connect_outside_the_driver_module_is_reported(gate):
    """083 US5: one module opens connections, and the gate is what keeps it one.

    A connect anywhere else skips the socket precheck, the bounded handshake, the
    back-off window and the expired-password recovery -- the four things the v0.14.0
    CI hangs were missing.
    """
    findings = gate.driver_connects(PACKAGE_FILE, DRIVER_CONNECT)

    assert len(findings) == 1, findings
    (finding,) = findings
    assert finding.check == "unguarded-dbapi-connect"
    assert finding.location == f"{PACKAGE_FILE}:5"
    assert "session" in finding.message


def test_the_driver_module_is_where_connects_belong(gate):
    assert gate.driver_connects(DRIVERS_FILE, DRIVER_CONNECT) == []


def test_the_old_connection_module_is_no_longer_exempt(gate):
    """`common/iris_connection.py` used to be the exemption; since 083 it delegates."""
    findings = gate.driver_connects(
        "iris_vector_rag/common/iris_connection.py", DRIVER_CONNECT
    )

    assert [f.location for f in findings] == [
        "iris_vector_rag/common/iris_connection.py:5"
    ]


def test_a_comment_about_connecting_is_not_connecting(gate):
    assert (
        gate.driver_connects(PACKAGE_FILE, "# never call iris.dbapi.connect(...)") == []
    )


def test_the_package_has_exactly_one_module_that_connects(gate):
    """The registered check, against the real tree, holds at zero findings."""
    assert "unguarded-dbapi-connect" in gate.CHECKS
    assert "unguarded-dbapi-connect" in gate.NO_BASELINE

    findings = gate.CHECKS["unguarded-dbapi-connect"]()

    assert findings == [], [f.location for f in findings]
    # And the exempt module really is the one that connects, so the check is not
    # passing because the pattern stopped matching anything.
    assert gate.driver_connects(DRIVERS_FILE, gate.read(ROOT / DRIVERS_FILE)) == []
    assert "dbapi.connect(" in gate.read(ROOT / DRIVERS_FILE)


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
