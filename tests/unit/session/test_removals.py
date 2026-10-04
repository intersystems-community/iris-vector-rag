"""What 0.15 deletes outright, and the names that stopped colliding.

A re-export nobody calls is not an API, it is a second answer to a question that
already had one. Those go now rather than after a deprecation cycle nobody would
read (FR-016). The renames exist so that "ConnectionManager" names one thing.

Feature: 083-iris-session (FR-016, SC-001, US5)
"""

import ast
import importlib
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[3] / "iris_vector_rag"


GONE = [
    ("iris_vector_rag.common.iris_connection", "IRISConnectionPool"),
    ("iris_vector_rag.common.iris_connection", "_PooledConnection"),
    ("iris_vector_rag.common.iris_connection_manager", "get_iris_connection_old"),
    ("iris_vector_rag.common.connection_pool", "get_pool"),
    ("iris_vector_rag.common.connection_pool", "initialize_pool"),
    ("iris_vector_rag.common.connection_pool", "close_pool"),
    ("iris_vector_rag.common.connection_manager", "ConnectionManager"),
]


@pytest.mark.parametrize(
    "module_name,attribute", GONE, ids=[f"{m.rsplit('.', 1)[-1]}.{a}" for m, a in GONE]
)
def test_the_name_is_gone(module_name, attribute):
    module = importlib.import_module(module_name)

    assert not hasattr(module, attribute), (
        f"{module_name}.{attribute} still exists; it had no caller and a second "
        "answer to 'how do I get a connection' is the bug this feature removes"
    )


def test_the_third_get_iris_connection_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("iris_vector_rag.common.iris_connector")

    assert not (PACKAGE / "common" / "iris_connector.py").exists()


def test_the_legacy_manager_is_reachable_under_its_new_name():
    from iris_vector_rag.common import connection_manager

    assert isinstance(connection_manager.LegacyConnectionManager, type)


def _classes_in_package():
    """Every class the package defines, as name -> the files defining it."""
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a file we do not own
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found.setdefault(node.name, []).append(
                    str(path.relative_to(PACKAGE.parent))
                )
    return found


def test_no_two_connection_classes_share_a_name():
    """SC-001: a name in a traceback must identify one class.

    Three unrelated `ConnectionManager` classes is how a reader ends up debugging
    the wrong file.
    """
    classes = _classes_in_package()
    collisions = {
        name: files
        for name, files in classes.items()
        if len(files) > 1
        and any(
            name.endswith(suffix)
            for suffix in ("Manager", "Pool", "Session", "Driver", "Connection")
        )
    }

    assert collisions == {}, f"connection classes sharing a name: {collisions}"


def test_the_pool_the_api_keeps_is_the_only_one_left():
    from iris_vector_rag.common import connection_pool

    assert isinstance(connection_pool.IRISConnectionPool, type)

    exported = importlib.import_module("iris_vector_rag.common")
    assert exported.IRISConnectionPool is connection_pool.IRISConnectionPool, (
        "iris_vector_rag.common must re-export the pool that survives, not the "
        "deleted one"
    )


def test_no_module_in_the_library_connects_with_the_driver_itself():
    """FR-018: `session/drivers.py` is the only place that calls a driver connect."""
    offenders = []
    allowed = {"iris_vector_rag/session/drivers.py"}
    patterns = ("iris.dbapi.connect(", "iris.createConnection(", "iris_dbapi.connect(")

    for path in sorted(PACKAGE.rglob("*.py")):
        relative = str(path.relative_to(PACKAGE.parent))
        if relative in allowed:
            continue
        text = path.read_text()
        for pattern in patterns:
            if pattern in text:
                offenders.append(f"{relative}: {pattern}")

    assert offenders == [], (
        "a direct driver connect outside session/drivers.py skips the socket "
        f"precheck and the bounded handshake: {offenders}"
    )
