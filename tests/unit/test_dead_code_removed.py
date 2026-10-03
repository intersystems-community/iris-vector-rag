"""The code the remediation plan (feature remove-dead-code) deleted stays deleted.

Each piece had no production caller and carried a hazard: `IRISStorage` could drop a
populated table and formatted filter keys into SQL, `IrisDevToolsBridge` ran
`DROP NAMESPACE` with no container guard, the regex entity extractor matched any two
words as a `Person`, and the pure-SQL `insert_vector` fallback had a signature that
disagreed with its call site. Decision records G3 and G4 settled the fate: delete, with
no deprecation release.
"""

import importlib
import inspect
from pathlib import Path

import pytest

import iris_vector_rag
from iris_vector_rag.common import db_vector_utils, iris_sql_utils

PACKAGE_ROOT = Path(iris_vector_rag.__file__).parent


@pytest.mark.parametrize(
    "module",
    [
        "iris_vector_rag.storage.enterprise_storage",
        "iris_vector_rag.testing.iris_devtools_bridge",
        "iris_vector_rag.embeddings.entity_extractor",
    ],
)
def test_deleted_module_is_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_storage_package_no_longer_exports_iris_storage():
    import iris_vector_rag.storage as storage

    assert not hasattr(storage, "IRISStorage")
    assert "IRISStorage" not in storage.__all__


def test_testing_package_no_longer_exports_the_bridge():
    import iris_vector_rag.testing as testing

    assert not hasattr(testing, "IrisDevToolsBridge")
    assert "IrisDevToolsBridge" not in getattr(testing, "__all__", [])


def test_db_vector_utils_has_no_pure_sql_fallbacks():
    """iris-vector-graph is a hard dependency, so the module re-exports it and nothing else."""
    source = inspect.getsource(db_vector_utils)
    assert "except ImportError" not in source
    for name in ("insert_vector", "create_hnsw_index", "vector_similarity_search"):
        assert getattr(db_vector_utils, name).__module__ == (
            "iris_vector_graph.dbapi_utils"
        )


def test_check_table_exists_swallowing_helper_is_gone():
    """It returned False for a dead connection; nothing but its own test called it."""
    assert not hasattr(iris_sql_utils.IRISSQLUtils, "check_table_exists")
    assert not hasattr(
        iris_sql_utils.IRISSQLUtils, "_check_table_exists_with_connection"
    )


def test_no_remaining_source_names_a_deleted_module():
    needles = (
        "enterprise_storage",
        "iris_devtools_bridge",
        "embeddings.entity_extractor",
    )
    offenders = [
        f"{path.relative_to(PACKAGE_ROOT)}: {needle}"
        for path in PACKAGE_ROOT.rglob("*.py")
        for needle in needles
        if needle in path.read_text(encoding="utf-8")
        and path.name != Path(__file__).name
    ]
    assert offenders == []
