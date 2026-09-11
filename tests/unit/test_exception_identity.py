"""One exception class per failure, whichever module a caller imports it from.

`VectorStoreConnectionError`, `VectorStoreDataError` and `VectorStoreCLOBError`
were each defined twice — once in `iris_vector_rag/exceptions.py` and again in
`iris_vector_rag/core/exceptions.py`. The two definitions were unrelated classes
with the same name, so `except` on one did not catch the other: the storage layer
raises the top-level `VectorStoreConnectionError` while `iris_vector_rag.core`
exported the core one, and `clob_handler` raised the core `VectorStoreCLOBError`
while `vector_store_iris` imported the top-level one. These tests pin the classes
to a single identity and keep the hierarchy a caller can catch broadly.
"""

from pathlib import Path

import pytest

import iris_vector_rag.core as core_pkg
import iris_vector_rag.core.exceptions as core_exc
import iris_vector_rag.exceptions as top_exc

DUPLICATED = [
    "VectorStoreError",
    "VectorStoreConnectionError",
    "VectorStoreDataError",
    "VectorStoreCLOBError",
]


@pytest.mark.parametrize("name", DUPLICATED)
def test_core_and_top_level_name_the_same_class(name):
    """The two modules must expose one class, not two classes with one name."""
    assert getattr(core_exc, name) is getattr(top_exc, name), (
        f"{name} is defined twice: iris_vector_rag.exceptions.{name} and "
        f"iris_vector_rag.core.exceptions.{name} are different classes, so "
        f"`except {name}` catches only the one the caller happened to import. "
        f"Define it once in iris_vector_rag/exceptions.py and re-export it."
    )


@pytest.mark.parametrize("name", DUPLICATED)
def test_the_core_package_reexports_that_same_class(name):
    """`from iris_vector_rag.core import ...` must give the class that is raised."""
    assert getattr(core_pkg, name) is getattr(top_exc, name), (
        f"iris_vector_rag.core.{name} is not the class the library raises, so "
        f"catching it around a vector store call silently misses the error."
    )


def test_rag_exception_is_one_class_too():
    """Pipelines import RAGException from core; it must stay the same class."""
    assert core_exc.RAGException is top_exc.RAGException


@pytest.mark.parametrize(
    "name",
    ["VectorStoreConnectionError", "VectorStoreDataError", "VectorStoreCLOBError"],
)
def test_every_vector_store_error_is_catchable_by_the_base(name):
    """`except VectorStoreError` must catch each of them, as core's base did."""
    cls = getattr(top_exc, name)
    assert issubclass(cls, top_exc.VectorStoreError), (
        f"{name} does not subclass VectorStoreError, so code that caught the "
        f"core hierarchy broadly stops catching it."
    )
    assert issubclass(top_exc.VectorStoreError, top_exc.RAGException), (
        "VectorStoreError must stay under RAGException: that is what "
        "iris_vector_rag/core/exceptions.py declared before the classes merged."
    )


def test_the_storage_layer_error_is_catchable_from_the_core_package():
    """What vector_store_iris raises, a core-package importer must be able to catch."""
    from iris_vector_rag.core import VectorStoreConnectionError as from_core
    from iris_vector_rag.storage.vector_store_iris import (
        VectorStoreConnectionError as from_storage,
    )

    with pytest.raises(from_core):
        raise from_storage("Failed to get IRIS connection: refused")


def test_the_clob_handler_error_is_catchable_from_the_top_level_module():
    """clob_handler raised the core class; vector_store_iris catches the top-level one."""
    from iris_vector_rag.core.exceptions import VectorStoreCLOBError as from_core
    from iris_vector_rag.exceptions import VectorStoreCLOBError as from_top

    with pytest.raises(from_top):
        raise from_core("Failed to read CLOB stream: closed")


def test_details_survive_the_merge():
    """The richer top-level signature is the one that is kept."""
    err = top_exc.VectorStoreConnectionError("no route", host="localhost", port=51972)

    assert err.message == "no route"
    assert err.details == {"host": "localhost", "port": 51972}
    assert "host=localhost" in str(err)


def test_core_exceptions_defines_no_duplicate_of_its_own():
    """The re-export must be a re-export: no class statement left behind."""
    source = Path(core_exc.__file__).read_text()

    for name in DUPLICATED:
        assert f"class {name}" not in source, (
            f"core/exceptions.py still defines {name}; it should import it from "
            f"iris_vector_rag.exceptions instead."
        )
