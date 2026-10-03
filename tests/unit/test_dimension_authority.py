"""One embedding dimension, from the active backend (G6 point 5).

ST-08  the constructor argument applies to the first SchemaManager too
ST-09  tables, migrations and the ivg engine all read the same value
ST-16  the relation-embedding table has no hardcoded 384
"""

from unittest.mock import MagicMock

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.embeddings.manager import EmbeddingManager
from iris_vector_rag.exceptions import EmbeddingError
from iris_vector_rag.storage.schema_manager import SchemaManager

WIDTH = 12  # nothing in the code base defaults to this


def _config(embeddings=None):
    cfg = MagicMock(spec=ConfigurationManager)
    cloud = MagicMock()
    cloud.vector.vector_dimension = 384  # the old source; must no longer decide
    cfg.get_cloud_config.return_value = cloud
    cfg.get_schema_prefix.return_value = "RAG"
    cfg.get_embedding_config.return_value = {"dimension": 384}
    values = {
        "embedding_model.name": "sentence-transformers/all-MiniLM-L6-v2",
        "storage:iris": {},
        "embeddings": (
            embeddings
            if embeddings is not None
            else {"provider": "stub", "dimension": WIDTH}
        ),
        "embedding:dimension": 384,
    }
    cfg.get.side_effect = lambda key, default=None: values.get(key, default)
    return cfg


def _schema(cfg=None, **kwargs):
    return SchemaManager(MagicMock(spec=ConnectionManager), cfg or _config(), **kwargs)


@pytest.fixture(autouse=True)
def _reset_config_loaded():
    saved = SchemaManager._config_loaded
    yield
    SchemaManager._config_loaded = saved


@pytest.mark.parametrize("already_loaded", [False, True])
def test_tables_take_the_backends_width_not_the_cloud_default(already_loaded):
    SchemaManager._config_loaded = already_loaded
    schema = _schema()
    assert schema.base_embedding_dimension == WIDTH
    for table in ("SourceDocuments", "DocumentChunks", "Entities"):
        assert schema.get_vector_dimension(table) == WIDTH


def test_entity_relationships_has_no_hardcoded_384():
    assert _schema().get_vector_dimension("EntityRelationships") == WIDTH


@pytest.mark.parametrize("already_loaded", [False, True])
def test_constructor_argument_applies_on_the_first_instance_too(already_loaded):
    """ST-08: only the branch taken when the config was already loaded used to honour it."""
    SchemaManager._config_loaded = already_loaded
    schema = _schema(base_embedding_dimension=64)
    assert schema.base_embedding_dimension == 64
    assert schema.get_vector_dimension("SourceDocuments") == 64


def test_construction_does_not_load_a_model():
    cfg = _config({"primary_backend": "sentence_transformers", "fallback_backends": []})
    schema = _schema(cfg)  # would raise if the constructor resolved the dimension
    assert schema._initialized is False


def test_a_dimension_nobody_can_determine_raises_instead_of_defaulting(monkeypatch):
    def refuse(name, device="cpu"):
        raise OSError("offline")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", refuse)
    schema = _schema(
        _config({"primary_backend": "sentence_transformers", "fallback_backends": []})
    )
    with pytest.raises(EmbeddingError, match="dimension"):
        schema.get_vector_dimension("SourceDocuments")


def test_create_source_documents_uses_the_authority():
    schema = _schema()
    cursor = MagicMock()
    assert schema._create_source_documents_table(cursor) is True
    sql = cursor.execute.call_args[0][0]
    assert f", {WIDTH})" in sql and "384" not in sql


def test_create_document_chunks_uses_the_authority():
    schema = _schema()
    cursor = MagicMock()
    assert schema._create_document_chunks_table(cursor) is True
    sql = cursor.execute.call_args_list[0][0][0]
    assert f", {WIDTH})" in sql and "384" not in sql


class _EightWide:
    def __init__(self, name, device="cpu"):
        pass

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, **_):
        return [[0.0] * 8 for _ in texts]


def _real_backend_config():
    return _config(
        {"primary_backend": "sentence_transformers", "fallback_backends": []}
    )


def test_vector_dimension_environment_variable_is_a_checked_override(monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _EightWide)
    monkeypatch.setenv("VECTOR_DIMENSION", "1024")
    manager = EmbeddingManager(_real_backend_config())
    with pytest.raises(EmbeddingError, match="VECTOR_DIMENSION.*1024.*8"):
        manager.get_embedding_dimension()


def test_vector_dimension_environment_variable_that_agrees_is_fine(monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _EightWide)
    monkeypatch.setenv("VECTOR_DIMENSION", "8")
    assert EmbeddingManager(_real_backend_config()).get_embedding_dimension() == 8
