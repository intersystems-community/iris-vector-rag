"""The settings of an attached table: column mapping and filter set (feature 092, FR-004).

Configuration supplies them under `storage:iris:`; constructor arguments override them
key by key. Nothing here touches a database.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.exceptions import FilterError, VectorStoreConfigurationError
from iris_vector_rag.storage.attached import ColumnMapping, FilterSpec, quote_identifier
from iris_vector_rag.storage.table_shape import ColumnRole


def test_an_unset_mapping_is_not_attached():
    mapping = ColumnMapping.from_settings({}, {})
    assert mapping.attached is False
    assert mapping.declared == {}


def test_configuration_declares_the_columns():
    mapping = ColumnMapping.from_settings(
        {
            "table_name": "MDT.EvidenceChunk",
            "id_col": "EvidenceId",
            "text_col": "Text",
            "embedding_col": "Embedding",
            "title_col": "Title",
            "parent_col": "ResourceId",
        },
        {},
    )
    assert mapping.attached is True
    assert mapping.declared == {
        ColumnRole.IDENTIFIER: "EvidenceId",
        ColumnRole.TEXT: "Text",
        ColumnRole.EMBEDDING: "Embedding",
        ColumnRole.TITLE: "Title",
        ColumnRole.PARENT: "ResourceId",
    }


def test_a_list_identifier_is_composite_and_a_one_item_list_is_not():
    composite = ColumnMapping.from_settings({"id_col": ["DocRef", "Seq"]}, {})
    assert composite.declared[ColumnRole.IDENTIFIER] == ("DocRef", "Seq")
    single = ColumnMapping.from_settings({"id_col": ["DocRef"]}, {})
    assert single.declared[ColumnRole.IDENTIFIER] == "DocRef"


def test_a_comma_separated_identifier_from_the_environment_is_a_list():
    # Environment variables arrive as strings: RAG_STORAGE__IRIS__ID_COL=DocRef,Seq
    mapping = ColumnMapping.from_settings({"id_col": "DocRef, Seq"}, {})
    assert mapping.declared[ColumnRole.IDENTIFIER] == ("DocRef", "Seq")


def test_constructor_arguments_override_configuration_key_by_key():
    mapping = ColumnMapping.from_settings(
        {"id_col": "EvidenceId", "text_col": "Text"}, {"text_col": "Department"}
    )
    assert mapping.declared[ColumnRole.TEXT] == "Department"
    assert mapping.declared[ColumnRole.IDENTIFIER] == "EvidenceId"


def test_an_override_of_none_keeps_the_configured_value():
    mapping = ColumnMapping.from_settings({"text_col": "Text"}, {"text_col": None})
    assert mapping.declared[ColumnRole.TEXT] == "Text"


@pytest.mark.parametrize("bad", ["Text; DROP", "a b", "x'"])
def test_a_column_that_is_not_an_identifier_is_refused(bad):
    with pytest.raises(VectorStoreConfigurationError, match="text_col"):
        ColumnMapping.from_settings({"text_col": bad}, {})


def test_filter_settings_read_lists_and_comma_strings():
    spec = FilterSpec.from_settings(
        {
            "filter_columns": "PatientId, EmbeddingModel",
            "required_filters": ["PatientId"],
        },
        {},
    )
    assert spec.filter_columns == ("PatientId", "EmbeddingModel")
    assert spec.required_filters == ("PatientId",)


def test_a_required_filter_must_also_be_a_filter_column():
    with pytest.raises(FilterError, match="DocDate"):
        FilterSpec.from_settings(
            {"filter_columns": ["PatientId"], "required_filters": ["DocDate"]}, {}
        )


def test_filter_settings_override_like_the_mapping():
    spec = FilterSpec.from_settings(
        {"filter_columns": ["PatientId"]}, {"filter_columns": ["PatientId", "Kind"]}
    )
    assert spec.filter_columns == ("PatientId", "Kind")
    assert spec.required_filters == ()


def test_quote_identifier_quotes_a_safe_name_and_refuses_anything_else():
    assert quote_identifier("Section") == '"Section"'
    with pytest.raises(ValueError):
        quote_identifier('Sec"tion')
