"""The section filter predicate: a bitmap-indexed column, or the metadata JSON LIKE.

Pure string building, no database. The column form is what the opt-in `section_label`
column uses; the metadata form must stay exactly what `_metadata_filter_predicate`
has always produced for the `section` key.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.storage.text_search import section_filter_predicate

META = "metadata"


def test_column_form_is_an_equality_on_the_column():
    assert section_filter_predicate("hpi", column="section_label") == (
        "section_label = 'hpi'"
    )


def test_quotes_in_the_label_are_doubled_in_the_column_form():
    assert section_filter_predicate("patient's hpi", column="section_label") == (
        "section_label = 'patient''s hpi'"
    )


def test_an_identifier_that_is_not_plain_is_refused():
    with pytest.raises(ValueError):
        section_filter_predicate("hpi", column="a; DROP")


def test_the_column_name_is_checked_before_any_sql_is_built():
    with pytest.raises(ValueError):
        section_filter_predicate("hpi", column="section_label) OR (1=1")


def test_metadata_form_matches_the_section_key_as_json_with_and_without_space():
    predicate = section_filter_predicate("hpi", metadata_expr=META)

    assert predicate == (
        f'({META} LIKE \'%"section":"hpi"%\' OR {META} LIKE \'%"section": "hpi"%\')'
    )


def test_metadata_form_doubles_quotes_in_the_label():
    predicate = section_filter_predicate("patient's hpi", metadata_expr=META)

    assert "patient''s hpi" in predicate
    assert "patient's hpi" not in predicate


def test_metadata_form_needs_the_metadata_expression():
    with pytest.raises(ValueError):
        section_filter_predicate("hpi")


def test_the_metadata_form_is_the_store_predicate_for_the_section_key():
    from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

    assert section_filter_predicate("hpi", metadata_expr=META) == (
        IRISVectorStore._metadata_filter_predicate({"section": "hpi"}, META)
    )
