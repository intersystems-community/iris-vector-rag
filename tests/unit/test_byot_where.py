"""Real-column filters compiled to a bound predicate (feature 092, US2: FR-006 to FR-009, FR-020).

`compile_where` turns a search's ``where`` into SQL text and parameters. The text
holds only quoted, checked column names and ``?`` markers; every value is a
parameter. The shapes are the teams' tables as IRIS recorded them.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from iris_vector_rag.exceptions import FilterError, RequiredFilterMissing
from iris_vector_rag.storage.attached import FilterSpec, compile_where
from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.table_shape import ColumnRole, TableShape

CATALOG = Path(__file__).resolve().parents[1] / "fixtures" / "byot" / "catalog"


@pytest.fixture(scope="module")
def stafeta() -> TableShape:
    snapshot = RecordedCatalog.from_fixture_dir(CATALOG).snapshot("BYOTStf", "DocChunk")
    return TableShape.from_snapshot(
        snapshot,
        declared={ColumnRole.IDENTIFIER: ("DocRef", "Seq"), ColumnRole.TEXT: "Text"},
    )


SPEC = FilterSpec(
    filter_columns=("PatientRef", "AuthoredOn", "ReceivedOn", "EmbeddingModel", "Seq"),
    required_filters=("PatientRef",),
)


def test_equality_is_a_bound_comparison(stafeta):
    sql, params = compile_where({"PatientRef": "Patient/2"}, stafeta, SPEC)
    assert sql == '"PatientRef" = ?'
    assert params == ["Patient/2"]


@pytest.mark.parametrize("op", ["=", "<", "<=", ">", ">="])
def test_each_comparison_operator(stafeta, op):
    sql, params = compile_where(
        {"PatientRef": "Patient/2", "Seq": (op, 3)}, stafeta, SPEC
    )
    assert sql == f'"PatientRef" = ? AND "Seq" {op} ?'
    assert params == ["Patient/2", 3]


def test_a_list_is_in(stafeta):
    sql, params = compile_where(
        {"PatientRef": ["Patient/1", "Patient/2"]}, stafeta, SPEC
    )
    assert sql == '"PatientRef" IN (?, ?)'
    assert params == ["Patient/1", "Patient/2"]


def test_an_empty_list_is_refused(stafeta):
    with pytest.raises(FilterError, match="PatientRef"):
        compile_where({"PatientRef": []}, stafeta, SPEC)


def test_dates_and_timestamps_are_cast_to_the_columns_type(stafeta):
    sql, params = compile_where(
        {
            "PatientRef": "Patient/2",
            "AuthoredOn": ("<", "2026-10-05"),
            "ReceivedOn": ("<=", dt.datetime(2026, 10, 5, 12, 0)),
        },
        stafeta,
        SPEC,
    )
    assert sql == (
        '"PatientRef" = ? AND "AuthoredOn" < CAST(? AS DATE) '
        'AND "ReceivedOn" <= CAST(? AS TIMESTAMP)'
    )
    # Dates go in as ISO text, which CAST reads the same on every driver.
    assert params == ["Patient/2", "2026-10-05", "2026-10-05 12:00:00"]


def test_a_date_object_is_iso_text(stafeta):
    _, params = compile_where(
        {"PatientRef": "Patient/2", "AuthoredOn": (">=", dt.date(2026, 9, 1))},
        stafeta,
        SPEC,
    )
    assert params[1] == "2026-09-01"


def test_a_key_outside_filter_columns_names_the_allowed_ones(stafeta):
    with pytest.raises(FilterError) as caught:
        compile_where({"PatientRef": "Patient/2", "Text": "x"}, stafeta, SPEC)
    message = str(caught.value)
    assert "Text" in message and "PatientRef" in message and "AuthoredOn" in message


def test_a_missing_required_filter_is_named(stafeta):
    with pytest.raises(RequiredFilterMissing, match="PatientRef"):
        compile_where({"Seq": 1}, stafeta, SPEC)


def test_no_where_at_all_is_still_missing_the_required_filter(stafeta):
    with pytest.raises(RequiredFilterMissing, match="PatientRef"):
        compile_where(None, stafeta, SPEC)


def test_without_required_filters_no_where_is_no_predicate(stafeta):
    assert compile_where(None, stafeta, FilterSpec(("Seq",), ())) == ("", [])
    assert compile_where({}, stafeta, FilterSpec(("Seq",), ())) == ("", [])


def test_keys_are_matched_without_regard_to_case(stafeta):
    sql, _ = compile_where({"patientref": "Patient/2"}, stafeta, SPEC)
    assert sql == '"PatientRef" = ?'


def test_a_filter_column_the_table_lacks_is_refused(stafeta):
    spec = FilterSpec(("PatientRef", "Ward"), ())
    with pytest.raises(FilterError, match="Ward"):
        compile_where({"Ward": "A"}, stafeta, spec)


def test_an_unknown_operator_is_refused(stafeta):
    with pytest.raises(FilterError, match="LIKE"):
        compile_where({"PatientRef": "Patient/2", "Seq": ("LIKE", "1%")}, stafeta, SPEC)


@pytest.mark.parametrize("value", [{"a": 1}, object(), b"bytes"])
def test_an_unsupported_value_type_is_refused(stafeta, value):
    with pytest.raises(FilterError, match="PatientRef"):
        compile_where({"PatientRef": value}, stafeta, SPEC)


def test_an_injection_attempt_stays_a_parameter(stafeta):
    sql, params = compile_where({"PatientRef": "x' OR 1=1 --"}, stafeta, SPEC)
    assert sql == '"PatientRef" = ?'
    assert params == ["x' OR 1=1 --"]


def test_the_predicate_never_starts_with_a_parenthesis(stafeta):
    # The DB-API miscounts parameters when a WHERE holds a doubled "((" (R2).
    sql, _ = compile_where(
        {"PatientRef": ["Patient/1"], "Seq": ("<", 2)}, stafeta, SPEC
    )
    assert not sql.startswith("(")
    assert "((" not in sql
