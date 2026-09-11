"""Rows are values, not driver objects, so they outlive the cursor they came from.

Feature: 083-iris-session (FR-011)
"""

import pytest

from iris_vector_rag.session.results import Rows


def test_rows_are_iterable_indexable_and_sized():
    rows = Rows(columns=("a", "b"), rows=((1, 2), (3, 4)))

    assert len(rows) == 2
    assert rows[0] == (1, 2)
    assert list(rows) == [(1, 2), (3, 4)]
    assert bool(rows) is True


def test_no_rows_is_falsy_and_empty():
    rows = Rows(columns=("a",), rows=())

    assert not rows
    assert len(rows) == 0
    assert list(rows) == []


def test_one_returns_none_rather_than_raising_about_row_objects():
    assert Rows(columns=("a",), rows=()).one() is None


def test_one_returns_the_single_row():
    assert Rows(columns=("a",), rows=((7,),)).one() == (7,)


def test_one_refuses_to_guess_between_two_rows():
    with pytest.raises(ValueError, match="2 rows"):
        Rows(columns=("a",), rows=((1,), (2,))).one()


def test_dicts_zips_the_columns():
    rows = Rows(columns=("doc_id", "text"), rows=(("d1", "hello"),))

    assert rows.dicts() == [{"doc_id": "d1", "text": "hello"}]


def test_rowcount_defaults_to_the_number_of_rows():
    assert Rows(columns=("a",), rows=((1,), (2,))).rowcount == 2


def test_rowcount_can_be_what_the_driver_reported():
    assert Rows(columns=(), rows=(), rowcount=5).rowcount == 5


def test_rows_hold_plain_tuples_even_when_built_from_other_sequences():
    rows = Rows(columns=["a", "b"], rows=[[1, 2]])

    assert rows.columns == ("a", "b")
    assert rows.rows == ((1, 2),)
    assert isinstance(rows[0], tuple)
