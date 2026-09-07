"""
Unit tests: metadata `filter` must narrow the vector search in SQL.

Background (opsreview upstream report, docs/upstream/ivr-metadata-filter-pushdown.md):
`similarity_search_by_embedding` built a LIKE predicate for the filter and then
discarded it, ranking the whole corpus, fetching top_k * 5 and filtering in
Python. On a shared table a small tenant's documents fall outside the global
page and are silently lost.

Contract asserted here:
- the filter predicate is present in the executed vector-search SQL
- the page size is top_k, not a fixed 5x over-fetch
- stream (LONGVARCHAR) metadata columns are wrapped in SUBSTRING, because
  LIKE on a stream column silently matches nothing in IRIS
- the Python exact-match check stays as a backstop for LIKE false positives,
  and only when it drops rows is a single wider page fetched
- non-string values are matched unquoted (JSON numbers / booleans)
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

DIM = 4
QUERY = [0.1, 0.2, 0.3, 0.4]
Row = Tuple[str, str, float]


class FakeCursor:
    """Records executed SQL and answers the store's queries deterministically."""

    def __init__(
        self,
        rows_for_page: Callable[[int], List[Row]],
        metadata: Dict[str, dict],
        metadata_type: str = "varchar",
    ):
        self.executed: List[Tuple[str, object]] = []
        self._rows_for_page = rows_for_page
        self._metadata = metadata
        self._metadata_type = metadata_type
        self._pending: list = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        flat = " ".join(sql.split()).upper()
        if "VECTOR_DOT_PRODUCT" in flat:
            top = int(re.search(r"SELECT TOP (\d+)", flat).group(1))
            self._pending = list(self._rows_for_page(top))
        elif "INFORMATION_SCHEMA.COLUMNS" in flat and "DATA_TYPE" in flat:
            self._pending = [(self._metadata_type,)]
        elif "WHERE DOC_ID IN" in flat:
            self._pending = [
                (d, json.dumps(self._metadata[d]))
                for d in (params or [])
                if d in self._metadata
            ]
        elif "%DICTIONARY.COMPILEDPROPERTY" in flat:
            self._pending = [None]
        else:  # COUNT(*) and other bookkeeping
            self._pending = [(0,)]

    def fetchall(self):
        rows, self._pending = self._pending, []
        return rows

    def fetchone(self):
        return self._pending[0] if self._pending else None

    def close(self):
        pass

    # --- helpers for assertions -------------------------------------------
    def vector_sqls(self) -> List[str]:
        return [s for s, _ in self.executed if "VECTOR_DOT_PRODUCT" in s.upper()]


def make_store(cursor: FakeCursor) -> IRISVectorStore:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    cm = MagicMock(spec=ConnectionManager)
    cm.get_connection.return_value = conn

    cfg = MagicMock(spec=ConfigurationManager)

    def cfg_get(key, default=None):
        if key == "embedding_model.name":
            return "sentence-transformers/all-MiniLM-L6-v2"
        if key == "storage:iris":
            return {}
        return default

    cfg.get.side_effect = cfg_get
    cfg.get_embedding_config.return_value = {"dimension": DIM}

    sm = MagicMock(spec=SchemaManager)
    sm.get_vector_dimension.return_value = DIM
    sm.get_current_schema_config.return_value = {
        "id_column": "doc_id",
        "text_column": "text_content",
        "metadata_column": "metadata",
    }
    return IRISVectorStore(cm, cfg, schema_manager=sm)


def _page(ids: List[str]) -> Callable[[int], List[Row]]:
    return lambda top: [
        (i, f"text {i}", 1.0 - n / 100) for n, i in enumerate(ids[:top])
    ]


@pytest.mark.unit
class TestFilterPushdown:
    def test_filter_predicate_is_in_sql_and_page_is_top_k(self):
        cursor = FakeCursor(
            _page(["d1", "d2"]),
            {"d1": {"source": "acme"}, "d2": {"source": "acme"}},
        )
        store = make_store(cursor)

        results = store.similarity_search_by_embedding(
            QUERY, top_k=5, filter={"source": "acme"}
        )

        sqls = cursor.vector_sqls()
        assert len(sqls) == 1, "one page is enough when the SQL predicate is exact"
        sql = sqls[0]
        assert '"source":"acme"' in sql and '"source": "acme"' in sql, sql
        assert " LIKE " in sql.upper(), "filter must be applied as a SQL predicate"
        assert re.search(
            r"SELECT TOP 5\b", sql
        ), f"page must be top_k, not an over-fetch: {sql}"
        assert [d.id for d, _ in results] == ["d1", "d2"]

    def test_no_filter_means_no_predicate(self):
        cursor = FakeCursor(_page(["d1"]), {"d1": {}})
        store = make_store(cursor)

        store.similarity_search_by_embedding(QUERY, top_k=3)

        sql = cursor.vector_sqls()[0]
        assert "LIKE" not in sql.upper()
        assert re.search(r"SELECT TOP 3\b", sql)

    def test_stream_metadata_column_is_wrapped_in_substring(self):
        cursor = FakeCursor(
            _page(["d1"]), {"d1": {"source": "acme"}}, metadata_type="longvarchar"
        )
        store = make_store(cursor)

        store.similarity_search_by_embedding(QUERY, top_k=5, filter={"source": "acme"})

        sql = cursor.vector_sqls()[0]
        assert re.search(
            r"SUBSTRING\(\s*metadata\s*,\s*1\s*,\s*\d+\s*\)\s+LIKE", sql, re.I
        ), sql

    def test_varchar_metadata_column_is_used_directly(self):
        cursor = FakeCursor(
            _page(["d1"]), {"d1": {"source": "acme"}}, metadata_type="varchar"
        )
        store = make_store(cursor)

        store.similarity_search_by_embedding(QUERY, top_k=5, filter={"source": "acme"})

        assert "SUBSTRING" not in cursor.vector_sqls()[0].upper()

    def test_backstop_drops_like_false_positive_and_widens_once(self):
        # d3 matches the LIKE pattern only through a nested key; the exact check must drop it.
        metadata = {
            "d1": {"source": "acme"},
            "d2": {"source": "acme"},
            "d3": {"source": "other", "parent": {"source": "acme"}},
            "d4": {"source": "acme"},
            "d5": {"source": "acme"},
            "d6": {"source": "acme"},
        }
        all_ids = ["d1", "d2", "d3", "d4", "d5", "d6"]
        cursor = FakeCursor(_page(all_ids), metadata)
        store = make_store(cursor)

        results = store.similarity_search_by_embedding(
            QUERY, top_k=5, filter={"source": "acme"}
        )

        ids = [d.id for d, _ in results]
        assert "d3" not in ids
        assert ids == [
            "d1",
            "d2",
            "d4",
            "d5",
            "d6",
        ], "a full-size page is delivered after widening"
        sqls = cursor.vector_sqls()
        assert (
            len(sqls) == 2
        ), "exactly one wider re-fetch after the backstop dropped rows"
        first_top = int(re.search(r"SELECT TOP (\d+)", sqls[0]).group(1))
        second_top = int(re.search(r"SELECT TOP (\d+)", sqls[1]).group(1))
        assert first_top == 5 and second_top > 5

    def test_short_result_without_dropped_rows_does_not_refetch(self):
        # The SQL page is short because the tenant simply has few documents.
        cursor = FakeCursor(_page(["d1"]), {"d1": {"source": "acme"}})
        store = make_store(cursor)

        results = store.similarity_search_by_embedding(
            QUERY, top_k=5, filter={"source": "acme"}
        )

        assert [d.id for d, _ in results] == ["d1"]
        assert len(cursor.vector_sqls()) == 1

    def test_non_string_values_match_json_literals(self):
        cursor = FakeCursor(_page(["d1"]), {"d1": {"page_number": 3, "source": "x"}})
        store = make_store(cursor)

        store.similarity_search_by_embedding(QUERY, top_k=5, filter={"page_number": 3})

        sql = cursor.vector_sqls()[0]
        assert '"page_number":3' in sql and '"page_number": 3' in sql, sql
        assert '"page_number":"3"' not in sql

    def test_single_quotes_in_values_are_escaped(self):
        cursor = FakeCursor(_page(["d1"]), {"d1": {"source": "o'brien"}})
        store = make_store(cursor)

        store.similarity_search_by_embedding(
            QUERY, top_k=5, filter={"source": "o'brien"}
        )

        sql = cursor.vector_sqls()[0]
        assert "o''brien" in sql and "o'brien\"" not in sql.replace("o''brien", "")
