"""Full-text search over the document table: the SQL, built as text.

The text leg of `text`, `hybrid` and `rrf` retrieval reads the document table through an
iFind index on its text column (`SchemaManager.ensure_text_index` creates it). Everything
here is pure: names and statements in, strings out, so the SQL can be tested without a
database and then run against a real one.

What was verified against IRIS 2026.1 (not assumed):

* ``CREATE INDEX name ON schema.table (col) AS %iFind.Index.Basic(LANGUAGE='en', LOWER=1)``
  works on a ``LONGVARCHAR`` column, builds over rows already in the table, and is
  reported by ``INFORMATION_SCHEMA.INDEXES`` under the name it was given.
* ``WHERE %ID %FIND search_index(IndexName, ?)`` takes the search string as a parameter.
  Several words are an AND; ``a OR b`` is supported; an empty string is SQLCODE -400.
* The index generates a rank procedure, ``<schema>.<class>_<IndexName>Rank(%ID, ?)``,
  which scores a row against a search string (TF-IDF).
* ``LIKE`` on the ``LONGVARCHAR`` text column matches nothing, which is why ivg's
  fallback never worked, and ``%FIND(col, ?)`` is not IRIS SQL (SQLCODE -359).
"""

from __future__ import annotations

import re
from typing import List, Optional

__all__ = [
    "TEXT_INDEX_NAME",
    "create_text_index_statement",
    "rank_procedure",
    "search_terms",
    "build_text_search_sql",
]

#: The iFind index this library keeps on the document table's text column.
TEXT_INDEX_NAME = "SourceDocumentsTextIdx"

#: Words that match nearly every document and would turn an ``OR`` query into a scan.
_STOPWORDS = frozenset(
    "a an and are as at be but by can did do does for from had has have how i if in "
    "into is it its me my no not of on or our so such than that the their then there "
    "these they this to was we were what when where which who whom why will with you "
    "your".split()
)

_WORD = re.compile(r"\w+", re.UNICODE)

#: The terms one query may put in the search string.
MAX_TERMS = 32

_IDENTIFIER = re.compile(r"^[A-Za-z_%][A-Za-z0-9_]*$")


def _checked(identifier: str) -> str:
    if not _IDENTIFIER.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return identifier


def create_text_index_statement(schema: str, table: str, column: str) -> str:
    """The DDL for the full-text index on ``column`` of ``schema.table``."""
    return (
        f"CREATE INDEX {TEXT_INDEX_NAME} ON {_checked(schema)}.{_checked(table)} "
        f"({_checked(column)}) AS %iFind.Index.Basic(LANGUAGE='en', LOWER=1)"
    )


def rank_procedure(schema: str, class_name: str, index: str = TEXT_INDEX_NAME) -> str:
    """The SQL name of the rank procedure iFind generates for ``index``.

    Built from the *class* name, not the table name: the two differ when a table name
    carries an underscore (``RAG.llm_cache`` is class ``RAG.llmcache``).
    """
    short_class = class_name.rpartition(".")[2]
    return f"{_checked(schema)}.{_checked(short_class)}_{_checked(index)}Rank"


def search_terms(query: str) -> str:
    """The iFind search string for a natural-language ``query``, or ``""`` for none.

    The words are joined with ``OR`` and the rank procedure orders the rows, so a
    question such as "what does insulin do?" finds the documents about insulin instead
    of requiring every word (stop words included) to appear. Punctuation, wildcards and
    the ``AND``/``OR``/``NOT`` operators a user might type are dropped; the search
    string is always a plain list of words. A query made only of stop words keeps them
    rather than searching for nothing.
    """
    words = [w.lower() for w in _WORD.findall(query or "")]
    kept = [w for w in words if w not in _STOPWORDS] or [
        w for w in words if w not in {"and", "or", "not"}
    ]
    unique: List[str] = list(dict.fromkeys(kept))[:MAX_TERMS]
    return " OR ".join(unique)


def build_text_search_sql(
    *,
    table: str,
    id_column: str,
    index_name: str,
    rank_proc: str,
    top_k: int,
    extra_where: Optional[str] = None,
) -> str:
    """``SELECT TOP n id, rank`` for one search string, best match first.

    Parameters, in order: the search string (rank), the search string (match), then the
    search string again (ordering). ``extra_where`` is a trusted predicate built by the
    caller (the metadata filter), never user text. It must be safe to follow ``AND``
    (each term parenthesised, as ``_metadata_filter_predicate`` builds them) and is
    not wrapped again: the IRIS DBAPI miscounts the parameters of a statement with a
    doubled ``((`` in its WHERE and a parameter in its ORDER BY ("Incorrect number of
    parameters").
    """
    if int(top_k) < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k!r}")
    where = f"%ID %FIND search_index({_checked(index_name)}, ?)"
    if extra_where:
        where += f" AND {extra_where}"
    rank = f"{rank_proc}(%ID, ?)"
    return (
        f"SELECT TOP {int(top_k)} {_checked(id_column)}, {rank} "
        f"FROM {table} WHERE {where} "
        f"ORDER BY CAST({rank} AS DOUBLE) DESC"
    )
