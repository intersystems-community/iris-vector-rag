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

import json
import re
from typing import Any, List, Optional

__all__ = [
    "TEXT_INDEX_NAME",
    "DEFAULT_TEXT_LANGUAGE",
    "STOPWORDS",
    "text_language",
    "create_text_index_statement",
    "rank_procedure",
    "search_terms",
    "TEXT_TERMS",
    "build_text_search_sql",
    "json_member_predicate",
    "section_filter_predicate",
]

#: The iFind index this library keeps on the document table's text column.
TEXT_INDEX_NAME = "SourceDocumentsTextIdx"

#: The language of the text index and of query terms when none is configured.
DEFAULT_TEXT_LANGUAGE = "en"

#: Words that match nearly every document and would turn an ``OR`` query into a scan.
_ENGLISH_STOPWORDS = frozenset(
    "a an and are as at be but by can did do does for from had has have how i if in "
    "into is it its me my no not of on or our so such than that the their then there "
    "these they this to was we were what when where which who whom why will with you "
    "your".split()
)

#: Stop words by language code. A language with none listed keeps every word.
STOPWORDS: dict[str, frozenset[str]] = {"en": _ENGLISH_STOPWORDS}

_OPERATORS = frozenset({"and", "or", "not"})

_WORD = re.compile(r"\w+", re.UNICODE)

_LANGUAGE = re.compile(r"^[a-z]{2}$")

#: The terms one query may put in the search string.
MAX_TERMS = 32

_IDENTIFIER = re.compile(r"^[A-Za-z_%][A-Za-z0-9_]*$")


def _checked(identifier: str) -> str:
    if not _IDENTIFIER.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return identifier


def text_language(language: Optional[str]) -> str:
    """``language`` as the two-letter code iFind takes, ``en`` if unset.

    Raises:
        ValueError: it is not two letters. The code goes into DDL as a literal.
    """
    code = (language or "").strip().lower() or DEFAULT_TEXT_LANGUAGE
    if not _LANGUAGE.match(code):
        raise ValueError(
            f"text language must be a two-letter code such as 'en' or 'cs', "
            f"got {language!r}"
        )
    return code


def create_text_index_statement(
    schema: str, table: str, column: str, language: Optional[str] = None
) -> str:
    """The DDL for the full-text index on ``column`` of ``schema.table``, in ``language``.

    An unset ``language`` is ``DEFAULT_TEXT_LANGUAGE``.
    """
    return (
        f"CREATE INDEX {TEXT_INDEX_NAME} ON {_checked(schema)}.{_checked(table)} "
        f"({_checked(column)}) "
        f"AS %iFind.Index.Basic(LANGUAGE='{text_language(language)}', LOWER=1)"
    )


def rank_procedure(schema: str, class_name: str, index: str = TEXT_INDEX_NAME) -> str:
    """The SQL name of the rank procedure iFind generates for ``index``.

    Built from the *class* name, not the table name: the two differ when a table name
    carries an underscore (``RAG.llm_cache`` is class ``RAG.llmcache``).
    """
    short_class = class_name.rpartition(".")[2]
    return f"{_checked(schema)}.{_checked(short_class)}_{_checked(index)}Rank"


#: How query words become iFind terms: whole words, or prefixes for inflected
#: languages (feature 092).
TEXT_TERMS = ("words", "prefix")


def _prefix(word: str) -> str:
    """A prefix wildcard for an inflected word: ``vysazena`` -> ``vysaz*``.

    Words under four letters stay whole. Longer ones keep the length less three
    letters, at least four and at most seven, so ``antikoagulace`` -> ``antikoa*``.
    The rule is the one the Štafeta team used with a Czech iFind index at Rakathon
    2026 (IRIS has no Czech stemmer), reimplemented here.
    """
    if len(word) < 4:
        return word
    return word[: max(4, min(len(word) - 3, 7))] + "*"


def search_terms(
    query: str, language: str = DEFAULT_TEXT_LANGUAGE, terms: str = "words"
) -> str:
    """The iFind search string for a natural-language ``query``, or ``""`` for none.

    The words are joined with ``OR`` and the rank procedure orders the rows, so a
    question such as "what does insulin do?" finds the documents about insulin instead
    of requiring every word (stop words included) to appear. Punctuation, wildcards and
    the ``AND``/``OR``/``NOT`` operators a user might type are dropped; the search
    string is always a plain list of words. A query made only of stop words keeps them
    rather than searching for nothing. Stop words are those of ``language``
    (``STOPWORDS``); a language with none listed keeps every word but the operators.
    With ``terms="prefix"`` each word longer than three letters becomes a prefix
    wildcard (see `_prefix`); the ``*`` is the only operator that reaches the index.
    """
    if terms not in TEXT_TERMS:
        raise ValueError(f"terms must be one of {', '.join(TEXT_TERMS)}, got {terms!r}")
    stopwords = STOPWORDS.get(text_language(language), frozenset())
    words = [w.lower() for w in _WORD.findall(query or "")]
    content = [w for w in words if w not in _OPERATORS]
    kept = [w for w in content if w not in stopwords] or content
    if terms == "prefix":
        kept = [_prefix(word) for word in kept]
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
    id_expression: Optional[str] = None,
) -> str:
    """``SELECT TOP n id, rank`` for one search string, best match first.

    Parameters, in order: the search string (rank), the search string (match), then the
    search string again (ordering); the parameters of ``extra_where``, if it has any,
    come after the match and before the ordering. ``extra_where`` is a trusted
    predicate built by the caller (the real-column and metadata filters), never user
    text. ``id_expression`` is selected instead of ``id_column`` when given: a trusted
    expression from checked identifiers (a composite identifier, feature 092). It must be safe to follow ``AND``
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
        f"SELECT TOP {int(top_k)} {id_expression or _checked(id_column)}, {rank} "
        f"FROM {table} WHERE {where} "
        f"ORDER BY CAST({rank} AS DOUBLE) DESC"
    )


def json_member_predicate(key: str, value: Any, metadata_expr: str) -> str:
    """LIKE predicate for the JSON member ``"key": value`` in ``metadata_expr``.

    Matches the member with or without a space after the colon (``json.dumps`` emits
    the space; hand-written JSON often does not). The literal is the JSON encoding of
    the value, with single quotes doubled for SQL. The closing quote of a string
    literal rules out prefix collisions; the caller's exact check removes the rest.
    """
    if not _IDENTIFIER.match(str(key)):
        raise ValueError(f"Invalid metadata filter key: {key!r}")
    literal = json.dumps(value).replace("'", "''")
    return (
        f"({metadata_expr} LIKE '%\"{key}\":{literal}%' "
        f"OR {metadata_expr} LIKE '%\"{key}\": {literal}%')"
    )


def section_filter_predicate(
    label: str,
    column: Optional[str] = None,
    metadata_expr: Optional[str] = None,
) -> str:
    """The SQL predicate that keeps the sections whose label is ``label``.

    With ``column`` (the opt-in ``section_label`` column, bitmap-indexed) it is an
    equality on that column; the column name is checked and the label's quotes are
    doubled. Without it, it is the metadata form: the LIKE over the JSON member
    ``section``, which is what ``IRISVectorStore`` has always used and which needs
    ``metadata_expr``. Both return the same rows for the labels SectionStore writes.
    """
    if column is not None:
        escaped = str(label).replace("'", "''")
        return f"{_checked(column)} = '{escaped}'"
    if metadata_expr is None:
        raise ValueError("the metadata form needs metadata_expr")
    return json_member_predicate("section", label, metadata_expr)
