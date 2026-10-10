"""BM25 index maintenance for iris-vector-graph 4.2.0.

ivg 4.2.0 changed the BM25 index to schema 2 (document frequency and a per-document
token record), which makes ``Insert`` and ``Delete`` cost in proportion to the
document instead of the index. An index built earlier is schema 1: ``Search`` keeps
working on it, but ``Insert`` and ``Delete`` raise an error naming ``Upgrade`` until
``engine.bm25_upgrade(name)`` has run once.

IVR's HybridGraphRAG pipeline owns no BM25 index (the text leg is ivg's ``kg_TXT`` over
the KG tables, or an injected :class:`~iris_vector_rag.pipelines.ivg_bm25_backend.IVGTextSearchBackend`
over an index the caller built), so nothing here runs implicitly. An operator, or
``SchemaManager.upgrade_iris_vector_graph``, calls :func:`upgrade_bm25_indexes`.

Scores change after the upgrade: IDF is computed at search time and the length
normalisation follows the BM25 formula (ivg CHANGELOG, "BM25 schema 2").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

#: The schema ivg 4.2.0 writes. A smaller one cannot be written to until upgraded.
CURRENT_SCHEMA = 2

# What an index's entry says happened (or, on a dry run, would happen).
UPGRADED = "upgraded"
ALREADY_CURRENT = "already_current"
WOULD_UPGRADE = "would_upgrade"
MISSING = "missing"
FAILED = "failed"

_NEEDS_IVG_42 = (
    "this iris-vector-graph has no engine.bm25_upgrade; BM25 schema 2 needs "
    "iris-vector-graph>=4.2.0,<5"
)


@dataclass(frozen=True)
class Bm25IndexResult:
    """What happened to one index."""

    name: str
    #: The schema the index had before; ``None`` when there is no such index.
    schema: Optional[int]
    action: str
    #: Documents in the index (``Info`` before the run, ``Upgrade``'s count after).
    docs: Optional[int] = None
    vocab: Optional[int] = None
    postings: Optional[int] = None
    error: Optional[str] = None

    @property
    def hint(self) -> str:
        """The call that upgrades this index by hand."""
        return f"engine.bm25_upgrade({self.name!r})"


class Bm25UpgradeError(RuntimeError):
    """One index could not be read or upgraded.

    ``name`` is the index, ``__cause__`` the error ivg raised, and ``report`` what had
    been done to the indexes before it (set by :func:`upgrade_bm25_indexes`).
    """

    def __init__(self, name: str, cause: BaseException, schema: Optional[int] = None):
        super().__init__(f"BM25 index {name!r}: {cause}")
        self.name = name
        self.schema = schema
        self.report: "Bm25UpgradeReport" = Bm25UpgradeReport()


@dataclass(frozen=True)
class Bm25UpgradeReport:
    """Every index's outcome, in the order they were handled."""

    results: Tuple[Bm25IndexResult, ...] = ()
    dry_run: bool = False

    def __getitem__(self, name: str) -> Bm25IndexResult:
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(name)

    def _named(self, action: str) -> Tuple[str, ...]:
        return tuple(r.name for r in self.results if r.action == action)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(r.name for r in self.results)

    @property
    def upgraded(self) -> Tuple[str, ...]:
        return self._named(UPGRADED)

    @property
    def pending(self) -> Tuple[str, ...]:
        """Schema-1 indexes a dry run found; a real run upgrades them."""
        return self._named(WOULD_UPGRADE)

    @property
    def current(self) -> Tuple[str, ...]:
        return self._named(ALREADY_CURRENT)

    @property
    def missing(self) -> Tuple[str, ...]:
        return self._named(MISSING)

    @property
    def failed(self) -> Tuple[str, ...]:
        return self._named(FAILED)

    @property
    def ok(self) -> bool:
        """No index failed. An absent or already-current index is not a failure."""
        return not self.failed


def bm25_index_names(engine: Any) -> Tuple[str, ...]:
    """The full-text (BM25) indexes the engine knows, by name.

    From ``engine.list_indexes()``, which ivg builds when the engine is created: an
    index another process makes afterwards is not in it. Pass ``names=`` to
    :func:`upgrade_bm25_indexes` for those.
    """
    return tuple(
        index.name
        for index in engine.list_indexes()
        if getattr(index, "type", None) == "fulltext"
    )


def _schema_of(info: Any) -> Optional[int]:
    """The schema ``Info`` reports; 1 when it reports none (ivg before 4.2.0)."""
    # ivg's wrapper adds "type" to the class's "{}", so an absent index is not an empty dict.
    if not info or "N" not in info:
        return None
    return int(info.get("schema", 1) or 1)


def _upgrade_one(engine: Any, name: str, dry_run: bool) -> Bm25IndexResult:
    """One index; an ivg failure is a :class:`Bm25UpgradeError`."""
    try:
        info = engine.bm25_info(name)
    except Exception as exc:  # the class is absent or the call is refused
        raise Bm25UpgradeError(name, exc) from exc
    schema = _schema_of(info)
    if schema is None:
        return Bm25IndexResult(name, None, MISSING)
    docs = info.get("N")
    if schema >= CURRENT_SCHEMA:
        return Bm25IndexResult(name, schema, ALREADY_CURRENT, docs=docs)
    if dry_run:
        return Bm25IndexResult(name, schema, WOULD_UPGRADE, docs=docs)
    try:
        answer = engine.bm25_upgrade(name) or {}
    except Exception as exc:
        raise Bm25UpgradeError(name, exc, schema=schema) from exc
    if answer.get("missing"):  # dropped between the read and the upgrade
        return Bm25IndexResult(name, None, MISSING)
    if answer.get("already"):  # another process got there first
        return Bm25IndexResult(name, schema, ALREADY_CURRENT, docs=docs)
    logger.info("BM25 index %r upgraded to schema %d: %s", name, CURRENT_SCHEMA, answer)
    return Bm25IndexResult(
        name,
        schema,
        UPGRADED,
        docs=answer.get("docs", docs),
        vocab=answer.get("vocab"),
        postings=answer.get("postings"),
    )


def upgrade_bm25_indexes(
    engine: Any,
    names: Optional[Iterable[str]] = None,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> Bm25UpgradeReport:
    """Bring BM25 indexes to schema 2.

    ``names`` limits the run (default: every full-text index the engine lists). Each
    index is read first and upgraded only if it is still schema 1, so the call can be
    repeated; ivg's own ``Upgrade`` is safe to re-run after an interruption.

    An index that cannot be read or upgraded raises :class:`Bm25UpgradeError`, whose
    ``report`` holds what was done before it. With ``continue_on_error=True`` the
    failure is recorded in the report (``action="failed"``, ``error``) and the
    remaining indexes still run; ``report.ok`` is then false.

    A dry run reads each index's ``Info`` and writes nothing.

    ``Upgrade`` is one pass over the posting lists and refuses writers while it runs
    (``Search`` keeps working); run it as a job, not on a request path.
    """
    if not dry_run and not callable(getattr(engine, "bm25_upgrade", None)):
        raise RuntimeError(_NEEDS_IVG_42)
    selected = tuple(names) if names is not None else bm25_index_names(engine)
    results: list = []
    for name in selected:
        try:
            results.append(_upgrade_one(engine, name, dry_run))
        except Bm25UpgradeError as failure:
            if not continue_on_error:
                failure.report = Bm25UpgradeReport(tuple(results), dry_run)
                raise
            logger.error("%s", failure)
            results.append(
                Bm25IndexResult(
                    name, failure.schema, FAILED, error=str(failure.__cause__)
                )
            )
    return Bm25UpgradeReport(results=tuple(results), dry_run=dry_run)
