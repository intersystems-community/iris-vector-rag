"""The physical shape of a document or chunk table (feature 082).

One module owns the answer to "which column holds the identifier / the text / the
metadata / the embedding, and what type is it". Before this module the answer was
re-derived at eleven call sites under two contradictory precedence rules, which is
how a table could be written through one column and read through another.

Everything here is a value: frozen, comparable, and built either from catalog rows
or from a recorded fixture. Nothing here holds a connection and nothing here emits
SQL, so `TableShape.from_snapshot` is a pure function and its tests need no
database.

Two IRIS facts shape the design, both measured rather than assumed (see
`specs/082-table-shape/research.md`):

* `INFORMATION_SCHEMA` reports a `VECTOR(FLOAT, 384)` column as `varchar`. That is
  a driver limitation, never a schema defect, so vector-ness is read from the class
  view instead and nothing in this package alters a column on the strength of a
  reported SQL type.
* `%Dictionary.CompiledProperty.Name` strips underscores (`chunk_text` becomes
  `chunktext`), so the class view is joined on `SqlFieldName`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from iris_vector_rag.storage.catalog import CatalogSnapshot

__all__ = [
    "CANONICAL_CHUNK_SHAPE",
    "CANONICAL_DOCUMENT_SHAPE",
    "CatalogUnavailable",
    "ColumnFact",
    "ColumnRole",
    "IdentifierNotWritable",
    "NotVectorSearchable",
    "RoleNotPresent",
    "ShapeKind",
    "ShapeUnresolvable",
    "TableNotFound",
    "TableShape",
    "TableShapeError",
    "VectorSpec",
]

CANONICAL_VECTOR_TYPE = "FLOAT"
VECTOR_CLASS_TYPE = "%Library.Vector"
STREAM_CLASS_PREFIX = "%Stream."
LONG_TEXT_SQL_TYPE = "longvarchar"


class ColumnRole(Enum):
    """The question a call site actually asks, rather than a column name."""

    IDENTIFIER = "identifier"
    TEXT = "text"
    TITLE = "title"
    METADATA = "metadata"
    EMBEDDING = "embedding"
    CREATED_AT = "created_at"


#: Role resolution order. The canonical column is first in every list, so
#: FR-003 ("canonical wins when a table carries both") is this data, in one place.
ROLE_PRECEDENCE: Mapping[ColumnRole, Sequence[str]] = {
    ColumnRole.IDENTIFIER: ("doc_id", "id", "document_id"),
    ColumnRole.TEXT: ("text_content", "content", "chunk_text", "abstract"),
    ColumnRole.TITLE: ("title", "source", "filename"),
    ColumnRole.METADATA: ("metadata",),
    ColumnRole.EMBEDDING: ("embedding", "chunk_embedding", "entity_embedding"),
    ColumnRole.CREATED_AT: ("created_at",),
}

#: Alternate spellings of the *same* fact, per role: the columns an older schema
#: generation used for what the canonical column holds now.
#:
#: This is deliberately narrower than ROLE_PRECEDENCE. `abstract` is a TEXT
#: candidate, because a table carrying only an abstract has to resolve somehow -- but
#: it is not an alias of `text_content`, since the canonical shape carries both and
#: they hold different things (a summary and the full text). Reading one where the
#: other was written would return the wrong text, quietly. Likewise `document_id` in
#: a chunk table is the parent reference, not another name for the chunk's own key.
#:
#: Only columns listed here take part in the mixed-table read fallback, and only
#: these make a table MIXED.
ROLE_ALIASES: Mapping[ColumnRole, Sequence[str]] = {
    ColumnRole.IDENTIFIER: ("id",),
    ColumnRole.TEXT: ("content", "chunk_text"),
    ColumnRole.TITLE: ("source", "filename"),
    ColumnRole.METADATA: (),
    ColumnRole.EMBEDDING: ("chunk_embedding", "entity_embedding"),
    ColumnRole.CREATED_AT: (),
}

#: A table that cannot answer these is not usable as a document table.
REQUIRED_ROLES = (ColumnRole.IDENTIFIER, ColumnRole.TEXT)

CANONICAL_COLUMNS: Mapping[ColumnRole, str] = {
    role: names[0] for role, names in ROLE_PRECEDENCE.items()
}


class ShapeKind(Enum):
    """Classification of the six observed shapes.

    Informational only: nothing branches on it, so a seventh observed shape works
    without being named here.
    """

    CANONICAL = "canonical"
    LEGACY_SERIAL = "legacy_serial"
    LEGACY_ID = "legacy_id"
    FIXTURE = "fixture"
    MIXED = "mixed"
    DOUBLE_VECTOR = "double_vector"
    UNKNOWN = "unknown"


# --- errors -----------------------------------------------------------------


class TableShapeError(Exception):
    """Base class, so a caller can catch the family.

    Every subclass names the schema and table it failed on: the first error message
    has to name the real cause, or a caller goes looking several layers away.
    """

    def __init__(self, message: str, schema: str = "", table: str = ""):
        self.schema = schema
        self.table = table
        if schema or table:
            message = f"{schema}.{table}: {message}"
        super().__init__(message)


class TableNotFound(TableShapeError):
    """No such table in this namespace."""


class ShapeUnresolvable(TableShapeError):
    """The table exists but a required role has no candidate column."""


class RoleNotPresent(TableShapeError):
    """A caller asked for a role this table does not have."""


class NotVectorSearchable(TableShapeError):
    """The embedding role is absent, or its column is not a VECTOR."""


class IdentifierNotWritable(TableShapeError):
    """A caller-supplied identifier was aimed at a database-generated column."""


class CatalogUnavailable(TableShapeError):
    """The catalog query itself failed; the driver error is the ``__cause__``."""


# --- values -----------------------------------------------------------------


@dataclass(frozen=True)
class VectorSpec:
    """What ``TO_VECTOR`` and the dimension check need, and nothing more."""

    element_type: str
    dimension: int

    @classmethod
    def from_parameters(
        cls, parameters: str, *, schema: str = "", table: str = "", column: str = ""
    ) -> "VectorSpec":
        """Parse ``%Dictionary.CompiledProperty.Parameters``.

        The string is a flat comma-separated key,value sequence with empty values
        preserved, e.g. ``CAPTION,,DATATYPE,FLOAT,...,LEN,384,...``. A missing
        ``DATATYPE`` or ``LEN`` is an error, never a default: the environment
        fallback this replaces is the reason a mistyped table name used to yield a
        plausible ``FLOAT`` and a failure several layers away.
        """
        parts = parameters.split(",") if parameters else []
        values = {
            parts[i].strip(): parts[i + 1].strip() for i in range(0, len(parts) - 1, 2)
        }
        element_type = values.get("DATATYPE") or ""
        length = values.get("LEN") or ""
        where = f" on column {column}" if column else ""
        if not element_type:
            raise ShapeUnresolvable(
                f"vector column{where} does not report DATATYPE; "
                "refusing to guess an element type",
                schema,
                table,
            )
        if not length.isdigit():
            raise ShapeUnresolvable(
                f"vector column{where} does not report a numeric LEN "
                f"(got {length!r}); refusing to guess a dimension",
                schema,
                table,
            )
        return cls(element_type=element_type.upper(), dimension=int(length))


@dataclass(frozen=True)
class ColumnFact:
    """One column as the catalog reports it, from both the SQL and class views."""

    name: str
    sql_type: str
    max_length: Optional[int] = None
    nullable: bool = True
    position: int = 0
    class_type: Optional[str] = None
    is_identity: bool = False
    is_key: bool = False
    vector: Optional[VectorSpec] = None

    @property
    def is_stream(self) -> bool:
        return bool(self.class_type and self.class_type.startswith(STREAM_CLASS_PREFIX))

    @property
    def is_long_text(self) -> bool:
        return self.sql_type == LONG_TEXT_SQL_TYPE

    @property
    def filterable_with_like(self) -> bool:
        """``LIKE`` on a long or stream column matches nothing, silently.

        So this is a correctness question, not an optimisation: a filter against a
        column where this is false has to be written with ``SUBSTRING``.
        """
        return not self.is_stream and not self.is_long_text


@dataclass(frozen=True)
class TableShape:
    """The physical shape of one table right now."""

    schema: str
    table: str
    class_name: str
    columns: Mapping[str, ColumnFact] = field(default_factory=dict)
    roles: Mapping[ColumnRole, str] = field(default_factory=dict)
    kind: ShapeKind = ShapeKind.UNKNOWN

    # -- construction --------------------------------------------------------

    @classmethod
    def from_snapshot(cls, snapshot: "CatalogSnapshot") -> "TableShape":
        """Build a shape from recorded catalog rows. Pure: no I/O, no environment."""
        schema, table = snapshot.schema, snapshot.table
        if not snapshot.columns:
            raise ShapeUnresolvable(
                "the catalog reports no columns for this table", schema, table
            )

        properties = {
            str(row.get("SqlFieldName", "")).lower(): row
            for row in snapshot.properties
            if row.get("SqlFieldName")
        }
        keys = {str(k).lower() for k in snapshot.key_columns}

        columns: dict[str, ColumnFact] = {}
        for row in sorted(snapshot.columns, key=lambda r: int(r["ORDINAL_POSITION"])):
            name = str(row["COLUMN_NAME"])
            key = name.lower()
            prop = properties.get(key, {})
            class_type = prop.get("Type")
            vector = None
            if class_type == VECTOR_CLASS_TYPE:
                vector = VectorSpec.from_parameters(
                    str(prop.get("Parameters") or ""),
                    schema=schema,
                    table=table,
                    column=name,
                )
            columns[key] = ColumnFact(
                name=name,
                sql_type=str(row["DATA_TYPE"]).lower(),
                max_length=_as_int(row.get("CHARACTER_MAXIMUM_LENGTH")),
                nullable=str(row.get("IS_NULLABLE", "YES")).upper() == "YES",
                position=int(row["ORDINAL_POSITION"]),
                class_type=str(class_type) if class_type else None,
                is_identity=_is_yes(row.get("IS_IDENTITY"))
                or _is_yes(row.get("AUTO_INCREMENT")),
                is_key=key in keys,
                vector=vector,
            )

        roles = _resolve_roles(columns, keys, schema, table)
        for role in roles:
            column = columns[roles[role]]
            if column.class_type is None:
                raise ShapeUnresolvable(
                    f"column {column.name} fills the {role.name} role but the class "
                    "view does not describe it, so its type is unknown",
                    schema,
                    table,
                )

        shape = cls(
            schema=schema,
            table=table,
            class_name=snapshot.class_name or f"{schema}.{table}",
            columns=columns,
            roles=roles,
        )
        return cls(
            schema=shape.schema,
            table=shape.table,
            class_name=shape.class_name,
            columns=shape.columns,
            roles=shape.roles,
            kind=_classify(shape),
        )

    # -- role access ---------------------------------------------------------

    def has(self, role: ColumnRole) -> bool:
        return role in self.roles

    def optional_column_for(self, role: ColumnRole) -> Optional[ColumnFact]:
        name = self.roles.get(role)
        return self.columns[name.lower()] if name else None

    def column_for(self, role: ColumnRole) -> ColumnFact:
        column = self.optional_column_for(role)
        if column is None:
            raise RoleNotPresent(
                f"no column fills the {role.name} role "
                f"(candidates: {', '.join(ROLE_PRECEDENCE[role])})",
                self.schema,
                self.table,
            )
        return column

    def legacy_column_for(self, role: ColumnRole) -> Optional[ColumnFact]:
        """The older spelling of the same fact, when the table carries both.

        This is what lets the read path fall back per row on a mixed table without
        turning the fallback into a precedence rule: ``column_for`` still returns the
        canonical column. Only ``ROLE_ALIASES`` counts, so `abstract` alongside
        `text_content` is not a legacy column.
        """
        bound = self.roles.get(role)
        if not bound:
            return None
        for candidate in ROLE_ALIASES[role]:
            if candidate != bound.lower() and candidate in self.columns:
                return self.columns[candidate]
        return None

    # -- derived answers -----------------------------------------------------

    def vector_spec(self) -> VectorSpec:
        column = self.optional_column_for(ColumnRole.EMBEDDING)
        if column is None:
            raise NotVectorSearchable(
                "table has no embedding column, so it cannot be searched by vector",
                self.schema,
                self.table,
            )
        if column.vector is None:
            raise NotVectorSearchable(
                f"column {column.name} is {column.class_type or column.sql_type}, "
                "not a VECTOR, so it cannot be searched by vector",
                self.schema,
                self.table,
            )
        return column.vector

    def writable_identifier(self) -> ColumnFact:
        column = self.column_for(ColumnRole.IDENTIFIER)
        if column.is_identity:
            raise IdentifierNotWritable(
                f"identifier column {column.name} is database-generated, so a "
                "caller-supplied identifier cannot be written to it",
                self.schema,
                self.table,
            )
        return column

    @property
    def is_canonical(self) -> bool:
        """Every role present, bound to its canonical column, with a FLOAT vector."""
        if set(self.roles) != set(ROLE_PRECEDENCE):
            return False
        if any(self.roles[role] != CANONICAL_COLUMNS[role] for role in self.roles):
            return False
        if any(self.legacy_column_for(role) is not None for role in self.roles):
            return False
        embedding = self.columns[CANONICAL_COLUMNS[ColumnRole.EMBEDDING]]
        return (
            embedding.vector is not None
            and embedding.vector.element_type == CANONICAL_VECTOR_TYPE
        )

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.table}"


# --- helpers ----------------------------------------------------------------


def _as_int(value) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_yes(value) -> bool:
    return str(value).upper() in ("YES", "TRUE", "1")


def _resolve_roles(
    columns: Mapping[str, ColumnFact], keys: set[str], schema: str, table: str
) -> Mapping[ColumnRole, str]:
    roles: dict[ColumnRole, str] = {}

    # A declared single-column key outranks name precedence: an identity column
    # cannot be written to, so picking the wrong identifier by name fails at insert
    # time rather than at resolution time.
    if len(keys) == 1:
        only_key = next(iter(keys))
        if only_key in columns:
            roles[ColumnRole.IDENTIFIER] = only_key

    for role, candidates in ROLE_PRECEDENCE.items():
        if role in roles:
            continue
        for candidate in candidates:
            if candidate in columns:
                roles[role] = candidate
                break

    for role in REQUIRED_ROLES:
        if role not in roles:
            raise ShapeUnresolvable(
                f"no column fills the required {role.name} role "
                f"(looked for: {', '.join(ROLE_PRECEDENCE[role])}; "
                f"found: {', '.join(c.name for c in columns.values())})",
                schema,
                table,
            )
    return roles


def _classify(shape: TableShape) -> ShapeKind:
    identifier = shape.column_for(ColumnRole.IDENTIFIER)
    text = shape.column_for(ColumnRole.TEXT)
    embedding = shape.optional_column_for(ColumnRole.EMBEDDING)

    if any(shape.legacy_column_for(role) is not None for role in shape.roles):
        return ShapeKind.MIXED
    if shape.is_canonical:
        return ShapeKind.CANONICAL
    if (
        identifier.name == "doc_id"
        and text.name == "text_content"
        and embedding is not None
        and embedding.vector is not None
        and embedding.vector.element_type != CANONICAL_VECTOR_TYPE
    ):
        return ShapeKind.DOUBLE_VECTOR
    if identifier.is_identity:
        return ShapeKind.LEGACY_SERIAL
    if identifier.name == "id" and text.name == "content":
        return ShapeKind.LEGACY_ID
    if identifier.name == "doc_id" and text.name == "content":
        return ShapeKind.FIXTURE
    return ShapeKind.UNKNOWN


# --- the shape every creation path emits ------------------------------------


def _canonical_shape(
    schema: str, table: str, columns: Sequence[ColumnFact]
) -> TableShape:
    by_name = {c.name.lower(): c for c in columns}
    roles = {role: name for role, name in CANONICAL_COLUMNS.items() if name in by_name}
    return TableShape(
        schema=schema,
        table=table,
        class_name=f"{schema}.{table}",
        columns=by_name,
        roles=roles,
        kind=ShapeKind.CANONICAL,
    )


CANONICAL_DOCUMENT_SHAPE = _canonical_shape(
    "RAG",
    "SourceDocuments",
    [
        ColumnFact("doc_id", "varchar", 255, False, 1, "%Library.String", is_key=True),
        ColumnFact("title", "varchar", 1000, True, 2, "%Library.String"),
        ColumnFact("abstract", "longvarchar", 2147483647, True, 3, "%Library.String"),
        ColumnFact(
            "text_content", "longvarchar", 2147483647, True, 4, "%Library.String"
        ),
        ColumnFact("authors", "longvarchar", 2147483647, True, 5, "%Library.String"),
        ColumnFact("keywords", "longvarchar", 2147483647, True, 6, "%Library.String"),
        ColumnFact(
            "embedding",
            "varchar",
            132863,
            True,
            7,
            VECTOR_CLASS_TYPE,
            vector=VectorSpec(CANONICAL_VECTOR_TYPE, 384),
        ),
        ColumnFact("metadata", "varchar", 2000, True, 8, "%Library.String"),
        ColumnFact("created_at", "timestamp", None, True, 9, "%Library.PosixTime"),
    ],
)

CANONICAL_CHUNK_SHAPE = _canonical_shape(
    "RAG",
    "DocumentChunks",
    [
        ColumnFact("doc_id", "varchar", 255, False, 1, "%Library.String", is_key=True),
        ColumnFact("document_id", "varchar", 255, True, 2, "%Library.String"),
        ColumnFact("chunk_text", "longvarchar", 2147483647, True, 3, "%Library.String"),
        ColumnFact(
            "chunk_embedding",
            "varchar",
            132863,
            True,
            4,
            VECTOR_CLASS_TYPE,
            vector=VectorSpec(CANONICAL_VECTOR_TYPE, 384),
        ),
        ColumnFact("metadata", "varchar", 2000, True, 5, "%Library.String"),
        ColumnFact("created_at", "timestamp", None, True, 6, "%Library.PosixTime"),
    ],
)
