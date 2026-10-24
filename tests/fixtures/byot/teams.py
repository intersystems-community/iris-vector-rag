"""Two Rakathon 2026 teams' chunk tables, rebuilt with synthetic rows (feature 092).

The DDL keeps each team's own column names, from their public repositories:
MedTeam's `EvidenceChunk` (`Plokar/Rakhaton2026`, `iris/vector_store.py`) and Štafeta's
`DocChunk` (`adambeniak/rakathon2026`, `backend/app/iris/vector_store.py`). The rows,
texts and vectors are made up and seeded, so every run builds the same tables. Each
team's own query is here too, as SQL, so a test can compare the library's answer with
the answer the team's code gives on the same rows. Spike S3 (ivr-research-notes,
`.scratch/rakathon-followup/research/s3-team-tables.md`) is where these were first run.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Dict, List, Sequence, Tuple

DIM = 384
MED_SCHEMA, MED_TABLE = "BYOTMed", "EvidenceChunk"
STF_SCHEMA, STF_TABLE = "BYOTStf", "DocChunk"
MED = f"{MED_SCHEMA}.{MED_TABLE}"
STF = f"{STF_SCHEMA}.{STF_TABLE}"
MODELS = ("mock:hashed@384", "gemini:gemini-embedding-001@384")
STF_MODEL = "mini-multilingual@384"
MED_PATIENTS = ("P01", "P02", "P03")
STF_PATIENTS = ("Patient/1", "Patient/2", "Patient/3", "Patient/4")

# Štafeta's as-of rule: authored before the board date, received no later than the cut-off.
AS_OF_AUTHORED_BEFORE = "2026-10-05"
AS_OF_RECEIVED_UNTIL = "2026-10-05 12:00:00"
STF_TEXT_QUERY = "Kdy byla vysazena antikoagulace?"
STF_IFIND_TERMS = "vysaz* OR antikoa*"  # what Štafeta's stemmer makes of that question

_MED_COLUMNS = (
    ("EvidenceId", "VARCHAR(200)"),
    ("PatientId", "VARCHAR(16)"),
    ("ResourceType", "VARCHAR(64)"),
    ("ResourceId", "VARCHAR(128)"),
    ("ChunkIndex", "INTEGER"),
    ("CharStart", "INTEGER"),
    ("CharEnd", "INTEGER"),
    ("DocDate", "DATE"),
    ("Title", "VARCHAR(500)"),
    ("Department", "VARCHAR(200)"),
    ("Text", "VARCHAR(10000)"),
    ("EmbeddingModel", "VARCHAR(100)"),
    ("Embedding", f"VECTOR(DOUBLE, {DIM})"),
)
MED_DDL = f"CREATE TABLE {MED} ({', '.join(f'{c} {t}' for c, t in _MED_COLUMNS)})"
MED_INDEXES = (
    f"CREATE INDEX EvidenceChunkPatientIdIdx ON {MED} (PatientId)",
    f"CREATE INDEX EvidenceChunkResourceIdIdx ON {MED} (ResourceId)",
)

STF_DDL = (
    f"CREATE TABLE {STF} (DocRef VARCHAR(200) NOT NULL, PatientRef VARCHAR(200), "
    "Seq INTEGER NOT NULL, AuthoredOn DATE, ReceivedOn TIMESTAMP, Text VARCHAR(32000), "
    f"Embedding VECTOR(DOUBLE, {DIM}), DocId VARCHAR(100), Title VARCHAR(500), "
    "CharStart INTEGER, CharEnd INTEGER, EmbeddingModel VARCHAR(200))"
)
STF_INDEXES = (
    f"CREATE INDEX DocChunkPatient ON TABLE {STF} (PatientRef)",
    f"CREATE INDEX DocChunkDocRef ON TABLE {STF} (DocRef)",
    f"CREATE INDEX DocChunkHNSW ON TABLE {STF} (Embedding) AS HNSW(Distance='Cosine')",
    f"CREATE INDEX DocChunkText ON TABLE {STF} (Text) "
    "AS %iFind.Index.Basic(LANGUAGE='cs', LOWER=1)",
)

_SNIPPETS = (
    "apixaban vysazen před biopsií",
    "alergie neguje",
    "CT hrudníku bez progrese",
    "ECOG 1 pacient v dobrém stavu",
    "histologie adenokarcinom",
    "chemoterapie cyklus zahájen",
    "kouří 10 cigaret denně",
    "léky metformin ramipril",
    "rodinná anamnéza karcinom prsu",
    "biopsie plic provedena bez komplikací",
    "antikoagulace obnovena po výkonu",
    "MR mozku negativní",
)


# --- vectors -------------------------------------------------------------------


def unit_vec(seed: int, dim: int = DIM) -> List[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def near(vec: Sequence[float], seed: int, noise: float = 0.15) -> List[float]:
    """``vec`` plus seeded noise, renormalised: a query close to ``vec``."""
    rng = random.Random(seed)
    w = [x + rng.gauss(0, noise / DIM**0.5) for x in vec]
    norm = sum(x * x for x in w) ** 0.5
    return [x / norm for x in w]


def blend(a: Sequence[float], b: Sequence[float], wa=0.7, wb=0.5) -> List[float]:
    w = [wa * x + wb * y for x, y in zip(a, b)]
    norm = sum(x * x for x in w) ** 0.5
    return [x / norm for x in w]


def vlit(vec: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _med_seed(patient: int, resource: int, chunk: int, model: int) -> int:
    return (patient * 1000 + resource * 10 + chunk) * 10 + model


def _stf_seed(patient: int, doc: int, seq: int) -> int:
    return 500000 + patient * 100 + doc * 10 + seq


#: Near MedTeam row P02-doc03#1 under the first model.
MED_QUERY = near(unit_vec(_med_seed(1, 3, 1, 0)), seed=7)
#: Between a Štafeta chunk the as-of rule keeps (doc 1) and one it drops (doc 4).
STF_QUERY = blend(unit_vec(_stf_seed(1, 1, 3)), unit_vec(_stf_seed(1, 4, 2)))


# --- build and drop --------------------------------------------------------------


def _run(cursor, sql: str, params=None) -> None:
    cursor.execute(sql, params or [])


def drop(connection) -> None:
    cursor = connection.cursor()
    try:
        for table in (MED, STF):
            _run(cursor, f"DROP TABLE IF EXISTS {table}")
        connection.commit()
    finally:
        cursor.close()


def create(connection) -> None:
    """Both tables, their indexes and their rows; drops them first."""
    drop(connection)
    cursor = connection.cursor()
    try:
        _run(cursor, MED_DDL)
        for statement in MED_INDEXES:
            _run(cursor, statement)
        _run(cursor, STF_DDL)
        for statement in STF_INDEXES:
            _run(cursor, statement)
        _seed(cursor)
        connection.commit()
    finally:
        cursor.close()


def _seed(cursor) -> None:
    rng = random.Random(42)
    insert_med = (
        f"INSERT INTO {MED} (EvidenceId, PatientId, ResourceType, ResourceId, ChunkIndex, "
        "CharStart, CharEnd, DocDate, Title, Department, Text, EmbeddingModel, Embedding) "
        f"VALUES (?,?,?,?,?,?,?,?,?,?,?,?,TO_VECTOR(?, DOUBLE, {DIM}))"
    )
    n = 0
    for pi, pid in enumerate(MED_PATIENTS):
        for ri in range(5):
            rid = f"{pid}-doc{ri:02d}"
            date = dt.date(2026, 9, 1) + dt.timedelta(days=rng.randint(0, 40))
            for ci in range(4):
                text = f"{rng.choice(_SNIPPETS)}; {rng.choice(_SNIPPETS)} (#{n})"
                for mi, model in enumerate(MODELS):
                    _run(
                        cursor,
                        insert_med,
                        [
                            f"{rid}#{ci}",
                            pid,
                            "DocumentReference",
                            rid,
                            ci,
                            ci * 100,
                            ci * 100 + 90,
                            date.isoformat(),
                            f"Zpráva {rid}",
                            rng.choice(["A-ONK", "RDG", "PAT-B"]),
                            text,
                            model,
                            vlit(unit_vec(_med_seed(pi, ri, ci, mi))),
                        ],
                    )
                    n += 1

    insert_stf = (
        f"INSERT INTO {STF} (DocRef, DocId, PatientRef, Seq, Title, AuthoredOn, ReceivedOn, "
        "Text, CharStart, CharEnd, EmbeddingModel, Embedding) "
        f"VALUES (?,?,?,?,?,?,?,?,?,?,?,TO_VECTOR(?, DOUBLE, {DIM}))"
    )
    m = 0
    for pi, pid in enumerate(STF_PATIENTS):
        for di in range(5):
            ref = f"DocumentReference/{pi}{di}"
            authored = dt.date(2026, 9, 1) + dt.timedelta(days=di * 9 + pi)
            received = dt.datetime(
                authored.year, authored.month, authored.day, 12
            ) + dt.timedelta(days=1 + pi % 2)
            for si in range(5):
                text = f"{rng.choice(_SNIPPETS)}; {rng.choice(_SNIPPETS)} (#{m})"
                _run(
                    cursor,
                    insert_stf,
                    [
                        ref,
                        f"doc_{pi}{di}",
                        pid,
                        si,
                        f"Dokument {pi}{di}",
                        authored.isoformat(),
                        received.isoformat(sep=" "),
                        text,
                        si * 100,
                        si * 100 + 90,
                        STF_MODEL,
                        vlit(unit_vec(_stf_seed(pi, di, si))),
                    ],
                )
                m += 1
    # The as-of rule must bite on the text leg too: a late chunk that matches the query.
    _run(
        cursor,
        f"UPDATE {STF} SET Text = ? WHERE DocRef = ? AND Seq = 2",
        ["apixaban vysazen před biopsií (pozdní zpráva)", "DocumentReference/14"],
    )


# --- the teams' own queries --------------------------------------------------------


def _rows(cursor, sql: str, params) -> List[tuple]:
    cursor.execute(sql, params)
    return [tuple(r) for r in cursor.fetchall()]


def med_team_topk(cursor, model: str, patient: str = "P02", k: int = 5):
    """MedTeam's `_select`: cosine, patient and model as real-column filters."""
    sql = (
        f"SELECT TOP {k} EvidenceId, VECTOR_COSINE(Embedding, TO_VECTOR(?, DOUBLE)) AS Score "
        f"FROM {MED} WHERE PatientId = ? AND EmbeddingModel = ? ORDER BY Score DESC"
    )
    rows = _rows(cursor, sql, [vlit(MED_QUERY), patient, model])
    return [(r[0], round(float(r[1]), 6)) for r in rows]


def stf_team_vector_topk(cursor, as_of: bool = True, k: int = 5):
    """Štafeta's vector leg with its scope (patient, model, and the as-of rule)."""
    scope = "PatientRef = ? AND EmbeddingModel = ?"
    params = [vlit(STF_QUERY), "Patient/2", STF_MODEL]
    if as_of:
        scope += " AND AuthoredOn < ? AND ReceivedOn <= ?"
        params += [AS_OF_AUTHORED_BEFORE, AS_OF_RECEIVED_UNTIL]
    sql = (
        f"SELECT TOP {k} DocRef, Seq, VECTOR_COSINE(Embedding, TO_VECTOR(?, DOUBLE)) AS Score "
        f"FROM {STF} WHERE {scope} ORDER BY Score DESC"
    )
    rows = _rows(cursor, sql, params)
    return [(f"{r[0]}#{r[1]}", round(float(r[2]), 6)) for r in rows]


def stf_team_ifind(cursor, as_of: bool = True) -> List[str]:
    """Štafeta's text leg: prefix terms against its Czech iFind index, same scope."""
    scope = "PatientRef = ? AND EmbeddingModel = ?"
    params: list = ["Patient/2", STF_MODEL]
    if as_of:
        scope += " AND AuthoredOn < ? AND ReceivedOn <= ?"
        params += [AS_OF_AUTHORED_BEFORE, AS_OF_RECEIVED_UNTIL]
    sql = (
        f"SELECT DocRef, Seq FROM {STF} WHERE {scope} "
        "AND %ID %FIND search_index(DocChunkText, ?, 0)"
    )
    rows = _rows(cursor, sql, params + [STF_IFIND_TERMS])
    return sorted(f"{r[0]}#{r[1]}" for r in rows)


# --- the read-only check (SC-004) ----------------------------------------------------


def snapshot(connection) -> Dict[str, Tuple]:
    """Columns, indexes and row counts of both tables, to compare before and after."""
    cursor = connection.cursor()
    try:
        out: Dict[str, Tuple] = {}
        for schema, table in ((MED_SCHEMA, MED_TABLE), (STF_SCHEMA, STF_TABLE)):
            cursor.execute(
                "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION",
                [schema, table],
            )
            columns = tuple(tuple(r) for r in cursor.fetchall())
            cursor.execute(
                "SELECT DISTINCT INDEX_NAME FROM INFORMATION_SCHEMA.INDEXES "
                "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
                [schema, table],
            )
            indexes = tuple(sorted(str(r[0]) for r in cursor.fetchall()))
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
            count = int(cursor.fetchone()[0])
            out[f"{schema}.{table}"] = (columns, indexes, count)
        return out
    finally:
        cursor.close()
