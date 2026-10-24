"""Does IRIS use an attached table's HNSW index, by measure and with a filter? (092 SC-007)

Builds a throwaway 5,000-row table with an HNSW index (Distance='Cosine') on the
project container, then times top-10 searches through `IRISVectorStore`: cosine and dot
product, with and without a patient filter, and reads the plan of each query shape with
EXPLAIN. Prints a table; drops the table afterwards. Not a test: the numbers depend on
the machine.

    IRIS_HOST=localhost IRIS_PORT=51972 IRIS_NAMESPACE=USER IRIS_USERNAME=_SYSTEM \\
    IRIS_PASSWORD=SYS python scripts/perf/byot_hnsw_probe.py
"""

from __future__ import annotations

import os
import random
import statistics
import time

import iris

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

TABLE = "BYOTPerf.Chunk"
ROWS, DIM, PATIENTS, RUNS, K = 5000, 384, 20, 20, 10


def unit(rng: random.Random) -> list:
    v = [rng.gauss(0, 1) for _ in range(DIM)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def vlit(v) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def build(conn) -> None:
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
    cur.execute(
        f"CREATE TABLE {TABLE} (ChunkId VARCHAR(20), PatientId VARCHAR(10), "
        f"Body VARCHAR(200), Vec VECTOR(DOUBLE, {DIM}))"
    )
    rng = random.Random(1)
    rows = [
        [f"c{i}", f"P{i % PATIENTS:02d}", f"chunk {i}", vlit(unit(rng))]
        for i in range(ROWS)
    ]
    cur.executemany(
        f"INSERT INTO {TABLE} VALUES (?, ?, ?, TO_VECTOR(?, DOUBLE, {DIM}))", rows
    )
    cur.execute(f"CREATE INDEX ChunkPatient ON {TABLE} (PatientId)")
    cur.execute(
        f"CREATE INDEX ChunkHNSW ON TABLE {TABLE} (Vec) AS HNSW(Distance='Cosine')"
    )
    conn.commit()
    cur.close()


def plan_uses_hnsw(conn, function: str, filtered: bool, query) -> bool:
    """EXPLAIN takes no parameters through the DB-API, so the plan is read with literals."""
    where = " WHERE PatientId = 'P03'" if filtered else ""
    sql = (
        f"EXPLAIN SELECT TOP {K} ChunkId, {function}(Vec, TO_VECTOR('{vlit(query)}', "
        f"DOUBLE, {DIM})) AS score FROM {TABLE}{where} ORDER BY score DESC"
    )
    cur = conn.cursor()
    try:
        cur.execute(sql)
        plan = " ".join(str(r[0]) for r in cur.fetchall())
    finally:
        cur.close()
    return "CHUNKHNSW" in plan.upper()


def main() -> None:
    conn = iris.connect(
        os.environ.get("IRIS_HOST", "localhost"),
        int(os.environ.get("IRIS_PORT", "51972")),
        os.environ.get("IRIS_NAMESPACE", "USER"),
        os.environ.get("IRIS_USERNAME", "_SYSTEM"),
        os.environ.get("IRIS_PASSWORD", "SYS"),
    )
    build(conn)
    try:
        config = ConfigurationManager()
        common = dict(
            table_name=TABLE,
            id_col="ChunkId",
            text_col="Body",
            embedding_col="Vec",
            filter_columns=["PatientId"],
        )
        rng = random.Random(7)
        queries = [unit(rng) for _ in range(RUNS)]
        print(
            f"{ROWS} rows, dim {DIM}, HNSW Distance='Cosine', top {K}, {RUNS} queries"
        )
        print(
            f"{'metric':8} {'filter':8} {'median ms':>10} {'p90 ms':>8} {'plan uses HNSW':>15}"
        )
        for metric, function in (
            ("cosine", "VECTOR_COSINE"),
            ("dot", "VECTOR_DOT_PRODUCT"),
        ):
            store = IRISVectorStore(
                ConnectionManager(config), config, metric=metric, **common
            )
            for filtered in (False, True):
                where = {"PatientId": "P03"} if filtered else None
                store.similarity_search_by_embedding(queries[0], K, where=where)  # warm
                times = []
                for q in queries:
                    start = time.perf_counter()
                    store.similarity_search_by_embedding(q, K, where=where)
                    times.append((time.perf_counter() - start) * 1000)
                times.sort()
                uses = plan_uses_hnsw(conn, function, filtered, queries[0])
                print(
                    f"{metric:8} {'patient' if filtered else 'none':8} "
                    f"{statistics.median(times):10.1f} {times[int(0.9 * len(times)) - 1]:8.1f} "
                    f"{str(uses):>15}"
                )
    finally:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
        conn.commit()
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
