"""
langchain-intersystems spike runner — answers Q1–Q8 from LANGCHAIN_INTERSYSTEMS_SPIKE.md

Run with:
    OPENAI_API_KEY=sk-... python docs/spike_runner.py

Container: iris-langchain-spike on port 13972
"""

import datetime
import inspect
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Connection config — dedicated spike container
# ---------------------------------------------------------------------------
CONNECT_KWARGS = {
    "hostname": os.environ.get("IRIS_HOSTNAME", "localhost"),
    "port": int(os.environ.get("IRIS_PORT", "13972")),
    "namespace": os.environ.get("IRIS_NAMESPACE", "USER"),
    "username": os.environ.get("IRIS_USERNAME", "_SYSTEM"),
    "password": os.environ.get("IRIS_PASSWORD", "SYS"),
}

RESULTS = {}


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def record(key, value):
    RESULTS[key] = value
    print(f"  → {key}: {value}")


# ---------------------------------------------------------------------------
# Embeddings — use HuggingFace if no OpenAI key (Q7 covers both)
# ---------------------------------------------------------------------------
def make_embeddings(provider="auto"):
    if provider == "openai" or (
        provider == "auto" and os.environ.get("OPENAI_API_KEY")
    ):
        from langchain_openai import OpenAIEmbeddings

        print("  Using OpenAIEmbeddings (text-embedding-3-small, dim=1536)")
        return OpenAIEmbeddings(model="text-embedding-3-small"), "openai", 1536
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        print("  Using HuggingFaceEmbeddings (all-MiniLM-L6-v2, dim=384, local)")
        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return emb, "huggingface", 384


# ---------------------------------------------------------------------------
# Wait for container
# ---------------------------------------------------------------------------
def wait_for_iris(max_wait=120):
    import iris.dbapi as dbapi

    print(f"\nWaiting for IRIS on port {CONNECT_KWARGS['port']}...")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            conn = dbapi.connect(**CONNECT_KWARGS)
            conn.close()
            print("  IRIS is ready ✓")
            return True
        except Exception as e:
            print(f"  not ready ({e.__class__.__name__}), retrying...")
            time.sleep(3)
    print("  IRIS never became ready — aborting")
    return False


# ---------------------------------------------------------------------------
# Q1 — Schema inspection
# ---------------------------------------------------------------------------
def q1_schema():
    section("Q1 — Schema: How does IRISVectorStore store metadata?")
    import iris.dbapi as dbapi
    from langchain_core.documents import Document
    from langchain_intersystems import IRISVectorStore

    embeddings, provider, dim = make_embeddings()

    vs = IRISVectorStore(
        embeddings,
        connect_kwargs=CONNECT_KWARGS,
        collection_name="spike_q1",
        replace_collection=True,
    )

    docs = [
        Document(
            page_content="IRIS vector search is fast.",
            metadata={
                "source": "test.pdf",
                "page": 1,
                "score": 0.95,
                "published": datetime.date(2024, 1, 15),
                "active": True,
            },
        ),
        Document(
            page_content="Metadata filtering maps to SQL WHERE clauses.",
            metadata={
                "source": "iris.pdf",
                "page": 2,
                "score": 0.88,
                "published": datetime.date(2024, 6, 1),
                "active": False,
            },
        ),
    ]
    vs.add_documents(docs)
    print("  Added 2 docs with mixed metadata types")

    # Inspect schema via DBAPI
    conn = dbapi.connect(**CONNECT_KWARGS)
    cur = conn.cursor()
    cur.execute(
        "SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH "
        "FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_NAME = 'spike_q1' "
        "ORDER BY ORDINAL_POSITION"
    )
    cols = cur.fetchall()
    cur.close()
    conn.close()

    print("\n  Table columns:")
    col_info = []
    for row in cols:
        print(f"    {row[0]:30s} {row[1]:20s} {row[2] or ''}")
        col_info.append({"name": row[0], "type": row[1], "max_len": row[2]})

    record("q1_schema_columns", col_info)
    record(
        "q1_schema_pattern",
        (
            "per-column"
            if any("source" in c["name"].lower() for c in col_info)
            else "json-blob"
        ),
    )


# ---------------------------------------------------------------------------
# Q2 — FHIR metadata compatibility
# ---------------------------------------------------------------------------
def q2_fhir_metadata():
    section("Q2 — FHIR Metadata Compatibility")
    from langchain_core.documents import Document
    from langchain_intersystems import IRISVectorStore, Predicate

    embeddings, provider, dim = make_embeddings()

    vs = IRISVectorStore(
        embeddings,
        connect_kwargs=CONNECT_KWARGS,
        collection_name="spike_q2_fhir",
        replace_collection=True,
    )

    docs = [
        Document(
            page_content=f"Patient {i} radiology note: chest X-ray shows clear lung fields.",
            metadata={
                "resource_type": "DiagnosticReport",
                "subject": f"Patient/1040682{i}",
                "status": "final",
                "category": "radiology" if i % 2 == 0 else "laboratory",
                "effective_date": datetime.date(2024, 1 + (i % 12), 1 + (i % 28)),
            },
        )
        for i in range(1, 11)
    ]

    try:
        vs.add_documents(docs)
        record("q2_add_fhir_docs", "SUCCESS — 10 FHIR-like docs added")
    except Exception as e:
        record("q2_add_fhir_docs", f"FAILED: {e}")
        return

    # Patient-scoped search
    try:
        r = vs.similarity_search("chest", filter={"subject": "Patient/10406821"}, k=5)
        record("q2_filter_by_subject", f"SUCCESS — {len(r)} results")
    except Exception as e:
        record("q2_filter_by_subject", f"FAILED: {e}")

    # STARTS_WITH on subject
    try:
        r = vs.similarity_search(
            "chest", filter={"subject": (Predicate.STARTS_WITH, "Patient/")}, k=5
        )
        record("q2_starts_with_patient", f"SUCCESS — {len(r)} results")
    except Exception as e:
        record("q2_starts_with_patient", f"FAILED: {e}")

    # Date range filter
    try:
        r = vs.similarity_search(
            "radiology",
            filter={
                Predicate.AND: [
                    {"resource_type": "DiagnosticReport"},
                    {
                        "effective_date": (
                            Predicate.BETWEEN,
                            datetime.date(2024, 1, 1),
                            datetime.date(2024, 6, 30),
                        )
                    },
                ]
            },
            k=5,
        )
        record("q2_date_range_filter", f"SUCCESS — {len(r)} results")
    except Exception as e:
        record("q2_date_range_filter", f"FAILED: {e}")

    # Category filter
    try:
        r = vs.similarity_search("report", filter={"category": "radiology"}, k=5)
        record("q2_category_filter", f"SUCCESS — {len(r)} results (expected ~5)")
    except Exception as e:
        record("q2_category_filter", f"FAILED: {e}")


# ---------------------------------------------------------------------------
# Q3 — Performance benchmarks
# ---------------------------------------------------------------------------
def q3_performance():
    section("Q3 — Performance on Realistic Data")
    import random
    import string

    from langchain_core.documents import Document
    from langchain_intersystems import IRISVectorStore

    embeddings, provider, dim = make_embeddings()

    def random_text(n=100):
        words = [
            "chest",
            "pain",
            "fever",
            "radiology",
            "lab",
            "patient",
            "diagnosis",
            "treatment",
            "medication",
            "blood",
            "pressure",
            "heart",
            "lung",
            "kidney",
        ]
        return " ".join(random.choices(words, k=n))

    # 1k doc ingest
    docs_1k = [
        Document(
            page_content=random_text(),
            metadata={
                "category": random.choice(["radiology", "lab", "notes"]),
                "doc_id": i,
            },
        )
        for i in range(200)
    ]

    vs = IRISVectorStore(
        embeddings,
        connect_kwargs=CONNECT_KWARGS,
        collection_name="spike_q3_perf",
        replace_collection=True,
    )

    t0 = time.time()
    # batch in chunks of 100 to avoid timeouts
    for i in range(0, len(docs_1k), 100):
        vs.add_documents(docs_1k[i : i + 100])
    ingest_time = time.time() - t0
    record("q3_ingest_200_seconds", round(ingest_time, 2))
    record("q3_ingest_200_docs_per_sec", round(200 / ingest_time, 1))

    # Unfiltered queries
    t0 = time.time()
    N = 20
    for _ in range(N):
        vs.similarity_search("chest X-ray findings", k=5)
    unfiltered_ms = (time.time() - t0) / N * 1000
    record("q3_unfiltered_query_p50_ms", round(unfiltered_ms, 1))

    # Filtered queries
    t0 = time.time()
    for _ in range(N):
        vs.similarity_search(
            "chest X-ray findings", k=5, filter={"category": "radiology"}
        )
    filtered_ms = (time.time() - t0) / N * 1000
    record("q3_filtered_query_p50_ms", round(filtered_ms, 1))
    record(
        "q3_filter_overhead_pct",
        round((filtered_ms - unfiltered_ms) / unfiltered_ms * 100, 1),
    )


# ---------------------------------------------------------------------------
# Q4 — replace_collection behavior
# ---------------------------------------------------------------------------
def q4_replace_collection():
    section("Q4 — replace_collection=True Behavior")
    import iris.dbapi as dbapi
    from langchain_core.documents import Document
    from langchain_intersystems import IRISVectorStore

    embeddings, _, _ = make_embeddings()

    docs = [
        Document(page_content=f"Document {i}", metadata={"idx": i}) for i in range(5)
    ]

    # First init + add
    vs = IRISVectorStore(
        embeddings,
        connect_kwargs=CONNECT_KWARGS,
        collection_name="spike_q4",
        replace_collection=True,
    )
    vs.add_documents(docs)

    conn = dbapi.connect(**CONNECT_KWARGS)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM spike_q4")
    count1 = cur.fetchone()[0]

    # Second init with replace_collection=True
    vs2 = IRISVectorStore(
        embeddings,
        connect_kwargs=CONNECT_KWARGS,
        collection_name="spike_q4",
        replace_collection=True,
    )
    vs2.add_documents(docs[:2])  # only 2 docs this time

    cur.execute("SELECT COUNT(*) FROM spike_q4")
    count2 = cur.fetchone()[0]
    cur.close()
    conn.close()

    record("q4_count_after_first_add", count1)
    record("q4_count_after_replace_and_add_2", count2)
    record(
        "q4_replace_behavior",
        "DROP+RECREATE (count reset)" if count2 == 2 else f"APPEND (count={count2})",
    )

    # Test replace_collection=False on existing
    try:
        vs3 = IRISVectorStore(
            embeddings,
            connect_kwargs=CONNECT_KWARGS,
            collection_name="spike_q4",
            replace_collection=False,
        )
        vs3.add_documents(docs[2:4])
        conn2 = dbapi.connect(**CONNECT_KWARGS)
        cur2 = conn2.cursor()
        cur2.execute("SELECT COUNT(*) FROM spike_q4")
        count3 = cur2.fetchone()[0]
        cur2.close()
        conn2.close()
        record("q4_count_after_no_replace_add_2_more", count3)
        record(
            "q4_no_replace_behavior",
            "APPEND" if count3 == 4 else f"unexpected={count3}",
        )
    except Exception as e:
        record("q4_no_replace_error", str(e))


# ---------------------------------------------------------------------------
# Q5 — MMR support
# ---------------------------------------------------------------------------
def q5_mmr():
    section("Q5 — MMR (max_marginal_relevance_search) Support")
    from langchain_core.documents import Document
    from langchain_intersystems import IRISVectorStore

    embeddings, _, _ = make_embeddings()

    vs = IRISVectorStore(
        embeddings,
        connect_kwargs=CONNECT_KWARGS,
        collection_name="spike_q5_mmr",
        replace_collection=True,
    )
    docs = [
        Document(
            page_content=f"Medical document {i} about chest conditions.",
            metadata={"i": i},
        )
        for i in range(10)
    ]
    vs.add_documents(docs)

    # Check method existence
    has_mmr = hasattr(vs, "max_marginal_relevance_search")
    record("q5_mmr_method_exists", has_mmr)

    if has_mmr:
        try:
            results = vs.max_marginal_relevance_search(
                "chest findings", k=5, fetch_k=20
            )
            record("q5_mmr_works", f"SUCCESS — {len(results)} results")
        except NotImplementedError as e:
            record("q5_mmr_works", f"NotImplementedError: {e}")
        except Exception as e:
            record("q5_mmr_works", f"FAILED: {type(e).__name__}: {e}")
    else:
        record("q5_mmr_works", "method not present")


# ---------------------------------------------------------------------------
# Q6 — Async support
# ---------------------------------------------------------------------------
def q6_async():
    section("Q6 — Async Support")
    import asyncio

    from langchain_core.documents import Document
    from langchain_intersystems import IRISVectorStore

    embeddings, _, _ = make_embeddings()

    vs = IRISVectorStore(
        embeddings,
        connect_kwargs=CONNECT_KWARGS,
        collection_name="spike_q6_async",
        replace_collection=True,
    )
    docs = [Document(page_content="Async test document.", metadata={})]
    vs.add_documents(docs)

    # asimilarity_search
    has_async_search = hasattr(vs, "asimilarity_search")
    record("q6_asimilarity_search_exists", has_async_search)
    if has_async_search:
        try:
            results = asyncio.run(vs.asimilarity_search("test", k=3))
            record("q6_asimilarity_search_works", f"SUCCESS — {len(results)} results")
        except NotImplementedError as e:
            record("q6_asimilarity_search_works", f"NotImplementedError: {e}")
        except Exception as e:
            record("q6_asimilarity_search_works", f"FAILED: {type(e).__name__}: {e}")

    # aadd_documents
    has_aadd = hasattr(vs, "aadd_documents")
    record("q6_aadd_documents_exists", has_aadd)
    if has_aadd:
        try:
            asyncio.run(
                vs.aadd_documents(
                    [Document(page_content="async add test", metadata={})]
                )
            )
            record("q6_aadd_documents_works", "SUCCESS")
        except NotImplementedError as e:
            record("q6_aadd_documents_works", f"NotImplementedError: {e}")
        except Exception as e:
            record("q6_aadd_documents_works", f"FAILED: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Q7 — Embedding model flexibility
# ---------------------------------------------------------------------------
def q7_embeddings():
    section("Q7 — Embedding Model Flexibility")
    from langchain_core.documents import Document
    from langchain_intersystems import IRISVectorStore

    results_by_provider = {}

    providers_to_test = [("huggingface", "all-MiniLM-L6-v2")]
    if os.environ.get("OPENAI_API_KEY"):
        providers_to_test.insert(0, ("openai", "text-embedding-3-small"))

    for provider_name, model_name in providers_to_test:
        try:
            if provider_name == "openai":
                from langchain_openai import OpenAIEmbeddings

                emb = OpenAIEmbeddings(model=model_name)
            else:
                from langchain_huggingface import HuggingFaceEmbeddings

                emb = HuggingFaceEmbeddings(model_name=model_name)

            vs = IRISVectorStore(
                emb,
                connect_kwargs=CONNECT_KWARGS,
                collection_name=f"spike_q7_{provider_name}",
                replace_collection=True,
            )
            docs = [
                Document(page_content="Embedding provider test document.", metadata={})
            ]
            vs.add_documents(docs)
            r = vs.similarity_search("test", k=1)
            results_by_provider[provider_name] = (
                f"SUCCESS (model={model_name}, results={len(r)})"
            )
        except Exception as e:
            results_by_provider[provider_name] = f"FAILED: {type(e).__name__}: {e}"

    for k, v in results_by_provider.items():
        record(f"q7_{k}_embeddings", v)


# ---------------------------------------------------------------------------
# Q8 — Comparison to langchain-iris (caretdev/Dmitry)
# ---------------------------------------------------------------------------
def q8_langchain_iris_comparison():
    section("Q8 — Comparison to langchain-iris (Dmitry/caretdev)")

    # Check if installed
    try:
        import langchain_iris

        version = getattr(langchain_iris, "__version__", "unknown")
        record("q8_langchain_iris_installed", f"YES — version={version}")

        # Check for metadata filtering (Predicate system)
        has_predicate = hasattr(langchain_iris, "Predicate") or any(
            "predicate" in str(m).lower() for m in dir(langchain_iris)
        )
        record("q8_langchain_iris_has_predicate_system", has_predicate)

        # Check similarity metrics
        has_similarity_metric = any(
            "metric" in str(m).lower() for m in dir(langchain_iris)
        )
        record("q8_langchain_iris_has_similarity_metric", has_similarity_metric)

    except ImportError:
        record("q8_langchain_iris_installed", "NO — not installed in spike venv")

    # Introspect langchain-intersystems itself
    from langchain_intersystems import IRISVectorStore, Predicate, SimilarityMetric

    record(
        "q8_langchain_intersystems_predicates",
        [p for p in dir(Predicate) if not p.startswith("_")],
    )
    record(
        "q8_langchain_intersystems_similarity_metrics",
        [m for m in dir(SimilarityMetric) if not m.startswith("_")],
    )

    # Source: DB-API vs Native API
    try:
        src = inspect.getsource(IRISVectorStore.__init__)
        uses_dbapi = "dbapi" in src.lower() or "connect_kwargs" in src
        uses_native = "iris.cls" in src.lower() or "native" in src.lower()
        record(
            "q8_langchain_intersystems_connection",
            "DB-API" if uses_dbapi else "Native API" if uses_native else "unknown",
        )
    except Exception:
        record("q8_langchain_intersystems_connection", "could not inspect source")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("langchain-intersystems spike runner")
    print(f"IRIS: {CONNECT_KWARGS['hostname']}:{CONNECT_KWARGS['port']}")
    print(
        f"OpenAI key: {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET (using HuggingFace)'}"
    )

    if not wait_for_iris():
        sys.exit(1)

    errors = []
    for fn in [
        q1_schema,
        q2_fhir_metadata,
        q3_performance,
        q4_replace_collection,
        q5_mmr,
        q6_async,
        q7_embeddings,
        q8_langchain_iris_comparison,
    ]:
        try:
            fn()
        except Exception as e:
            import traceback

            print(f"\n  !! {fn.__name__} CRASHED: {e}")
            traceback.print_exc()
            errors.append(f"{fn.__name__}: {e}")

    section("RESULTS SUMMARY")
    for k, v in RESULTS.items():
        print(f"  {k}: {v}")

    # Write JSON for SPIKE_NOTES
    out = "/Users/tdyar/ws/iris-vector-rag-private/docs/spike_results.json"
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\n  Results written to {out}")

    if errors:
        print(f"\n  {len(errors)} question(s) had errors:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("\n  All questions completed successfully ✓")
