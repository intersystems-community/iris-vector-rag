"""PL-13: the fallback id for a document with no id must not depend on the process.

`hash()` of a string is salted per process (PYTHONHASHSEED), so an id built from it
differs between runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types

from iris_vector_rag.pipelines.multi_query_rrf import MultiQueryRRFPipeline


def _idless(content):
    """A document-like object with no id (a Document refuses an empty one)."""
    return types.SimpleNamespace(id="", page_content=content, metadata={})


SNIPPET = (
    ""
    "import types;from iris_vector_rag.pipelines.multi_query_rrf import fallback_document_id;"
    "print(fallback_document_id(types.SimpleNamespace(id='', page_content='the same text', metadata={})))"
)


def _id_in_fresh_process(seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run(
        [sys.executable, "-c", SNIPPET],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_fallback_id_is_identical_across_processes():
    assert _id_in_fresh_process("1") == _id_in_fresh_process("2")


def test_fallback_id_differs_for_different_content():
    from iris_vector_rag.pipelines.multi_query_rrf import fallback_document_id

    a = _idless("x" * 100 + "a")
    b = _idless("x" * 100 + "b")
    assert fallback_document_id(a) != fallback_document_id(b)


def test_fusion_keeps_idless_documents_apart_and_merges_duplicates():
    pipeline = MultiQueryRRFPipeline.__new__(MultiQueryRRFPipeline)
    pipeline.rrf_k = 60
    one = _idless("first")
    two = _idless("second")
    again = _idless("first")
    fused = pipeline._reciprocal_rank_fusion([[one, two], [again]], top_k=5)
    assert sorted(d.page_content for d in fused) == ["first", "second"]
