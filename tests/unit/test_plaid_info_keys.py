"""PLAIDSearcher reads `Graph.KG.PLAIDSearch.Info` in the keys it actually returns.

Info (ivg 4.1.0, and the vendored copy in `colbert_iris/sp/`) reports the centroid count
as `nlist` and the document count as `indexed`; `nCentroids` is only the `^PLAID`
subscript those are read from. The `nCentroids` read this replaced always answered 0, so
every search on a built index raised PLAIDNotBuiltError -- hidden until ivg 4.x, because
the integration suite errored in setup before reaching a search.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iris_vector_rag.pipelines.colbert_iris.plaid import PLAIDSearcher

BUILT_4X = {"type": "plaid", "indexed": 200, "nlist": 8, "dim": 16, "totalTokens": 1}


def _searcher(info=None, error=None):
    engine = MagicMock()
    if error is not None:
        engine.plaid_info.side_effect = error
    else:
        engine.plaid_info.return_value = info
    return PLAIDSearcher(object(), index_name="idx", _engine=engine), engine


def test_a_built_4x_index_is_built():
    searcher, engine = _searcher(BUILT_4X)

    assert searcher._centroids_built() is True
    engine.plaid_info.assert_called_once_with("idx")


@pytest.mark.parametrize(
    "info",
    [
        {},  # Info's answer for a name with no ^PLAID entry
        {"type": "plaid", "indexed": 0, "nlist": 0, "dim": 16, "totalTokens": 0},
    ],
)
def test_an_absent_or_empty_index_is_not_built(info):
    searcher, _ = _searcher(info)

    assert searcher._centroids_built() is False


def test_the_global_subscript_name_is_not_the_json_key():
    """`nCentroids` names the ^PLAID node, never a key Info returns."""
    searcher, _ = _searcher({"nCentroids": 8, "nDocs": 200})

    assert searcher._centroids_built() is False


def test_a_failed_info_call_is_not_built():
    searcher, _ = _searcher(error=RuntimeError("class does not exist"))

    assert searcher._centroids_built() is False
