"""No fake vectors: the embedding manager's failure policy (G6) and correctness fixes.

OE-01  a failed model load never becomes random vectors
OE-02  every backend failing never becomes hash vectors
OE-03  the dimension is the active backend's, not a default
OE-04  HuggingFace pooling ignores padding
OE-10  the configured batch size is applied
OE-13  clear_cache finds what the manager cached; stats divide by what they count
OE-14  the OpenAI client has a timeout and chunks its input
"""

import sys
import types
from typing import Any, Dict, List

import pytest

from iris_vector_rag.embeddings import manager as em
from iris_vector_rag.embeddings.manager import EmbeddingManager
from iris_vector_rag.exceptions import EmbeddingError


class _Config:
    """The two calls EmbeddingManager makes on a configuration manager."""

    def __init__(self, embeddings: Dict[str, Any], **flat: Any):
        self._embeddings = embeddings
        self._flat = flat

    def get(self, key: str, default: Any = None) -> Any:
        if key == "embeddings":
            return self._embeddings
        return self._flat.get(key, default)


@pytest.fixture(autouse=True)
def _clean_cache():
    em._SENTENCE_TRANSFORMER_CACHE.clear()
    em._CACHE_STATS.clear()
    yield
    em._SENTENCE_TRANSFORMER_CACHE.clear()
    em._CACHE_STATS.clear()


class FakeSentenceTransformer:
    """A model that loads and encodes; records every construction."""

    loads: List[str] = []

    def __init__(self, name, device="cpu", dimension=8):
        type(self).loads.append(name)
        self.dimension = dimension

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, convert_to_tensor=False, show_progress_bar=False, **_):
        return [[float(len(t))] * self.dimension for t in texts]


@pytest.fixture
def working_st(monkeypatch):
    FakeSentenceTransformer.loads = []
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    return FakeSentenceTransformer


@pytest.fixture
def broken_st(monkeypatch):
    calls: List[str] = []

    def refuse(name, device="cpu"):
        calls.append(name)
        raise OSError(f"cannot reach the hub for {name}")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", refuse)
    return calls


class FakeOpenAI:
    instances: List["FakeOpenAI"] = []

    def __init__(self, api_key=None, timeout=None, **kw):
        self.api_key, self.timeout = api_key, timeout
        self.requests: List[List[str]] = []
        self.width = 1536
        self.fail = False
        type(self).instances.append(self)
        self.embeddings = types.SimpleNamespace(create=self._create)

    def _create(self, input, model):
        if self.fail:
            raise RuntimeError("openai is down")
        self.requests.append(list(input))
        data = [types.SimpleNamespace(embedding=[0.5] * self.width) for _ in input]
        return types.SimpleNamespace(data=data)


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.instances = []
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    return FakeOpenAI


# ---------------------------------------------------------------------------
# OE-01: a failed load is an error, not a stub
# ---------------------------------------------------------------------------


def test_failed_model_load_raises_and_names_the_backend(broken_st):
    mgr = EmbeddingManager(
        _Config({"primary_backend": "sentence_transformers", "fallback_backends": []})
    )
    with pytest.raises(EmbeddingError) as info:
        mgr.embed_texts(["hello"])
    message = str(info.value)
    assert "sentence_transformers" in message
    assert "cannot reach the hub" in message
    assert "offline" in message.lower()


def test_failed_load_is_not_cached_as_a_model(broken_st):
    mgr = EmbeddingManager(
        _Config({"primary_backend": "sentence_transformers", "fallback_backends": []})
    )
    with pytest.raises(EmbeddingError):
        mgr.embed_texts(["a"])
    assert em._SENTENCE_TRANSFORMER_CACHE == {}


def test_failed_load_falls_back_to_a_configured_real_backend(broken_st, fake_openai):
    mgr = EmbeddingManager(
        _Config(
            {
                "primary_backend": "sentence_transformers",
                "fallback_backends": ["openai"],
                "openai": {"api_key": "k", "model_name": "text-embedding-ada-002"},
            }
        )
    )
    vectors = mgr.embed_texts(["hello"])
    assert len(vectors[0]) == 1536
    assert fake_openai.instances[-1].requests == [["hello"]]


def test_every_backend_failing_reports_every_reason(broken_st, monkeypatch):
    def down(**kwargs):
        raise RuntimeError("openai is down")

    monkeypatch.setattr("openai.OpenAI", down)
    mgr = EmbeddingManager(
        _Config(
            {
                "primary_backend": "sentence_transformers",
                "fallback_backends": ["openai"],
                "openai": {"api_key": "k"},
            }
        )
    )
    with pytest.raises(EmbeddingError) as info:
        mgr.embed_texts(["hello"])
    message = str(info.value)
    assert "sentence_transformers" in message and "openai" in message
    assert "cannot reach the hub" in message and "openai is down" in message


# ---------------------------------------------------------------------------
# OE-02: every backend failing is an error, not hash vectors
# ---------------------------------------------------------------------------


def test_primary_failing_at_call_time_with_no_fallback_raises(working_st):
    mgr = EmbeddingManager(
        _Config({"primary_backend": "sentence_transformers", "fallback_backends": []})
    )

    def explode(texts):
        raise RuntimeError("GPU fell over")

    mgr._embedding_functions["sentence_transformers"] = explode
    with pytest.raises(EmbeddingError, match="GPU fell over"):
        mgr.embed_texts(["x"])


def test_no_fallback_function_remains():
    assert not hasattr(EmbeddingManager, "_create_fallback_function")


def test_unknown_backend_is_an_error_not_silence():
    mgr = EmbeddingManager(
        _Config({"primary_backend": "nonesuch", "fallback_backends": []})
    )
    with pytest.raises(EmbeddingError, match="nonesuch"):
        mgr.embed_texts(["x"])


# ---------------------------------------------------------------------------
# The test embedder is reachable only by name
# ---------------------------------------------------------------------------


def test_provider_stub_gives_the_test_embedder_at_the_configured_dimension(broken_st):
    mgr = EmbeddingManager(_Config({"provider": "stub", "dimension": 12}))
    vectors = mgr.embed_texts(["a", "b"])
    assert [len(v) for v in vectors] == [12, 12]
    assert mgr.get_embedding_dimension() == 12
    assert mgr.model_key == "stub"
    assert broken_st == [], "the stub provider must not touch sentence-transformers"


def test_primary_backend_stub_is_the_same_thing():
    mgr = EmbeddingManager(_Config({"primary_backend": "stub", "dimension": 5}))
    assert len(mgr.embed_text("x")) == 5


def test_stub_without_a_dimension_is_refused():
    with pytest.raises(EmbeddingError, match="dimension"):
        EmbeddingManager(_Config({"provider": "stub"})).embed_texts(["x"])


def test_stub_cannot_be_a_fallback(broken_st):
    with pytest.raises(ValueError, match="stub"):
        EmbeddingManager(
            _Config(
                {
                    "primary_backend": "sentence_transformers",
                    "fallback_backends": ["stub"],
                    "dimension": 4,
                }
            )
        )


def test_the_default_configuration_never_yields_the_test_embedder(broken_st):
    mgr = EmbeddingManager(_Config({}))
    with pytest.raises(EmbeddingError):
        mgr.embed_texts(["x"])


# ---------------------------------------------------------------------------
# OE-03 and G6 point 5: one dimension, the active backend's own
# ---------------------------------------------------------------------------


def test_openai_primary_reports_the_openai_width_not_384(fake_openai):
    mgr = EmbeddingManager(
        _Config(
            {
                "primary_backend": "openai",
                "fallback_backends": [],
                "openai": {"api_key": "k", "model_name": "text-embedding-ada-002"},
            }
        )
    )
    assert mgr.get_embedding_dimension() == 1536
    assert fake_openai.instances[-1].requests == [], "a known model needs no probe"


def test_sentence_transformers_width_comes_from_the_model(working_st):
    mgr = EmbeddingManager(
        _Config({"sentence_transformers": {"model_name": "my-org/wide-model"}})
    )
    assert mgr.get_embedding_dimension() == 8


def test_unknown_openai_model_is_probed_once(fake_openai):
    mgr = EmbeddingManager(
        _Config(
            {
                "primary_backend": "openai",
                "fallback_backends": [],
                "openai": {"api_key": "k", "model_name": "brand-new-model"},
            }
        )
    )
    client = fake_openai.instances[-1]
    client.width = 2048
    assert mgr.get_embedding_dimension() == 2048
    assert mgr.get_embedding_dimension() == 2048
    assert len(client.requests) == 1


def test_configured_dimension_that_matches_is_accepted(working_st):
    mgr = EmbeddingManager(_Config({"dimension": 8}))
    assert mgr.get_embedding_dimension() == 8


def test_configured_dimension_that_disagrees_with_the_model_raises(working_st):
    mgr = EmbeddingManager(_Config({"dimension": 384}))
    with pytest.raises(EmbeddingError, match="384.*8|8.*384"):
        mgr.get_embedding_dimension()


def test_a_fallback_of_a_different_width_is_refused(broken_st, fake_openai):
    """The dimension authority is one value; a backend that disagrees is a failure."""
    mgr = EmbeddingManager(
        _Config(
            {
                "primary_backend": "sentence_transformers",
                "fallback_backends": ["openai"],
                "openai": {"api_key": "k", "model_name": "text-embedding-ada-002"},
                "dimension": 384,
            }
        )
    )
    with pytest.raises(EmbeddingError, match="1536"):
        mgr.embed_texts(["x"])


def test_dimension_is_computed_once(working_st):
    mgr = EmbeddingManager(_Config({}))
    model = em._get_cached_sentence_transformer("all-MiniLM-L6-v2", "cpu")
    asked = []
    original = model.get_sentence_embedding_dimension
    model.get_sentence_embedding_dimension = lambda: asked.append(1) or original()
    assert mgr.get_embedding_dimension() == 8
    assert mgr.get_embedding_dimension() == 8
    assert len(asked) == 1


# ---------------------------------------------------------------------------
# Production code carries no mock detection and no second copy of the stub
# ---------------------------------------------------------------------------


def test_a_mock_model_is_not_special_cased(monkeypatch):
    from unittest.mock import Mock

    mock_model = Mock()
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", Mock(return_value=mock_model)
    )
    assert em._get_cached_sentence_transformer("m", "cpu") is mock_model
    assert em._SENTENCE_TRANSFORMER_CACHE["m:cpu"] is mock_model


def test_no_stub_sentence_transformer_in_the_package():
    import pathlib

    import iris_vector_rag

    root = pathlib.Path(iris_vector_rag.__file__).parent
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if "StubSentenceTransformer" in p.read_text()
        or "unittest.mock" in p.read_text()
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# OE-04: HuggingFace pooling
# ---------------------------------------------------------------------------


def _hf_fakes(monkeypatch):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        vocab = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

        def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
            rows = [[self.vocab[w] for w in t.split()] for t in texts]
            longest = max(len(r) for r in rows)
            ids = torch.tensor([r + [0] * (longest - len(r)) for r in rows])
            mask = torch.tensor([[1] * len(r) + [0] * (longest - len(r)) for r in rows])
            return {"input_ids": ids, "attention_mask": mask}

    class Model:
        table = torch.tensor(
            [
                [9.0, 9.0, 9.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
        config = types.SimpleNamespace(hidden_size=3)

        def __call__(self, input_ids, attention_mask):
            return types.SimpleNamespace(last_hidden_state=self.table[input_ids])

    monkeypatch.setattr(
        "iris_vector_rag.common.huggingface_utils.download_huggingface_model",
        lambda name: (Tokenizer(), Model()),
    )


def test_huggingface_vector_does_not_depend_on_batch_neighbours(monkeypatch):
    _hf_fakes(monkeypatch)
    mgr = EmbeddingManager(
        _Config(
            {
                "primary_backend": "huggingface",
                "fallback_backends": [],
                "huggingface": {"model_name": "fake"},
            }
        )
    )
    alone = mgr.embed_texts(["a b"])[0]
    with_long_neighbour = mgr.embed_texts(["a b", "a b c d e a b c"])[0]
    assert alone == pytest.approx(with_long_neighbour)


def test_huggingface_vectors_are_unit_length(monkeypatch):
    _hf_fakes(monkeypatch)
    mgr = EmbeddingManager(
        _Config(
            {
                "primary_backend": "huggingface",
                "fallback_backends": [],
                "huggingface": {"model_name": "fake"},
            }
        )
    )
    for vector in mgr.embed_texts(["a b", "c d e"]):
        assert sum(x * x for x in vector) == pytest.approx(1.0)


def test_huggingface_width_comes_from_the_model_config(monkeypatch):
    _hf_fakes(monkeypatch)
    mgr = EmbeddingManager(
        _Config(
            {
                "primary_backend": "huggingface",
                "fallback_backends": [],
                "huggingface": {"model_name": "fake"},
            }
        )
    )
    assert mgr.get_embedding_dimension() == 3


# ---------------------------------------------------------------------------
# OE-10: batch size
# ---------------------------------------------------------------------------


def test_manager_splits_a_large_call_by_the_configured_batch_size():
    seen: List[int] = []

    mgr = EmbeddingManager(
        _Config({"provider": "stub", "dimension": 4, "batch_size": 100})
    )
    inner = mgr._embedding_functions["stub"]

    def spying(texts):
        seen.append(len(texts))
        return inner(texts)

    mgr._embedding_functions["stub"] = spying
    vectors = mgr.embed_texts([f"t{i}" for i in range(250)])
    assert len(vectors) == 250
    assert seen == [100, 100, 50]


def test_iris_embedding_entry_applies_the_config_batch_size():
    from iris_vector_rag.embeddings import iris_embedding as ie
    from iris_vector_rag.testing.embedder import TestEmbedder

    seen: List[int] = []

    class Recording(TestEmbedder):
        def encode(self, texts, **kw):
            seen.append(len(texts))
            return super().encode(texts, **kw)

    ie.configure_embedding(
        name="batchy", model_name="m-batchy", batch_size=32, device_preference="cpu"
    )
    em._SENTENCE_TRANSFORMER_CACHE["m-batchy:cpu"] = Recording(dimension=4)
    result = ie.embed_texts("batchy", [f"t{i}" for i in range(70)])
    assert len(result.embeddings) == 70
    assert seen == [32, 32, 6]


# ---------------------------------------------------------------------------
# OE-13: cache clearing and stats
# ---------------------------------------------------------------------------


def test_clear_cache_removes_whatever_device_key_the_model_was_cached_under():
    from iris_vector_rag.embeddings import iris_embedding as ie

    ie.configure_embedding(
        name="cuda-config", model_name="m-cuda", device_preference="cuda"
    )
    em._SENTENCE_TRANSFORMER_CACHE["m-cuda:cuda"] = object()
    em._SENTENCE_TRANSFORMER_CACHE["other:cuda"] = object()
    result = em.clear_cache("cuda-config")
    assert result.models_cleared == 1
    assert "m-cuda:cuda" not in em._SENTENCE_TRANSFORMER_CACHE
    assert "other:cuda" in em._SENTENCE_TRANSFORMER_CACHE


def test_average_embedding_time_divides_by_embedding_calls():
    em._record_cache_miss("cfg", "cpu", 10.0)
    em._record_embedding_time("cfg", 30.0)
    em._record_cache_hit("cfg")
    em._record_embedding_time("cfg", 10.0)
    em._record_cache_hit("cfg")
    em._record_cache_hit("cfg")  # hits with no timed call must not dilute the average
    assert em.get_cache_stats("cfg").avg_embedding_time_ms == pytest.approx(20.0)


def test_aggregate_average_embedding_time_is_reported():
    em._record_cache_miss("a", "cpu", 1.0)
    em._record_embedding_time("a", 10.0)
    em._record_cache_miss("b", "cpu", 1.0)
    em._record_embedding_time("b", 30.0)
    assert em.get_cache_stats().avg_embedding_time_ms == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# OE-14 (manager half): the OpenAI client
# ---------------------------------------------------------------------------


def test_openai_client_has_a_timeout(fake_openai):
    EmbeddingManager(
        _Config(
            {
                "primary_backend": "openai",
                "fallback_backends": [],
                "openai": {"api_key": "k"},
            }
        )
    )
    assert fake_openai.instances[-1].timeout == 60


def test_openai_timeout_is_configurable(fake_openai):
    EmbeddingManager(
        _Config(
            {
                "primary_backend": "openai",
                "fallback_backends": [],
                "openai": {"api_key": "k", "timeout": 5},
            }
        )
    )
    assert fake_openai.instances[-1].timeout == 5


def test_openai_input_is_chunked(fake_openai):
    mgr = EmbeddingManager(
        _Config(
            {
                "primary_backend": "openai",
                "fallback_backends": [],
                "openai": {
                    "api_key": "k",
                    "batch_size": 10,
                    "model_name": "text-embedding-ada-002",
                },
            }
        )
    )
    vectors = mgr.embed_texts([f"t{i}" for i in range(25)])
    assert len(vectors) == 25
    assert [len(r) for r in fake_openai.instances[-1].requests] == [10, 10, 5]


def test_openai_short_response_is_an_error(fake_openai):
    mgr = EmbeddingManager(
        _Config(
            {
                "primary_backend": "openai",
                "fallback_backends": [],
                "openai": {"api_key": "k", "model_name": "text-embedding-ada-002"},
            }
        )
    )
    client = fake_openai.instances[-1]
    client.embeddings.create = lambda input, model: types.SimpleNamespace(data=[])
    with pytest.raises(EmbeddingError, match="returned 0"):
        mgr.embed_texts(["x"])
