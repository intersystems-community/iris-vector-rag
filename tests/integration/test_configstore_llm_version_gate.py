"""An IRIS without ConfigStore says so, with its version (spec 093, US2 scenario 6).

Runs on the project container (2026.1), which predates AI Hub's
%ConfigStore.Configuration, so it runs everywhere, GitHub CI included.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.common.configstore_llm import read_llm_entry
from iris_vector_rag.exceptions import LLMConfigError

pytestmark = pytest.mark.integration


def test_iris_without_configstore_names_the_version(iris_connection):
    import iris

    version = iris.createIRIS(iris_connection).classMethodValue(
        "%SYSTEM.Version", "GetNumber"
    )
    if not str(version).startswith("2026.1"):
        pytest.fail(f"this test expects the 2026.1 project container, got {version}")

    with pytest.raises(LLMConfigError) as caught:
        read_llm_entry("opsreview", iris_connection)

    message = str(caught.value)
    assert "AI.LLM.opsreview" in message
    assert "2026.3" in message
    assert str(version) in message
