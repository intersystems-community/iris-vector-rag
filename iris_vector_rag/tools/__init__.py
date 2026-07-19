"""
iris_vector_rag.tools — optional submodule requiring iris_llm.

Only loaded on explicit ``from iris_vector_rag.tools import ...`` — never
auto-imported by ``import iris_vector_rag``.

Raises ``ImportError`` with install instructions if iris_llm is not installed.
"""

from __future__ import annotations

try:
    from iris_vector_rag.tools.graphrag import GraphRAGToolSet

    __all__ = ["GraphRAGToolSet"]
except ImportError as _e:
    raise ImportError(
        "iris_vector_rag.tools requires the iris_llm wheel. "
        "Install with: pip install iris_llm-*.whl\n"
        f"Original error: {_e}"
    ) from _e
